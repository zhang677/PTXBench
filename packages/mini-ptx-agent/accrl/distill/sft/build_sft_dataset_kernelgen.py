#!/usr/bin/env python3
"""Rebuild the three-message dataset used for the published ``kernelgen`` run.

Each reasoning record points at an original Gemini trajectory and turn.  The
training row intentionally keeps only the trajectory's system message, first
user message, and the selected Gemini answer.  Synthesized reasoning is
prepended to that answer inside Qwen ``<think>`` tags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to write parquet") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def load_tokenizer(model_name: str | None) -> Any:
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"warning: failed to load tokenizer {model_name}: {exc}")
        return None


def chat_template_payload(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def token_count(tokenizer: Any, messages: list[dict[str, Any]]) -> int | None:
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.apply_chat_template(
            chat_template_payload(messages),
            add_special_tokens=False,
            tokenize=True,
        )
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            return len(encoded["input_ids"])
        return len(encoded)
    except (RuntimeError, TypeError, ValueError):
        return None


def parse_qwen_chat_template(rendered: str) -> list[dict[str, str]]:
    start_token = "<|im_start|>"
    end_token = "<|im_end|>"
    messages: list[dict[str, str]] = []
    position = 0
    while position < len(rendered):
        start = rendered.find(start_token, position)
        if start == -1:
            if rendered[position:].strip():
                raise ValueError("unexpected trailing text after final chat-template message")
            break
        if rendered[position:start].strip():
            raise ValueError("unexpected text before chat-template message")
        role_start = start + len(start_token)
        role_end = rendered.find("\n", role_start)
        if role_end == -1:
            raise ValueError("chat-template message is missing role newline")
        role = rendered[role_start:role_end]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat-template role: {role!r}")
        content_start = role_end + 1
        content_end = rendered.find(end_token, content_start)
        if content_end == -1:
            raise ValueError(f"chat-template message for role {role!r} is missing end token")
        messages.append({"role": role, "content": rendered[content_start:content_end]})
        position = content_end + len(end_token)
        if position < len(rendered) and rendered[position] == "\n":
            position += 1
    return messages


def export_messages_from_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    rendered = tokenizer.apply_chat_template(
        chat_template_payload(messages),
        add_special_tokens=False,
        tokenize=False,
    )
    parsed_messages = parse_qwen_chat_template(rendered)
    if len(parsed_messages) != len(messages):
        raise ValueError(
            f"chat-template roundtrip changed message count: {len(messages)} -> {len(parsed_messages)}"
        )
    normalized: list[dict[str, Any]] = []
    character_delta = 0
    for index, (original, parsed) in enumerate(zip(messages, parsed_messages, strict=True)):
        if parsed["role"] != original["role"]:
            raise ValueError(
                f"chat-template roundtrip changed role at message {index}: "
                f"{original['role']!r} -> {parsed['role']!r}"
            )
        updated = dict(original)
        updated["content"] = parsed["content"]
        normalized.append(updated)
        character_delta += len(parsed["content"]) - len(original["content"])
    return normalized, character_delta


def export_rows_from_chat_template(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            messages, character_delta = export_messages_from_chat_template(tokenizer, row["messages"])
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"failed to export chat-template messages for row {row['id']}: {exc}") from exc
        updated = dict(row)
        metadata = dict(updated.get("metadata") or {})
        metadata.update(
            {
                "chat_template_messages_exported": True,
                "chat_template_content_char_delta": character_delta,
            }
        )
        updated["messages"] = messages
        updated["metadata"] = metadata
        normalized_rows.append(updated)
    return normalized_rows


def filter_rows_by_max_tokens(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        tokens = token_count(tokenizer, row["messages"])
        if tokens is None:
            raise SystemExit(f"failed to count tokens for row {row['id']}")
        if tokens <= max_tokens:
            kept.append(row)
            continue
        metadata = row.get("metadata") or {}
        filtered.append(
            {
                "id": row["id"],
                "total_tokens": tokens,
                "max_tokens": max_tokens,
                "trajectory_path": metadata.get("trajectory_path", ""),
                "turn": metadata.get("turn", ""),
                "definition_name": metadata.get("definition_name", ""),
                "reason": "over_max_tokens",
            }
        )
    return kept, filtered


def summarize_ints(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    values = sorted(values)

    def percentile(fraction: float) -> int:
        return values[round((len(values) - 1) * fraction)]

    return {
        "min": values[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def assistant_index_for_turn(turn: int) -> int:
    return 2 + 2 * turn


def make_target(reasoning: str, gemini_output: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n{gemini_output.strip()}"


def resolve_trajectory_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_file():
        return path
    data_root = os.environ.get("PTXBENCH_DATA_ROOT")
    if data_root and "eval_runs" in path.parts:
        eval_runs_index = path.parts.index("eval_runs")
        relocated = Path(data_root).expanduser().joinpath(*path.parts[eval_runs_index:])
        if relocated.is_file():
            return relocated
    return path


def convert_record(
    record: dict[str, Any],
    *,
    reasoning_field: str,
    include_distill_system: bool,
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    trajectory_path = resolve_trajectory_path(str(metadata["trajectory_path"]))
    turn = int(metadata["turn"])
    trajectory = load_json(trajectory_path)
    messages = trajectory.get("messages") or []
    assistant_index = assistant_index_for_turn(turn)
    if len(messages) <= assistant_index:
        raise ValueError(f"{trajectory_path}: missing assistant message for turn {turn}")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError(f"{trajectory_path}: expected first messages to be system,user")
    if messages[assistant_index].get("role") != "assistant":
        raise ValueError(f"{trajectory_path}: message {assistant_index} is not assistant")

    reasoning = (record.get(reasoning_field) or "").strip()
    if not reasoning:
        raise ValueError(f"empty reasoning field: {reasoning_field}")
    gemini_output = messages[assistant_index].get("content") or ""
    if not gemini_output.strip():
        raise ValueError(f"{trajectory_path}: empty Gemini output at turn {turn}")

    experiment_id = metadata.get("exp_id") or metadata.get("trajectory_id") or "unknown"
    metadata.update(
        {
            "exp_id": experiment_id,
            "reasoning_field": reasoning_field,
            "target_source_field": reasoning_field,
            "target_format": "qwen_think_wrapped_plus_gemini_output",
            "prompt_source": "gemini_system_and_first_user",
            "assistant_target_source": "trajectory_actual_gemini_output",
            "trajectory_assistant_index": assistant_index,
            "teacher_hidden_thinking_available": bool((record.get("thinking") or "").strip()),
            "system_prompt_sha256": sha256_text(messages[0].get("content", "")),
            "first_user_prompt_sha256": sha256_text(messages[1].get("content", "")),
            "gemini_output_sha256": sha256_text(gemini_output),
            "reasoning_sha256": sha256_text(reasoning),
            "reasoning_chars": len(reasoning),
            "gemini_output_chars": len(gemini_output),
        }
    )
    if include_distill_system:
        metadata["distill_system_prompt"] = record.get("system_prompt", "")
    row_id = (
        f"{metadata.get('run_id', 'unknown')}_{experiment_id}_"
        f"{metadata.get('definition_name', 'unknown')}_t{turn}"
    )
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": messages[0].get("content", "")},
            {"role": "user", "content": messages[1].get("content", "")},
            {"role": "assistant", "content": make_target(reasoning, gemini_output)},
        ],
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning-field", choices=["reasoning", "thinking"], default="reasoning")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--normalize-with-chat-template", action="store_true")
    parser.add_argument("--include-distill-system", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pairs.is_file():
        raise SystemExit(f"missing pairs JSONL: {args.pairs}")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for line_number, record in enumerate(load_jsonl(args.pairs), start=1):
        try:
            rows.append(
                convert_record(
                    record,
                    reasoning_field=args.reasoning_field,
                    include_distill_system=args.include_distill_system,
                )
            )
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            dropped.append({"line": line_number, "reason": str(exc)})
    if not rows:
        raise SystemExit("no rows exported")

    tokenizer = load_tokenizer(args.tokenizer)
    if (args.max_tokens is not None or args.normalize_with_chat_template) and tokenizer is None:
        raise SystemExit("--max-tokens and --normalize-with-chat-template require a loadable --tokenizer")
    if args.normalize_with_chat_template:
        rows = export_rows_from_chat_template(rows, tokenizer=tokenizer)

    built_rows = len(rows)
    max_token_filtered: list[dict[str, Any]] = []
    if args.max_tokens is not None:
        rows, max_token_filtered = filter_rows_by_max_tokens(
            rows,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )
        if not rows:
            raise SystemExit("no rows exported after applying --max-tokens")

    write_parquet(args.output, rows)
    token_lengths = [token_count(tokenizer, row["messages"]) for row in rows]
    stats = {
        "num_rows": len(rows),
        "num_dropped": len(dropped),
        "total_tokens": summarize_ints([length for length in token_lengths if length is not None]),
        "assistant_chars": summarize_ints([len(row["messages"][-1]["content"]) for row in rows]),
    }
    manifest = {
        "builder": "accrl.distill.sft.build_sft_dataset_kernelgen",
        "pairs_path": str(args.pairs),
        "pairs_sha256": file_sha256(args.pairs),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "reasoning_field": args.reasoning_field,
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "normalize_with_chat_template": args.normalize_with_chat_template,
        "target_format": "gemini_base_prompt_first_user_to_reasoned_gemini_output",
        "built_rows_before_max_token_filter": built_rows,
        "num_rows": len(rows),
        "num_dropped": len(dropped),
        "dropped": dropped,
        "max_token_filtered_rows": len(max_token_filtered),
        "max_token_filtered": max_token_filtered,
        "stats": stats,
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    write_json(args.output.with_suffix(".length_stats.json"), stats)
    print(f"wrote rows: {len(rows)} -> {args.output}")
    print(f"dropped: {len(dropped)}")
    print(f"max-token filtered rows: {len(max_token_filtered)}")
    print(f"manifest: {args.output.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
