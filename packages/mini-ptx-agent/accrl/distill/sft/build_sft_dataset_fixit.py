#!/usr/bin/env python3
"""Build fix-it SFT data from pair-reasoning JSONL plus one kernel-pairs CSV.

Each output row is a Qwen-style chat:

    system    original trajectory system prompt
    user      original first task prompt
    assistant wrong/error kernel
    user      wrong/error kernel evaluation log
    assistant <think>teacher reasoning</think> plus the correct kernel

The final assistant message is the only message marked with step_loss_mask=1.
The wrong kernel, error log, source trajectory, and source turn come from the
CSV emitted by collect_success_kernel_pairs.py.

If ``--max-tokens`` is provided, rows whose final composed Qwen chat exceeds
that token budget are excluded from the output.

By default, each row is first rendered through the tokenizer's chat template and
parsed back into ``messages`` before writing. For Qwen-style templates this
materializes the template's handling of prior assistant thinking instead of
preserving raw source trajectory text. Pass ``--no-normalize-with-chat-template``
to preserve raw messages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_FORMAT = "fixit_qwen_original_prompt_wrong_kernel_log_to_reasoned_correct_kernel"
FORBIDDEN_PARQUET_TOKENS = ("<my_reasoning>", "</my_reasoning>")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    data_root = os.environ.get("PTXBENCH_DATA_ROOT")
    if data_root and "eval_runs" in path.parts:
        eval_runs_index = path.parts.index("eval_runs")
        relocated = Path(data_root).expanduser().joinpath(*path.parts[eval_runs_index:])
        if relocated.exists():
            return relocated
    if path.is_absolute():
        return path
    return base_dir / path


def path_key(value: str, *, base_dir: Path) -> str:
    return str(resolve_path(value, base_dir=base_dir))


def read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.read_text(errors="replace")


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def load_kernel_pairs_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {
            "wrong_kernel_path",
            "wrong_log_path",
            "wrong_trajectory_path",
            "wrong_turn",
            "correct_kernel_path",
        }
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required CSV columns: {', '.join(sorted(missing))}")

        rows = list(reader)

    by_correct: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = path_key(row["correct_kernel_path"], base_dir=path.parent)
        if key in by_correct:
            duplicates.append(key)
        by_correct[key] = row
    if duplicates:
        raise ValueError(f"{path}: duplicate correct_kernel_path rows, first few: {duplicates[:5]}")
    return by_correct


def csv_wrong_turn(row: dict[str, str]) -> int:
    raw = (row.get("wrong_turn") or "").strip()
    if not raw:
        raise ValueError("kernel-pairs CSV row has blank wrong_turn")
    return int(raw)


def csv_wrong_trajectory_path(row: dict[str, str], csv_dir: Path) -> Path:
    raw = (row.get("wrong_trajectory_path") or "").strip()
    if not raw:
        raise ValueError("kernel-pairs CSV row has blank wrong_trajectory_path")
    return resolve_path(raw, base_dir=csv_dir)


def role_content(message: dict[str, Any]) -> tuple[str, str]:
    role = str(message.get("role") or "")
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True)
    return role, content


def original_system_and_task(trajectory: dict[str, Any], trajectory_path: Path) -> tuple[str, str]:
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{trajectory_path}: missing messages list")

    system_prompt = ""
    task_prompt = ""
    for message in messages:
        role, content = role_content(message)
        if role == "system" and not system_prompt:
            system_prompt = content
        elif role == "user" and not task_prompt:
            task_prompt = content
        if system_prompt and task_prompt:
            break
    if not system_prompt:
        raise ValueError(f"{trajectory_path}: missing original system message")
    if not task_prompt:
        raise ValueError(f"{trajectory_path}: missing original first user message")
    return system_prompt, task_prompt


def assistant_and_feedback_from_trajectory(
    trajectory: dict[str, Any],
    *,
    wrong_turn: int,
    trajectory_path: Path,
) -> tuple[str, str]:
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{trajectory_path}: missing messages list")

    assistant_seen = 0
    for index, message in enumerate(messages):
        role, content = role_content(message)
        if role != "assistant":
            continue
        if assistant_seen != wrong_turn:
            assistant_seen += 1
            continue

        feedback = ""
        for next_message in messages[index + 1 :]:
            next_role, next_content = role_content(next_message)
            if next_role == "user":
                feedback = next_content
                break
        if not feedback:
            raise ValueError(f"{trajectory_path}: no user feedback after assistant turn {wrong_turn}")
        return content, feedback
    raise ValueError(f"{trajectory_path}: assistant turn {wrong_turn} out of range")


def code_block(code: str) -> str:
    stripped = code.strip()
    if "```" in stripped:
        return stripped
    return f"```cpp\n{stripped}\n```"


def prior_assistant_content(
    *,
    trajectory: dict[str, Any],
    trajectory_path: Path,
    wrong_kernel_path: Path,
    wrong_turn: int,
) -> str:
    try:
        assistant, _ = assistant_and_feedback_from_trajectory(
            trajectory, wrong_turn=wrong_turn, trajectory_path=trajectory_path
        )
        return assistant
    except Exception:
        return code_block(read_text(wrong_kernel_path, "wrong kernel"))


def feedback_content(
    *,
    trajectory: dict[str, Any],
    trajectory_path: Path,
    wrong_log_path: Path,
    wrong_turn: int,
) -> str:
    try:
        _, feedback = assistant_and_feedback_from_trajectory(
            trajectory, wrong_turn=wrong_turn, trajectory_path=trajectory_path
        )
        return feedback
    except Exception:
        return read_text(wrong_log_path, "wrong log")


def build_final_assistant(reasoning: str, correct_code: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n```cpp\n{correct_code.strip()}\n```"



def assert_no_forbidden_parquet_tokens(row_id: str, messages: list[dict[str, Any]]) -> None:
    for index, message in enumerate(messages):
        content = message.get("content") or ""
        if not isinstance(content, str):
            continue
        for token in FORBIDDEN_PARQUET_TOKENS:
            if token in content:
                raise ValueError(
                    f"forbidden token {token!r} in output message {index} for row {row_id}"
                )

def make_message(role: str, content: str, step_loss_mask: int) -> dict[str, Any]:
    return {"role": role, "content": content, "step_loss_mask": step_loss_mask}


def record_correct_kernel_path(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    value = metadata.get("correct_kernel_path")
    if not value:
        raise ValueError("reasoning record metadata is missing correct_kernel_path")
    return str(value)


def build_row(
    *,
    record: dict[str, Any],
    csv_row: dict[str, str],
    csv_dir: Path,
    source_label: str,
) -> dict[str, Any]:
    reasoning = (record.get("reasoning") or "").strip()
    if not reasoning:
        raise ValueError("empty reasoning field: reasoning")

    wrong_kernel_path = resolve_path(csv_row["wrong_kernel_path"], base_dir=csv_dir)
    wrong_log_path = resolve_path(csv_row["wrong_log_path"], base_dir=csv_dir)
    correct_kernel_path = resolve_path(csv_row["correct_kernel_path"], base_dir=csv_dir)
    wrong_turn = csv_wrong_turn(csv_row)
    trajectory_path = csv_wrong_trajectory_path(csv_row, csv_dir)
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"missing wrong trajectory: {trajectory_path}")

    trajectory = load_json(trajectory_path)
    if not isinstance(trajectory, dict):
        raise ValueError(f"{trajectory_path}: expected JSON object")
    system_prompt, task_prompt = original_system_and_task(trajectory, trajectory_path)
    wrong_assistant = prior_assistant_content(
        trajectory=trajectory,
        trajectory_path=trajectory_path,
        wrong_kernel_path=wrong_kernel_path,
        wrong_turn=wrong_turn,
    )
    wrong_feedback = feedback_content(
        trajectory=trajectory,
        trajectory_path=trajectory_path,
        wrong_log_path=wrong_log_path,
        wrong_turn=wrong_turn,
    )
    correct_code = read_text(correct_kernel_path, "correct kernel")

    messages = [
        make_message("system", system_prompt, 0),
        make_message("user", task_prompt, 0),
        make_message("assistant", wrong_assistant, 0),
        make_message("user", wrong_feedback, 0),
        make_message("assistant", build_final_assistant(reasoning, correct_code), 1),
    ]

    rp_metadata = dict(record.get("metadata") or {})
    run_id = rp_metadata.get("run_id") or Path(csv_row.get("exp_dir", "")).name or "unknown_run"
    exp_id = rp_metadata.get("exp_id") or csv_row.get("trajectory_id") or "unknown_exp"
    correct_version = csv_row.get("correct_kernel_version") or rp_metadata.get("correct_kernel_version", "")
    row_id = f"{run_id}_{exp_id}_kv{correct_version}_{source_label}"
    assert_no_forbidden_parquet_tokens(row_id, messages)
    metadata = {
        **rp_metadata,
        **{k: v for k, v in csv_row.items() if k not in rp_metadata},
        "source_label": source_label,
        "source_reasoning_model": rp_metadata.get("reasoning_model", ""),
        "reasoning_field": "reasoning",
        "target_source_field": "reasoning",
        "target_format": TARGET_FORMAT,
        "teacher_hidden_thinking_available": bool((record.get("thinking") or "").strip()),
        "wrong_trajectory_path": csv_row["wrong_trajectory_path"],
        "wrong_turn": wrong_turn,
        "wrong_kernel_path": csv_row["wrong_kernel_path"],
        "wrong_log_path": csv_row["wrong_log_path"],
        "correct_kernel_path": csv_row["correct_kernel_path"],
        "system_prompt_sha256": sha256_text(system_prompt),
        "task_prompt_sha256": sha256_text(task_prompt),
        "wrong_assistant_sha256": sha256_text(wrong_assistant),
        "wrong_feedback_sha256": sha256_text(wrong_feedback),
        "reasoning_sha256": sha256_text(reasoning),
        "correct_kernel_sha256": file_sha256(correct_kernel_path),
        "reasoning_chars": len(reasoning),
        "correct_kernel_chars": len(correct_code),
        "num_messages": len(messages),
    }
    return {"id": row_id, "messages": messages, "metadata": metadata}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to write parquet; use .jsonl output if unavailable") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        print(f"warning: failed to load tokenizer from {model_path}: {exc}")
        return None


def token_count(tokenizer: Any, messages: list[dict[str, Any]]) -> int | None:
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in messages],
            add_special_tokens=False,
            tokenize=True,
        )
        if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
            return len(encoded["input_ids"])
        return len(encoded)
    except Exception:
        return None


def chat_template_payload(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def parse_qwen_chat_template(rendered: str) -> list[dict[str, str]]:
    start_token = "<|im_start|>"
    end_token = "<|im_end|>"
    messages: list[dict[str, str]] = []
    pos = 0

    while pos < len(rendered):
        start = rendered.find(start_token, pos)
        if start == -1:
            if rendered[pos:].strip():
                raise ValueError("unexpected trailing text after final chat-template message")
            break
        if rendered[pos:start].strip():
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

        pos = content_end + len(end_token)
        if pos < len(rendered) and rendered[pos] == "\n":
            pos += 1

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
            f"chat-template roundtrip changed message count: "
            f"{len(messages)} -> {len(parsed_messages)}"
        )

    normalized: list[dict[str, Any]] = []
    char_delta = 0
    for index, (original, parsed) in enumerate(zip(messages, parsed_messages, strict=True)):
        if parsed["role"] != original["role"]:
            raise ValueError(
                f"chat-template roundtrip changed role at message {index}: "
                f"{original['role']!r} -> {parsed['role']!r}"
            )
        updated = dict(original)
        updated["content"] = parsed["content"]
        normalized.append(updated)
        char_delta += len(parsed["content"]) - len(original["content"])
    return normalized, char_delta


def export_rows_from_chat_template(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            messages, char_delta = export_messages_from_chat_template(
                tokenizer, row["messages"]
            )
        except Exception as exc:
            raise SystemExit(f"failed to export chat-template messages for row {row['id']}: {exc}") from exc
        updated = dict(row)
        metadata = dict(updated.get("metadata") or {})
        metadata.update(
            {
                "chat_template_messages_exported": True,
                "chat_template_message_char_delta": char_delta,
                "chat_template_messages_sha256": sha256_text(
                    json.dumps(chat_template_payload(messages), sort_keys=True)
                ),
            }
        )
        updated["messages"] = messages
        updated["metadata"] = metadata
        normalized_rows.append(updated)
    return normalized_rows


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    idx = round((len(values) - 1) * pct)
    return sorted(values)[idx]


def summarize_ints(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": sorted_values[-1],
        "mean": statistics.fmean(values),
    }


def dataset_stats(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    prompt_chars = [sum(len(m["content"]) for m in row["messages"][:-1]) for row in rows]
    target_chars = [len(row["messages"][-1]["content"]) for row in rows]
    message_counts = [len(row["messages"]) for row in rows]
    token_lengths = [token_count(tokenizer, row["messages"]) for row in rows]
    token_lengths_int = [x for x in token_lengths if x is not None]
    stats: dict[str, Any] = {
        "num_rows": len(rows),
        "prompt_chars": summarize_ints(prompt_chars),
        "target_chars": summarize_ints(target_chars),
        "messages": summarize_ints(message_counts),
    }
    if token_lengths_int:
        stats["total_tokens"] = summarize_ints(token_lengths_int)
        stats["top_10_longest"] = sorted(
            (
                {
                    "id": row["id"],
                    "total_tokens": tokens,
                    "target_chars": len(row["messages"][-1]["content"]),
                }
                for row, tokens in zip(rows, token_lengths, strict=True)
                if tokens is not None
            ),
            key=lambda item: item["total_tokens"],
            reverse=True,
        )[:10]
    return stats


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
                "correct_kernel_path": metadata.get("correct_kernel_path", ""),
                "wrong_trajectory_path": metadata.get("wrong_trajectory_path", ""),
                "wrong_turn": metadata.get("wrong_turn", ""),
                "reason": "over_max_tokens",
            }
        )
    return kept, filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", "--pairs", dest="pairs_jsonl", type=Path, required=True)
    parser.add_argument("--kernel-pairs-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .parquet or .jsonl")
    parser.add_argument("--source-label", default="fixit")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Only write rows whose final composed Qwen chat is at most this many tokens.",
    )
    parser.set_defaults(normalize_with_chat_template=True)
    parser.add_argument(
        "--no-normalize-with-chat-template",
        dest="normalize_with_chat_template",
        action="store_false",
        help="Write raw constructed messages without chat-template normalization.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle output rows after all filtering and before writing.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed used with --shuffle for reproducible output order.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on the first dropped row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    records = load_jsonl(args.pairs_jsonl)
    kernel_pairs = load_kernel_pairs_csv(args.kernel_pairs_csv)

    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_ids: Counter[str] = Counter()
    for line_index, record in enumerate(records, 1):
        try:
            correct_key = path_key(
                record_correct_kernel_path(record),
                base_dir=args.pairs_jsonl.parent,
            )
            csv_row = kernel_pairs.get(correct_key)
            if csv_row is None:
                raise ValueError(f"no kernel-pairs CSV row for correct_kernel_path={correct_key}")
            row = build_row(
                record=record,
                csv_row=csv_row,
                csv_dir=args.kernel_pairs_csv.parent,
                source_label=args.source_label,
            )
            seen_ids[row["id"]] += 1
            if seen_ids[row["id"]] > 1:
                row["id"] = f"{row['id']}_{seen_ids[row['id']]}"
            rows.append(row)
        except Exception as exc:
            item = {"line": line_index, "reason": str(exc)}
            metadata = record.get("metadata") if isinstance(record, dict) else None
            if isinstance(metadata, dict):
                item["correct_kernel_path"] = metadata.get("correct_kernel_path", "")
                item["exp_id"] = metadata.get("exp_id", "")
            if args.strict:
                raise
            dropped.append(item)
            print(f"drop line {line_index}: {exc}")

    if not rows:
        raise SystemExit("no rows exported")

    tokenizer = load_tokenizer(args.tokenizer)
    if (args.max_tokens is not None or args.normalize_with_chat_template) and tokenizer is None:
        raise SystemExit(
            "--max-tokens and default chat-template normalization require --tokenizer to load successfully; "
            "pass --no-normalize-with-chat-template to disable normalization"
        )

    if args.normalize_with_chat_template:
        rows = export_rows_from_chat_template(rows, tokenizer=tokenizer)

    max_token_filtered: list[dict[str, Any]] = []
    built_rows = len(rows)
    if args.max_tokens is not None:
        rows, max_token_filtered = filter_rows_by_max_tokens(
            rows,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )
        if not rows:
            raise SystemExit("no rows exported after applying --max-tokens")

    if args.shuffle:
        random.Random(args.shuffle_seed).shuffle(rows)

    if args.output.suffix == ".parquet":
        write_parquet(args.output, rows)
    elif args.output.suffix == ".jsonl":
        write_jsonl(args.output, rows)
    else:
        raise SystemExit("output must end with .parquet or .jsonl")

    stats = dataset_stats(rows, tokenizer)
    stats_path = args.output.with_suffix(args.output.suffix + ".length_stats.json")
    write_json(stats_path, stats)

    manifest = {
        "builder": "accrl.distill.sft.build_sft_dataset_fixit",
        "target_format": TARGET_FORMAT,
        "pairs_jsonl": str(args.pairs_jsonl),
        "kernel_pairs_csv": str(args.kernel_pairs_csv),
        "output": str(args.output),
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "normalize_with_chat_template": args.normalize_with_chat_template,
        "shuffle": args.shuffle,
        "shuffle_seed": args.shuffle_seed if args.shuffle else None,
        "pairs_jsonl_sha256": file_sha256(args.pairs_jsonl),
        "kernel_pairs_csv_sha256": file_sha256(args.kernel_pairs_csv),
        "output_sha256": file_sha256(args.output),
        "reasoning_field": "reasoning",
        "source_label": args.source_label,
        "input_records": len(records),
        "kernel_pair_rows": len(kernel_pairs),
        "built_rows_before_max_token_filter": built_rows,
        "output_rows": len(rows),
        "dropped_rows": len(dropped),
        "dropped": dropped,
        "max_token_filtered_rows": len(max_token_filtered),
        "max_token_filtered": max_token_filtered,
        "length_stats_path": str(stats_path),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    write_json(manifest_path, manifest)

    print(f"input records: {len(records)}")
    print(f"wrote rows:    {len(rows)} -> {args.output}")
    print(f"dropped rows:  {len(dropped)}")
    if args.normalize_with_chat_template:
        print("normalized with chat template: yes")
    if args.shuffle:
        print(f"shuffled rows: yes (seed={args.shuffle_seed})")
    if args.max_tokens is not None:
        print(f"max-token filtered rows: {len(max_token_filtered)}")
    print(f"manifest:      {manifest_path}")
    print(f"length stats:  {stats_path}")


if __name__ == "__main__":
    main()
