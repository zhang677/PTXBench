#!/usr/bin/env python3
"""Build an SFT parquet from reasoning-pairs, cutting composed Qwen chats.

This is a single-source SFT builder like ``build_sft_dataset_single.py``, but
the token budget is applied to the final composed Qwen conversation, not to the
raw ``reasoning_pairs.input`` text. For each row:

1. Compose messages from ``--gemini-turns`` and ``--pairs``.
2. Count tokens with ``tokenizer.apply_chat_template(..., tokenize=True)``.
3. If the row is over ``--max-tokens``, remove prior Gemini turn pairs
   (assistant kernel + user verdict) from oldest to newest until the row is
   within budget or no prior turns remain.
4. Write the parquet plus a JSON cut report.

Only records whose ``metadata.definition_name`` starts with ``--def-prefix`` are
included. The report records both the pair turn and the resolved Gemini kernel
turn because the pair's turn value may be an ordinal into the conversation
rather than the exact ``turn`` value stored in ``gemini_turns``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

TurnKey = tuple[str, str, str, int]
SessionKey = tuple[str, str, str]


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def reasoning_pair_key(record: dict[str, Any]) -> TurnKey:
    metadata = record.get("metadata") or {}
    try:
        return (
            str(metadata["run_id"]),
            str(metadata["exp_id"]),
            str(metadata["definition_name"]),
            int(metadata["turn"]),
        )
    except KeyError as e:
        raise ValueError(f"reasoning_pairs record missing metadata key: {e}") from e


def index_reasoning_pairs(
    records: list[dict[str, Any]], source_name: str
) -> dict[TurnKey, dict[str, Any]]:
    out: dict[TurnKey, dict[str, Any]] = {}
    duplicates: Counter[TurnKey] = Counter()
    for record in records:
        key = reasoning_pair_key(record)
        duplicates[key] += 1
        if key not in out:
            out[key] = record
    dupes = [key for key, count in duplicates.items() if count > 1]
    if dupes:
        raise ValueError(
            f"{source_name} has duplicate canonical keys, first few: {dupes[:5]}"
        )
    return out


def index_gemini_turns_by_session(
    records: list[dict[str, Any]],
) -> dict[SessionKey, list[dict[str, Any]]]:
    by_session: dict[SessionKey, list[dict[str, Any]]] = {}
    seen: set[TurnKey] = set()
    duplicates: list[TurnKey] = []
    for record in records:
        sess = (
            str(record["run_id"]),
            str(record["exp_id"]),
            str(record["definition_name"]),
        )
        turn = int(record["turn"])
        key = (sess[0], sess[1], sess[2], turn)
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        by_session.setdefault(sess, []).append(record)
    if duplicates:
        raise ValueError(
            f"gemini_turns has duplicate (run_id, exp_id, definition_name, turn) entries, "
            f"first few: {duplicates[:5]}"
        )
    for turns in by_session.values():
        turns.sort(key=lambda r: int(r["turn"]))
    return by_session


def load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"warning: failed to load tokenizer from {model_path}: {e}")
        return None


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int | None:
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_special_tokens=False, tokenize=True
        )
        if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
            return len(encoded["input_ids"])
        return len(encoded)
    except Exception:
        return None


def require_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    count = token_count(tokenizer, messages)
    if count is None:
        raise RuntimeError("token counting failed; --tokenizer is required for cutting")
    return count


def build_qwen_think_target(reasoning: str, raw_assistant_content: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n{raw_assistant_content}"


def resolve_kernel_index(
    *,
    session_turns: list[dict[str, Any]],
    pair_turn: int,
) -> tuple[int, str]:
    exact = [
        idx for idx, turn in enumerate(session_turns) if int(turn["turn"]) == pair_turn
    ]
    if exact:
        return exact[0], "exact"
    if 0 <= pair_turn < len(session_turns):
        return pair_turn, "ordinal"
    raise ValueError(
        f"pair turn {pair_turn} is neither a Gemini turn id nor an ordinal "
        f"for session turn ids {[int(t['turn']) for t in session_turns]}"
    )


def compose_messages(
    *,
    session_turns: list[dict[str, Any]],
    kernel_index: int,
    reasoning: str,
    removed_prior_indices: set[int],
) -> list[dict[str, str]]:
    if not session_turns:
        raise ValueError("empty session_turns")
    if kernel_index >= len(session_turns):
        raise ValueError(
            f"kernel_index={kernel_index} out of range for session of length {len(session_turns)}"
        )

    first = session_turns[0]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": first["system_prompt"]},
        {"role": "user", "content": first["task_prompt"]},
    ]
    for idx in range(kernel_index):
        if idx in removed_prior_indices:
            continue
        t = session_turns[idx]
        messages.append({"role": "assistant", "content": t["raw_assistant_content"]})
        messages.append({"role": "user", "content": t["raw_feedback_content"]})

    final = build_qwen_think_target(
        reasoning, session_turns[kernel_index]["raw_assistant_content"]
    )
    messages.append({"role": "assistant", "content": final})
    return messages


def cut_messages_to_budget(
    *,
    tokenizer: Any,
    session_turns: list[dict[str, Any]],
    kernel_index: int,
    reasoning: str,
    max_tokens: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    removed: set[int] = set()
    messages = compose_messages(
        session_turns=session_turns,
        kernel_index=kernel_index,
        reasoning=reasoning,
        removed_prior_indices=removed,
    )
    before_tokens = require_token_count(tokenizer, messages)
    token_count_after = before_tokens
    cuts: list[dict[str, Any]] = []

    for prior_index in range(kernel_index):
        if token_count_after <= max_tokens:
            break
        removed.add(prior_index)
        messages = compose_messages(
            session_turns=session_turns,
            kernel_index=kernel_index,
            reasoning=reasoning,
            removed_prior_indices=removed,
        )
        token_count_after = require_token_count(tokenizer, messages)
        cuts.append(
            {
                "conversation_index": prior_index,
                "gemini_turn": int(session_turns[prior_index]["turn"]),
                "sections": ["Kernel", "Verdict"],
                "tokens_after_cut": token_count_after,
            }
        )

    info = {
        "before_tokens": before_tokens,
        "after_tokens": token_count_after,
        "already_within_budget": before_tokens <= max_tokens,
        "still_over_budget": token_count_after > max_tokens,
        "cuts": cuts,
        "included_prior_turns": [
            {
                "conversation_index": idx,
                "gemini_turn": int(session_turns[idx]["turn"]),
            }
            for idx in range(kernel_index)
            if idx not in removed
        ],
    }
    return messages, info


def make_messages_row(
    *,
    session_turns: list[dict[str, Any]],
    kernel_index: int,
    reasoning_record: dict[str, Any],
    reasoning: str,
    messages: list[dict[str, str]],
    cut_info: dict[str, Any],
    turn_resolution: str,
    source_model: str,
    source_label: str,
    reasoning_field: str,
) -> dict[str, Any]:
    gemini = session_turns[kernel_index]
    key = (
        str(gemini["run_id"]),
        str(gemini["exp_id"]),
        str(gemini["definition_name"]),
        int(gemini["turn"]),
    )
    rp_metadata = dict(reasoning_record.get("metadata") or {})
    metadata = {
        **rp_metadata,
        "source_reasoning_model": source_model,
        "source_label": source_label,
        "reasoning_field": reasoning_field,
        "target_source_field": reasoning_field,
        "target_format": "qwen_think_wrapped",
        "prompt_source": "gemini_turns_with_cut",
        "pair_metadata_turn": rp_metadata.get("turn"),
        "kernel_conversation_index": kernel_index,
        "kernel_gemini_turn": int(gemini["turn"]),
        "turn_resolution": turn_resolution,
        "num_prior_turns_before_cut": kernel_index,
        "num_prior_turns_after_cut": len(cut_info["included_prior_turns"]),
        "included_prior_turns": cut_info["included_prior_turns"],
        "cut_turns": cut_info["cuts"],
        "tokens_before_cut": cut_info["before_tokens"],
        "tokens_after_cut": cut_info["after_tokens"],
        "still_over_budget": cut_info["still_over_budget"],
        "teacher_hidden_thinking_available": bool(
            (reasoning_record.get("thinking") or "").strip()
        ),
        "system_prompt_sha256": sha256_text(session_turns[0]["system_prompt"]),
        "task_prompt_sha256": sha256_text(session_turns[0]["task_prompt"]),
        "kernel_code_sha256": sha256_text(gemini.get("kernel_code", "")),
        "reasoning_sha256": sha256_text(reasoning),
        "reasoning_chars": len(reasoning),
        "kernel_passed": gemini.get("passed"),
        "kernel_speedup": gemini.get("speedup"),
    }
    return {
        "id": f"{key[0]}_{key[1]}_{key[2]}_t{key[3]}_{source_label}",
        "messages": messages,
        "metadata": metadata,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow is required; run this through the Miles image") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


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
    reasoning_chars = [len(row["messages"][-1]["content"]) for row in rows]
    prompt_chars = [
        sum(len(m["content"]) for m in row["messages"][:-1]) for row in rows
    ]
    num_messages = [len(row["messages"]) for row in rows]
    token_lengths = [token_count(tokenizer, row["messages"]) for row in rows]
    token_lengths_int = [x for x in token_lengths if x is not None]
    stats: dict[str, Any] = {
        "num_rows": len(rows),
        "reasoning_chars": summarize_ints(reasoning_chars),
        "prompt_chars": summarize_ints(prompt_chars),
        "num_messages": summarize_ints(num_messages),
    }
    if token_lengths_int:
        stats["total_tokens"] = summarize_ints(token_lengths_int)
        longest = sorted(
            (
                {
                    "id": row["id"],
                    "total_tokens": tokens,
                    "reasoning_chars": len(row["messages"][-1]["content"]),
                }
                for row, tokens in zip(rows, token_lengths, strict=True)
                if tokens is not None
            ),
            key=lambda x: x["total_tokens"],
            reverse=True,
        )
        stats["top_10_longest"] = longest[:10]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemini-turns",
        type=Path,
        required=True,
        help="Path to gemini_turns_*.jsonl with original system_prompt/task_prompt/kernel_code",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Path to a single reasoning_pairs.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Stem for output files (default: derived from --pairs basename)",
    )
    parser.add_argument("--tokenizer", default="/data/local/models/qwen3.5_9B")
    parser.add_argument(
        "--reasoning-field",
        choices=["reasoning", "thinking"],
        default="reasoning",
    )
    parser.add_argument("--min-reasoning-chars", type=int, default=200)
    parser.add_argument(
        "--source-model",
        default="GLM-5.1",
        help="Label written into source_reasoning_model",
    )
    parser.add_argument(
        "--source-label",
        default="glm",
        help="Short label written into source_label and the row id suffix",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="maximum final chat tokens after cutting prior kernel/verdict pairs",
    )
    parser.add_argument(
        "--def-prefix",
        default="",
        help="only include records whose metadata.definition_name starts with this prefix",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional cut report path (defaults to <output-dir>/<stem>.cut_report.json)",
    )
    args = parser.parse_args()

    if not args.gemini_turns.is_file():
        raise SystemExit(f"missing gemini turns file: {args.gemini_turns}")
    if not args.pairs.is_file():
        raise SystemExit(f"missing pairs file: {args.pairs}")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    stem = args.output_name or args.pairs.stem
    gemini_by_session = index_gemini_turns_by_session(load_jsonl(args.gemini_turns))
    pair_records = index_reasoning_pairs(load_jsonl(args.pairs), args.source_label)
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer is None:
        raise SystemExit("--tokenizer is required and must load successfully")

    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    report_records: list[dict[str, Any]] = []
    prefix_filtered = 0
    modified_rows = 0
    already_within_budget_rows = 0
    still_over_budget_rows = 0

    for key in sorted(pair_records):
        record = pair_records[key]
        metadata = record.get("metadata") or {}
        definition_name = str(metadata.get("definition_name") or "")
        if args.def_prefix and not definition_name.startswith(args.def_prefix):
            prefix_filtered += 1
            continue

        sess = (key[0], key[1], key[2])
        pair_turn = key[3]
        session_turns = gemini_by_session.get(sess)
        if session_turns is None:
            dropped.append({"key": list(key), "reason": "missing_gemini_session"})
            continue
        try:
            kernel_index, turn_resolution = resolve_kernel_index(
                session_turns=session_turns, pair_turn=pair_turn
            )
        except ValueError as e:
            dropped.append({"key": list(key), "reason": str(e)})
            continue

        reasoning = (record.get(args.reasoning_field) or "").strip()
        if len(reasoning) < args.min_reasoning_chars:
            dropped.append(
                {"key": list(key), "reason": f"{args.reasoning_field}_too_short"}
            )
            continue

        messages, cut_info = cut_messages_to_budget(
            tokenizer=tokenizer,
            session_turns=session_turns,
            kernel_index=kernel_index,
            reasoning=reasoning,
            max_tokens=args.max_tokens,
        )
        if cut_info["cuts"]:
            modified_rows += 1
        if cut_info["already_within_budget"]:
            already_within_budget_rows += 1
        if cut_info["still_over_budget"]:
            still_over_budget_rows += 1

        row = make_messages_row(
            session_turns=session_turns,
            kernel_index=kernel_index,
            reasoning_record=record,
            reasoning=reasoning,
            messages=messages,
            cut_info=cut_info,
            turn_resolution=turn_resolution,
            source_model=args.source_model,
            source_label=args.source_label,
            reasoning_field=args.reasoning_field,
        )
        rows.append(row)

        gemini_turn = int(session_turns[kernel_index]["turn"])
        report_records.append(
            {
                "key": list(key),
                "id": row["id"],
                "run_id": key[0],
                "exp_id": key[1],
                "definition_name": key[2],
                "pair_metadata_turn": pair_turn,
                "kernel_conversation_index": kernel_index,
                "kernel_gemini_turn": gemini_turn,
                "turn_resolution": turn_resolution,
                "before_tokens": cut_info["before_tokens"],
                "after_tokens": cut_info["after_tokens"],
                "already_within_budget": cut_info["already_within_budget"],
                "still_over_budget": cut_info["still_over_budget"],
                "cuts": cut_info["cuts"],
                "included_prior_turns": cut_info["included_prior_turns"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / f"{stem}.parquet"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    stats_path = args.output_dir / f"{stem}.length_stats.json"
    report_path = args.report or args.output_dir / f"{stem}.cut_report.json"

    write_parquet(parquet_path, rows)
    write_json(stats_path, dataset_stats(rows, tokenizer))

    report = {
        "builder": "accrl.distill.sft.build_sft_dataset_single_with_cut",
        "gemini_turns_path": str(args.gemini_turns),
        "pairs_path": str(args.pairs),
        "output_path": str(parquet_path),
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "def_prefix": args.def_prefix,
        "summary": {
            "input_pair_records": len(pair_records),
            "prefix_filtered_rows": prefix_filtered,
            "output_rows": len(rows),
            "dropped_rows": len(dropped),
            "already_within_budget_rows": already_within_budget_rows,
            "modified_rows": modified_rows,
            "still_over_budget_rows": still_over_budget_rows,
        },
        "dropped_records": dropped,
        "records": report_records,
    }
    write_json(report_path, report)

    manifest = {
        "builder": "accrl.distill.sft.build_sft_dataset_single_with_cut",
        "prompt_source": "gemini_turns_with_cut",
        "gemini_turns_path": str(args.gemini_turns),
        "gemini_turns_sha256": file_sha256(args.gemini_turns),
        "pairs_path": str(args.pairs),
        "pairs_sha256": file_sha256(args.pairs),
        "cut_report_path": str(report_path),
        "cut_report_sha256": file_sha256(report_path),
        "reasoning_field": args.reasoning_field,
        "target_source_field": args.reasoning_field,
        "target_format": "qwen_think_wrapped",
        "min_reasoning_chars": args.min_reasoning_chars,
        "source_model": args.source_model,
        "source_label": args.source_label,
        "max_tokens": args.max_tokens,
        "def_prefix": args.def_prefix,
        "num_gemini_turns": sum(len(v) for v in gemini_by_session.values()),
        "num_gemini_sessions": len(gemini_by_session),
        "num_pair_records": len(pair_records),
        "num_prefix_filtered": prefix_filtered,
        "num_kept_keys": len(rows),
        "num_dropped_keys": len(dropped),
        "dropped_keys": dropped,
        "outputs": {
            "parquet": {
                "path": str(parquet_path),
                "sha256": file_sha256(parquet_path),
                "rows": len(rows),
            },
        },
    }
    write_json(manifest_path, manifest)

    print(f"input pair records: {len(pair_records)}")
    print(f"prefix filtered   : {prefix_filtered}")
    print(f"wrote rows        : {len(rows)} -> {parquet_path}")
    print(f"modified rows     : {modified_rows}")
    print(f"still over budget : {still_over_budget_rows}")
    print(f"manifest          : {manifest_path}")
    print(f"length stats      : {stats_path}")
    print(f"cut report        : {report_path}")


if __name__ == "__main__":
    main()
