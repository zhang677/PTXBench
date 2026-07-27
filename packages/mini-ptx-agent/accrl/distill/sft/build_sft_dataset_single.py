#!/usr/bin/env python3
"""Build an SFT parquet from a single reasoning-pairs jsonl.

Single-source variant of ``build_sft_dataset_v2``: no glm/kimi intersection.
For each reasoning-pair record at (run_id, exp_id, definition_name, turn=N),
replay the full Gemini multi-turn history up to and including turn N, with
the final assistant message containing the synthesized reasoning wrapped in
``<think>...</think>`` followed by the Gemini turn-N kernel response:

    system    = gemini_turn0.system_prompt
    user      = gemini_turn0.task_prompt
    [for i in 0..N-1]
        assistant = gemini_turn_i.raw_assistant_content
        user      = gemini_turn_i.raw_feedback_content
    assistant = <think>\\n{reasoning_N}\\n</think>\\n
                + gemini_turn_N.raw_assistant_content       # SFT target

Outputs (in ``--output-dir``):
- ``<stem>.parquet`` — chat ``messages`` schema rows.
- ``<stem>.manifest.json``
- ``<stem>.length_stats.json``

where ``<stem>`` defaults to the ``--pairs`` file basename (without
``.jsonl``) and can be overridden with ``--output-name``.
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
    for sess, turns in by_session.items():
        turns.sort(key=lambda r: int(r["turn"]))
        expected = list(range(len(turns)))
        actual = [int(r["turn"]) for r in turns]
        if actual != expected:
            raise ValueError(
                f"gemini session {sess} has non-contiguous turns: {actual}"
            )
    return by_session


def build_qwen_think_target(reasoning: str, raw_assistant_content: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n{raw_assistant_content}"


def build_messages(
    *,
    session_turns: list[dict[str, Any]],
    turn_n: int,
    reasoning: str,
) -> list[dict[str, str]]:
    if not session_turns:
        raise ValueError("empty session_turns")
    if turn_n >= len(session_turns):
        raise ValueError(
            f"turn_n={turn_n} out of range for session of length {len(session_turns)}"
        )
    first = session_turns[0]
    messages: list[dict[str, str]] = [
        {"role": "system", "content": first["system_prompt"]},
        {"role": "user", "content": first["task_prompt"]},
    ]
    for i in range(turn_n):
        t = session_turns[i]
        messages.append({"role": "assistant", "content": t["raw_assistant_content"]})
        messages.append({"role": "user", "content": t["raw_feedback_content"]})
    final = build_qwen_think_target(
        reasoning, session_turns[turn_n]["raw_assistant_content"]
    )
    messages.append({"role": "assistant", "content": final})
    return messages


def make_messages_row(
    *,
    session_turns: list[dict[str, Any]],
    turn_n: int,
    reasoning_record: dict[str, Any],
    reasoning: str,
    source_model: str,
    source_label: str,
    reasoning_field: str,
) -> dict[str, Any]:
    gemini = session_turns[turn_n]
    key = (
        str(gemini["run_id"]),
        str(gemini["exp_id"]),
        str(gemini["definition_name"]),
        int(gemini["turn"]),
    )
    messages = build_messages(
        session_turns=session_turns, turn_n=turn_n, reasoning=reasoning
    )
    rp_metadata = dict(reasoning_record.get("metadata") or {})
    metadata = {
        **rp_metadata,
        "source_reasoning_model": source_model,
        "source_label": source_label,
        "reasoning_field": reasoning_field,
        "target_source_field": reasoning_field,
        "target_format": "qwen_think_wrapped",
        "prompt_source": "gemini_turns",
        "num_prior_turns": turn_n,
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
    args = parser.parse_args()

    if not args.gemini_turns.is_file():
        raise SystemExit(f"missing gemini turns file: {args.gemini_turns}")
    if not args.pairs.is_file():
        raise SystemExit(f"missing pairs file: {args.pairs}")

    stem = args.output_name or args.pairs.stem

    gemini_by_session = index_gemini_turns_by_session(load_jsonl(args.gemini_turns))
    pair_records = index_reasoning_pairs(load_jsonl(args.pairs), args.source_label)

    kept_keys: list[TurnKey] = []
    dropped: list[dict[str, Any]] = []
    for key in sorted(pair_records):
        sess = (key[0], key[1], key[2])
        turn_n = key[3]
        session_turns = gemini_by_session.get(sess)
        if session_turns is None or turn_n >= len(session_turns):
            dropped.append({"key": list(key), "reason": "missing_in_gemini_turns"})
            continue
        record = pair_records[key]
        if (
            len((record.get(args.reasoning_field) or "").strip())
            < args.min_reasoning_chars
        ):
            dropped.append(
                {"key": list(key), "reason": f"{args.reasoning_field}_too_short"}
            )
            continue
        kept_keys.append(key)

    rows = [
        make_messages_row(
            session_turns=gemini_by_session[(key[0], key[1], key[2])],
            turn_n=key[3],
            reasoning_record=pair_records[key],
            reasoning=(pair_records[key].get(args.reasoning_field) or "").strip(),
            source_model=args.source_model,
            source_label=args.source_label,
            reasoning_field=args.reasoning_field,
        )
        for key in kept_keys
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / f"{stem}.parquet"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    stats_path = args.output_dir / f"{stem}.length_stats.json"

    write_parquet(parquet_path, rows)

    tokenizer = load_tokenizer(args.tokenizer)
    write_json(stats_path, dataset_stats(rows, tokenizer))

    manifest = {
        "builder": "accrl.distill.sft.build_sft_dataset_single",
        "prompt_source": "gemini_turns",
        "gemini_turns_path": str(args.gemini_turns),
        "gemini_turns_sha256": file_sha256(args.gemini_turns),
        "pairs_path": str(args.pairs),
        "pairs_sha256": file_sha256(args.pairs),
        "reasoning_field": args.reasoning_field,
        "target_source_field": args.reasoning_field,
        "target_format": "qwen_think_wrapped",
        "min_reasoning_chars": args.min_reasoning_chars,
        "source_model": args.source_model,
        "source_label": args.source_label,
        "num_gemini_turns": sum(len(v) for v in gemini_by_session.values()),
        "num_gemini_sessions": len(gemini_by_session),
        "num_pair_records": len(pair_records),
        "num_kept_keys": len(kept_keys),
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

    print(f"wrote rows: {len(rows)} -> {parquet_path}")
    print(f"manifest:    {manifest_path}")
    print(f"length stats: {stats_path}")


if __name__ == "__main__":
    main()
