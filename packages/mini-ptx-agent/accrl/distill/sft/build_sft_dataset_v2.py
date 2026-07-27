#!/usr/bin/env python3
"""Build SFT datasets that preserve the multi-turn Gemini history.

For each reasoning-pair record at (exp_id, definition_name, turn=N), build a
chat conversation that replays the *entire* prior Gemini interaction for the
same session and ends with one assistant message that contains both the
synthesized reasoning (wrapped in ``<think>...</think>``) and the gemini
turn-N kernel response:

    system    = gemini_turn0.system_prompt
    user      = gemini_turn0.task_prompt
    [for i in 0..N-1]
        assistant = gemini_turn_i.raw_assistant_content
        user      = gemini_turn_i.raw_feedback_content
    assistant = <think>\\n{reasoning_N}\\n</think>\\n
                + gemini_turn_N.raw_assistant_content       # SFT target

Outputs (in ``--output-dir``):
- ``{glm,kimi,mixed}_reasoning_raw.jsonl`` — flat inspection schema:
  ``{system_prompt, task_prompt, prior_turns, reasoning, kernel_code,
     exp_id, definition_name, turn, speedup, passed, source_model,
     source_label}``  where ``prior_turns`` is a list of
  ``{turn, assistant, feedback, passed, speedup, kernel_code}`` for turns
  0..N-1.
- ``{glm,kimi,mixed}_intersection.parquet`` — chat ``messages`` schema with the
  full multi-turn history as described above.
- ``manifest.json`` and ``length_stats.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


CANONICAL_KEY_FIELDS = ("exp_id", "definition_name", "turn")
TARGET_FORMAT = "qwen_think_plus_gemini_response"
FORBIDDEN_REASONING_TAGS = ("<think>", "</think>", "<my_reasoning>", "</my_reasoning>")


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


def reasoning_pair_key(record: dict[str, Any]) -> tuple[str, str, int]:
    metadata = record.get("metadata") or {}
    try:
        return (
            str(metadata["exp_id"]),
            str(metadata["definition_name"]),
            int(metadata["turn"]),
        )
    except KeyError as e:
        raise ValueError(f"reasoning_pairs record missing metadata key: {e}") from e


def gemini_turn_key(record: dict[str, Any]) -> tuple[str, str, int]:
    try:
        return (
            str(record["exp_id"]),
            str(record["definition_name"]),
            int(record["turn"]),
        )
    except KeyError as e:
        raise ValueError(f"gemini_turns record missing key: {e}") from e


def index_reasoning_pairs(
    records: list[dict[str, Any]], source_name: str
) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicates: Counter[tuple[str, str, int]] = Counter()
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
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group gemini turns by (exp_id, definition_name), sorted by turn ascending.

    Also detects duplicate (exp_id, definition_name, turn) entries.
    """
    by_session: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, int]] = set()
    duplicates: list[tuple[str, str, int]] = []
    for record in records:
        sess = (str(record["exp_id"]), str(record["definition_name"]))
        turn = int(record["turn"])
        triple = (sess[0], sess[1], turn)
        if triple in seen:
            duplicates.append(triple)
            continue
        seen.add(triple)
        by_session.setdefault(sess, []).append(record)
    if duplicates:
        raise ValueError(
            f"gemini_turns has duplicate (exp_id, definition_name, turn) entries, "
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


def sanitize_reasoning(raw_reasoning: Any) -> str:
    """Normalize the visible reasoning field before wrapping it in Qwen tags.

    A few Kimi rows contain the full model transcript, e.g. hidden thinking
    followed by ``</think> <my_reasoning>...``.  For this SFT target we only want
    the visible reasoning payload, then we add exactly one outer
    ``<think>...</think>`` pair ourselves.
    """
    text = str(raw_reasoning or "").strip()
    if "<my_reasoning>" in text:
        text = text.split("<my_reasoning>", 1)[1]
        if "</my_reasoning>" in text:
            text = text.split("</my_reasoning>", 1)[0]
        text = text.strip()
    return text


def has_forbidden_reasoning_tags(reasoning: str) -> bool:
    return any(tag in reasoning for tag in FORBIDDEN_REASONING_TAGS)


def build_qwen_think_target(reasoning: str, raw_assistant_content: str) -> str:
    """Final assistant content: <think>{reasoning}</think>\\n + gemini's kernel response."""
    return f"<think>\n{reasoning.strip()}\n</think>\n{raw_assistant_content.strip()}"


def build_prior_assistant_context(raw_assistant_content: str) -> str:
    """Represent prior Gemini assistant turns in Qwen thinking-chat format.

    We do not have Gemini hidden thinking for prior turns, so the thinking block
    is intentionally empty.  Making it explicit keeps the full chat template used
    for eval aligned with Miles' qwen3 per-message SFT tokenization.
    """
    return f"<think>\n\n</think>\n\n{raw_assistant_content.strip()}"


def build_messages(
    *,
    session_turns: list[dict[str, Any]],
    turn_n: int,
    reasoning: str,
) -> list[dict[str, Any]]:
    """Construct the multi-turn chat conversation up to and including turn_n.

    Layout:
        system    = session_turns[0].system_prompt
        user      = session_turns[0].task_prompt
        for i in 0..turn_n - 1:
            assistant = session_turns[i].raw_assistant_content
            user      = session_turns[i].raw_feedback_content
        assistant = <think>{reasoning}</think>\\n + session_turns[turn_n].raw_assistant_content
    """
    if not session_turns:
        raise ValueError("empty session_turns")
    if turn_n >= len(session_turns):
        raise ValueError(f"turn_n={turn_n} out of range for session of length {len(session_turns)}")
    first = session_turns[0]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": first["system_prompt"], "step_loss_mask": 0},
        {"role": "user", "content": first["task_prompt"], "step_loss_mask": 0},
    ]
    for i in range(turn_n):
        t = session_turns[i]
        messages.append(
            {
                "role": "assistant",
                "content": build_prior_assistant_context(t["raw_assistant_content"]),
                "step_loss_mask": 0,
            }
        )
        messages.append({"role": "user", "content": t["raw_feedback_content"], "step_loss_mask": 0})
    final = build_qwen_think_target(reasoning, session_turns[turn_n]["raw_assistant_content"])
    messages.append({"role": "assistant", "content": final, "step_loss_mask": 1})
    return messages


def build_prior_turns(
    session_turns: list[dict[str, Any]], turn_n: int
) -> list[dict[str, Any]]:
    """Flat-schema list of {turn, assistant, feedback, passed, speedup, kernel_code}
    for turns 0..turn_n - 1.
    """
    out: list[dict[str, Any]] = []
    for i in range(turn_n):
        t = session_turns[i]
        out.append(
            {
                "turn": int(t["turn"]),
                "assistant": t["raw_assistant_content"],
                "feedback": t["raw_feedback_content"],
                "passed": t.get("passed"),
                "speedup": t.get("speedup"),
                "kernel_code": t.get("kernel_code", ""),
            }
        )
    return out


def make_raw_row(
    *,
    session_turns: list[dict[str, Any]],
    turn_n: int,
    reasoning: str,
    source_model: str,
    source_label: str,
) -> dict[str, Any]:
    gemini = session_turns[turn_n]
    return {
        "system_prompt": session_turns[0]["system_prompt"],
        "task_prompt": session_turns[0]["task_prompt"],
        "prior_turns": build_prior_turns(session_turns, turn_n),
        "reasoning": reasoning,
        "kernel_code": gemini.get("kernel_code", ""),
        "exp_id": gemini["exp_id"],
        "definition_name": gemini["definition_name"],
        "turn": int(gemini["turn"]),
        "speedup": gemini.get("speedup"),
        "passed": gemini.get("passed"),
        "source_model": source_model,
        "source_label": source_label,
    }


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
    key = (str(gemini["exp_id"]), str(gemini["definition_name"]), int(gemini["turn"]))
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
        "target_format": TARGET_FORMAT,
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
        "id": f"{key[0]}_{key[1]}_t{key[2]}_{source_label}",
        "messages": messages,
        "metadata": metadata,
    }


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


def token_count(tokenizer: Any, messages: list[dict[str, Any]]) -> int | None:
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
    # Prompt = everything except the final (target) assistant message
    prompt_chars = [
        sum(len(m["content"]) for m in row["messages"][:-1]) for row in rows
    ]
    num_messages = [len(row["messages"]) for row in rows]
    token_lengths = [token_count(tokenizer, row["messages"]) for row in rows]
    token_lengths_int = [x for x in token_lengths if x is not None]
    source_counts = Counter(
        row.get("metadata", {}).get("source_label", "unknown") for row in rows
    )
    stats: dict[str, Any] = {
        "num_rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
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
                    "source_label": row.get("metadata", {}).get("source_label"),
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


def validate_rows(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        row_id = row.get("id")
        if not isinstance(messages, list) or not messages:
            errors.append({"id": row_id, "reason": "missing_messages"})
            continue
        for idx, message in enumerate(messages[:-1]):
            if message.get("role") == "assistant" and message.get("step_loss_mask") != 0:
                errors.append({"id": row_id, "message_index": idx, "reason": "prior_assistant_not_masked"})
            if message.get("role") == "assistant" and not str(message.get("content", "")).startswith(
                "<think>\n\n</think>\n\n"
            ):
                errors.append({"id": row_id, "message_index": idx, "reason": "prior_assistant_missing_empty_think"})
        final = messages[-1]
        final_content = str(final.get("content", ""))
        if final.get("role") != "assistant":
            errors.append({"id": row_id, "reason": "final_message_not_assistant"})
        if final.get("step_loss_mask") != 1:
            errors.append({"id": row_id, "reason": "final_assistant_not_supervised"})
        if final_content.count("<think>") != 1:
            errors.append({"id": row_id, "reason": "final_think_tag_count", "count": final_content.count("<think>")})
        if final_content.count("</think>") != 1:
            errors.append(
                {"id": row_id, "reason": "final_end_think_tag_count", "count": final_content.count("</think>")}
            )
        if "</think>" in final_content:
            reasoning_part, visible_part = final_content.split("</think>", 1)
            reasoning_part = reasoning_part.removeprefix("<think>").strip()
            if has_forbidden_reasoning_tags(reasoning_part):
                errors.append({"id": row_id, "reason": "reasoning_still_contains_tags"})
            if not visible_part.strip():
                errors.append({"id": row_id, "reason": "empty_visible_after_think"})
            if "```" not in visible_part:
                errors.append({"id": row_id, "reason": "visible_response_missing_code_fence"})
        else:
            errors.append({"id": row_id, "reason": "missing_end_think"})
    if errors:
        raise ValueError(f"{name} validation failed, first errors: {errors[:10]}")
    return errors


def read_provenance(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "provenance.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gemini-turns",
        type=Path,
        required=True,
        help="Path to gemini_turns_*.jsonl with original system_prompt/task_prompt/kernel_code",
    )
    parser.add_argument("--glm-dir", type=Path, default=None)
    parser.add_argument("--kimi-dir", type=Path, default=None)
    parser.add_argument(
        "--glm-pairs",
        type=Path,
        default=None,
        help="Override glm reasoning_pairs path (default: <glm-dir>/reasoning_pairs.jsonl)",
    )
    parser.add_argument(
        "--kimi-pairs",
        type=Path,
        default=None,
        help="Override kimi reasoning_pairs path (default: <kimi-dir>/reasoning_pairs.jsonl)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="/data/local/models/qwen3.5_9B")
    parser.add_argument(
        "--reasoning-field",
        choices=["reasoning", "thinking"],
        default="reasoning",
    )
    parser.add_argument("--min-reasoning-chars", type=int, default=200)
    parser.add_argument(
        "--glm-source-model",
        default="GLM-5.1",
        help="Label written into source_reasoning_model for the glm side",
    )
    parser.add_argument(
        "--kimi-source-model",
        default="Kimi-K2.6",
        help="Label written into source_reasoning_model for the kimi side",
    )
    args = parser.parse_args()

    if not args.gemini_turns.is_file():
        raise SystemExit(f"missing gemini turns file: {args.gemini_turns}")

    glm_pairs = args.glm_pairs or (args.glm_dir / "reasoning_pairs.jsonl")
    kimi_pairs = args.kimi_pairs or (args.kimi_dir / "reasoning_pairs.jsonl")
    if not glm_pairs.is_file():
        raise SystemExit(f"missing GLM pairs: {glm_pairs}")
    if not kimi_pairs.is_file():
        raise SystemExit(f"missing Kimi pairs: {kimi_pairs}")

    gemini_by_session = index_gemini_turns_by_session(load_jsonl(args.gemini_turns))
    glm_records = index_reasoning_pairs(load_jsonl(glm_pairs), "glm")
    kimi_records = index_reasoning_pairs(load_jsonl(kimi_pairs), "kimi")

    common_keys = sorted(set(glm_records) & set(kimi_records))

    kept_keys: list[tuple[str, str, int]] = []
    glm_reasoning_by_key: dict[tuple[str, str, int], str] = {}
    kimi_reasoning_by_key: dict[tuple[str, str, int], str] = {}
    dropped: list[dict[str, Any]] = []
    for key in common_keys:
        glm = glm_records[key]
        kimi = kimi_records[key]
        sess = (key[0], key[1])
        turn_n = key[2]
        session_turns = gemini_by_session.get(sess)
        if session_turns is None or turn_n >= len(session_turns):
            dropped.append({"key": list(key), "reason": "missing_in_gemini_turns"})
            continue
        # Sanity: synthesizer prompts should agree across glm/kimi for the same key
        if glm.get("system_prompt", "") != kimi.get("system_prompt", ""):
            dropped.append({"key": list(key), "reason": "synth_system_prompt_mismatch"})
            continue
        if glm.get("input", "") != kimi.get("input", ""):
            dropped.append({"key": list(key), "reason": "synth_input_mismatch"})
            continue
        glm_reasoning = sanitize_reasoning(glm.get(args.reasoning_field))
        kimi_reasoning = sanitize_reasoning(kimi.get(args.reasoning_field))
        if has_forbidden_reasoning_tags(glm_reasoning):
            dropped.append({"key": list(key), "reason": f"glm_{args.reasoning_field}_contains_tags"})
            continue
        if has_forbidden_reasoning_tags(kimi_reasoning):
            dropped.append({"key": list(key), "reason": f"kimi_{args.reasoning_field}_contains_tags"})
            continue
        if len(glm_reasoning) < args.min_reasoning_chars:
            dropped.append(
                {"key": list(key), "reason": f"glm_{args.reasoning_field}_too_short"}
            )
            continue
        if len(kimi_reasoning) < args.min_reasoning_chars:
            dropped.append(
                {"key": list(key), "reason": f"kimi_{args.reasoning_field}_too_short"}
            )
            continue
        glm_reasoning_by_key[key] = glm_reasoning
        kimi_reasoning_by_key[key] = kimi_reasoning
        kept_keys.append(key)

    def session_for(key: tuple[str, str, int]) -> list[dict[str, Any]]:
        return gemini_by_session[(key[0], key[1])]

    # Build flat raw rows
    glm_raw = [
        make_raw_row(
            session_turns=session_for(key),
            turn_n=key[2],
            reasoning=glm_reasoning_by_key[key],
            source_model=args.glm_source_model,
            source_label="glm",
        )
        for key in kept_keys
    ]
    kimi_raw = [
        make_raw_row(
            session_turns=session_for(key),
            turn_n=key[2],
            reasoning=kimi_reasoning_by_key[key],
            source_model=args.kimi_source_model,
            source_label="kimi",
        )
        for key in kept_keys
    ]
    mixed_raw = [row for pair in zip(glm_raw, kimi_raw, strict=True) for row in pair]

    # Build messages-format rows for SFT
    glm_rows = [
        make_messages_row(
            session_turns=session_for(key),
            turn_n=key[2],
            reasoning_record=glm_records[key],
            reasoning=glm_reasoning_by_key[key],
            source_model=args.glm_source_model,
            source_label="glm",
            reasoning_field=args.reasoning_field,
        )
        for key in kept_keys
    ]
    kimi_rows = [
        make_messages_row(
            session_turns=session_for(key),
            turn_n=key[2],
            reasoning_record=kimi_records[key],
            reasoning=kimi_reasoning_by_key[key],
            source_model=args.kimi_source_model,
            source_label="kimi",
            reasoning_field=args.reasoning_field,
        )
        for key in kept_keys
    ]
    mixed_rows = [
        row for pair in zip(glm_rows, kimi_rows, strict=True) for row in pair
    ]

    validate_rows(glm_rows, "glm_rows")
    validate_rows(kimi_rows, "kimi_rows")
    validate_rows(mixed_rows, "mixed_rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = {
        "glm": args.output_dir / "glm_intersection.parquet",
        "kimi": args.output_dir / "kimi_intersection.parquet",
        "mixed": args.output_dir / "mixed_intersection.parquet",
    }
    raw_paths = {
        "glm": args.output_dir / "glm_reasoning_raw.jsonl",
        "kimi": args.output_dir / "kimi_reasoning_raw.jsonl",
        "mixed": args.output_dir / "mixed_reasoning_raw.jsonl",
    }

    write_parquet(parquet_paths["glm"], glm_rows)
    write_parquet(parquet_paths["kimi"], kimi_rows)
    write_parquet(parquet_paths["mixed"], mixed_rows)

    write_jsonl(raw_paths["glm"], glm_raw)
    write_jsonl(raw_paths["kimi"], kimi_raw)
    write_jsonl(raw_paths["mixed"], mixed_raw)

    tokenizer = load_tokenizer(args.tokenizer)
    report = {
        "glm": dataset_stats(glm_rows, tokenizer),
        "kimi": dataset_stats(kimi_rows, tokenizer),
        "mixed": dataset_stats(mixed_rows, tokenizer),
    }
    write_json(args.output_dir / "length_stats.json", report)

    manifest = {
        "builder": "accrl.distill.sft.build_sft_dataset_v2",
        "prompt_source": "gemini_turns",
        "gemini_turns_path": str(args.gemini_turns),
        "gemini_turns_sha256": file_sha256(args.gemini_turns),
        "reasoning_field": args.reasoning_field,
        "target_source_field": args.reasoning_field,
        "target_format": TARGET_FORMAT,
        "min_reasoning_chars": args.min_reasoning_chars,
        "glm_dir": str(args.glm_dir),
        "kimi_dir": str(args.kimi_dir),
        "glm_pairs_path": str(glm_pairs),
        "kimi_pairs_path": str(kimi_pairs),
        "glm_pairs_sha256": file_sha256(glm_pairs),
        "kimi_pairs_sha256": file_sha256(kimi_pairs),
        "glm_provenance": read_provenance(args.glm_dir),
        "kimi_provenance": read_provenance(args.kimi_dir),
        "num_gemini_turns": sum(len(v) for v in gemini_by_session.values()),
        "num_gemini_sessions": len(gemini_by_session),
        "num_glm_records": len(glm_records),
        "num_kimi_records": len(kimi_records),
        "num_common_keys": len(common_keys),
        "num_kept_keys": len(kept_keys),
        "num_dropped_common_keys": len(dropped),
        "dropped_common_keys": dropped,
        "only_glm_keys": [
            list(key) for key in sorted(set(glm_records) - set(kimi_records))
        ],
        "only_kimi_keys": [
            list(key) for key in sorted(set(kimi_records) - set(glm_records))
        ],
        "outputs": {
            **{
                f"{name}_parquet": {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "rows": len(
                        {"glm": glm_rows, "kimi": kimi_rows, "mixed": mixed_rows}[name]
                    ),
                }
                for name, path in parquet_paths.items()
            },
            **{
                f"{name}_raw_jsonl": {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "rows": len(
                        {"glm": glm_raw, "kimi": kimi_raw, "mixed": mixed_raw}[name]
                    ),
                }
                for name, path in raw_paths.items()
            },
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(f"wrote GLM rows:   {len(glm_rows)} -> {parquet_paths['glm']}")
    print(f"wrote Kimi rows:  {len(kimi_rows)} -> {parquet_paths['kimi']}")
    print(f"wrote mixed rows: {len(mixed_rows)} -> {parquet_paths['mixed']}")
    print(f"wrote GLM raw:    {len(glm_raw)} -> {raw_paths['glm']}")
    print(f"wrote Kimi raw:   {len(kimi_raw)} -> {raw_paths['kimi']}")
    print(f"wrote mixed raw:  {len(mixed_raw)} -> {raw_paths['mixed']}")
    print(f"manifest:    {args.output_dir / 'manifest.json'}")
    print(f"length stats: {args.output_dir / 'length_stats.json'}")


if __name__ == "__main__":
    main()
