#!/usr/bin/env python3
"""Build controlled SFT datasets from two distill experiments.

This builder is intentionally conservative:
- it matches samples by (exp_id, definition_name, turn)
- it only keeps samples whose system prompt and user prompt are byte-identical
- it writes separate GLM, Kimi, and mixed parquet files
- it writes a manifest and length report next to the outputs
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


def canonical_key(record: dict[str, Any]) -> tuple[str, str, int]:
    metadata = record.get("metadata") or {}
    try:
        return (
            str(metadata["exp_id"]),
            str(metadata["definition_name"]),
            int(metadata["turn"]),
        )
    except KeyError as e:
        raise ValueError(f"record missing metadata key for canonical id: {e}") from e


def index_records(records: list[dict[str, Any]], source_name: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicates: Counter[tuple[str, str, int]] = Counter()
    for record in records:
        key = canonical_key(record)
        duplicates[key] += 1
        if key not in out:
            out[key] = record
    dupes = [key for key, count in duplicates.items() if count > 1]
    if dupes:
        raise ValueError(f"{source_name} has duplicate canonical keys, first few: {dupes[:5]}")
    return out


def build_qwen_think_target(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def make_messages(record: dict[str, Any], reasoning_field: str) -> list[dict[str, str]]:
    reasoning = (record.get(reasoning_field) or "").strip()
    if not reasoning:
        raise ValueError(f"empty {reasoning_field}")
    return [
        {"role": "system", "content": record.get("system_prompt", "")},
        {"role": "user", "content": record["input"]},
        {"role": "assistant", "content": build_qwen_think_target(reasoning)},
    ]


def make_row(
    record: dict[str, Any],
    *,
    source_model: str,
    source_label: str,
    reasoning_field: str,
) -> dict[str, Any]:
    key = canonical_key(record)
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "source_reasoning_model": source_model,
            "source_label": source_label,
            "reasoning_field": reasoning_field,
            "target_source_field": reasoning_field,
            "target_format": "qwen_think_wrapped",
            "teacher_hidden_thinking_available": bool((record.get("thinking") or "").strip()),
            "system_prompt_sha256": sha256_text(record.get("system_prompt", "")),
            "input_sha256": sha256_text(record.get("input", "")),
            "reasoning_sha256": sha256_text((record.get(reasoning_field) or "").strip()),
            "reasoning_chars": len((record.get(reasoning_field) or "").strip()),
        }
    )
    return {
        "id": f"{key[0]}_{key[1]}_t{key[2]}_{source_label}",
        "messages": make_messages(record, reasoning_field),
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


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int | None:
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=True)
        # transformers >= 4.45 defaults apply_chat_template to return_dict=True, which
        # yields a BatchEncoding (UserDict-based, NOT a plain dict). Use Mapping
        # duck-typing so we work across versions and pull input_ids when available.
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


def dataset_stats(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    reasoning_chars = [len(row["messages"][-1]["content"]) for row in rows]
    prompt_chars = [len(row["messages"][0]["content"]) + len(row["messages"][1]["content"]) for row in rows]
    token_lengths = [token_count(tokenizer, row["messages"]) for row in rows]
    token_lengths_int = [x for x in token_lengths if x is not None]
    source_counts = Counter(row.get("metadata", {}).get("source_label", "unknown") for row in rows)
    stats: dict[str, Any] = {
        "num_rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "reasoning_chars": summarize_ints(reasoning_chars),
        "prompt_chars": summarize_ints(prompt_chars),
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


def read_provenance(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "provenance.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-dir", type=Path, required=True)
    parser.add_argument("--kimi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="/data/local/models/qwen3.5_9B")
    parser.add_argument(
        "--reasoning-field",
        choices=["reasoning", "thinking"],
        default="reasoning",
        help="Distill field to wrap inside Qwen <think>...</think>; main experiments should use reasoning.",
    )
    parser.add_argument("--min-reasoning-chars", type=int, default=200)
    parser.add_argument("--write-jsonl", action="store_true")
    args = parser.parse_args()

    glm_pairs = args.glm_dir / "reasoning_pairs.jsonl"
    kimi_pairs = args.kimi_dir / "reasoning_pairs.jsonl"
    if not glm_pairs.is_file():
        raise SystemExit(f"missing GLM pairs: {glm_pairs}")
    if not kimi_pairs.is_file():
        raise SystemExit(f"missing Kimi pairs: {kimi_pairs}")

    glm_records = index_records(load_jsonl(glm_pairs), "glm")
    kimi_records = index_records(load_jsonl(kimi_pairs), "kimi")
    common_keys = sorted(set(glm_records) & set(kimi_records))

    kept_keys: list[tuple[str, str, int]] = []
    dropped: list[dict[str, Any]] = []
    for key in common_keys:
        glm = glm_records[key]
        kimi = kimi_records[key]
        if glm.get("system_prompt", "") != kimi.get("system_prompt", ""):
            dropped.append({"key": key, "reason": "system_prompt_mismatch"})
            continue
        if glm.get("input", "") != kimi.get("input", ""):
            dropped.append({"key": key, "reason": "input_mismatch"})
            continue
        if len((glm.get(args.reasoning_field) or "").strip()) < args.min_reasoning_chars:
            dropped.append({"key": key, "reason": f"glm_{args.reasoning_field}_too_short"})
            continue
        if len((kimi.get(args.reasoning_field) or "").strip()) < args.min_reasoning_chars:
            dropped.append({"key": key, "reason": f"kimi_{args.reasoning_field}_too_short"})
            continue
        kept_keys.append(key)

    glm_rows = [
        make_row(glm_records[key], source_model="GLM-5.1", source_label="glm", reasoning_field=args.reasoning_field)
        for key in kept_keys
    ]
    kimi_rows = [
        make_row(kimi_records[key], source_model="Kimi-K2.6", source_label="kimi", reasoning_field=args.reasoning_field)
        for key in kept_keys
    ]
    mixed_rows = [row for pair in zip(glm_rows, kimi_rows, strict=True) for row in pair]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "glm": args.output_dir / "glm_intersection.parquet",
        "kimi": args.output_dir / "kimi_intersection.parquet",
        "mixed": args.output_dir / "mixed_intersection.parquet",
    }
    write_parquet(outputs["glm"], glm_rows)
    write_parquet(outputs["kimi"], kimi_rows)
    write_parquet(outputs["mixed"], mixed_rows)
    if args.write_jsonl:
        write_jsonl(args.output_dir / "glm_intersection.jsonl", glm_rows)
        write_jsonl(args.output_dir / "kimi_intersection.jsonl", kimi_rows)
        write_jsonl(args.output_dir / "mixed_intersection.jsonl", mixed_rows)

    tokenizer = load_tokenizer(args.tokenizer)
    report = {
        "glm": dataset_stats(glm_rows, tokenizer),
        "kimi": dataset_stats(kimi_rows, tokenizer),
        "mixed": dataset_stats(mixed_rows, tokenizer),
    }
    write_json(args.output_dir / "length_stats.json", report)

    manifest = {
        "builder": "accrl.distill.sft.build_sft_dataset",
        "reasoning_field": args.reasoning_field,
        "target_source_field": args.reasoning_field,
        "target_format": "qwen_think_wrapped",
        "min_reasoning_chars": args.min_reasoning_chars,
        "glm_dir": str(args.glm_dir),
        "kimi_dir": str(args.kimi_dir),
        "glm_pairs_sha256": file_sha256(glm_pairs),
        "kimi_pairs_sha256": file_sha256(kimi_pairs),
        "glm_provenance": read_provenance(args.glm_dir),
        "kimi_provenance": read_provenance(args.kimi_dir),
        "num_glm_records": len(glm_records),
        "num_kimi_records": len(kimi_records),
        "num_common_keys": len(common_keys),
        "num_kept_keys": len(kept_keys),
        "num_dropped_common_keys": len(dropped),
        "dropped_common_keys": dropped,
        "only_glm_keys": [list(key) for key in sorted(set(glm_records) - set(kimi_records))],
        "only_kimi_keys": [list(key) for key in sorted(set(kimi_records) - set(glm_records))],
        "outputs": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len({"glm": glm_rows, "kimi": kimi_rows, "mixed": mixed_rows}[name]),
            }
            for name, path in outputs.items()
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(f"wrote GLM rows:   {len(glm_rows)} -> {outputs['glm']}")
    print(f"wrote Kimi rows:  {len(kimi_rows)} -> {outputs['kimi']}")
    print(f"wrote mixed rows: {len(mixed_rows)} -> {outputs['mixed']}")
    print(f"manifest: {args.output_dir / 'manifest.json'}")
    print(f"length stats: {args.output_dir / 'length_stats.json'}")


if __name__ == "__main__":
    main()
