#!/usr/bin/env python3
"""Inspect SFT parquet datasets produced by the distill pipeline."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_METADATA_KEYS = [
    "run_id",
    "exp_id",
    "definition_name",
    "turn",
    "source_label",
    "source_reasoning_model",
    "target_format",
    "target_source_field",
    "reasoning_chars",
    "tokens_before_cut",
    "tokens_after_cut",
    "prompt_replaced",
    "still_over_budget",
    "kernel_passed",
    "kernel_speedup",
    "passed",
    "speedup",
]

DISTRIBUTION_KEYS = [
    "run_id",
    "definition_name",
    "source_label",
    "target_format",
    "turn_resolution",
    "prompt_replaced",
    "still_over_budget",
    "kernel_passed",
    "passed",
]

NUMERIC_METADATA_KEYS = [
    "reasoning_chars",
    "tokens_before_cut",
    "tokens_after_cut",
    "tokens_after_cut_before_prompt_replacement",
    "num_prior_turns_before_cut",
    "num_prior_turns_after_cut",
    "kernel_speedup",
    "speedup",
]


def load_rows(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow is required for parquet inspection; run in the Miles image") from e

    parquet_file = pq.ParquetFile(path)
    rows = parquet_file.read().to_pylist()
    return parquet_file, rows


def percentile(sorted_values: list[float], pct: float) -> float:
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def summarize_numbers(values: list[int | float]) -> dict[str, float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return {}
    cleaned.sort()
    return {
        "min": cleaned[0],
        "p50": percentile(cleaned, 0.50),
        "p90": percentile(cleaned, 0.90),
        "p95": percentile(cleaned, 0.95),
        "p99": percentile(cleaned, 0.99),
        "max": cleaned[-1],
        "mean": statistics.fmean(cleaned),
    }


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def print_numeric_summary(label: str, values: list[int | float]) -> None:
    stats = summarize_numbers(values)
    if not stats:
        return
    fields = " ".join(f"{k}={format_number(v)}" for k, v in stats.items())
    print(f"{label}: {fields}")


def message_char_counts(row: dict[str, Any]) -> dict[str, int]:
    messages = row.get("messages") or []
    prompt_chars = sum(len(message.get("content") or "") for message in messages[:-1])
    assistant_chars = len(messages[-1].get("content") or "") if messages else 0
    return {
        "messages": len(messages),
        "prompt_chars": prompt_chars,
        "assistant_chars": assistant_chars,
        "total_chars": prompt_chars + assistant_chars,
    }


def scalar_label(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, sort_keys=True)[:120]


def metadata_value(row: dict[str, Any], key: str) -> Any:
    metadata = row.get("metadata") or {}
    return metadata.get(key)


def row_sort_value(row: dict[str, Any], key: str) -> Any:
    counts = message_char_counts(row)
    if key in counts:
        return (2, float(counts[key]), "")
    value = metadata_value(row, key)
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, bool):
        return (2, float(value), "")
    if isinstance(value, (int, float)):
        return (2, float(value), "")
    return (1, 0.0, str(value))


def collect_metadata_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        keys.update(metadata.keys())
    return sorted(keys)


def print_schema(parquet_file: Any) -> None:
    print("\nSchema")
    print(parquet_file.schema)


def print_file_summary(path: Path, parquet_file: Any, rows: list[dict[str, Any]]) -> None:
    metadata = parquet_file.metadata
    top_level_columns = getattr(parquet_file, "schema_arrow", parquet_file.schema).names
    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size:,} bytes")
    print(f"Rows: {metadata.num_rows:,}")
    print(f"Row groups: {metadata.num_row_groups:,}")
    print(f"Columns: {', '.join(top_level_columns)}")
    print(f"Loaded rows: {len(rows):,}")


def print_distribution_summary(rows: list[dict[str, Any]], *, top_k: int) -> None:
    print("\nMetadata distributions")
    for key in DISTRIBUTION_KEYS:
        counter = Counter(
            scalar_label(metadata_value(row, key))
            for row in rows
            if metadata_value(row, key) is not None
        )
        if not counter:
            continue
        pairs = ", ".join(f"{value}={count}" for value, count in counter.most_common(top_k))
        print(f"{key}: {pairs}")


def print_length_summary(rows: list[dict[str, Any]]) -> None:
    counts = [message_char_counts(row) for row in rows]
    print("\nLength stats")
    print_numeric_summary("messages", [item["messages"] for item in counts])
    print_numeric_summary("prompt_chars", [item["prompt_chars"] for item in counts])
    print_numeric_summary("assistant_chars", [item["assistant_chars"] for item in counts])
    print_numeric_summary("total_chars", [item["total_chars"] for item in counts])
    for key in NUMERIC_METADATA_KEYS:
        values = [metadata_value(row, key) for row in rows]
        print_numeric_summary(key, values)


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, limit // 2)
    head = text[:half].rstrip()
    tail = text[-half:].lstrip()
    return f"{head}\n... <truncated {len(text) - len(head) - len(tail):,} chars> ...\n{tail}"


def indent_block(text: str, *, prefix: str = "  ") -> str:
    return textwrap.indent(text, prefix)


def print_row_preview(
    row: dict[str, Any],
    *,
    row_idx: int,
    preview_chars: int,
    metadata_keys: list[str] | None,
    show_all_metadata: bool,
    show_messages: bool,
) -> None:
    messages = row.get("messages") or []
    counts = message_char_counts(row)
    print(f"\nRow {row_idx}: {row.get('id')}")
    print(
        "chars: "
        f"messages={counts['messages']} prompt={counts['prompt_chars']:,} "
        f"assistant={counts['assistant_chars']:,} total={counts['total_chars']:,}"
    )

    metadata = row.get("metadata") or {}
    if show_all_metadata:
        keys = sorted(metadata.keys())
    else:
        keys = metadata_keys or DEFAULT_METADATA_KEYS
    selected = {key: metadata.get(key) for key in keys if key in metadata}
    if selected:
        print("metadata:")
        for key, value in selected.items():
            print(f"  {key}: {scalar_label(value)}")

    if not show_messages:
        return

    print("messages:")
    for msg_idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content") or ""
        print(f"  [{msg_idx}] role={role} chars={len(content):,}")
        preview = truncate_text(content, preview_chars)
        print(indent_block(preview))


def parse_row_indices(values: list[str], *, num_rows: int) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                start_s, end_s = part.split(":", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else num_rows
                out.extend(range(start, min(end, num_rows)))
            else:
                idx = int(part)
                if idx < 0:
                    idx += num_rows
                out.append(idx)
    return [idx for idx in dict.fromkeys(out) if 0 <= idx < num_rows]


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    if args.row:
        return parse_row_indices(args.row, num_rows=len(rows))

    if args.id_contains:
        matches = [
            idx
            for idx, row in enumerate(rows)
            if all(needle in str(row.get("id", "")) for needle in args.id_contains)
        ]
        return matches[: args.rows]

    if args.sample:
        rng = random.Random(args.seed)
        return sorted(rng.sample(range(len(rows)), min(args.sample, len(rows))))

    if args.sort_by:
        indexed = list(enumerate(rows))
        indexed.sort(key=lambda item: row_sort_value(item[1], args.sort_by), reverse=args.descending)
        return [idx for idx, _ in indexed[: args.rows]]

    return list(range(min(args.rows, len(rows))))


def write_json_output(rows: list[dict[str, Any]], indices: list[int]) -> None:
    payload = [{"row_idx": idx, **rows[idx]} for idx in indices]
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Input .parquet file")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows to preview")
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        help="Row index, comma list, or Python-style half-open range such as 0,2,5:8",
    )
    parser.add_argument(
        "--id-contains",
        action="append",
        default=[],
        help="Only preview rows whose id contains this substring; repeat to require multiple substrings",
    )
    parser.add_argument("--sample", type=int, default=0, help="Preview a deterministic random sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sort-by",
        default=None,
        help=(
            "Preview rows sorted by a metric or metadata field, e.g. total_chars, "
            "assistant_chars, reasoning_chars, tokens_after_cut, speedup"
        ),
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Use ascending order with --sort-by; default is descending",
    )
    parser.add_argument("--preview-chars", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=12, help="Max values per distribution line")
    parser.add_argument("--schema", action="store_true", help="Print parquet schema")
    parser.add_argument("--all-metadata", action="store_true", help="Print every metadata key for preview rows")
    parser.add_argument(
        "--metadata-key",
        action="append",
        default=None,
        help="Metadata key to print for preview rows; repeat to customize the metadata section",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip file and aggregate summaries")
    parser.add_argument("--no-messages", action="store_true", help="Skip message previews")
    parser.add_argument("--json", action="store_true", help="Emit selected rows as JSON instead of human text")
    args = parser.parse_args()
    args.descending = not args.ascending

    if not args.parquet.is_file():
        raise SystemExit(f"missing parquet file: {args.parquet}")
    if args.rows < 0:
        raise SystemExit("--rows must be non-negative")
    if args.sample < 0:
        raise SystemExit("--sample must be non-negative")

    parquet_file, rows = load_rows(args.parquet)
    indices = select_rows(rows, args)

    if args.json:
        write_json_output(rows, indices)
        return

    if not args.no_summary:
        print_file_summary(args.parquet, parquet_file, rows)
        print(f"Metadata keys: {', '.join(collect_metadata_keys(rows))}")
        if args.schema:
            print_schema(parquet_file)
        print_distribution_summary(rows, top_k=args.top_k)
        print_length_summary(rows)

    if indices:
        print("\nSelected rows")
        for idx in indices:
            print_row_preview(
                rows[idx],
                row_idx=idx,
                preview_chars=args.preview_chars,
                metadata_keys=args.metadata_key,
                show_all_metadata=args.all_metadata,
                show_messages=not args.no_messages,
            )
    else:
        print("\nSelected rows: none")


if __name__ == "__main__":
    main()
