#!/usr/bin/env python3
"""Balance fix-it kernel-pair rows before parquet construction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, required=True)
    parser.add_argument("--cap-per-definition", type=int, required=True)
    return parser.parse_args()


def speedup_from_trace(trace: dict) -> float | None:
    speedup = (
        trace.get("evaluation", {})
        .get("performance", {})
        .get("speedup_factor")
    )
    if speedup is None:
        return None
    try:
        return float(speedup)
    except (TypeError, ValueError):
        return None


def record_entries(success_dir: Path) -> list[dict]:
    record_path = success_dir / "record.json"
    if not record_path.is_file():
        return []
    try:
        records = json.loads(record_path.read_text())
    except json.JSONDecodeError:
        return []
    return records if isinstance(records, list) else []


def min_speedup_for_version(success_dir: Path, version: int) -> float | None:
    speedups: list[float] = []
    for entry in record_entries(success_dir):
        try:
            entry_version = int(entry.get("version"))
        except (TypeError, ValueError):
            continue
        if entry_version != version:
            continue
        for trace in entry.get("traces", []):
            speedup = speedup_from_trace(trace)
            if speedup is not None:
                speedups.append(speedup)
    return min(speedups) if speedups else None


def row_min_speedup(row: dict[str, str]) -> float | None:
    try:
        version = int(row["correct_kernel_version"])
    except (KeyError, TypeError, ValueError):
        return None
    success_dir = Path(row["exp_dir"]) / "success" / row["trajectory_id"]
    return min_speedup_for_version(success_dir, version)


def main() -> None:
    args = parse_args()
    if args.cap_per_definition <= 0:
        raise SystemExit("--cap-per-definition must be positive")

    with args.input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise SystemExit(f"{args.input_csv}: missing CSV header")

    selected_fieldnames = fieldnames + [
        "selection_min_speedup",
        "selection_rank_in_definition",
    ]
    by_definition: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_speedup = 0
    below_threshold = 0

    for row in rows:
        speedup = row_min_speedup(row)
        if speedup is None:
            missing_speedup += 1
            continue
        if speedup <= args.min_speedup:
            below_threshold += 1
            continue
        enriched = dict(row)
        enriched["selection_min_speedup"] = f"{speedup:.12g}"
        by_definition[row.get("definition", "")].append(enriched)

    selected: list[dict[str, str]] = []
    print("definition,eligible,selected,cutoff_speedup")
    for definition in sorted(by_definition):
        group = sorted(
            by_definition[definition],
            key=lambda row: (
                -float(row["selection_min_speedup"]),
                row.get("trajectory_id", ""),
                row.get("correct_kernel_version", ""),
                row.get("correct_kernel_path", ""),
            ),
        )
        keep = group[: args.cap_per_definition]
        for rank, row in enumerate(keep, 1):
            row["selection_rank_in_definition"] = str(rank)
        selected.extend(keep)
        cutoff = keep[-1]["selection_min_speedup"] if keep else ""
        print(f"{definition},{len(group)},{len(keep)},{cutoff}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"input_rows={len(rows)}")
    print(f"missing_speedup={missing_speedup}")
    print(f"filtered_speedup_lte_{args.min_speedup:g}={below_threshold}")
    print(f"selected_rows={len(selected)}")
    print(f"wrote_csv={args.output_csv}")


if __name__ == "__main__":
    main()
