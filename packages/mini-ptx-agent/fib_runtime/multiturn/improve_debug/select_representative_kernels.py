#!/usr/bin/env python3
"""Select representative error and correct kernels for debug iteration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ERROR_CATEGORIES = {"Runtime error", "Kernel Execution Timeout"}


def first_message(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    for line in log_path.read_text(errors="replace").splitlines():
        if any(
            token in line
            for token in (
                "CUDA error",
                "TIMEOUT",
                "Runtime error",
                "torch.AcceleratorError",
                "Evaluation timeout",
                "PASSED",
                "Kernel is correct",
            )
        ):
            return line.strip()
    return ""


def select(root: Path, per_category: int, correct_count: int) -> list[dict[str, object]]:
    csv_path = root / "figures" / "turn_correctness_arch.csv"
    kernel_root = root / "kernels"
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            category = row["correctness"]
            if category not in ERROR_CATEGORIES and category != "Correct":
                continue
            exp = row["trajectory_id"]
            turn = int(row["turn"])
            kernel_path = kernel_root / exp / f"kernel_t{turn}.cu"
            log_path = kernel_root / exp / f"log_t{turn}.txt"
            if not kernel_path.exists():
                continue
            rows.append(
                {
                    "category": category,
                    "exp": exp,
                    "turn": turn,
                    "kernel": str(kernel_path),
                    "log": str(log_path),
                    "message": first_message(log_path),
                }
            )

    selected: list[dict[str, object]] = []
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)

    for category in ("Runtime error", "Kernel Execution Timeout", "Correct"):
        target_count = correct_count if category == "Correct" else per_category
        if target_count <= 0:
            continue
        by_exp: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in by_category[category]:
            by_exp[str(row["exp"])].append(row)
        for exp_rows in by_exp.values():
            exp_rows.sort(key=lambda item: int(item["turn"]))

        while sum(1 for item in selected if item["category"] == category) < target_count:
            added = False
            for exp in sorted(by_exp):
                if sum(1 for item in selected if item["category"] == category) >= target_count:
                    break
                if by_exp[exp]:
                    selected.append(by_exp[exp].pop(0))
                    added = True
            if not added:
                break

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128"),
    )
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument(
        "--correct-count",
        type=int,
        default=0,
        help="Number of Correct kernels to append if available.",
    )
    parser.add_argument("--output", type=Path, default=Path("selected_error_kernels.json"))
    args = parser.parse_args()

    selected = select(args.eval_root, args.per_category, args.correct_count)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    print(f"wrote {len(selected)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
