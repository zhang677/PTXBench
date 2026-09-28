#!/usr/bin/env python3
"""Enrich correct-kernel collector output for reasoning synthesis.

Input is the unchanged collect_correct_kernels.py CSV. This script derives the
selected trajectory and turn from kernel_path, recovers prompt_tag from the run
plan, and reads profiling feedback from trajectory evaluation messages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


KERNEL_TURN_RE = re.compile(r"^kernel_t(?P<turn>\d+)\.cu$")

FIELDNAMES = [
    "model",
    "arch",
    "definition",
    "workload",
    "exp_dir",
    "test_path",
    "kernel_path",
    "correct_kernel_path",
    "trajectory_id",
    "turn",
    "prompt_tag",
    "sass_arch_tag",
    "correctness",
    "speedup",
    "turn_csv",
    "trajectory_path",
    "feedback_raw_output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--dropped-csv",
        type=Path,
        default=None,
        help="Rows that could not be enriched; defaults to <output-csv>.dropped.csv",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_kernel_identity(kernel_path: Path) -> tuple[str, int]:
    match = KERNEL_TURN_RE.match(kernel_path.name)
    if not match:
        raise ValueError(f"kernel_path does not end with kernel_tN.cu: {kernel_path}")
    trajectory_id = kernel_path.parent.name
    if not trajectory_id.startswith("exp_"):
        raise ValueError(f"kernel_path parent is not exp_NNN: {kernel_path}")
    return trajectory_id, int(match.group("turn"))


def exp_index(trajectory_id: str) -> int:
    if not trajectory_id.startswith("exp_"):
        raise ValueError(f"not an exp id: {trajectory_id}")
    return int(trajectory_id.removeprefix("exp_"))


def prompt_tag_from_plan(exp_dir: Path, trajectory_id: str) -> str:
    plan_path = exp_dir / "plan.json"
    if plan_path.is_file():
        data = load_json(plan_path)
        plan = data.get("plan") if isinstance(data, dict) else data
        if isinstance(plan, list):
            target_index = exp_index(trajectory_id)
            for item in plan:
                if not isinstance(item, dict):
                    continue
                if item.get("exp_index") == target_index:
                    value = item.get("prompt_tag")
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    summary_path = exp_dir / "summary.json"
    if summary_path.is_file():
        data = load_json(summary_path)
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                if item.get("exp_name") == trajectory_id:
                    value = item.get("prompt_tag")
                    if isinstance(value, str) and value.strip():
                        return value.strip()

    raise ValueError(f"could not recover prompt_tag for {exp_dir}/{trajectory_id}")


def turn_csv_row(exp_dir: Path, trajectory_id: str, turn: int) -> dict[str, str]:
    path = exp_dir / "figures" / "turn_correctness_arch.csv"
    if not path.is_file():
        raise ValueError(f"missing turn CSV: {path}")
    for row in read_csv_rows(path):
        if row.get("trajectory_id") == trajectory_id and row.get("turn") == str(turn):
            return dict(row, turn_csv=str(path))
    raise ValueError(f"missing turn row in {path}: {trajectory_id} turn={turn}")


def feedback_raw_output(trajectory_path: Path, turn: int) -> str:
    data = load_json(trajectory_path)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{trajectory_path}: missing messages list")

    assistant_turn = -1
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_turn += 1
        if assistant_turn != turn:
            continue

        if index + 1 >= len(messages):
            raise ValueError(f"{trajectory_path}: no evaluation after assistant turn {turn}")
        evaluation = messages[index + 1]
        if not isinstance(evaluation, dict):
            raise ValueError(f"{trajectory_path}: malformed evaluation after turn {turn}")
        extra = evaluation.get("extra")
        if not isinstance(extra, dict) or extra.get("event") != "evaluation":
            raise ValueError(f"{trajectory_path}: next message after turn {turn} is not evaluation")
        raw_output = extra.get("raw_output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ValueError(f"{trajectory_path}: missing extra.raw_output after turn {turn}")
        return raw_output

    raise ValueError(f"{trajectory_path}: assistant turn {turn} not found")


def enrich_row(row: dict[str, str]) -> dict[str, str]:
    exp_dir = Path(os.path.expandvars(row["exp_dir"])).expanduser()
    kernel_path = Path(os.path.expandvars(row["kernel_path"])).expanduser()
    trajectory_id, turn = parse_kernel_identity(kernel_path)
    prompt_tag = prompt_tag_from_plan(exp_dir, trajectory_id)
    turn_row = turn_csv_row(exp_dir, trajectory_id, turn)
    trajectory_path = exp_dir / "trajectories" / f"{trajectory_id}.json"
    feedback = feedback_raw_output(trajectory_path, turn)

    return {
        **{name: row.get(name, "") for name in FIELDNAMES},
        "kernel_path": str(kernel_path),
        "correct_kernel_path": str(kernel_path),
        "trajectory_id": trajectory_id,
        "turn": str(turn),
        "prompt_tag": prompt_tag,
        "sass_arch_tag": turn_row.get("sass_arch_tag", ""),
        "correctness": turn_row.get("correctness", ""),
        "speedup": turn_row.get("speedup", row.get("speedup", "")),
        "turn_csv": turn_row["turn_csv"],
        "trajectory_path": str(trajectory_path),
        "feedback_raw_output": feedback,
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dropped_path = args.dropped_csv or args.output_csv.with_suffix(
        args.output_csv.suffix + ".dropped.csv"
    )

    rows = read_csv_rows(args.input_csv)
    enriched: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        try:
            enriched.append(enrich_row(row))
        except Exception as exc:
            dropped.append({**row, "csv_row": str(index), "drop_reason": str(exc)})

    write_csv(args.output_csv, enriched, FIELDNAMES)
    dropped_fields = list(rows[0].keys()) + ["csv_row", "drop_reason"] if rows else [
        "csv_row",
        "drop_reason",
    ]
    write_csv(dropped_path, dropped, dropped_fields)

    print(f"input rows:    {len(rows)}")
    print(f"enriched rows: {len(enriched)} -> {args.output_csv}")
    print(f"dropped rows:  {len(dropped)} -> {dropped_path}")
    if dropped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
