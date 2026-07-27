#!/usr/bin/env python3
"""Convert masked-kernel CSV rows into a prompt config JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MIGRATION_CSV = Path("/home/ubuntu/AccRL/benchmark/migration.csv")
REQUIRED_COLUMNS = ("speedup", "masked_kernel_path", "definition", "workload", "test_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prompt_configs-style JSON from a masked-kernels CSV. "
            "Each CSV row becomes one JSON item whose target_speedup is the "
            "row speedup multiplied by --speedup-factor."
        )
    )
    parser.add_argument(
        "--speedup-factor",
        type=float,
        default=1.0,
        help="Multiplier applied to each CSV speedup when writing target_speedup. Defaults to 1.0.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input masked-kernels CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        required=True,
        help="num_trajectories value for every output item.",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        required=True,
        help="num_turns value for every output item.",
    )
    parser.add_argument(
        "--prompt-tag",
        required=True,
        help="prompt_tag value for every output item.",
    )
    parser.add_argument(
        "--migration-csv",
        type=Path,
        default=DEFAULT_MIGRATION_CSV,
        help=f"CSV mapping old definition/workload names to latest names. Defaults to {DEFAULT_MIGRATION_CSV}.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation level. Defaults to 4.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be >= 1")
    if args.num_turns < 1:
        raise ValueError("--num-turns must be >= 1")
    if args.speedup_factor <= 0:
        raise ValueError("--speedup-factor must be > 0")
    if args.indent < 0:
        raise ValueError("--indent must be >= 0")
    if not args.migration_csv.is_file():
        raise ValueError(f"--migration-csv does not exist: {args.migration_csv}")


def load_migrations(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"old_definition", "old_workload", "new_definition", "new_workload"}
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

        migrations: dict[tuple[str, str], tuple[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            old_definition = row["old_definition"].strip()
            old_workload = row["old_workload"].strip()
            new_definition = row["new_definition"].strip()
            new_workload = row["new_workload"].strip()
            if not (old_definition and old_workload and new_definition and new_workload):
                raise ValueError(f"incomplete migration row at {path}:{line_number}")
            migrations[(old_definition, old_workload)] = (new_definition, new_workload)
        return migrations


def latest_definition_workload(
    row: dict[str, str],
    migrations: dict[tuple[str, str], tuple[str, str]],
    *,
    line_number: int,
) -> tuple[str, str]:
    definition = row.get("definition", "").strip()
    workload = row.get("workload", "").strip()
    if not definition:
        raise ValueError(f"missing definition at CSV line {line_number}")
    if not workload:
        raise ValueError(f"missing workload at CSV line {line_number}")
    return migrations.get((definition, workload), (definition, workload))


def row_test_path(row: dict[str, str], *, line_number: int) -> str:
    test_path_text = row.get("test_path", "").strip()
    if not test_path_text:
        raise ValueError(f"missing test_path at CSV line {line_number}")
    test_path = Path(test_path_text).expanduser()
    if not test_path.is_file():
        raise ValueError(f"test_path does not exist at CSV line {line_number}: {test_path}")
    return str(test_path.resolve())


def build_config(
    rows: list[dict[str, str]],
    *,
    num_trajectories: int,
    num_turns: int,
    prompt_tag: str,
    speedup_factor: float,
    migrations: dict[tuple[str, str], tuple[str, str]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        speedup_text = row.get("speedup", "").strip()
        masked_kernel_path = row.get("masked_kernel_path", "").strip()
        if not speedup_text:
            raise ValueError(f"missing speedup at CSV line {index}")
        if not masked_kernel_path:
            raise ValueError(f"missing masked_kernel_path at CSV line {index}")

        try:
            target_speedup = float(speedup_text) * speedup_factor
        except ValueError as exc:
            raise ValueError(f"invalid speedup {speedup_text!r} at CSV line {index}") from exc
        definition, _workload = latest_definition_workload(row, migrations, line_number=index)
        test_path = row_test_path(row, line_number=index)

        items.append(
            {
                "num_trajectories": num_trajectories,
                "num_turns": num_turns,
                "target_speedup": target_speedup,
                "prompt_tag": prompt_tag,
                "masked_kernel_path": masked_kernel_path,
                "definition": definition,
                "test_path": str(test_path),
            }
        )
    return items


def write_json(path: Path, items: list[dict[str, Any]], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=indent) + "\n")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        fieldnames, rows = read_rows(args.input_csv)
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{args.input_csv} is missing required column(s): {missing}")

        migrations = load_migrations(args.migration_csv)
        items = build_config(
            rows,
            num_trajectories=args.num_trajectories,
            num_turns=args.num_turns,
            prompt_tag=args.prompt_tag,
            speedup_factor=args.speedup_factor,
            migrations=migrations,
        )
        write_json(args.output, items, args.indent)
        print(f"wrote {len(items)} prompt config entries to {args.output}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
