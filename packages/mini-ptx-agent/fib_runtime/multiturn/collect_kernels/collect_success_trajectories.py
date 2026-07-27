#!/usr/bin/env python3
"""Collect trajectory paths for successful experiments in eval run folders."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


FIELDNAMES = [
    "run_dir",
    "run_name",
    "exp_name",
    "trajectory_path",
    "success_dir",
    "success_kernel_count",
    "success_kernel_paths",
]


@dataclass(frozen=True)
class SuccessTrajectory:
    run_dir: Path
    exp_name: str
    trajectory_path: Path
    success_dir: Path
    kernel_paths: tuple[Path, ...]

    def to_row(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "run_name": self.run_dir.name,
            "exp_name": self.exp_name,
            "trajectory_path": str(self.trajectory_path),
            "success_dir": str(self.success_dir),
            "success_kernel_count": str(len(self.kernel_paths)),
            "success_kernel_paths": ";".join(str(path) for path in self.kernel_paths),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Given one or more eval run directories, write a CSV containing the "
            "trajectory paths for experiments with real success kernels."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Eval run directories, for example /home/ubuntu/AccRL-exps/eval_runs/2026-0608-1500.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-missing-trajectories",
        action="store_true",
        help=(
            "Include rows for success dirs whose trajectories/exp_NNN.json file "
            "is missing. By default these are reported to stderr and skipped."
        ),
    )
    return parser.parse_args()


def exp_sort_key(path: Path) -> tuple[int, str]:
    if path.name.startswith("exp_"):
        try:
            return int(path.name.removeprefix("exp_")), path.name
        except ValueError:
            pass
    return sys.maxsize, path.name


def collect_success_trajectories(
    run_dir: Path,
    include_missing_trajectories: bool,
) -> list[SuccessTrajectory]:
    run_dir = run_dir.expanduser().resolve(strict=False)
    success_root = run_dir / "success"
    trajectories_root = run_dir / "trajectories"

    if not run_dir.is_dir():
        print(f"warning: run directory does not exist: {run_dir}", file=sys.stderr)
        return []
    if not success_root.is_dir():
        print(f"warning: missing success directory: {success_root}", file=sys.stderr)
        return []

    rows: list[SuccessTrajectory] = []
    for success_dir in sorted((path for path in success_root.iterdir() if path.is_dir()), key=exp_sort_key):
        kernel_paths = tuple(sorted(success_dir.glob("kernel_v*.cu")))
        if not kernel_paths:
            continue

        trajectory_path = trajectories_root / f"{success_dir.name}.json"
        if not trajectory_path.is_file() and not include_missing_trajectories:
            print(f"warning: missing trajectory for {success_dir}: {trajectory_path}", file=sys.stderr)
            continue

        rows.append(
            SuccessTrajectory(
                run_dir=run_dir,
                exp_name=success_dir.name,
                trajectory_path=trajectory_path,
                success_dir=success_dir,
                kernel_paths=kernel_paths,
            )
        )

    return rows


def write_rows(path: Path, rows: list[SuccessTrajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(row.to_row() for row in rows)


def main() -> None:
    args = parse_args()
    rows: list[SuccessTrajectory] = []
    for run_dir in args.run_dirs:
        rows.extend(collect_success_trajectories(run_dir, args.include_missing_trajectories))

    write_rows(args.output, rows)
    print(f"wrote {len(rows)} success trajectories to {args.output}")


if __name__ == "__main__":
    main()
