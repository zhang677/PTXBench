#!/usr/bin/env python3
"""Collect correct per-turn kernels from selected multiturn eval runs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ARCH_SASS_TAGS = {
    "hopper": "H",
    "blackwell": "B",
}


def expand_path(value: str) -> Path:
    """Resolve paths containing PTXBench environment variables."""
    return Path(os.path.expandvars(value)).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect kernels whose per-turn CSV rows are Correct, have the expected "
            "dynamically verified SASS tag, and exceed --min-speedup."
        )
    )
    parser.add_argument(
        "selected_runs_csv",
        nargs="?",
        type=Path,
        help="CSV with model,arch,definition,workload,exp_dir,test_path rows.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        required=True,
        help="Only include kernels with speedup strictly greater than this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def expected_arch_sass_tag(arch: str) -> str:
    key = arch.strip().lower()
    if key not in ARCH_SASS_TAGS:
        supported = ", ".join(sorted(ARCH_SASS_TAGS))
        raise ValueError(f"unsupported arch {arch!r}; supported values: {supported}")
    return ARCH_SASS_TAGS[key]


def arch_sass_tag_matches(actual: str, expected: str) -> bool:
    tags = {tag.strip() for tag in actual.split(",") if tag.strip()}
    return expected in tags


def ensure_kernels_dir(exp_dir: Path) -> None:
    kernels_dir = exp_dir / "kernels"
    if kernels_dir.is_dir():
        return

    raise FileNotFoundError(
        f"{kernels_dir} is missing; extract the KernelGen source-data bundle "
        "before collecting kernels"
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_rows(selected_runs_csv: Path, min_speedup: float) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []

    for run_row in read_csv_rows(selected_runs_csv):
        test_path = run_row.get("test_path", "").strip()
        if not test_path:
            print(f"warning: missing test_path for exp_dir {run_row.get('exp_dir', '?')}; skipping", file=sys.stderr)
            continue
        test_path_obj = expand_path(test_path)
        if not test_path_obj.is_file():
            print(f"warning: test_path does not exist: {test_path}; skipping", file=sys.stderr)
            continue

        exp_dir = expand_path(run_row["exp_dir"])
        expected_tag = expected_arch_sass_tag(run_row["arch"])
        turn_csv = exp_dir / "figures" / "turn_correctness_arch.csv"
        if not turn_csv.is_file():
            print(f"warning: missing {turn_csv}; skipping", file=sys.stderr)
            continue

        ensure_kernels_dir(exp_dir)

        for turn_row in read_csv_rows(turn_csv):
            if turn_row.get("correctness") != "Correct":
                continue
            if turn_row.get("sass_verification_status") != "dynamic_present":
                continue

            try:
                speedup = float(turn_row.get("speedup", ""))
            except ValueError:
                continue

            if speedup <= min_speedup:
                continue
            if not arch_sass_tag_matches(
                turn_row.get("arch_sass_tag", ""), expected_tag
            ):
                continue

            trajectory_id = turn_row["trajectory_id"]
            turn = int(turn_row["turn"])
            kernel_path = exp_dir / "kernels" / trajectory_id / f"kernel_t{turn}.cu"
            if not kernel_path.is_file():
                print(f"warning: missing {kernel_path}; skipping", file=sys.stderr)
                continue

            output_rows.append(
                {
                    "model": run_row["model"],
                    "arch": run_row["arch"],
                    "definition": run_row["definition"],
                    "workload": run_row["workload"],
                    "exp_dir": str(exp_dir),
                    "test_path": str(test_path_obj.resolve()),
                    "kernel_path": str(kernel_path),
                    "speedup": str(speedup),
                }
            )

    return output_rows


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["model", "arch", "definition", "workload", "exp_dir", "test_path", "kernel_path", "speedup"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.selected_runs_csv, args.min_speedup)
    write_output(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
