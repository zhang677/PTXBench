#!/usr/bin/env python3
"""Select failed per-turn kernels from AccRL multiturn eval runs.

The source of truth for turn labels is:

    <exp_dir>/figures/turn_correctness_arch.csv

If that CSV is missing, this script generates it with the repository-root
benchmark/export_turn_correctness_arch.py. If <exp_dir>/kernels does not exist,
it extracts per-turn kernels and logs with analyze_kernel_per_turn.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


MINI_PTX_AGENT_ROOT = Path(__file__).resolve().parents[3]
PTXBENCH_ROOT = MINI_PTX_AGENT_ROOT.parents[1]
MULTITURN_ROOT = Path(__file__).resolve().parents[1]
EXPORT_TURN_CORRECTNESS = (
    PTXBENCH_ROOT / "benchmark" / "export_turn_correctness_arch.py"
)
ANALYZE_KERNEL_PER_TURN = MULTITURN_ROOT / "analyze_kernel_per_turn.py"
TURN_CSV_REL = Path("figures") / "turn_correctness_arch.csv"
DEFAULT_FAILURE_LABELS = (
    "Runtime error",
    "Kernel Execution Timeout",
    "Numerical error",
    "Compilation error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select generated kernels whose turn_correctness_arch.csv label is "
            "one of the requested failure labels."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--selected-runs-csv",
        type=Path,
        help="CSV with exp_dir rows, usually tasks/selected-runs-*.csv.",
    )
    source.add_argument(
        "--exp-dir",
        type=Path,
        help="Single eval run directory.",
    )
    parser.add_argument(
        "--arch",
        help="Architecture for --exp-dir, or fallback when selected-runs rows lack arch.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV containing selected failed kernels and metadata.",
    )
    parser.add_argument(
        "--failure-label",
        action="append",
        dest="failure_labels",
        help=(
            "Correctness label to select. May be repeated. Defaults to runtime, "
            "timeout, numerical, and compilation errors."
        ),
    )
    parser.add_argument(
        "--force-turn-csv",
        action="store_true",
        help="Regenerate turn_correctness_arch.csv even when it already exists.",
    )
    parser.add_argument(
        "--force-kernels",
        action="store_true",
        help="Regenerate per-turn kernels/logs even when the kernels directory exists.",
    )
    parser.add_argument(
        "--allow-non-fenced-turns",
        action="store_true",
        help=(
            "Include failed turns whose assistant message does not end with a closing "
            "code fence. By default these malformed turns are skipped."
        ),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def load_run_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.exp_dir is not None:
        row = {"exp_dir": str(args.exp_dir.expanduser())}
        if args.arch:
            row["arch"] = args.arch
        return [row]

    fieldnames, rows = read_csv_rows(args.selected_runs_csv)
    if "exp_dir" not in fieldnames:
        raise ValueError(f"{args.selected_runs_csv} is missing required exp_dir column")
    for row in rows:
        for field in ("exp_dir", "test_path"):
            if row.get(field):
                row[field] = os.path.expandvars(row[field])
    return rows


def first_nonempty(rows: list[dict[str, str]], field: str) -> str:
    for row in rows:
        value = row.get(field, "").strip()
        if value:
            return value
    return ""


def group_rows_by_exp_dir(rows: list[dict[str, str]]) -> dict[Path, list[dict[str, str]]]:
    grouped: dict[Path, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        exp_dir = row.get("exp_dir", "").strip()
        if not exp_dir:
            continue
        grouped[Path(exp_dir).expanduser()].append(row)
    return dict(sorted(grouped.items(), key=lambda item: str(item[0])))


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_turn_csv(
    exp_dir: Path,
    metadata_rows: list[dict[str, str]],
    *,
    fallback_arch: str | None,
    force: bool,
) -> Path:
    turn_csv = exp_dir / TURN_CSV_REL
    if turn_csv.is_file() and not force:
        return turn_csv

    arch = first_nonempty(metadata_rows, "arch") or (fallback_arch or "")
    if not arch:
        raise ValueError(
            f"{exp_dir} is missing {TURN_CSV_REL} and no arch is available to generate it"
        )

    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as f:
        temp_csv = Path(f.name)
        fieldnames = sorted({key for row in metadata_rows for key in row.keys()} | {"arch", "exp_dir"})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row = dict(metadata_rows[0]) if metadata_rows else {}
        row["exp_dir"] = str(exp_dir)
        row["arch"] = arch
        writer.writerow(row)

    try:
        cmd = [
            sys.executable,
            str(EXPORT_TURN_CORRECTNESS),
            "--experiments-csv",
            str(temp_csv),
            "--skip-sass-verification",
        ]
        if force:
            cmd.append("--force")
        run_command(cmd)
    finally:
        temp_csv.unlink(missing_ok=True)

    if not turn_csv.is_file():
        raise FileNotFoundError(f"expected generated CSV does not exist: {turn_csv}")
    return turn_csv


def ensure_kernels(exp_dir: Path, *, force: bool) -> Path:
    kernels_dir = exp_dir / "kernels"
    if kernels_dir.is_dir() and not force:
        return kernels_dir

    run_command(
        [
            sys.executable,
            str(ANALYZE_KERNEL_PER_TURN),
            "--run-dir",
            str(exp_dir),
        ]
    )
    if not kernels_dir.is_dir():
        raise FileNotFoundError(f"expected generated kernels directory does not exist: {kernels_dir}")
    return kernels_dir


def assistant_turn_ends_with_code_fence(exp_dir: Path, trajectory_id: str, turn: str) -> bool:
    try:
        turn_index = int(turn)
    except ValueError:
        return False

    trajectory_path = exp_dir / "trajectories" / f"{trajectory_id}.json"
    with trajectory_path.open() as f:
        trajectory = json.load(f)

    assistant_turn = 0
    for message in trajectory.get("messages", []):
        if message.get("role") != "assistant":
            continue
        if assistant_turn == turn_index:
            return (message.get("content") or "").rstrip().endswith("```")
        assistant_turn += 1

    return False


def selected_rows_for_run(
    exp_dir: Path,
    metadata_rows: list[dict[str, str]],
    turn_csv: Path,
    failure_labels: set[str],
    *,
    require_closing_code_fence: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    turn_rows = read_csv_rows(turn_csv)[1]
    metadata = metadata_rows[0] if metadata_rows else {}
    output_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for turn_row in turn_rows:
        correctness = turn_row.get("correctness", "").strip()
        if correctness not in failure_labels:
            continue

        trajectory_id = turn_row.get("trajectory_id", "").strip()
        turn = turn_row.get("turn", "").strip()
        if not trajectory_id or not turn:
            warnings.append(f"skipping malformed row in {turn_csv}: {turn_row}")
            continue

        if require_closing_code_fence:
            try:
                if not assistant_turn_ends_with_code_fence(exp_dir, trajectory_id, turn):
                    warnings.append(
                        "skipping assistant turn without closing code fence: "
                        f"{exp_dir}/trajectories/{trajectory_id}.json turn {turn}"
                    )
                    continue
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                warnings.append(
                    "skipping turn with unreadable trajectory: "
                    f"{exp_dir}/trajectories/{trajectory_id}.json turn {turn}: {exc}"
                )
                continue

        kernel_path = exp_dir / "kernels" / trajectory_id / f"kernel_t{turn}.cu"
        log_path = exp_dir / "kernels" / trajectory_id / f"log_t{turn}.txt"
        if not kernel_path.is_file():
            warnings.append(f"skipping missing kernel: {kernel_path}")
            continue
        if not log_path.is_file():
            warnings.append(f"skipping missing error log: {log_path}")
            continue

        output_rows.append(
            {
                "exp_dir": str(exp_dir),
                "model": metadata.get("model", ""),
                "arch": metadata.get("arch", ""),
                "definition": metadata.get("definition", ""),
                "workload": metadata.get("workload", ""),
                "test_path": metadata.get("test_path", ""),
                "trajectory_id": trajectory_id,
                "turn": turn,
                "sass_arch_tag": turn_row.get("sass_arch_tag", ""),
                "turn_csv": str(turn_csv),
                "error_kernel_path": str(kernel_path),
                "error_log_path": str(log_path),
            }
        )

    return output_rows, warnings


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "exp_dir",
        "model",
        "arch",
        "definition",
        "workload",
        "test_path",
        "trajectory_id",
        "turn",
        "sass_arch_tag",
        "turn_csv",
        "error_kernel_path",
        "error_log_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    failure_labels = set(args.failure_labels or DEFAULT_FAILURE_LABELS)
    run_rows = load_run_rows(args)
    rows_by_exp_dir = group_rows_by_exp_dir(run_rows)

    selected: list[dict[str, str]] = []
    warnings: list[str] = []
    for exp_dir, metadata_rows in rows_by_exp_dir.items():
        turn_csv = ensure_turn_csv(
            exp_dir,
            metadata_rows,
            fallback_arch=args.arch,
            force=args.force_turn_csv,
        )
        ensure_kernels(exp_dir, force=args.force_kernels)
        rows, run_warnings = selected_rows_for_run(
            exp_dir,
            metadata_rows,
            turn_csv,
            failure_labels,
            require_closing_code_fence=not args.allow_non_fenced_turns,
        )
        selected.extend(rows)
        warnings.extend(run_warnings)

    write_output(args.output_csv, selected)
    print(f"runs={len(rows_by_exp_dir)}")
    print(f"failure_labels={','.join(sorted(failure_labels))}")
    print(f"require_closing_code_fence={not args.allow_non_fenced_turns}")
    print(f"selected_kernels={len(selected)}")
    print(f"wrote_csv={args.output_csv}")
    if warnings:
        print(f"warnings={len(warnings)}", file=sys.stderr)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
