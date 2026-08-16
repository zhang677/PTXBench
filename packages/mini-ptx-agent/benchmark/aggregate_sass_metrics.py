#!/usr/bin/env python3
"""Build plot-viewer architecture metrics from native per-turn SASS tags."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

ARCH_SASS_TAGS = {
    "hopper": "H",
    "blackwell": "B",
}
DEFAULT_TURN_LIMITS = (1, 4, 8)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "plot_viewer_v1" / "data" / "arch_instruction_correctness_by_release_date.csv"
)
DEFAULT_MODEL_DATES = Path(__file__).resolve().parent / "plot_viewer_v1" / "data" / "model_release_dates.csv"
OUTPUT_FIELDS = [
    "model",
    "arch",
    "required_arch_sass_tag",
    "workload",
    "workload_name",
    "definition",
    "date",
    "turn_limit",
    "n_trajectories",
    "correct_with_arch",
    "correct_with_arch_rate",
    "n_correct_turns",
    "n_unknown_correct_turns",
    "tag_evidence",
]
CONCLUSIVE_STATUSES = {
    "dynamic_present",
    "cubin_sass_absent",
    "dynamic_not_executed",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def split_tags(value: str) -> set[str]:
    return {tag.strip() for tag in value.split(",") if tag.strip()}


def load_model_dates(path: Path) -> dict[str, str]:
    fieldnames, rows = read_csv(path)
    missing = {"model", "date"}.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return {row["model"]: row["date"] for row in rows}


def aggregate_experiment(
    experiment: dict[str, str],
    *,
    model_dates: dict[str, str],
    turn_limits: tuple[int, ...],
) -> list[dict[str, object]]:
    arch = experiment["arch"].strip().lower()
    if arch not in ARCH_SASS_TAGS:
        supported = ", ".join(sorted(ARCH_SASS_TAGS))
        raise ValueError(f"unsupported arch {experiment['arch']!r}; expected one of: {supported}")
    model = experiment["model"].strip()
    date = experiment.get("date", "").strip() or model_dates.get(model, "")
    if not date:
        raise ValueError(f"no release date for model {model!r}")

    exp_dir = expand_path(experiment["exp_dir"])
    turn_csv = exp_dir / "figures" / "turn_correctness_arch.csv"
    fieldnames, rows = read_csv(turn_csv)
    required = {
        "trajectory_id",
        "turn",
        "correctness",
        "sass_arch_tag",
        "sass_verification_status",
    }
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{turn_csv} is not a native SASS export; missing: {', '.join(sorted(missing))}")

    by_trajectory: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        trajectory_id = row.get("trajectory_id", "").strip()
        if trajectory_id:
            by_trajectory[trajectory_id].append(row)

    required_tag = ARCH_SASS_TAGS[arch]
    output = []
    for turn_limit in turn_limits:
        correct_with_arch = 0
        correct_turns = 0
        unknown_correct_turns = 0
        for trajectory_rows in by_trajectory.values():
            selected = [row for row in trajectory_rows if int(row["turn"]) < turn_limit]
            correct_rows = [row for row in selected if row["correctness"] == "Correct"]
            correct_turns += len(correct_rows)
            unknown_correct_turns += sum(
                row.get("sass_verification_status", "") not in CONCLUSIVE_STATUSES for row in correct_rows
            )
            if any(
                row.get("sass_verification_status", "") == "dynamic_present"
                and required_tag in split_tags(row.get("sass_arch_tag", ""))
                for row in correct_rows
            ):
                correct_with_arch += 1

        n_trajectories = len(by_trajectory)
        output.append(
            {
                "model": model,
                "arch": arch,
                "required_arch_sass_tag": required_tag,
                "workload": experiment["workload"],
                "workload_name": experiment.get("workload_name", "").strip() or experiment["definition"],
                "definition": experiment["definition"],
                "date": date,
                "turn_limit": turn_limit,
                "n_trajectories": n_trajectories,
                "correct_with_arch": correct_with_arch,
                "correct_with_arch_rate": (correct_with_arch / n_trajectories if n_trajectories else 0.0),
                "n_correct_turns": correct_turns,
                "n_unknown_correct_turns": unknown_correct_turns,
                "tag_evidence": "dynamic_sass",
            }
        )
    return output


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def parse_turn_limits(value: str) -> tuple[int, ...]:
    limits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not limits or any(limit < 1 for limit in limits):
        raise argparse.ArgumentTypeError("turn limits must be positive integers")
    return limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        required=True,
        help="CSV with model, arch, definition, workload, and exp_dir columns",
    )
    parser.add_argument("--model-release-dates", type=Path, default=DEFAULT_MODEL_DATES)
    parser.add_argument("--turn-limits", type=parse_turn_limits, default=DEFAULT_TURN_LIMITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, experiments = read_csv(args.experiments_csv)
    required = {"model", "arch", "definition", "workload", "exp_dir"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{args.experiments_csv} is missing required columns: {', '.join(sorted(missing))}")
    model_dates = load_model_dates(args.model_release_dates)
    rows = []
    for experiment in experiments:
        rows.extend(
            aggregate_experiment(
                experiment,
                model_dates=model_dates,
                turn_limits=args.turn_limits,
            )
        )
    write_csv_atomic(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
