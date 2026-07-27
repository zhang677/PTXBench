#!/usr/bin/env python3
"""Analyze trajectory throughput from the profile-scaling watcher logs.

Metrics:
  * First-pass throughput:
      trajectory completion events in the first launcher pass
      / first launcher-pass wall time.
  * Whole-experiment throughput:
      unique planned trajectories in the final successful audit
      / watcher elapsed time from the terminal experiment-timing line.

The whole-experiment numerator intentionally excludes repeated resume attempts.
Their cost remains included in the elapsed-time denominator.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "profile-scaling-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TIMESTAMP = r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)"
RUNNER_START_RE = re.compile(
    rf"^{TIMESTAMP} starting eval runner with mixed resume/fresh roots"
)
RUNNER_EXIT_RE = re.compile(rf"^{TIMESTAMP} eval launcher exited rc=\d+")
COMPLETION_RE = re.compile(r"^\[(?P<completed>\d+)/(?P<total>\d+)\]")
FINAL_TIMING_RE = re.compile(
    rf"^{TIMESTAMP} experiment timing MAX_PARALLEL=(?P<workers>\d+) "
    rf"start=(?P<start>\S+) end=(?P<end>\S+) elapsed=(?P<elapsed>\S+) "
    rf"elapsed_seconds=(?P<elapsed_seconds>\d+)"
)
FINAL_AUDIT_RE = re.compile(r"^[A-Za-z0-9_]+: ok plan=(?P<plan>\d+)\b")


@dataclass(frozen=True)
class Metrics:
    workers: int
    log_path: str
    planned_trajectories: int
    first_pass_completed: int
    first_pass_start_utc: str
    first_pass_end_utc: str
    first_pass_elapsed_seconds: int
    first_pass_throughput_per_hour: float
    experiment_start_utc: str
    experiment_end_utc: str
    experiment_elapsed_seconds: int
    whole_experiment_throughput_per_hour: float
    all_completion_events: int
    retry_completion_events: int


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, UTC_FORMAT).replace(tzinfo=timezone.utc)


def parse_log(path: Path) -> Metrics:
    lines = path.read_text(encoding="utf-8").splitlines()

    final_timing_index = None
    final_timing = None
    for index, line in enumerate(lines):
        if match := FINAL_TIMING_RE.search(line):
            final_timing_index = index
            final_timing = match

    if final_timing_index is None or final_timing is None:
        raise ValueError(f"{path}: missing terminal experiment-timing line")

    # A tmux pane can contain an interrupted watcher invocation before the
    # completed invocation. Scope launcher parsing to the experiment start
    # recorded by the final timing line so stale pane history is ignored.
    experiment_start = parse_utc(final_timing.group("start"))
    first_start_index = None
    first_start = None
    first_exit_index = None
    first_exit = None

    for index, line in enumerate(lines[: final_timing_index + 1]):
        if first_start is None and (match := RUNNER_START_RE.search(line)):
            timestamp = match.group("timestamp")
            if parse_utc(timestamp) >= experiment_start:
                first_start_index = index
                first_start = timestamp
            continue
        if (
            first_start is not None
            and first_exit is None
            and (match := RUNNER_EXIT_RE.search(line))
        ):
            first_exit_index = index
            first_exit = match.group("timestamp")

    if first_start_index is None or first_start is None:
        raise ValueError(f"{path}: missing first runner-start line")
    if first_exit_index is None or first_exit is None:
        raise ValueError(f"{path}: missing first runner-exit line")

    first_pass_completions = [
        match
        for line in lines[first_start_index + 1 : first_exit_index]
        if (match := COMPLETION_RE.search(line))
    ]
    if not first_pass_completions:
        raise ValueError(f"{path}: no first-pass trajectory completion lines")

    all_completions = [
        match
        for line in lines[first_start_index + 1 : final_timing_index]
        if (match := COMPLETION_RE.search(line))
    ]

    all_roots_indices = [
        index
        for index, line in enumerate(lines[: final_timing_index + 1])
        if " all roots complete" in line
    ]
    if not all_roots_indices:
        raise ValueError(f"{path}: missing all-roots-complete line")
    all_roots_index = all_roots_indices[-1]
    audit_start_indices = [
        index
        for index, line in enumerate(lines[:all_roots_index])
        if " no active eval process found; auditing roots" in line
    ]
    audit_start_index = audit_start_indices[-1] if audit_start_indices else 0
    planned_trajectories = sum(
        int(match.group("plan"))
        for line in lines[audit_start_index:all_roots_index]
        if (match := FINAL_AUDIT_RE.search(line))
    )
    if planned_trajectories == 0:
        planned_trajectories = max(
            int(match.group("total")) for match in first_pass_completions
        )

    first_pass_elapsed_seconds = int(
        (parse_utc(first_exit) - parse_utc(first_start)).total_seconds()
    )
    experiment_elapsed_seconds = int(final_timing.group("elapsed_seconds"))
    if first_pass_elapsed_seconds <= 0 or experiment_elapsed_seconds <= 0:
        raise ValueError(f"{path}: elapsed times must be positive")

    first_pass_completed = len(first_pass_completions)
    all_completion_events = len(all_completions)

    return Metrics(
        workers=int(final_timing.group("workers")),
        log_path=str(path),
        planned_trajectories=planned_trajectories,
        first_pass_completed=first_pass_completed,
        first_pass_start_utc=first_start,
        first_pass_end_utc=first_exit,
        first_pass_elapsed_seconds=first_pass_elapsed_seconds,
        first_pass_throughput_per_hour=(
            first_pass_completed * 3600 / first_pass_elapsed_seconds
        ),
        experiment_start_utc=final_timing.group("start"),
        experiment_end_utc=final_timing.group("end"),
        experiment_elapsed_seconds=experiment_elapsed_seconds,
        whole_experiment_throughput_per_hour=(
            planned_trajectories * 3600 / experiment_elapsed_seconds
        ),
        all_completion_events=all_completion_events,
        retry_completion_events=max(0, all_completion_events - first_pass_completed),
    )


def write_csv(metrics: list[Metrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(metrics[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in metrics:
            row = asdict(item)
            row["first_pass_throughput_per_hour"] = (
                f"{item.first_pass_throughput_per_hour:.6f}"
            )
            row["whole_experiment_throughput_per_hour"] = (
                f"{item.whole_experiment_throughput_per_hour:.6f}"
            )
            writer.writerow(row)


def plot_metric(
    metrics: list[Metrics],
    attribute: str,
    title: str,
    output_path: Path,
) -> None:
    workers = [item.workers for item in metrics]
    values = [getattr(item, attribute) for item in metrics]

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(workers, values, marker="o", linewidth=2.0, markersize=7)
    for worker, value in zip(workers, values, strict=True):
        axis.annotate(
            f"{value:.2f}",
            (worker, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axis.set_title(title)
    axis.set_xlabel("w (MAX_PARALLEL)")
    axis.set_ylabel("Trajectories per hour")
    axis.set_xscale("log", base=2)
    axis.set_xticks(workers, labels=[str(worker) for worker in workers])
    value_span = max(values) - min(values)
    vertical_padding = max(0.1, value_span * 0.2)
    axis.set_ylim(
        min(values) - vertical_padding,
        max(values) + vertical_padding,
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def print_table(metrics: list[Metrics]) -> None:
    print(
        f"{'w':>4}  {'first pass':>12}  {'whole exp':>12}  "
        f"{'first-pass s':>12}  {'experiment s':>12}  {'retries':>7}"
    )
    for item in metrics:
        print(
            f"{item.workers:>4}  "
            f"{item.first_pass_throughput_per_hour:>12.3f}  "
            f"{item.whole_experiment_throughput_per_hour:>12.3f}  "
            f"{item.first_pass_elapsed_seconds:>12}  "
            f"{item.experiment_elapsed_seconds:>12}  "
            f"{item.retry_completion_events:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze first-pass and whole-experiment trajectory throughput."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="Watcher log paths (default: raw_logs/profile-w*.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results"),
        help="Directory for the CSV and figures (default: analysis_results)",
    )
    args = parser.parse_args()

    log_paths = args.logs or sorted(
        path
        for path in Path("raw_logs").glob("profile-w*.txt")
        if re.fullmatch(r"profile-w\d+\.txt", path.name)
    )
    if not log_paths:
        parser.error("no input logs found")

    metrics = sorted((parse_log(path) for path in log_paths), key=lambda item: item.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "trajectory_throughput.csv"
    first_pass_figure = (
        args.output_dir / "first_pass_trajectory_throughput.png"
    )
    whole_experiment_figure = (
        args.output_dir / "whole_experiment_trajectory_throughput.png"
    )

    write_csv(metrics, csv_path)
    plot_metric(
        metrics,
        "first_pass_throughput_per_hour",
        "First-pass trajectory throughput",
        first_pass_figure,
    )
    plot_metric(
        metrics,
        "whole_experiment_throughput_per_hour",
        "Whole-experiment trajectory throughput",
        whole_experiment_figure,
    )

    print_table(metrics)
    print(f"\nCSV: {csv_path}")
    print(f"First-pass figure: {first_pass_figure}")
    print(f"Whole-experiment figure: {whole_experiment_figure}")


if __name__ == "__main__":
    main()
