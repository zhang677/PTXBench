#!/usr/bin/env python3
"""Analyze the matched profiling-GPU sweep at w=48 and MAX_PROFILES=g."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from analyze_trajectory_throughput import Metrics, parse_log

import matplotlib.pyplot as plt


LOG_NAME_RE = re.compile(r"profile-w48(?:-g(?P<g>\d+))?\.txt")


@dataclass(frozen=True)
class GpuMetrics:
    g: int
    workers: int
    max_profiles: int
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


def infer_g(path: Path) -> int:
    match = LOG_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"{path}: expected profile-w48.txt or profile-w48-g<G>.txt"
        )
    return int(match.group("g") or 4)


def add_gpu_fields(item: Metrics, g: int) -> GpuMetrics:
    return GpuMetrics(
        g=g,
        workers=item.workers,
        max_profiles=g,
        log_path=item.log_path,
        planned_trajectories=item.planned_trajectories,
        first_pass_completed=item.first_pass_completed,
        first_pass_start_utc=item.first_pass_start_utc,
        first_pass_end_utc=item.first_pass_end_utc,
        first_pass_elapsed_seconds=item.first_pass_elapsed_seconds,
        first_pass_throughput_per_hour=item.first_pass_throughput_per_hour,
        experiment_start_utc=item.experiment_start_utc,
        experiment_end_utc=item.experiment_end_utc,
        experiment_elapsed_seconds=item.experiment_elapsed_seconds,
        whole_experiment_throughput_per_hour=(
            item.whole_experiment_throughput_per_hour
        ),
        all_completion_events=item.all_completion_events,
        retry_completion_events=item.retry_completion_events,
    )


def write_csv(metrics: list[GpuMetrics], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(metrics[0])),
            lineterminator="\n",
        )
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
    metrics: list[GpuMetrics],
    attribute: str,
    title: str,
    path: Path,
    *,
    start_y_at_zero: bool = False,
) -> None:
    g_values = [item.g for item in metrics]
    values = [getattr(item, attribute) for item in metrics]

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(
        g_values,
        values,
        marker="o",
        linewidth=2.0,
        markersize=7,
    )
    for g, value in zip(g_values, values, strict=True):
        axis.annotate(
            f"{value:.2f}",
            (g, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    axis.set_title(title)
    axis.set_xlabel("g (profiling GPUs; MAX_PROFILES=g)")
    axis.set_ylabel("Trajectories per hour")
    axis.set_xticks(g_values)
    axis.set_ylim(
        0 if start_y_at_zero else min(values) - 0.18,
        max(values) * 1.05 if start_y_at_zero else max(values) + 0.18,
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def print_table(metrics: list[GpuMetrics]) -> None:
    print(
        f"{'g':>3}  {'first pass':>12}  {'whole exp':>12}  "
        f"{'first-pass s':>12}  {'experiment s':>12}  {'retries':>7}"
    )
    for item in metrics:
        print(
            f"{item.g:>3}  "
            f"{item.first_pass_throughput_per_hour:>12.3f}  "
            f"{item.whole_experiment_throughput_per_hour:>12.3f}  "
            f"{item.first_pass_elapsed_seconds:>12}  "
            f"{item.experiment_elapsed_seconds:>12}  "
            f"{item.retry_completion_events:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze trajectory throughput for the matched w=48, "
            "MAX_PROFILES=g profiling-GPU sweep."
        )
    )
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help=(
            "Watcher logs (default: raw_logs/profile-w48-g2.txt, "
            "profile-w48.txt, profile-w48-g6.txt, profile-w48-g8.txt)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results"),
        help="Directory for the CSV and figure (default: analysis_results)",
    )
    args = parser.parse_args()

    log_paths = args.logs or [
        Path("raw_logs/profile-w48-g2.txt"),
        Path("raw_logs/profile-w48.txt"),
        Path("raw_logs/profile-w48-g6.txt"),
        Path("raw_logs/profile-w48-g8.txt"),
    ]
    missing = [path for path in log_paths if not path.is_file()]
    if missing:
        parser.error(f"missing input logs: {', '.join(map(str, missing))}")

    metrics = sorted(
        (
            add_gpu_fields(parse_log(path), infer_g(path))
            for path in log_paths
        ),
        key=lambda item: item.g,
    )
    if any(item.workers != 48 for item in metrics):
        parser.error("all sweep logs must have MAX_PARALLEL=48")
    if len({item.g for item in metrics}) != len(metrics):
        parser.error("duplicate g values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "gpu_scaling_trajectory_throughput.csv"
    first_pass_figure = (
        args.output_dir / "gpu_scaling_first_pass_trajectory_throughput.png"
    )
    whole_experiment_figure = (
        args.output_dir
        / "gpu_scaling_whole_experiment_trajectory_throughput.png"
    )
    write_csv(metrics, csv_path)
    plot_metric(
        metrics,
        "first_pass_throughput_per_hour",
        "First-pass trajectory throughput vs. profiling GPUs (w=48)",
        first_pass_figure,
        start_y_at_zero=True,
    )
    plot_metric(
        metrics,
        "whole_experiment_throughput_per_hour",
        "Whole-experiment trajectory throughput vs. profiling GPUs (w=48)",
        whole_experiment_figure,
    )

    print_table(metrics)
    print(f"\nCSV: {csv_path}")
    print(f"First-pass figure: {first_pass_figure}")
    print(f"Whole-experiment figure: {whole_experiment_figure}")


if __name__ == "__main__":
    main()
