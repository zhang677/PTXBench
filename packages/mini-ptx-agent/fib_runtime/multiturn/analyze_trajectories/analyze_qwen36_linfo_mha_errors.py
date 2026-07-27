#!/usr/bin/env python3
"""Analyze error types for the four Qwen3.6-27B MHA debug runs.

The script reads the existing figures/turn_correctness_arch.csv files for the
linfo and Qwen3.6-27B MHA runs, writes auditable CSV summaries, and
plots the per-problem error-type distribution.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


RUN_ROOT = Path("/home/ubuntu/AccRL-exps/eval_runs")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "qwen36_linfo_mha_error_analysis"
CONDITIONS = ("qwen36-27b-linfo", "qwen36-27b")
RUNS = [
    (
        "mha_d128",
        "MHA d128",
        "qwen36-27b-linfo-mha-d128",
        "qwen36-27b-mha-d128",
    ),
    (
        "mha_d128_causal",
        "MHA d128 causal",
        "qwen36-27b-linfo-mha-d128-causal",
        "qwen36-27b-mha-d128-causal",
    ),
    (
        "mha_bwd_d128",
        "MHA bwd d128",
        "qwen36-27b-linfo-mha-bwd-d128",
        "qwen36-27b-mha-bwd-d128",
    ),
    (
        "mha_bwd_d128_causal",
        "MHA bwd d128 causal",
        "qwen36-27b-linfo-mha-bwd-d128-causal",
        "qwen36-27b-mha-bwd-d128-causal",
    ),
]
OUTCOME_ORDER = [
    "Compilation error",
    "Runtime error",
    "Kernel Execution Timeout",
    "Numerical error",
    "Extraction error",
    "Profiling Service Timeout",
    "Sanitize Timeout",
    "Other error",
    "Correct",
]
OUTCOME_COLORS = {
    "Correct": "#4d9b69",
    "Compilation error": "#d18f2f",
    "Runtime error": "#b84e4e",
    "Numerical error": "#7b65b5",
    "Kernel Execution Timeout": "#5b8ab8",
    "Extraction error": "#8b8f97",
    "Profiling Service Timeout": "#71a6a1",
    "Sanitize Timeout": "#c56b9b",
    "Other error": "#5a5a5a",
}


@dataclass(frozen=True)
class RunSpec:
    problem_id: str
    problem_label: str
    condition: str
    run_dir: Path

    @property
    def csv_path(self) -> Path:
        return self.run_dir / "figures" / "turn_correctness_arch.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=RUN_ROOT,
        help="Directory containing eval run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV, markdown, and plot artifacts.",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot raw counts instead of fractions.",
    )
    return parser.parse_args()


def build_run_specs(run_root: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for problem_id, problem_label, linfo_dir, base_dir in RUNS:
        specs.append(
            RunSpec(
                problem_id=problem_id,
                problem_label=problem_label,
                condition=CONDITIONS[0],
                run_dir=run_root / linfo_dir,
            )
        )
        specs.append(
            RunSpec(
                problem_id=problem_id,
                problem_label=problem_label,
                condition=CONDITIONS[1],
                run_dir=run_root / base_dir,
            )
        )
    return specs


def read_correctness_counts(csv_path: Path) -> Counter[str]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    counts: Counter[str] = Counter()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "correctness" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} missing required correctness column")
        for row in reader:
            correctness = (row.get("correctness") or "").strip()
            if correctness:
                counts[correctness] += 1
    return counts


def ordered_outcomes(counters: dict[tuple[str, str], Counter[str]]) -> list[str]:
    observed = set()
    for counter in counters.values():
        observed.update(counter)
    ordered = [outcome for outcome in OUTCOME_ORDER if outcome in observed]
    ordered.extend(sorted(observed.difference(OUTCOME_ORDER)))
    return ordered


def write_distribution_csv(
    path: Path,
    specs: list[RunSpec],
    counters: dict[tuple[str, str], Counter[str]],
    outcomes: list[str],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem_id",
                "problem_label",
                "condition",
                "run_dir",
                "total",
                "correctness",
                "count",
                "fraction",
            ],
        )
        writer.writeheader()
        for spec in specs:
            counter = counters[(spec.problem_id, spec.condition)]
            total = sum(counter.values())
            for outcome in outcomes:
                count = counter.get(outcome, 0)
                writer.writerow(
                    {
                        "problem_id": spec.problem_id,
                        "problem_label": spec.problem_label,
                        "condition": spec.condition,
                        "run_dir": str(spec.run_dir),
                        "total": total,
                        "correctness": outcome,
                        "count": count,
                        "fraction": f"{count / total:.6f}" if total else "0.000000",
                    }
                )


def write_summary_csv(
    path: Path,
    specs: list[RunSpec],
    counters: dict[tuple[str, str], Counter[str]],
    outcomes: list[str],
) -> None:
    fieldnames = [
        "problem_id",
        "problem_label",
        "condition",
        "run_dir",
        "total",
        *[f"{outcome}_count" for outcome in outcomes],
        *[f"{outcome}_fraction" for outcome in outcomes],
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            counter = counters[(spec.problem_id, spec.condition)]
            total = sum(counter.values())
            row = {
                "problem_id": spec.problem_id,
                "problem_label": spec.problem_label,
                "condition": spec.condition,
                "run_dir": str(spec.run_dir),
                "total": total,
            }
            for outcome in outcomes:
                count = counter.get(outcome, 0)
                row[f"{outcome}_count"] = count
                row[f"{outcome}_fraction"] = f"{count / total:.6f}" if total else "0.000000"
            writer.writerow(row)


def aggregate_by_condition(
    specs: list[RunSpec],
    counters: dict[tuple[str, str], Counter[str]],
) -> dict[str, Counter[str]]:
    aggregates = {condition: Counter() for condition in CONDITIONS}
    for spec in specs:
        aggregates[spec.condition].update(counters[(spec.problem_id, spec.condition)])
    return aggregates


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%" if total else "0.0%"


def write_markdown_summary(
    path: Path,
    specs: list[RunSpec],
    counters: dict[tuple[str, str], Counter[str]],
    outcomes: list[str],
    distribution_csv: Path,
    summary_csv: Path,
    plot_path: Path,
) -> None:
    aggregates = aggregate_by_condition(specs, counters)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    aggregate_rows: list[list[object]] = []
    for condition in CONDITIONS:
        total = sum(aggregates[condition].values())
        aggregate_rows.append(
            [
                condition,
                total,
                *[
                    f"{aggregates[condition].get(outcome, 0)} ({pct(aggregates[condition].get(outcome, 0), total)})"
                    for outcome in outcomes
                ],
            ]
        )

    delta_rows: list[list[object]] = []
    linfo_total = sum(aggregates[CONDITIONS[0]].values())
    base_total = sum(aggregates[CONDITIONS[1]].values())
    for outcome in outcomes:
        linfo_count = aggregates[CONDITIONS[0]].get(outcome, 0)
        base_count = aggregates[CONDITIONS[1]].get(outcome, 0)
        linfo_frac = linfo_count / linfo_total if linfo_total else 0.0
        base_frac = base_count / base_total if base_total else 0.0
        delta_rows.append(
            [
                outcome,
                linfo_count,
                base_count,
                linfo_count - base_count,
                f"{100.0 * (linfo_frac - base_frac):+.1f} pp",
            ]
        )

    per_problem_rows: list[list[object]] = []
    for spec in specs:
        counter = counters[(spec.problem_id, spec.condition)]
        total = sum(counter.values())
        per_problem_rows.append(
            [
                spec.problem_label,
                spec.condition,
                total,
                counter.get("Compilation error", 0),
                counter.get("Runtime error", 0),
                counter.get("Kernel Execution Timeout", 0),
                counter.get("Numerical error", 0),
                counter.get("Correct", 0),
            ]
        )

    source_rows = [
        [spec.problem_label, spec.condition, spec.csv_path]
        for spec in specs
    ]

    content = "\n\n".join(
        [
            "# Qwen3.6-27B linfo vs qwen36 MHA error types",
            f"Generated: {generated}",
            "Rows are read from each run's `figures/turn_correctness_arch.csv`.",
            f"Plot: `{plot_path.name}`",
            f"Long CSV: `{distribution_csv.name}`",
            f"Wide CSV: `{summary_csv.name}`",
            "## Aggregate over the four MHA problems",
            markdown_table(["Condition", "Total", *outcomes], aggregate_rows),
            "## linfo minus qwen36-27b delta",
            markdown_table(
                ["Correctness", "linfo", "qwen36-27b", "count_delta", "fraction_delta"],
                delta_rows,
            ),
            "## Per-problem key counts",
            markdown_table(
                [
                    "Problem",
                    "Condition",
                    "Total",
                    "Compilation",
                    "Runtime",
                    "Kernel Timeout",
                    "Numerical",
                    "Correct",
                ],
                per_problem_rows,
            ),
            "## Source CSVs",
            markdown_table(["Problem", "Condition", "CSV"], source_rows),
            "",
        ]
    )
    path.write_text(content)


def make_plot(
    path: Path,
    specs: list[RunSpec],
    counters: dict[tuple[str, str], Counter[str]],
    outcomes: list[str],
    use_counts: bool,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    problem_labels = [label for _, label, _, _ in RUNS]
    problem_ids = [problem_id for problem_id, _, _, _ in RUNS]
    fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=160)
    bar_width = 0.34
    group_gap = 1.22
    centers = [idx * group_gap for idx in range(len(problem_ids))]
    offsets = {CONDITIONS[0]: -bar_width / 2, CONDITIONS[1]: bar_width / 2}
    max_count = max((sum(counter.values()) for counter in counters.values()), default=1)

    for problem_id, center in zip(problem_ids, centers):
        for condition in CONDITIONS:
            counter = counters[(problem_id, condition)]
            total = sum(counter.values())
            bottom = 0.0
            x = center + offsets[condition]
            for outcome in outcomes:
                count = counter.get(outcome, 0)
                height = count if use_counts else (count / total if total else 0.0)
                if height <= 0:
                    continue
                ax.bar(
                    x,
                    height,
                    width=bar_width,
                    bottom=bottom,
                    color=OUTCOME_COLORS.get(outcome, "#999999"),
                    edgecolor="none",
                    linewidth=0,
                )
                bottom += height
            ax.text(
                x,
                -max_count * 0.045 if use_counts else -0.055,
                "linfo" if condition == CONDITIONS[0] else "qwen36",
                ha="center",
                va="top",
                fontsize=8.5,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels(problem_labels)
    ax.set_xlabel("Problem")
    if use_counts:
        ax.set_ylabel("Turn count")
        ax.set_ylim(-max_count * 0.09, max_count * 1.04)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.set_title("Qwen3.6-27B linfo vs qwen36 MHA error types")
    else:
        ax.set_ylabel("Fraction of turns")
        ax.set_ylim(-0.095, 1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set_title("Qwen3.6-27B linfo vs qwen36 MHA error-type fractions")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=OUTCOME_COLORS.get(outcome, "#999999"),
            edgecolor="none",
            label=outcome,
        )
        for outcome in outcomes
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    specs = build_run_specs(args.run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counters: dict[tuple[str, str], Counter[str]] = {}
    for spec in specs:
        counters[(spec.problem_id, spec.condition)] = read_correctness_counts(spec.csv_path)
    outcomes = ordered_outcomes(counters)

    distribution_csv = args.output_dir / "error_distribution_by_problem.csv"
    summary_csv = args.output_dir / "error_distribution_summary.csv"
    plot_name = (
        "qwen36_linfo_vs_qwen36_mha_error_types_counts.png"
        if args.counts
        else "qwen36_linfo_vs_qwen36_mha_error_types.png"
    )
    plot_path = args.output_dir / plot_name
    markdown_path = args.output_dir / "summary.md"

    write_distribution_csv(distribution_csv, specs, counters, outcomes)
    write_summary_csv(summary_csv, specs, counters, outcomes)
    make_plot(plot_path, specs, counters, outcomes, args.counts)
    write_markdown_summary(
        markdown_path,
        specs,
        counters,
        outcomes,
        distribution_csv,
        summary_csv,
        plot_path,
    )

    print(f"Wrote {distribution_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
