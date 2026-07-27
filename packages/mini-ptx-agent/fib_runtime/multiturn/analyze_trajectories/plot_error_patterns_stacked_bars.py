#!/usr/bin/env python3
"""Plot before/after-SFT error pattern stacks by definition."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_CSV = (
    Path(__file__).resolve().parent
    / "qwen36-27b-error-distribution"
    / "error_distribution.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "qwen36-27b-error-distribution"
GROUP_LABELS = {
    "left": "Before SFT",
    "right": "After SFT",
}
DEFAULT_DEFINITIONS = [
    "mha_with_lse_d128",
    "mha_with_lse_d128_causal",
    "mha_bwd_d128",
    "mha_bwd_d128_causal",
]
DEFINITION_LABELS = {
    "gemm_n7168_k5120": "GEMM",
    "mha_with_lse_d128": "MHA",
    "mha_with_lse_d128_causal": "MHA causal",
    "mha_bwd_d128": "MHA bwd",
    "mha_bwd_d128_causal": "MHA bwd causal",
}
OUTCOME_ORDER = [
    "Compilation error",
    "Runtime error",
    "Numerical error",
    "Kernel Execution Timeout",
    "Extraction error",
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
    "Other error": "#5a5a5a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="error_patterns_stacked_by_definition",
        help="Output filename stem for PNG, SVG, and plotted CSV.",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot raw counts instead of fractions.",
    )
    parser.add_argument(
        "--definition",
        action="append",
        dest="definitions",
        help="Definition to include. Defaults to the four MHA definitions.",
    )
    return parser.parse_args()


def read_definition_rows(
    path: Path,
    selected_definitions: list[str],
) -> tuple[list[str], dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], int]]:
    data: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    totals: dict[tuple[str, str], int] = {}
    definitions: list[str] = []
    seen_definitions: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"scope", "definition", "group", "correctness", "count", "total", "fraction"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            if row["scope"] != "definition":
                continue
            definition = row["definition"]
            if definition not in selected_definitions:
                continue
            group = row["group"]
            if group not in GROUP_LABELS:
                continue
            if definition not in seen_definitions:
                seen_definitions.add(definition)
                definitions.append(definition)
            key = (definition, group)
            data[key][row["correctness"]] = float(row["count"])
            totals[key] = int(row["total"])
    return definitions, data, totals


def write_plotted_csv(
    path: Path,
    definitions: list[str],
    data: dict[tuple[str, str], dict[str, float]],
    totals: dict[tuple[str, str], int],
) -> None:
    fieldnames = [
        "definition",
        "group",
        "label",
        "total",
        *[f"{outcome}_count" for outcome in OUTCOME_ORDER],
        *[f"{outcome}_fraction" for outcome in OUTCOME_ORDER],
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for definition in definitions:
            for group in ("left", "right"):
                key = (definition, group)
                total = totals.get(key, 0)
                row = {
                    "definition": definition,
                    "group": group,
                    "label": GROUP_LABELS[group],
                    "total": total,
                }
                for outcome in OUTCOME_ORDER:
                    count = int(data.get(key, {}).get(outcome, 0))
                    row[f"{outcome}_count"] = count
                    row[f"{outcome}_fraction"] = f"{count / total:.6f}" if total else "0.000000"
                writer.writerow(row)


def make_plot(
    definitions: list[str],
    data: dict[tuple[str, str], dict[str, float]],
    totals: dict[tuple[str, str], int],
    png_path: Path,
    use_counts: bool,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(12.5, 6.0), dpi=160)
    bar_width = 0.34
    group_gap = 1.18
    x_centers = [idx * group_gap for idx in range(len(definitions))]
    offsets = {"left": -bar_width / 2, "right": bar_width / 2}

    for definition, center in zip(definitions, x_centers):
        for group in ("left", "right"):
            key = (definition, group)
            total = totals.get(key, 0)
            bottom = 0.0
            x = center + offsets[group]
            for outcome in OUTCOME_ORDER:
                count = data.get(key, {}).get(outcome, 0.0)
                height = count if use_counts else (count / total if total else 0.0)
                if height <= 0:
                    continue
                ax.bar(
                    x,
                    height,
                    width=bar_width,
                    bottom=bottom,
                    color=OUTCOME_COLORS[outcome],
                    edgecolor="none",
                    linewidth=0,
                )
                bottom += height
        divider_top = (
            max(totals.get((definition, "left"), 0), totals.get((definition, "right"), 0))
            if use_counts
            else 1.0
        )
        ax.vlines(
            center,
            0,
            divider_top,
            colors="white",
            linewidth=1.2,
            zorder=5,
        )
        for group in ("left", "right"):
            x = center + offsets[group]
            ax.text(
                x,
                -0.045 if not use_counts else -max(totals.values()) * 0.04,
                "Before" if group == "left" else "After",
                ha="center",
                va="top",
                fontsize=8.5,
                rotation=0,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([DEFINITION_LABELS.get(value, value) for value in definitions])
    ax.set_xlabel("Definition")
    if use_counts:
        ax.set_ylabel("Turn count")
        ax.set_title("Error Pattern Counts by Definition")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    else:
        ax.set_ylim(-0.085, 1.0)
        ax.set_ylabel("Fraction of turns")
        ax.set_title("Error Pattern Fractions by Definition")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=OUTCOME_COLORS[outcome], edgecolor="none", label=outcome)
        for outcome in OUTCOME_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_definitions = args.definitions or DEFAULT_DEFINITIONS
    definitions, data, totals = read_definition_rows(args.input_csv, selected_definitions)
    if not definitions:
        raise SystemExit("No definition-scope rows found.")

    stem = args.output_stem + ("_counts" if args.counts else "_fractions")
    plotted_csv = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    write_plotted_csv(plotted_csv, definitions, data, totals)
    make_plot(definitions, data, totals, png_path, args.counts)
    print(f"Wrote {plotted_csv}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
