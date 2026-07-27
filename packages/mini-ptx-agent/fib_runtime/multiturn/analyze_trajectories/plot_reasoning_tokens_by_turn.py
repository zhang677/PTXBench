#!/usr/bin/env python3
"""Plot mean reasoning tokens by turn before and after SFT.

The default input is the per-turn CSV produced by
compare_qwen36_27b_reasoning_sft.py. Means are computed from non-empty
reasoning_tokens values. By default, the x-axis is restricted to turns present
in the SFT group for the selected definitions, matching the 8-turn
2026-0624-0939 evals.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_INPUT_CSV = (
    Path(__file__).resolve().parent
    / "qwen36-27b-reasoning-2026-0624-0939"
    / "reasoning_turns.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "qwen36-27b-reasoning-2026-0624-0939"
)
GROUP_LABELS = {
    "baseline": "Before SFT",
    "sft": "After SFT",
}
GROUP_COLORS = {
    "mha_with_lse_d128": "#2f6f9f",
    "mha_with_lse_d128_causal": "#c5523c",
    "mha_bwd_d128": "#5f8f3a",
    "mha_bwd_d128_causal": "#7a5aa6",
}
GROUP_LINESTYLES = {
    "baseline": (0, (5, 3)),
    "sft": "solid",
}
DEFINITION_LABELS = {
    "mha_with_lse_d128": "MHA fwd d128",
    "mha_with_lse_d128_causal": "MHA fwd d128 causal",
    "mha_bwd_d128": "MHA bwd d128",
    "mha_bwd_d128_causal": "MHA bwd d128 causal",
}
DEFAULT_DEFINITIONS = tuple(DEFINITION_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="mean_reasoning_tokens_by_turn_mha_definitions",
        help="Output filename stem for PNG, SVG, and plotted CSV.",
    )
    parser.add_argument(
        "--definition",
        action="append",
        dest="definitions",
        help="Definition to include. Can be passed multiple times. Defaults to the four MHA definitions.",
    )
    parser.add_argument(
        "--include-baseline-only-turns",
        action="store_true",
        help="Include turns that appear only in baseline rows.",
    )
    return parser.parse_args()


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_turn_tokens(
    path: Path,
    definitions: set[str],
) -> dict[str, dict[str, dict[int, list[float]]]]:
    grouped: dict[str, dict[str, dict[int, list[float]]]] = {
        definition: {
            "baseline": defaultdict(list),
            "sft": defaultdict(list),
        }
        for definition in sorted(definitions)
    }
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"definition", "group", "turn", "reasoning_tokens"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            definition = row["definition"].strip()
            if definition not in grouped:
                continue
            group = row["group"].strip()
            if group not in grouped[definition]:
                continue
            try:
                turn = int(row["turn"])
            except ValueError:
                continue
            tokens = as_float(row["reasoning_tokens"])
            if tokens is None:
                continue
            grouped[definition][group][turn].append(tokens)
    return grouped


def build_plot_rows(
    grouped: dict[str, dict[str, dict[int, list[float]]]],
    include_baseline_only_turns: bool,
) -> list[dict[str, str]]:
    if include_baseline_only_turns:
        turns = sorted(
            {
                turn
                for by_group in grouped.values()
                for values_by_turn in by_group.values()
                for turn in values_by_turn
            }
        )
    else:
        turns = sorted(
            {
                turn
                for by_group in grouped.values()
                for turn in by_group["sft"]
            }
        )

    rows: list[dict[str, str]] = []
    for definition in sorted(grouped):
        for turn in turns:
            for group in ("baseline", "sft"):
                values = grouped[definition][group].get(turn, [])
                rows.append(
                    {
                        "definition": definition,
                        "definition_label": DEFINITION_LABELS.get(definition, definition),
                        "turn": str(turn),
                        "group": group,
                        "group_label": GROUP_LABELS[group],
                        "rows": str(len(values)),
                        "mean_reasoning_tokens": (
                            f"{mean(values):.6f}" if values else ""
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "definition",
        "definition_label",
        "turn",
        "group",
        "group_label",
        "rows",
        "mean_reasoning_tokens",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict[str, str]], png_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    turns = sorted({int(row["turn"]) for row in rows})
    definitions = list(dict.fromkeys(row["definition"] for row in rows))
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=160)

    row_by_key = {
        (row["definition"], row["group"], int(row["turn"])): row for row in rows
    }
    for definition in definitions:
        for group in ("baseline", "sft"):
            y_values: list[float | None] = []
            for turn in turns:
                row = row_by_key.get((definition, group, turn))
                value = row["mean_reasoning_tokens"] if row else ""
                y_values.append(float(value) if value else None)
            label = f"{DEFINITION_LABELS.get(definition, definition)} - {GROUP_LABELS[group]}"
            ax.plot(
                turns,
                y_values,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=GROUP_COLORS.get(definition, "#555555"),
                linestyle=GROUP_LINESTYLES[group],
                label=label,
            )

    ax.set_title("Mean Reasoning Tokens by Turn for MHA Definitions")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean reasoning tokens")
    ax.set_xticks(turns)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    definitions = set(args.definitions or DEFAULT_DEFINITIONS)
    grouped = read_turn_tokens(args.input_csv, definitions)
    rows = build_plot_rows(grouped, args.include_baseline_only_turns)
    if not rows:
        raise SystemExit("No rows with reasoning_tokens were found.")

    plotted_csv = args.output_dir / f"{args.output_stem}.csv"
    png_path = args.output_dir / f"{args.output_stem}.png"
    write_csv(plotted_csv, rows)
    make_plot(rows, png_path)

    print(f"Wrote {plotted_csv}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
