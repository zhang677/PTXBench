#!/usr/bin/env python3
"""Render the principal horizon-metric tables as a presentation-ready SVG."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path


METRIC_FIELDS = [
    "correct_turns",
    "correctness_rate",
    "correct_and_use_instruction_rate",
    "trajectory_correctness_rate",
    "trajectory_correct_and_use_instruction_rate",
    "best_speedup",
    "correct_turn_speedup_geomean",
]

HEADER_LABELS = {
    "group": ["Model / stage"],
    "parquet_rows": ["Parquet", "rows"],
    "n_trajectories": ["Traj."],
    "n_turns": ["Turns"],
    "correct_turns": ["Correct", "turns"],
    "correctness_rate": ["Turn", "correctness"],
    "correct_and_use_instruction_rate": ["Turn correct", "+ instruction"],
    "trajectory_correctness_rate": ["Trajectory", "correctness"],
    "trajectory_correct_and_use_instruction_rate": ["Trajectory correct", "+ instruction"],
    "best_speedup": ["Best", "speedup"],
    "correct_turn_speedup_geomean": ["Correct-turn", "speedup gmean"],
}


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_table(lines: list[str], header_index: int) -> list[dict[str, str]]:
    headers = cells(lines[header_index])
    rows = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        values = cells(line)
        rows.append(dict(zip(headers, values)))
    return rows


def next_table_header(lines: list[str], start: int, end: int | None = None) -> int:
    stop = len(lines) if end is None else end
    return next(
        index
        for index in range(start, stop)
        if lines[index].startswith("| ") and "group" in cells(lines[index])
    )


def load_principal_tables(
    summary_path: Path,
) -> list[tuple[str, list[dict[str, str]]]]:
    lines = summary_path.read_text().splitlines()
    overall_start = next(
        index
        for index, line in enumerate(lines)
        if line.endswith("-Prompt Overall Metrics")
    )
    collective_start = next(
        index
        for index, line in enumerate(lines)
        if line.endswith("-Definition Collective Metrics")
    )
    detail_start = next(
        index
        for index, line in enumerate(lines)
        if "Definition / Prompt-Tag" in line
    )
    overall_rows = parse_table(
        lines, next_table_header(lines, overall_start, collective_start)
    )
    has_parquet_rows = bool(overall_rows and "parquet_rows" in overall_rows[0])
    parquet_rows_by_group = (
        {row["group"]: row["parquet_rows"] for row in overall_rows}
        if has_parquet_rows
        else {}
    )
    tables = [
        (
            lines[overall_start].removeprefix("## "),
            overall_rows,
        )
    ]
    for index in range(collective_start, detail_start):
        line = lines[index]
        if not line.startswith("### `"):
            continue
        title = line.removeprefix("### `").removesuffix("`")
        header_index = next_table_header(lines, index + 1, detail_start)
        rows = parse_table(lines, header_index)
        if has_parquet_rows:
            for row in rows:
                row["parquet_rows"] = parquet_rows_by_group[row["group"]]
        tables.append((title, rows))
    if len(tables) < 2:
        raise ValueError(
            f"Expected an overall table and at least one definition table in "
            f"{summary_path}, found {len(tables)}"
        )
    return tables


def triplet(value: str) -> list[float]:
    parts = value.split(" / ")
    if len(parts) != 3:
        raise ValueError(f"Expected a three-horizon metric, got {value!r}")
    return [float(part) for part in parts]


def winner_classes(
    rows: list[dict[str, str]], qwen_groups: set[str]
) -> dict[tuple[int, str, int], str]:
    winners = {}
    for field in METRIC_FIELDS:
        values = [triplet(row[field]) for row in rows]
        qwen_indices = [
            index for index, row in enumerate(rows) if row["group"] in qwen_groups
        ]
        for horizon_index in range(3):
            overall_best = max(value[horizon_index] for value in values)
            qwen_best = (
                max(values[index][horizon_index] for index in qwen_indices)
                if qwen_indices
                else 0.0
            )
            for row_index, value in enumerate(values):
                is_overall = overall_best > 0 and math.isclose(
                    value[horizon_index], overall_best, rel_tol=1e-12, abs_tol=1e-12
                )
                is_qwen = (
                    row_index in qwen_indices
                    and qwen_best > 0
                    and math.isclose(
                        value[horizon_index], qwen_best, rel_tol=1e-12, abs_tol=1e-12
                    )
                )
                if is_overall and is_qwen:
                    winners[(row_index, field, horizon_index)] = "both"
                elif is_overall:
                    winners[(row_index, field, horizon_index)] = "overall"
                elif is_qwen:
                    winners[(row_index, field, horizon_index)] = "qwen"
    return winners


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: float = 13,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#24292f",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">'
        f"{html.escape(value)}</text>"
    )


def render(
    summary_path: Path,
    output_path: Path,
    title: str,
    qwen_groups: set[str],
) -> None:
    tables = load_principal_tables(summary_path)
    has_parquet_rows = bool(
        tables[0][1] and "parquet_rows" in tables[0][1][0]
    )
    count_fields = [
        *(["parquet_rows"] if has_parquet_rows else []),
        "n_trajectories",
        "n_turns",
    ]
    columns = ["group", *count_fields, *METRIC_FIELDS]
    width = 2500
    margin = 34
    group_width = 300
    count_width = 94
    metric_width = (
        width - margin * 2 - group_width - count_width * len(count_fields)
    ) / len(METRIC_FIELDS)
    column_widths = [group_width, *([count_width] * len(count_fields))] + [
        metric_width
    ] * len(METRIC_FIELDS)
    x_positions = [margin]
    for column_width in column_widths[:-1]:
        x_positions.append(x_positions[-1] + column_width)

    header_height = 72
    row_height = 36
    table_title_height = 31
    table_gap = 24
    top = 122
    table_heights = [
        table_title_height + header_height + row_height * len(rows)
        for _name, rows in tables
    ]
    height = int(top + sum(table_heights) + table_gap * (len(tables) - 1) + 40)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,Arial,sans-serif}.grid{stroke:#d0d7de;stroke-width:1;shape-rendering:crispEdges}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        svg_text(margin, 38, title, size=26, weight=700),
        svg_text(
            margin,
            64,
            f"{len(tables)} tables; every metric is shown at ≤1, ≤4, and ≤8 turns.",
            size=14,
            fill="#57606a",
        ),
        svg_text(margin, 90, "Bold gold: best overall", size=13, weight=700, fill="#9a6700"),
    ]
    if qwen_groups:
        parts.extend(
            [
                svg_text(margin + 210, 90, "Bold blue: best Qwen-derived", size=13, weight=700, fill="#0969da"),
                svg_text(margin + 490, 90, "Bold purple: both", size=13, weight=700, fill="#8250df"),
                svg_text(
                    margin + 660,
                    90,
                    "All-zero metric/horizon slices are not bolded.",
                    size=13,
                    fill="#57606a",
                ),
            ]
        )
    else:
        parts.append(
            svg_text(
                margin + 210,
                90,
                "All-zero metric/horizon slices are not bolded.",
                size=13,
                fill="#57606a",
            )
        )

    y = top
    winner_fill = {"overall": "#9a6700", "qwen": "#0969da", "both": "#8250df"}
    for table_index, (table_name, rows) in enumerate(tables):
        parts.append(svg_text(margin, y + 21, table_name, size=18, weight=700))
        table_y = y + table_title_height
        table_width = sum(column_widths)
        table_height = header_height + row_height * len(rows)
        parts.append(
            f'<rect x="{margin}" y="{table_y}" width="{table_width:.1f}" height="{table_height:.1f}" rx="5" fill="#ffffff" stroke="#8c959f"/>'
        )
        parts.append(
            f'<rect x="{margin}" y="{table_y}" width="{table_width:.1f}" height="{header_height}" rx="5" fill="#f6f8fa"/>'
        )

        for column_index, field in enumerate(columns):
            x = x_positions[column_index]
            cell_width = column_widths[column_index]
            for line_index, label in enumerate(HEADER_LABELS[field]):
                parts.append(
                    svg_text(
                        x + cell_width / 2,
                        table_y + 19 + line_index * 15,
                        label,
                        size=12,
                        anchor="middle",
                        weight=700,
                    )
                )
            if field in METRIC_FIELDS:
                for horizon_index, horizon in enumerate(("≤1", "≤4", "≤8")):
                    parts.append(
                        svg_text(
                            x + cell_width * (horizon_index + 0.5) / 3,
                            table_y + 60,
                            horizon,
                            size=11,
                            anchor="middle",
                            weight=600,
                            fill="#57606a",
                        )
                    )
            if column_index:
                parts.append(
                    f'<line class="grid" x1="{x:.1f}" y1="{table_y:.1f}" x2="{x:.1f}" y2="{table_y + table_height:.1f}"/>'
                )
        parts.append(
            f'<line class="grid" x1="{margin}" y1="{table_y + header_height:.1f}" x2="{margin + table_width:.1f}" y2="{table_y + header_height:.1f}"/>'
        )

        winners = winner_classes(rows, qwen_groups)
        for row_index, row in enumerate(rows):
            row_y = table_y + header_height + row_index * row_height
            if row_index % 2:
                parts.append(
                    f'<rect x="{margin}" y="{row_y}" width="{table_width:.1f}" height="{row_height}" fill="#fbfcfd"/>'
                )
            baseline = row_y + 23
            group = row["group"]
            if group in qwen_groups:
                parts.append(
                    f'<rect x="{margin + 8}" y="{row_y + 10}" width="16" height="16" rx="4" fill="#ddf4ff" stroke="#54aeff"/>'
                )
                parts.append(svg_text(margin + 16, baseline - 1, "Q", size=9, anchor="middle", weight=700, fill="#0969da"))
                group_x = margin + 31
            else:
                group_x = margin + 9
            parts.append(svg_text(group_x, baseline, group, size=12.5, weight=600))
            for count_offset, field in enumerate(count_fields, start=1):
                parts.append(
                    svg_text(
                        x_positions[count_offset]
                        + column_widths[count_offset] / 2,
                        baseline,
                        row[field],
                        size=12.5,
                        anchor="middle",
                    )
                )
            metric_start = 1 + len(count_fields)
            for metric_offset, field in enumerate(
                METRIC_FIELDS, start=metric_start
            ):
                values = row[field].split(" / ")
                x = x_positions[metric_offset]
                cell_width = column_widths[metric_offset]
                for horizon_index, value in enumerate(values):
                    winner = winners.get((row_index, field, horizon_index))
                    parts.append(
                        svg_text(
                            x + cell_width * (horizon_index + 0.5) / 3,
                            baseline,
                            value,
                            size=12.2,
                            anchor="middle",
                            weight=700 if winner else 400,
                            fill=winner_fill.get(winner, "#24292f"),
                        )
                    )
            if row_index < len(rows) - 1:
                parts.append(
                    f'<line class="grid" x1="{margin}" y1="{row_y + row_height:.1f}" x2="{margin + table_width:.1f}" y2="{row_y + row_height:.1f}"/>'
                )
        y += table_heights[table_index] + table_gap

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--qwen-group", action="append", default=[])
    args = parser.parse_args()
    render(args.summary, args.output, args.title, set(args.qwen_group))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
