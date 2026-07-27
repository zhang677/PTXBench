#!/usr/bin/env python3
"""Aggregate exporter-produced turn correctness labels for the g sweep."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


PREFERRED_CATEGORY_ORDER = [
    "Correct",
    "Compilation error",
    "Extraction error",
    "Kernel Execution Timeout",
    "Numerical error",
    "Other error",
    "Profiling Service Timeout",
    "Runtime error",
    "Sanitize Timeout",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"g", "exp_dir"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"{path}: missing columns {', '.join(sorted(missing))}"
        )
    return rows


def collect(
    manifest_rows: list[dict[str, str]],
) -> tuple[dict[int, Counter[str]], dict[int, int]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    run_counts: Counter[int] = Counter()
    seen_paths: set[Path] = set()

    for item in manifest_rows:
        g = int(item["g"])
        csv_path = (
            Path(item["exp_dir"]) / "figures" / "turn_correctness_arch.csv"
        )
        if csv_path in seen_paths:
            raise ValueError(f"duplicate CSV in manifest: {csv_path}")
        seen_paths.add(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"missing exporter output: {csv_path}; run "
                "benchmark/export_turn_correctness_arch.py first"
            )

        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "trajectory_id",
                "turn",
                "correctness",
                "speedup",
                "arch_tag",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{csv_path}: missing columns "
                    f"{', '.join(sorted(missing))}"
                )
            rows = list(reader)

        counts[g].update(
            (row.get("correctness") or "").strip() or "Unknown"
            for row in rows
        )
        run_counts[g] += 1

    return dict(counts), dict(run_counts)


def ordered_categories(counts: dict[int, Counter[str]]) -> list[str]:
    observed = {category for counter in counts.values() for category in counter}
    categories = [
        category
        for category in PREFERRED_CATEGORY_ORDER
        if category in observed
    ]
    categories.extend(sorted(observed.difference(categories)))
    return categories


def cell(count: int, total: int) -> str:
    return f"{count} ({count / total:.2%})"


def write_csv(
    path: Path,
    counts: dict[int, Counter[str]],
    categories: list[str],
) -> None:
    g_values = sorted(counts)
    fieldnames = ["correctness"]
    for g in g_values:
        fieldnames.extend([f"g{g}_count", f"g{g}_percent"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for category in categories:
            row: dict[str, str | int] = {"correctness": category}
            for g in g_values:
                total = sum(counts[g].values())
                count = counts[g][category]
                row[f"g{g}_count"] = count
                row[f"g{g}_percent"] = f"{count / total:.6%}"
            writer.writerow(row)


def write_markdown(
    path: Path,
    manifest_path: Path,
    counts: dict[int, Counter[str]],
    run_counts: dict[int, int],
    categories: list[str],
) -> None:
    g_values = sorted(counts)
    totals = {g: sum(counts[g].values()) for g in g_values}
    lines = [
        "# GPU-scaling turn error distribution",
        "",
        (
            "Source: exporter-produced "
            "`figures/turn_correctness_arch.csv` files routed by "
            f"`{manifest_path}`."
        ),
        "",
        "| correctness | "
        + " | ".join(f"g={g}" for g in g_values)
        + " |",
        "| --- | " + " | ".join("---:" for _ in g_values) + " |",
    ]
    for category in categories:
        lines.append(
            f"| {category} | "
            + " | ".join(
                cell(counts[g][category], totals[g]) for g in g_values
            )
            + " |"
        )
    lines.append(
        "| **Total exported turns** | "
        + " | ".join(str(totals[g]) for g in g_values)
        + " |"
    )
    lines.extend(
        [
            "",
            "Coverage: "
            + ", ".join(
                f"`g={g}` has {run_counts[g]} runs and {totals[g]} rows"
                for g in g_values
            )
            + ".",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis_results/gpu_scaling_export_experiments.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results"),
    )
    args = parser.parse_args()

    counts, run_counts = collect(read_manifest(args.manifest))
    categories = ordered_categories(counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "gpu_scaling_error_distribution.csv"
    markdown_path = args.output_dir / "gpu_scaling_error_distribution.md"
    write_csv(csv_path, counts, categories)
    write_markdown(
        markdown_path,
        args.manifest,
        counts,
        run_counts,
        categories,
    )
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
