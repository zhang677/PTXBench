#!/usr/bin/env python3
"""Compare turn-level error distributions between two eval sets.

Defaults compare the five shared Hopper evals for Qwen3.6-27B and the
2026-0624-0939 SFT run family:

  Qwen3.6-27B vs Qwen3.6-27B-fixit-v2-glm

The source of truth is each eval run's figures/turn_correctness_arch.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXPERIMENTS_CSV = Path("/home/ubuntu/AccRL/benchmark/experiments.csv")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "qwen36-27b-error-distribution"
)
DEFAULT_LEFT_MODEL = "Qwen3.6-27B"
DEFAULT_RIGHT_MODEL = "Qwen3.6-27B-fixit-v2-glm"
DEFAULT_RIGHT_RUN_FRAGMENT = "2026-0624-0939"
DEFAULT_DEFINITIONS = {
    "gemm_n7168_k5120",
    "mha_with_lse_d128",
    "mha_with_lse_d128_causal",
    "mha_bwd_d128",
    "mha_bwd_d128_causal",
}


@dataclass(frozen=True)
class Experiment:
    group: str
    model: str
    arch: str
    definition: str
    workload: str
    exp_dir: Path

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.arch, self.definition, self.workload)

    @property
    def csv_path(self) -> Path:
        return self.exp_dir / "figures" / "turn_correctness_arch.csv"


@dataclass(frozen=True)
class TurnRow:
    group: str
    model: str
    arch: str
    definition: str
    workload: str
    exp_dir: Path
    trajectory_id: str
    turn: int
    correctness: str
    speedup: str
    arch_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare correctness/error distributions across paired eval runs."
    )
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        default=DEFAULT_EXPERIMENTS_CSV,
        help="Registry CSV with model, arch, definition, workload, exp_dir columns.",
    )
    parser.add_argument("--left-model", default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--right-model", default=DEFAULT_RIGHT_MODEL)
    parser.add_argument(
        "--right-run-fragment",
        default=DEFAULT_RIGHT_RUN_FRAGMENT,
        help="Optional substring required in right-side exp_dir values.",
    )
    parser.add_argument(
        "--definition",
        action="append",
        dest="definitions",
        help="Definition to include. Can be passed multiple times. Defaults to the five 2026-0624-0939 comparison evals.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=None,
        help="Only include turns with turn < this value.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV, JSON, and markdown summary.",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include Correct rows in the *_errors_only.csv output too.",
    )
    return parser.parse_args()


def read_experiment_registry(
    path: Path,
    left_model: str,
    right_model: str,
    definitions: set[str],
    right_run_fragment: str,
) -> list[Experiment]:
    required = {"model", "arch", "definition", "workload", "exp_dir"}
    experiments: list[Experiment] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            model = row["model"].strip()
            definition = row["definition"].strip()
            exp_dir = Path(row["exp_dir"].strip())
            if definition not in definitions:
                continue
            if model == left_model:
                group = "left"
            elif model == right_model:
                if right_run_fragment and right_run_fragment not in str(exp_dir):
                    continue
                group = "right"
            else:
                continue
            experiments.append(
                Experiment(
                    group=group,
                    model=model,
                    arch=row["arch"].strip(),
                    definition=definition,
                    workload=row["workload"].strip(),
                    exp_dir=exp_dir,
                )
            )
    return experiments


def pair_experiments(experiments: list[Experiment]) -> list[tuple[Experiment, Experiment]]:
    by_key: dict[tuple[str, str, str], dict[str, list[Experiment]]] = defaultdict(
        lambda: {"left": [], "right": []}
    )
    for experiment in experiments:
        by_key[experiment.key][experiment.group].append(experiment)

    pairs: list[tuple[Experiment, Experiment]] = []
    problems: list[str] = []
    for key in sorted(by_key):
        grouped = by_key[key]
        left = grouped["left"]
        right = grouped["right"]
        if len(left) != 1 or len(right) != 1:
            problems.append(
                f"{key}: expected exactly one left and one right run, got left={len(left)} right={len(right)}"
            )
            continue
        pairs.append((left[0], right[0]))

    if problems:
        raise ValueError("Unpaired or ambiguous experiments:\n" + "\n".join(problems))
    if not pairs:
        raise ValueError("No paired experiments found.")
    return pairs


def read_turn_rows(experiment: Experiment, turn_limit: int | None) -> list[TurnRow]:
    if not experiment.csv_path.exists():
        raise FileNotFoundError(f"Missing turn correctness CSV: {experiment.csv_path}")

    rows: list[TurnRow] = []
    with experiment.csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"trajectory_id", "turn", "correctness"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{experiment.csv_path} missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            turn = int(row["turn"])
            if turn_limit is not None and turn >= turn_limit:
                continue
            correctness = (row.get("correctness") or "").strip() or "Unknown"
            rows.append(
                TurnRow(
                    group=experiment.group,
                    model=experiment.model,
                    arch=experiment.arch,
                    definition=experiment.definition,
                    workload=experiment.workload,
                    exp_dir=experiment.exp_dir,
                    trajectory_id=row["trajectory_id"].strip(),
                    turn=turn,
                    correctness=correctness,
                    speedup=(row.get("speedup") or "").strip(),
                    arch_tag=(row.get("arch_tag") or "").strip(),
                )
            )
    return rows


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def write_distribution_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "scope",
        "arch",
        "definition",
        "workload",
        "group",
        "model",
        "exp_dir",
        "turn",
        "correctness",
        "count",
        "total",
        "fraction",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_distribution_rows(turn_rows: list[TurnRow]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    def add_counter(scope: str, key: tuple[object, ...], rows: list[TurnRow]) -> None:
        counts = Counter(row.correctness for row in rows)
        total = sum(counts.values())
        sample = rows[0]
        turn_value = key[-1] if scope.endswith("_by_turn") else ""
        for correctness, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output.append(
                {
                    "scope": scope,
                    "arch": sample.arch,
                    "definition": sample.definition if "definition" in scope else "",
                    "workload": sample.workload if "definition" in scope else "",
                    "group": sample.group,
                    "model": sample.model,
                    "exp_dir": str(sample.exp_dir) if "definition" in scope else "",
                    "turn": turn_value,
                    "correctness": correctness,
                    "count": count,
                    "total": total,
                    "fraction": f"{pct(count, total):.6f}",
                }
            )

    groupings: dict[tuple[str, tuple[object, ...]], list[TurnRow]] = defaultdict(list)
    for row in turn_rows:
        groupings[
            (
                "overall",
                (row.group, row.model),
            )
        ].append(row)
        groupings[
            (
                "overall_by_turn",
                (row.group, row.model, row.turn),
            )
        ].append(row)
        groupings[
            (
                "definition",
                (row.arch, row.definition, row.workload, row.group, row.model, str(row.exp_dir)),
            )
        ].append(row)
        groupings[
            (
                "definition_by_turn",
                (
                    row.arch,
                    row.definition,
                    row.workload,
                    row.group,
                    row.model,
                    str(row.exp_dir),
                    row.turn,
                ),
            )
        ].append(row)

    for (scope, key), grouped_rows in sorted(groupings.items()):
        add_counter(scope, key, grouped_rows)
    return output


def write_errors_only_csv(
    path: Path,
    rows: list[TurnRow],
    include_correct: bool,
) -> None:
    fieldnames = [
        "group",
        "model",
        "arch",
        "definition",
        "workload",
        "exp_dir",
        "trajectory_id",
        "turn",
        "correctness",
        "speedup",
        "arch_tag",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not include_correct and row.correctness == "Correct":
                continue
            writer.writerow(
                {
                    "group": row.group,
                    "model": row.model,
                    "arch": row.arch,
                    "definition": row.definition,
                    "workload": row.workload,
                    "exp_dir": row.exp_dir,
                    "trajectory_id": row.trajectory_id,
                    "turn": row.turn,
                    "correctness": row.correctness,
                    "speedup": row.speedup,
                    "arch_tag": row.arch_tag,
                }
            )


def summarize_counts(rows: list[TurnRow]) -> dict[tuple[str, str], Counter[str]]:
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        counters[(row.group, row.model)][row.correctness] += 1
    return counters


def first_counter_for_group(
    counters: dict[tuple[str, str], Counter[str]],
    group: str,
) -> tuple[tuple[str, str], Counter[str]]:
    matches = [(key, counter) for key, counter in counters.items() if key[0] == group]
    if len(matches) != 1:
        raise ValueError(f"Expected one {group!r} overall counter, found {len(matches)}")
    return matches[0]


def write_markdown_summary(
    path: Path,
    pairs: list[tuple[Experiment, Experiment]],
    rows: list[TurnRow],
    outputs: dict[str, Path],
    turn_limit: int | None,
    title: str = "Qwen3.6-27B Error Distribution Comparison",
) -> None:
    counters = summarize_counts(rows)
    left_key, left_counter = first_counter_for_group(counters, "left")
    right_key, right_counter = first_counter_for_group(counters, "right")
    left_total = sum(left_counter.values())
    right_total = sum(right_counter.values())
    categories = sorted(set(left_counter) | set(right_counter), key=lambda c: (c != "Correct", c))

    lines = [
        f"# {title}",
        "",
        f"- left: `{left_key[1]}`",
        f"- right: `{right_key[1]}`",
        f"- turn_limit: `{turn_limit if turn_limit is not None else 'all'}`",
        f"- paired evals: `{len(pairs)}`",
        "",
        "## Paired Runs",
        "",
        "| definition | left exp_dir | right exp_dir |",
        "| --- | --- | --- |",
    ]
    for left, right in pairs:
        lines.append(f"| `{left.definition}` | `{left.exp_dir}` | `{right.exp_dir}` |")

    lines.extend(
        [
            "",
            "## Overall Distribution",
            "",
            "| correctness | left count | left pct | right count | right pct | delta pct |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category in categories:
        left_count = left_counter[category]
        right_count = right_counter[category]
        left_fraction = pct(left_count, left_total)
        right_fraction = pct(right_count, right_total)
        lines.append(
            f"| {category} | {left_count} | {left_fraction:.2%} | "
            f"{right_count} | {right_fraction:.2%} | {right_fraction - left_fraction:+.2%} |"
        )

    if outputs:
        lines.extend(
            [
                "",
                "## Outputs",
                "",
            ]
        )
        for name, output_path in outputs.items():
            lines.append(f"- {name}: `{output_path}`")
    lines.append("")

    path.write_text("\n".join(lines))


def report_slug(experiment: Experiment) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in experiment.definition
    )


def write_mha_reports(
    output_dir: Path,
    pairs: list[tuple[Experiment, Experiment]],
    rows: list[TurnRow],
    turn_limit: int | None,
) -> dict[str, Path]:
    reports_dir = output_dir / "mha_eval_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Path] = {}

    for left, right in pairs:
        if not left.definition.startswith("mha_"):
            continue
        pair_rows = [
            row
            for row in rows
            if (row.arch, row.definition, row.workload) == left.key
        ]
        report_path = reports_dir / f"{report_slug(left)}_error_distribution_summary.md"
        write_markdown_summary(
            report_path,
            [(left, right)],
            pair_rows,
            outputs={},
            turn_limit=turn_limit,
            title=f"{left.definition} Error Distribution Comparison",
        )
        reports[left.definition] = report_path

    return reports


def main() -> None:
    args = parse_args()
    definitions = set(args.definitions or DEFAULT_DEFINITIONS)

    experiments = read_experiment_registry(
        args.experiments_csv,
        left_model=args.left_model,
        right_model=args.right_model,
        definitions=definitions,
        right_run_fragment=args.right_run_fragment,
    )
    pairs = pair_experiments(experiments)

    turn_rows: list[TurnRow] = []
    for left, right in pairs:
        turn_rows.extend(read_turn_rows(left, args.turn_limit))
        turn_rows.extend(read_turn_rows(right, args.turn_limit))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_csv = args.output_dir / "error_distribution.csv"
    errors_only_csv = args.output_dir / "error_rows.csv"
    summary_md = args.output_dir / "error_distribution_summary.md"
    manifest_json = args.output_dir / "manifest.json"

    write_distribution_csv(distribution_csv, build_distribution_rows(turn_rows))
    write_errors_only_csv(errors_only_csv, turn_rows, include_correct=args.include_correct)
    mha_reports = write_mha_reports(args.output_dir, pairs, turn_rows, args.turn_limit)
    outputs = {
        "distribution_csv": distribution_csv,
        "error_rows_csv": errors_only_csv,
        "summary_md": summary_md,
        "manifest_json": manifest_json,
    }
    write_markdown_summary(summary_md, pairs, turn_rows, outputs, args.turn_limit)
    manifest_json.write_text(
        json.dumps(
            {
                "left_model": args.left_model,
                "right_model": args.right_model,
                "right_run_fragment": args.right_run_fragment,
                "turn_limit": args.turn_limit,
                "definitions": sorted(definitions),
                "paired_runs": [
                    {
                        "arch": left.arch,
                        "definition": left.definition,
                        "workload": left.workload,
                        "left_exp_dir": str(left.exp_dir),
                        "right_exp_dir": str(right.exp_dir),
                    }
                    for left, right in pairs
                ],
                "outputs": {name: str(output_path) for name, output_path in outputs.items()},
                "mha_reports": {
                    definition: str(output_path)
                    for definition, output_path in sorted(mha_reports.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"Wrote {distribution_csv}")
    print(f"Wrote {errors_only_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {manifest_json}")
    for definition, report_path in sorted(mha_reports.items()):
        print(f"Wrote {definition}: {report_path}")


if __name__ == "__main__":
    main()
