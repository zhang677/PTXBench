#!/usr/bin/env python3
"""Plot the Fixit outcome distribution by stage from source artifacts."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTCOME_LABELS = [
    "Correct",
    "Compilation error",
    "Runtime error",
    "Numerical error",
    "Kernel Execution Timeout",
    "Other error",
    "Extraction error",
]

STAGE_COLORS = {
    "Before fixit": "#4C78A8",
    "Teacher": "#F58518",
    "After fixit": "#54A24B",
}

DEFAULT_EVAL_ROOT = Path("/home/ubuntu/AccRL-exps/eval_runs")
DEFAULT_AUDIT_CSV = Path(
    "/home/ubuntu/AccRL-exps/tasks/test_scale_gemini_fixit_wrong_kernel_table.csv"
)
DEFAULT_AFTER_RUN_DIRS = [
    DEFAULT_EVAL_ROOT / "2026-0629-2229-mha-bwd-d128",
    DEFAULT_EVAL_ROOT / "2026-0629-2229-mha-bwd-d128-causal",
    DEFAULT_EVAL_ROOT / "2026-0629-2229-mha-d128",
    DEFAULT_EVAL_ROOT / "2026-0629-2229-mha-d128-causal",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_turn_index(run_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = load_csv(run_dir / "figures" / "turn_correctness_arch.csv")
    return {(row["trajectory_id"], row["turn"]): row for row in rows}


def is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def final_teacher_outcome(row: dict[str, str], *, max_fix_turn: int) -> str:
    outcomes = [row.get(f"fix_turn_{turn}", "").strip() for turn in range(max_fix_turn + 1)]
    if any(is_number(outcome) for outcome in outcomes):
        return "Correct"
    return next((outcome for outcome in reversed(outcomes) if outcome and outcome != "none"), "Other error")


def compute_before_and_teacher_counts(
    audit_csv: Path,
    eval_root: Path,
    *,
    max_teacher_fix_turn: int,
) -> tuple[Counter[str], Counter[str]]:
    audit_rows = load_csv(audit_csv)
    source_turns: dict[Path, dict[tuple[str, str], dict[str, str]]] = {}
    before_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()

    for row in audit_rows:
        source_run = eval_root / row["source_run"]
        if source_run not in source_turns:
            source_turns[source_run] = load_turn_index(source_run)

        source_key = (row["source_exp"], row["source_turn"])
        try:
            source_turn = source_turns[source_run][source_key]
        except KeyError as exc:
            raise KeyError(
                f"Missing source turn {source_key} in {source_run / 'figures' / 'turn_correctness_arch.csv'}"
            ) from exc

        before_counts[source_turn["correctness"]] += 1
        teacher_counts[final_teacher_outcome(row, max_fix_turn=max_teacher_fix_turn)] += 1

    return before_counts, teacher_counts


def compute_after_counts(after_run_dirs: list[Path], *, max_after_turn: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for run_dir in after_run_dirs:
        for row in load_csv(run_dir / "figures" / "turn_correctness_arch.csv"):
            if int(row["turn"]) <= max_after_turn:
                counts[row["correctness"]] += 1
    return counts


def compute_stage_counts(
    audit_csv: Path,
    eval_root: Path,
    after_run_dirs: list[Path],
    *,
    max_teacher_fix_turn: int,
    max_after_turn: int,
) -> dict[str, Counter[str]]:
    before_counts, teacher_counts = compute_before_and_teacher_counts(
        audit_csv,
        eval_root,
        max_teacher_fix_turn=max_teacher_fix_turn,
    )
    after_counts = compute_after_counts(after_run_dirs, max_after_turn=max_after_turn)
    return {
        "Before fixit": before_counts,
        "Teacher": teacher_counts,
        "After fixit": after_counts,
    }


def plot(stage_counts: dict[str, Counter[str]], output: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    x_positions = list(range(len(OUTCOME_LABELS)))
    width = 0.25

    for stage_index, (stage_name, counts) in enumerate(stage_counts.items()):
        denominator = sum(counts.values())
        if denominator == 0:
            raise ValueError(f"No rows found for stage {stage_name!r}")
        values = [
            counts.get(outcome_label, 0) / denominator * 100
            for outcome_label in OUTCOME_LABELS
        ]
        offsets = [x + (stage_index - 1) * width for x in x_positions]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=f"{stage_name} (n={denominator})",
            color=STAGE_COLORS[stage_name],
        )
        for bar, value in zip(bars, values):
            if value >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{value:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Percentage of kernels / attempts")
    ax.set_xlabel("Outcome type")
    ax.set_title("Fixit outcome distribution by stage")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(OUTCOME_LABELS, rotation=25, ha="right")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncols=3, loc="upper right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Fixit outcome distribution grouped bar chart."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("fixit_outcome_distribution.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI. The recovered original used 180.",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
        help="Gemini fixit wrong-kernel audit CSV.",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Root containing source eval run directories.",
    )
    parser.add_argument(
        "--after-run-dir",
        type=Path,
        action="append",
        default=None,
        help="Fixit-v2 GLM after-fixit run dir. Repeat to override defaults.",
    )
    parser.add_argument(
        "--max-teacher-fix-turn",
        type=int,
        default=4,
        help="Highest fix_turn_N column to use for Gemini teacher outcomes.",
    )
    parser.add_argument(
        "--max-after-turn",
        type=int,
        default=3,
        help="Highest attempted turn to count from after-fixit run dirs.",
    )
    args = parser.parse_args()

    after_run_dirs = args.after_run_dir or DEFAULT_AFTER_RUN_DIRS
    stage_counts = compute_stage_counts(
        args.audit_csv,
        args.eval_root,
        after_run_dirs,
        max_teacher_fix_turn=args.max_teacher_fix_turn,
        max_after_turn=args.max_after_turn,
    )
    for stage_name, counts in stage_counts.items():
        summary = ", ".join(
            f"{outcome}={counts.get(outcome, 0)}" for outcome in OUTCOME_LABELS
        )
        print(f"{stage_name} (n={sum(counts.values())}): {summary}")

    plot(stage_counts, args.output, dpi=args.dpi)
    print(args.output)


if __name__ == "__main__":
    main()
