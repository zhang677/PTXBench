#!/usr/bin/env python3
"""Plot adjacent-turn alluvial diagrams for eval trajectory status labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


DEFAULT_RUNS = [
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-d128"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-d128-causal"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-bwd-d128"),
    Path("/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-bwd-d128-causal"),
]

STATUS_ORDER = [
    "Correct",
    "Numerical error",
    "Compilation error",
    "Kernel Execution Timeout",
    "Runtime error",
    "Sanitize Timeout",
    "Profiling Service Timeout",
    "Extraction error",
    "Other error",
]

STATUS_COLORS = {
    "Correct": "#2ca02c",
    "Numerical error": "#d62728",
    "Compilation error": "#ff7f0e",
    "Kernel Execution Timeout": "#9467bd",
    "Runtime error": "#1f77b4",
    "Sanitize Timeout": "#8c564b",
    "Profiling Service Timeout": "#e377c2",
    "Extraction error": "#7f7f7f",
    "Other error": "#bcbd22",
}


def classify_turn(content: str) -> str:
    """Mirror the repository-root benchmark/export_turn_correctness_arch.py labels."""
    if "PASSED" in content:
        return "Correct"
    if "INCORRECT_NUMERICAL" in content or "Result is incorrect" in content:
        return "Numerical error"
    if "Failed to compile kernel" in content:
        return "Compilation error"
    if re.search(r"Timed out after \d+(?:\.\d+)?s waiting for sanitize", content):
        return "Sanitize Timeout"
    if "returncode 137" in content:
        return "Profiling Service Timeout"
    if (
        "Kernel execution timed out" in content
        or "Evaluation timeout after" in content
        or "memcheck timed out" in content
        or re.search(r"\]\s+TIMEOUT\b", content)
    ):
        return "Kernel Execution Timeout"
    if "Could not extract" in content:
        return "Extraction error"
    if "RUNTIME_ERROR" in content or "CUDA error" in content or "CU error" in content:
        return "Runtime error"
    return "Other error"


def extract_turn_sequence(traj: dict) -> list[str]:
    seq: list[str] = []
    first_user = True
    for i, msg in enumerate(traj.get("messages", [])):
        if msg.get("role") != "user" or i == 0:
            continue
        if first_user:
            first_user = False
            continue
        seq.append(classify_turn(msg.get("content", "")))
    return seq


def run_label(run_dir: Path) -> str:
    name = run_dir.name
    if "note-feedback2" in name:
        task_name = name.split("8defs-", 1)[-1]
        if "-d128" in task_name:
            return "note-feedback2-d128-8defs"
        if "-d96" in task_name:
            return "note-feedback2-d96-8defs"
        return "note-feedback2-d96-8defs"
    if "retrieved-notes" in name:
        return "retrieved-notes-d128-4defs"
    return name


def load_turn_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    plan_by_exp = {}
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        with plan_path.open() as f:
            plan = json.load(f)
        plan_by_exp = {int(item["exp_index"]): item for item in plan.get("plan", [])}

    for traj_path in sorted((run_dir / "trajectories").glob("exp_*.json")):
        if ".bak" in traj_path.name:
            continue
        with traj_path.open() as f:
            traj = json.load(f)
        exp_index = int(traj_path.stem.split("_", 1)[1])
        meta = plan_by_exp.get(exp_index, {})
        for turn, status in enumerate(extract_turn_sequence(traj)):
            rows.append(
                {
                    "run": run_label(run_dir),
                    "run_dir": str(run_dir),
                    "trajectory": traj_path.stem,
                    "exp_index": exp_index,
                    "turn": turn,
                    "status": status,
                    "definition": meta.get("definition", ""),
                    "prompt_tag": meta.get("prompt_tag", ""),
                }
            )
    return rows


def ordered_statuses(statuses: set[str]) -> list[str]:
    ordered = [status for status in STATUS_ORDER if status in statuses]
    ordered.extend(sorted(statuses.difference(STATUS_ORDER)))
    return ordered


def stacked_positions(totals: dict[str, int], order: list[str], total: int) -> dict[str, tuple[float, float]]:
    gap = 0.025
    usable = 0.78 - gap * max(0, len(order) - 1)
    y = 0.86
    positions = {}
    for status in order:
        height = usable * totals.get(status, 0) / total
        positions[status] = (y - height, y)
        y -= height + gap
    return positions


def add_flow(ax, x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float, color: str):
    dx = (x1 - x0) * 0.45
    verts = [
        (x0, y0a),
        (x0 + dx, y0a),
        (x1 - dx, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - dx, y1b),
        (x0 + dx, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=0.42))


def plot_alluvial(
    title: str,
    left_label: str,
    right_label: str,
    transitions: Counter[tuple[str, str]],
    output_path: Path | None = None,
):
    total = sum(transitions.values())
    statuses = {status for edge in transitions for status in edge}
    order = ordered_statuses(statuses)
    left_totals: dict[str, int] = defaultdict(int)
    right_totals: dict[str, int] = defaultdict(int)
    for (src, dst), count in transitions.items():
        left_totals[src] += count
        right_totals[dst] += count

    left_pos = stacked_positions(left_totals, order, total)
    right_pos = stacked_positions(right_totals, order, total)
    left_offsets = {status: left_pos[status][0] for status in order}
    right_offsets = {status: right_pos[status][0] for status in order}
    usable = 0.78 - 0.025 * max(0, len(order) - 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.99,
        f"{title} (n={total})",
        fontsize=13,
        weight="bold",
        ha="center",
        va="top",
    )
    ax.text(0.16, 0.90, left_label, fontsize=11, ha="center")
    ax.text(0.84, 0.90, right_label, fontsize=11, ha="center")

    sorted_edges = sorted(transitions.items(), key=lambda kv: (order.index(kv[0][0]), order.index(kv[0][1])))
    for (src, dst), count in sorted_edges:
        height = usable * count / total
        y0a = left_offsets[src]
        y0b = y0a + height
        y1a = right_offsets[dst]
        y1b = y1a + height
        add_flow(ax, 0.20, 0.80, y0a, y0b, y1a, y1b, STATUS_COLORS.get(src, "#999999"))
        left_offsets[src] += height
        right_offsets[dst] += height

    for side, x, positions, totals, ha in [
        ("left", 0.12, left_pos, left_totals, "right"),
        ("right", 0.80, right_pos, right_totals, "left"),
    ]:
        for status in order:
            y0, y1 = positions[status]
            if math.isclose(y0, y1):
                continue
            color = STATUS_COLORS.get(status, "#999999")
            rect_x = x if side == "left" else x
            ax.add_patch(Rectangle((rect_x, y0), 0.08, y1 - y0, facecolor=color, edgecolor="white", lw=1.0))
            label_x = rect_x - 0.012 if side == "left" else rect_x + 0.092
            ax.text(label_x, (y0 + y1) / 2, f"{status} ({totals[status]})", fontsize=9.5, ha=ha, va="center")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_pair(run: str, pair: int, transitions: Counter[tuple[str, str]], output_path: Path | None = None):
    return plot_alluvial(
        title=f"{run}: turn {pair} -> {pair + 1}",
        left_label=f"turn {pair}",
        right_label=f"turn {pair + 1}",
        transitions=transitions,
        output_path=output_path,
    )


def plot_aggregate(run: str, transitions: Counter[tuple[str, str]], output_path: Path | None = None):
    return plot_alluvial(
        title=f"{run}: pooled adjacent-turn transitions",
        left_label="turn t",
        right_label="turn t+1",
        transitions=transitions,
        output_path=output_path,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures/alluvial_turn_pairs"))
    parser.add_argument("runs", nargs="*", type=Path, default=DEFAULT_RUNS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in args.output_dir.glob("*/turn_*_to_*.png"):
        stale_png.unlink()

    all_turn_rows: list[dict[str, object]] = []
    for run_dir in args.runs:
        all_turn_rows.extend(load_turn_rows(run_dir))

    write_csv(
        args.output_dir / "turn_statuses.csv",
        all_turn_rows,
        ["run", "run_dir", "trajectory", "exp_index", "turn", "status", "definition", "prompt_tag"],
    )

    labels_by_run_traj: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    for row in all_turn_rows:
        labels_by_run_traj[
            (str(row["run"]), str(row["run_dir"]), str(row["trajectory"]))
        ][int(row["turn"])] = str(row["status"])

    transitions_by_run_pair: dict[tuple[str, int], Counter[tuple[str, str]]] = defaultdict(Counter)
    transitions_by_run: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for (run, _run_dir, _trajectory), labels in labels_by_run_traj.items():
        for turn in range(max(labels.keys(), default=-1)):
            if turn in labels and turn + 1 in labels:
                transitions_by_run_pair[(run, turn)][(labels[turn], labels[turn + 1])] += 1
                transitions_by_run[run][(labels[turn], labels[turn + 1])] += 1

    transition_rows = []
    for (run, turn), transitions in sorted(transitions_by_run_pair.items()):
        for (src, dst), count in sorted(transitions.items()):
            transition_rows.append(
                {
                    "run": run,
                    "turn_from": turn,
                    "turn_to": turn + 1,
                    "status_from": src,
                    "status_to": dst,
                    "count": count,
                }
            )
    write_csv(
        args.output_dir / "pair_transitions.csv",
        transition_rows,
        ["run", "turn_from", "turn_to", "status_from", "status_to", "count"],
    )

    aggregate_rows = []
    for run, transitions in sorted(transitions_by_run.items()):
        for (src, dst), count in sorted(transitions.items()):
            aggregate_rows.append(
                {
                    "run": run,
                    "status_from": src,
                    "status_to": dst,
                    "count": count,
                }
            )
    write_csv(
        args.output_dir / "aggregate_transitions.csv",
        aggregate_rows,
        ["run", "status_from", "status_to", "count"],
    )

    pdfs: dict[str, PdfPages] = {}
    try:
        for (run, turn), transitions in sorted(transitions_by_run_pair.items()):
            safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run)
            fig = plot_pair(run, turn, transitions)
            pdf = pdfs.get(run)
            if pdf is None:
                pdf_path = args.output_dir / safe_run / "all_turn_pairs.pdf"
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf = PdfPages(pdf_path)
                pdfs[run] = pdf
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    finally:
        for pdf in pdfs.values():
            pdf.close()

    combined_aggregate_pdf = args.output_dir / "all_runs_aggregate.pdf"
    if combined_aggregate_pdf.exists():
        combined_aggregate_pdf.unlink()
    for run, transitions in sorted(transitions_by_run.items()):
        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run)
        fig = plot_aggregate(
            run,
            transitions,
            args.output_dir / safe_run / "all_turns_aggregate.png",
        )
        plt.close(fig)

    print(f"Wrote {len(all_turn_rows)} turn rows")
    print(f"Wrote {len(transition_rows)} transition rows")
    print(f"Wrote {len(aggregate_rows)} aggregate transition rows")
    print(f"Wrote plots under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
