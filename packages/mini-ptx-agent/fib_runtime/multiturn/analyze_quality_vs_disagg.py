"""Plot per-trajectory quality vs. kernel disaggregation factor.

Inputs: one or more pattern_analysis directories (each containing ted_matrix.npz
produced by analyze_patterns_batch.py). The sibling `trajectories/exp_NNN.json`
files (at pattern_analysis.parent/trajectories/) supply per-turn success and
speedup.

Per trajectory:
  success_rate = num_success_turns / num_total_turns_attempted
  quality       = success_rate * exp(aggregate_speedup)  over passed turns
                  where aggregate is either sum (default) or max
  disagg_factor = (2 / (n*(n-1))) * sum_{i<j} d_ij       over kernels in TED
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# Events in trajectory messages that represent a completed turn attempt.
# A turn is "successful" only when event == "evaluation" AND all_passed is True.
_TURN_EVENTS = ("evaluation", "test_failed")

_LABEL_RE = re.compile(r"^(exp_\d+)/kernel_t\d+\.cu$")


@dataclass
class TrajectoryPoint:
    source_dir: str
    exp_id: str
    num_total_turns: int
    num_success_turns: int
    sum_speedup: float
    max_speedup: float
    quality: float
    quality_metric: str  # "sum" or "max"
    disagg_factor: float
    num_kernels_in_ted: int


def load_ted(pattern_analysis_dir: Path) -> tuple[np.ndarray, list[str]]:
    npz_path = pattern_analysis_dir / "ted_matrix.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"ted_matrix.npz not found in {pattern_analysis_dir}. "
            "Run analyze_patterns_batch.py on the run dir first."
        )
    data = np.load(npz_path, allow_pickle=True)
    matrix = np.asarray(data["matrix"])
    labels = [str(x) for x in data["labels"].tolist()]
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(
            f"ted_matrix shape {matrix.shape} does not match labels length {len(labels)}"
        )
    return matrix, labels


def group_labels_by_exp(labels: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        m = _LABEL_RE.match(label)
        if not m:
            continue
        groups.setdefault(m.group(1), []).append(idx)
    return groups


def compute_disagg(matrix: np.ndarray, indices: list[int]) -> float:
    n = len(indices)
    if n < 2:
        return float("nan")
    sub = matrix[np.ix_(indices, indices)]
    iu = np.triu_indices(n, k=1)
    return float(sub[iu].sum() * 2.0 / (n * (n - 1)))


def parse_trajectory(traj_path: Path) -> list[tuple[bool, Optional[float]]]:
    with traj_path.open() as f:
        traj = json.load(f)
    evals: list[tuple[bool, Optional[float]]] = []
    for msg in traj.get("messages", []):
        if msg.get("role") != "user":
            continue
        extra = msg.get("extra") or {}
        if extra.get("event") not in _TURN_EVENTS:
            continue
        all_passed = bool(extra.get("all_passed", False))
        min_speedup = extra.get("min_speedup")
        if isinstance(min_speedup, (int, float)):
            min_speedup = float(min_speedup)
        else:
            min_speedup = None
        evals.append((all_passed, min_speedup))
    return evals


def compute_quality(
    evals: list[tuple[bool, Optional[float]]],
    metric: str = "sum",
) -> tuple[int, int, float, float, float]:
    """Returns (num_total, num_success, sum_speedup, max_speedup, quality)."""
    num_total = len(evals)
    if num_total == 0:
        return 0, 0, 0.0, 0.0, float("nan")
    passed_speedups = [s for ok, s in evals if ok and s is not None]
    num_success = len(passed_speedups)
    sum_speedup = float(sum(passed_speedups))
    max_speedup = float(max(passed_speedups)) if passed_speedups else 0.0
    success_rate = num_success / num_total
    agg = sum_speedup if metric == "sum" else max_speedup
    quality = success_rate * math.exp(agg)
    return num_total, num_success, sum_speedup, max_speedup, quality


def collect_points(
    pattern_analysis_dir: Path,
    trajectories_dir: Path,
    source_label: str,
    quality_metric: str = "sum",
) -> list[TrajectoryPoint]:
    matrix, labels = load_ted(pattern_analysis_dir)
    groups = group_labels_by_exp(labels)
    points: list[TrajectoryPoint] = []
    for exp_id in sorted(groups):
        indices = groups[exp_id]
        traj_path = trajectories_dir / f"{exp_id}.json"
        if not traj_path.exists():
            warnings.warn(f"[{source_label}] missing trajectory for {exp_id}: {traj_path}")
            continue
        evals = parse_trajectory(traj_path)
        num_total, num_success, sum_s, max_s, quality = compute_quality(evals, quality_metric)
        if num_total == 0:
            warnings.warn(f"[{source_label}] {exp_id}: no turn events in trajectory, skipping")
            continue
        disagg = compute_disagg(matrix, indices)
        points.append(
            TrajectoryPoint(
                source_dir=source_label,
                exp_id=exp_id,
                num_total_turns=num_total,
                num_success_turns=num_success,
                sum_speedup=sum_s,
                max_speedup=max_s,
                quality=quality,
                quality_metric=quality_metric,
                disagg_factor=disagg,
                num_kernels_in_ted=len(indices),
            )
        )
    return points


def _assign_colors(source_labels: list[str]) -> dict[str, tuple]:
    import matplotlib.pyplot as plt

    uniq = list(dict.fromkeys(source_labels))
    cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(uniq)}


def plot_points(points: list[TrajectoryPoint], out_png: Path, annotate: bool, metric: str = "sum") -> None:
    import matplotlib.pyplot as plt

    plotable = [p for p in points if not math.isnan(p.disagg_factor)]
    dropped = len(points) - len(plotable)
    if dropped:
        warnings.warn(f"{dropped} trajectories dropped from plot (disagg NaN, <2 kernels)")
    if not plotable:
        raise RuntimeError("No trajectories left to plot")

    colors = _assign_colors([p.source_dir for p in plotable])

    fig, ax = plt.subplots(figsize=(9, 7))
    by_src: dict[str, list[TrajectoryPoint]] = {}
    for p in plotable:
        by_src.setdefault(p.source_dir, []).append(p)

    for src, pts in by_src.items():
        xs = [p.disagg_factor for p in pts]
        ys = [p.quality for p in pts]
        ax.scatter(xs, ys, color=colors[src], label=src, alpha=0.75, s=40, edgecolors="black", linewidths=0.3)

    if annotate:
        for p in plotable:
            ax.annotate(
                p.exp_id, (p.disagg_factor, p.quality),
                fontsize=6, xytext=(3, 3), textcoords="offset points",
            )

    ys_all = [p.quality for p in plotable if p.quality > 0]
    if ys_all and max(ys_all) / min(ys_all) > 100:
        ax.set_yscale("log")

    agg_label = "Σ speedup_passed" if metric == "sum" else "max speedup_passed"
    ax.set_xlabel("Disaggregation factor  (mean pairwise CFG TED)")
    ax.set_ylabel(f"Trajectory quality  (success_rate · exp({agg_label}))")
    ax.set_title(f"Trajectory quality vs. kernel disaggregation  [metric={metric}]")
    ax.grid(True, linestyle=":", alpha=0.5)
    if len(by_src) >= 2:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_csv(points: list[TrajectoryPoint], out_csv: Path) -> None:
    fields = list(asdict(points[0]).keys()) if points else [
        "source_dir", "exp_id", "num_total_turns", "num_success_turns",
        "sum_speedup", "quality", "disagg_factor", "num_kernels_in_ted",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow(asdict(p))


def _default_trajectories_dir(pattern_analysis_dir: Path) -> Path:
    return pattern_analysis_dir.parent / "trajectories"


def _resolve_source_label(pattern_analysis_dir: Path) -> str:
    # Use the run-dir name (parent of pattern_analysis) for a friendlier legend.
    return pattern_analysis_dir.parent.name or pattern_analysis_dir.name


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "pattern_analysis_dirs", nargs="+", type=Path,
        help="One or more pattern_analysis directories (each containing ted_matrix.npz).",
    )
    ap.add_argument(
        "--trajectories-dir", action="append", type=Path, default=None,
        help="Override the sibling trajectories dir. May be repeated to set per-input "
             "(positional with pattern_analysis_dirs).",
    )
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Directory to write plot+CSV (default: first pattern_analysis dir).")
    ap.add_argument("--output-name", default="quality_vs_disagg",
                    help="Basename for the output files (default: quality_vs_disagg).")
    ap.add_argument("--annotate", action="store_true",
                    help="Draw exp_id next to each point.")
    ap.add_argument("--quality-metric", choices=["sum", "max"], default="sum",
                    help="Aggregate speedup inside exp(): 'sum' = prod(exp(s_i)) over passed, "
                         "'max' = exp(best passed speedup). Default: sum.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    pa_dirs: list[Path] = [p.resolve() for p in args.pattern_analysis_dirs]
    if args.trajectories_dir is None:
        traj_dirs = [_default_trajectories_dir(p) for p in pa_dirs]
    else:
        traj_dirs = [t.resolve() for t in args.trajectories_dir]
        if len(traj_dirs) != len(pa_dirs):
            ap.error(
                f"--trajectories-dir supplied {len(traj_dirs)} times but "
                f"{len(pa_dirs)} pattern_analysis_dirs were given"
            )

    # Build source labels, disambiguate duplicates.
    raw_labels = [_resolve_source_label(p) for p in pa_dirs]
    seen: dict[str, int] = {}
    labels: list[str] = []
    for lab in raw_labels:
        n = seen.get(lab, 0)
        seen[lab] = n + 1
        labels.append(lab if n == 0 else f"{lab}#{n+1}")

    all_points: list[TrajectoryPoint] = []
    for pa_dir, traj_dir, label in zip(pa_dirs, traj_dirs, labels):
        print(f"[{label}] pattern_analysis={pa_dir}")
        print(f"[{label}] trajectories={traj_dir}")
        pts = collect_points(pa_dir, traj_dir, label, args.quality_metric)
        print(f"[{label}] collected {len(pts)} trajectory points")
        all_points.extend(pts)

    if not all_points:
        print("No points collected.", file=sys.stderr)
        return 1

    out_dir = args.output_dir.resolve() if args.output_dir else pa_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{args.output_name}.png"
    out_csv = out_dir / f"{args.output_name}.csv"

    write_csv(all_points, out_csv)
    print(f"wrote {out_csv}  ({len(all_points)} rows)")
    plot_points(all_points, out_png, annotate=args.annotate, metric=args.quality_metric)
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
