#!/usr/bin/env python3
"""Cross-run CFG TED matrix.

For two eval runs (run_a, run_b), compute the **rectangular** M×N matrix of
normalized Zhang-Shasha tree edit distances between every CFG-hash
representative in run_a and every representative in run_b. Save the matrix
as a compressed NPZ and render a rectangular heatmap.

Within-run pairs are *not* computed (they already live in each run's own
`pattern_analysis/ted_matrix.npz`). The resulting matrix is asymmetric —
different row/column populations — so both the storage format and the
heatmap differ from the within-run analysis in analyze_patterns_batch.py.

Usage:
    python analyze_patterns_cross_run.py <run_a> <run_b>
        [--output-dir DIR] [--jobs N] [--reference-fns PATH]
        [--tag-a NAME] [--tag-b NAME]

`run_a` / `run_b` may be either the run root (containing
`kernels/exp_*/kernel_*.cu` and `pattern_analysis/`) or the `kernels/`
subdirectory directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from zss import simple_distance

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_patterns_batch import (  # noqa: E402
    DEFAULT_REFERENCE_FNS,
    KERNEL_IDX_RE,
    ZssCache,
    _pick_representatives,
    analyze_one,
    discover_kernels,
    get_model_name,
    load_reference_fn_names,
)

CROSS_MATRIX_FILENAME = "ted_matrix.npz"
CROSS_HEATMAP_FILENAME = "ted_heatmap.png"
CROSS_SUMMARY_FILENAME = "cross_summary.json"


# ---------------------------------------------------------------------------
# Input resolution.

def resolve_run_paths(run_dir: Path) -> tuple[Path, Path]:
    """Return (run_root, kernels_root) for a user-supplied run directory.

    Accepts either the run root (which contains `kernels/` and
    `pattern_analysis/` side by side) or the `kernels/` subdirectory.
    """
    run_dir = run_dir.resolve()
    if (run_dir / "kernels").is_dir():
        return run_dir, run_dir / "kernels"
    if run_dir.name == "kernels":
        return run_dir.parent, run_dir
    # Fall back: treat as a generic kernel tree. `pattern_analysis` may not
    # exist yet; the cache lookup will miss and we'll rebuild.
    return run_dir, run_dir


def default_tag(run_root: Path) -> str:
    name = run_root.name
    # Strip a leading YYYY- or YYYY-MM- prefix to shorten labels in plots.
    parts = name.split("-")
    if parts and parts[0].isdigit() and len(parts[0]) == 4:
        return "-".join(parts[1:]) or name
    return name


# ---------------------------------------------------------------------------
# Record loading.

def load_or_build_records(
    run_root: Path,
    kernels_root: Path,
    reference_names: set[str],
    jobs: int,
    correct_only: bool = False,
) -> list[dict]:
    """Prefer `<run_root>/pattern_analysis[_correct]/kernels.jsonl` if present,
    else build fresh records from kernel sources."""
    cache_subdir = "pattern_analysis_correct" if correct_only else "pattern_analysis"
    cached = run_root / cache_subdir / "kernels.jsonl"
    if cached.is_file():
        with cached.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"[cross-ted] loaded {len(records)} cached records from {cached}")
        return records

    if correct_only:
        source_root = run_root / "success"
        if not source_root.is_dir():
            sys.exit(
                f"[cross-ted] --correct-only: success/ not found at {source_root}"
            )
    else:
        source_root = kernels_root

    kernels = discover_kernels(source_root)
    if not kernels:
        sys.exit(f"[cross-ted] no kernel_*.cu found under {source_root}")
    print(f"[cross-ted] analyzing {len(kernels)} kernels under {source_root}")
    tasks = [(p, reference_names) for p in kernels]
    if jobs > 1 and len(tasks) > 1:
        with Pool(jobs) as pool:
            records = pool.map(analyze_one, tasks)
    else:
        records = [analyze_one(t) for t in tasks]
    records.sort(key=lambda r: (r["exp"], r["family"], r["index"]))
    return records


# ---------------------------------------------------------------------------
# Cross-run distance.

def qualify_record(rec: dict, tag: str) -> dict:
    """Return a shallow copy of `rec` with a run-tagged kernel_id so two
    runs can share a single `ZssCache` without id collisions."""
    out = dict(rec)
    out["kernel_id"] = f"{tag}/{rec['kernel_id']}"
    return out


def cross_distances(
    reps_a: list[dict],
    reps_b: list[dict],
    cache: ZssCache,
) -> np.ndarray:
    """Compute the M×N matrix of normalized TED distances between every
    rep in reps_a (rows) and every rep in reps_b (columns).
    """
    M, N = len(reps_a), len(reps_b)
    out = np.zeros((M, N), dtype=np.float32)
    t0 = time.time()
    # Pre-fetch row trees once; each is reused N times.
    za_all = [cache.get(r["kernel_id"]) for r in reps_a]
    zb_all = [cache.get(r["kernel_id"]) for r in reps_b]
    size_a = [max(r["cfg_node_count"], 1) for r in reps_a]
    size_b = [max(r["cfg_node_count"], 1) for r in reps_b]
    for i in range(M):
        for j in range(N):
            raw = int(simple_distance(za_all[i], zb_all[j]))
            out[i, j] = raw / max(size_a[i], size_b[j])
    dt = time.time() - t0
    print(f"[cross-ted] computed {M}×{N}={M*N} pairs in {dt:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Heatmap expansion + plotting.

def _kid_sort_key(kid: str) -> tuple[str, str, int]:
    exp, name = kid.split("/", 1)
    m = KERNEL_IDX_RE.search(name)
    family = m.group(1) if m else ""
    idx = int(m.group(2)) if m else -1
    return (exp, family, idx)


def _exp_structure(kids: list[str]) -> tuple[list[int], list[str]]:
    """Given kernel_ids already sorted by (exp, family, index), return
    (offsets, exp_names) where offsets marks the row index at which each
    exp block starts (last entry is len(kids))."""
    exp_names: list[str] = []
    offsets: list[int] = [0]
    prev: str | None = None
    for i, kid in enumerate(kids):
        exp = kid.split("/", 1)[0]
        if exp != prev:
            if prev is not None:
                offsets.append(i)
            exp_names.append(exp)
            prev = exp
    offsets.append(len(kids))
    return offsets, exp_names


def _expand_cross_by_exp(
    reps_a: list[dict],
    members_a: dict[str, list[str]],
    reps_b: list[dict],
    members_b: dict[str, list[str]],
    matrix: np.ndarray,
    tag_a: str,
    tag_b: str,
) -> tuple[
    list[str], list[str], np.ndarray,
    list[int], list[int], list[str], list[str],
]:
    """Expand the rep-level M×N matrix into a per-kernel R×C matrix where
    rows are ordered by (exp, family, index) in run A and columns same in B.

    Uses the original (un-qualified) kernel_ids for member lookup since
    `members_*` keys are the pre-qualification rep ids. Returns
    (row_labels, col_labels, M_full, row_exp_offsets, col_exp_offsets,
    row_exp_names, col_exp_names).
    """
    def _kid_to_rep_idx(reps, members):
        out = {}
        for rep_idx, rep in enumerate(reps):
            for kid in members[rep["kernel_id"]]:
                out[kid] = rep_idx
        return out

    kid_to_rep_a = _kid_to_rep_idx(reps_a, members_a)
    kid_to_rep_b = _kid_to_rep_idx(reps_b, members_b)
    row_kids = sorted(kid_to_rep_a.keys(), key=_kid_sort_key)
    col_kids = sorted(kid_to_rep_b.keys(), key=_kid_sort_key)
    row_labels = [f"{tag_a}/{kid}" for kid in row_kids]
    col_labels = [f"{tag_b}/{kid}" for kid in col_kids]
    row_exp_offsets, row_exp_names = _exp_structure(row_kids)
    col_exp_offsets, col_exp_names = _exp_structure(col_kids)

    row_reps = np.array([kid_to_rep_a[kid] for kid in row_kids])
    col_reps = np.array([kid_to_rep_b[kid] for kid in col_kids])
    M_full = matrix[row_reps[:, None], col_reps[None, :]]
    return (row_labels, col_labels, M_full,
            row_exp_offsets, col_exp_offsets,
            row_exp_names, col_exp_names)


def plot_cross_heatmap(
    reps_a: list[dict],
    members_a: dict[str, list[str]],
    reps_b: list[dict],
    members_b: dict[str, list[str]],
    matrix: np.ndarray,
    run_root_a: Path,
    run_root_b: Path,
    tag_a: str,
    tag_b: str,
    out_path: Path,
) -> None:
    (row_labels, col_labels, M_full,
     row_exp_offsets, col_exp_offsets,
     row_exp_names, col_exp_names) = _expand_cross_by_exp(
        reps_a, members_a, reps_b, members_b, matrix, tag_a, tag_b,
    )
    R, C = M_full.shape
    if R == 0 or C == 0:
        print("[cross-ted] empty matrix — nothing to plot")
        return

    cmap = plt.get_cmap("viridis").copy()
    width = max(6.0, min(24.0, C * 0.18 + 2))
    height = max(6.0, min(24.0, R * 0.18 + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(
        M_full, cmap=cmap, vmin=0.0, vmax=1.0,
        interpolation="nearest", aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, aspect=25)
    cbar.set_label("cfg_ted_norm", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Black lines at exp_* block boundaries on each axis.
    for off in row_exp_offsets[1:-1]:
        ax.axhline(off - 0.5, color="black", linewidth=0.6, alpha=0.7)
    for off in col_exp_offsets[1:-1]:
        ax.axvline(off - 0.5, color="black", linewidth=0.6, alpha=0.7)

    # One tick per exp, centred on its block.
    row_centres = [
        (row_exp_offsets[i] + row_exp_offsets[i + 1] - 1) / 2
        for i in range(len(row_exp_names))
    ]
    col_centres = [
        (col_exp_offsets[i] + col_exp_offsets[i + 1] - 1) / 2
        for i in range(len(col_exp_names))
    ]
    ax.set_yticks(row_centres)
    ax.set_yticklabels(row_exp_names, fontsize=8)
    ax.set_xticks(col_centres)
    ax.set_xticklabels(col_exp_names, fontsize=8, rotation=90)
    ax.tick_params(axis="both", length=0)

    model_a = get_model_name(run_root_a)
    model_b = get_model_name(run_root_b)
    title = (
        f"{model_a} ({tag_a}) vs {model_b} ({tag_b})\n"
        f"Cross-run CFG TED — {R} × {C} kernels "
        f"({len(reps_a)} × {len(reps_b)} reps)"
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(f"run A — {tag_a}", fontsize=11)
    ax.set_xlabel(f"run B — {tag_b}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[cross-ted] wrote heatmap to {out_path}")


# ---------------------------------------------------------------------------
# NPZ writer + summary.

def save_cross_matrix(
    matrix: np.ndarray,
    reps_a: list[dict],
    reps_b: list[dict],
    tag_a: str,
    tag_b: str,
    out_path: Path,
) -> None:
    np.savez_compressed(
        out_path,
        matrix=matrix,
        row_labels=np.array(
            [f"{tag_a}/{r['kernel_id']}" for r in reps_a], dtype=object,
        ),
        col_labels=np.array(
            [f"{tag_b}/{r['kernel_id']}" for r in reps_b], dtype=object,
        ),
        row_run=np.array(tag_a),
        col_run=np.array(tag_b),
    )
    print(f"[cross-ted] saved {matrix.shape[0]}x{matrix.shape[1]} matrix "
          f"to {out_path}")


def load_cross_matrix(
    reps_a: list[dict],
    reps_b: list[dict],
    tag_a: str,
    tag_b: str,
    out_path: Path,
) -> np.ndarray | None:
    """Reload the cached rep-level M×N matrix if it matches the current
    rep sets. Returns None when the cache is missing or the labels don't
    match — re-indexes if the cached order differs from the current one.
    """
    if not out_path.is_file():
        return None
    data = np.load(out_path, allow_pickle=True)
    cached_rows = [str(x) for x in data["row_labels"]]
    cached_cols = [str(x) for x in data["col_labels"]]
    current_rows = [f"{tag_a}/{r['kernel_id']}" for r in reps_a]
    current_cols = [f"{tag_b}/{r['kernel_id']}" for r in reps_b]
    if (set(cached_rows) != set(current_rows)
            or set(cached_cols) != set(current_cols)):
        return None
    M = data["matrix"]
    if cached_rows == current_rows and cached_cols == current_cols:
        out = np.ascontiguousarray(M, dtype=np.float32)
    else:
        row_pos = {lbl: i for i, lbl in enumerate(cached_rows)}
        col_pos = {lbl: i for i, lbl in enumerate(cached_cols)}
        row_idx = np.array([row_pos[r] for r in current_rows])
        col_idx = np.array([col_pos[c] for c in current_cols])
        out = np.ascontiguousarray(
            M[row_idx[:, None], col_idx[None, :]], dtype=np.float32,
        )
    print(f"[cross-ted] loaded cached {out.shape[0]}×{out.shape[1]} "
          f"matrix from {out_path}")
    return out


def summarize(
    matrix: np.ndarray,
    reps_a: list[dict],
    reps_b: list[dict],
    members_a: dict[str, list[str]],
    members_b: dict[str, list[str]],
    tag_a: str,
    tag_b: str,
) -> dict:
    flat = matrix.reshape(-1)
    i_min, j_min = np.unravel_index(int(np.argmin(matrix)), matrix.shape)
    i_max, j_max = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    row_kernels = sum(len(members_a[r["kernel_id"]]) for r in reps_a)
    col_kernels = sum(len(members_b[r["kernel_id"]]) for r in reps_b)
    return {
        "row_run": tag_a,
        "col_run": tag_b,
        "row_reps": len(reps_a),
        "col_reps": len(reps_b),
        "row_kernels": row_kernels,
        "col_kernels": col_kernels,
        "min": float(flat.min()),
        "mean": float(flat.mean()),
        "max": float(flat.max()),
        "closest_pair": {
            "row": f"{tag_a}/{reps_a[int(i_min)]['kernel_id']}",
            "col": f"{tag_b}/{reps_b[int(j_min)]['kernel_id']}",
            "dist": float(matrix[i_min, j_min]),
        },
        "farthest_pair": {
            "row": f"{tag_a}/{reps_a[int(i_max)]['kernel_id']}",
            "col": f"{tag_b}/{reps_b[int(j_max)]['kernel_id']}",
            "dist": float(matrix[i_max, j_max]),
        },
    }


# ---------------------------------------------------------------------------
# Driver.

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] if __doc__ else "",
    )
    ap.add_argument("run_a", type=Path,
                    help="First eval run (root or kernels/ dir)")
    ap.add_argument("run_b", type=Path,
                    help="Second eval run (root or kernels/ dir)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write ted_matrix.npz + heatmap. "
                         "Default: <run_a>/pattern_analysis_cross_<tagB>/")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--reference-fns", type=Path,
                    default=DEFAULT_REFERENCE_FNS)
    ap.add_argument("--tag-a", type=str, default=None,
                    help="Short label for run A (default: run_a dir name)")
    ap.add_argument("--tag-b", type=str, default=None,
                    help="Short label for run B (default: run_b dir name)")
    ap.add_argument("--correct-only", action="store_true",
                    help="Use only kernels that passed evaluation. Loads from "
                         "<run_root>/pattern_analysis_correct/kernels.jsonl "
                         "(falls back to <run_root>/success/). Default output "
                         "dir gets a _correct suffix.")
    args = ap.parse_args()

    run_root_a, kernels_a = resolve_run_paths(args.run_a)
    run_root_b, kernels_b = resolve_run_paths(args.run_b)
    tag_a = args.tag_a or default_tag(run_root_a)
    tag_b = args.tag_b or default_tag(run_root_b)

    suffix = "_correct" if args.correct_only else ""
    out_dir: Path = args.output_dir or (
        run_root_a / f"pattern_analysis_cross_{tag_b}{suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_names = load_reference_fn_names(args.reference_fns)

    records_a = load_or_build_records(
        run_root_a, kernels_a, reference_names, args.jobs,
        correct_only=args.correct_only,
    )
    records_b = load_or_build_records(
        run_root_b, kernels_b, reference_names, args.jobs,
        correct_only=args.correct_only,
    )

    reps_a, members_a = _pick_representatives(records_a)
    reps_b, members_b = _pick_representatives(records_b)
    if not reps_a or not reps_b:
        sys.exit(
            f"[cross-ted] empty rep set: |A|={len(reps_a)}, |B|={len(reps_b)}"
        )
    print(f"[cross-ted] run A ({tag_a}): {len(records_a)} kernels, "
          f"{len(reps_a)} reps")
    print(f"[cross-ted] run B ({tag_b}): {len(records_b)} kernels, "
          f"{len(reps_b)} reps")

    matrix_path = out_dir / CROSS_MATRIX_FILENAME
    matrix = load_cross_matrix(reps_a, reps_b, tag_a, tag_b, matrix_path)
    if matrix is None:
        reps_a_q = [qualify_record(r, tag_a) for r in reps_a]
        reps_b_q = [qualify_record(r, tag_b) for r in reps_b]
        cache = ZssCache(reps_a_q + reps_b_q)
        matrix = cross_distances(reps_a_q, reps_b_q, cache)
        save_cross_matrix(matrix, reps_a, reps_b, tag_a, tag_b, matrix_path)

    plot_cross_heatmap(
        reps_a, members_a, reps_b, members_b, matrix,
        run_root_a, run_root_b, tag_a, tag_b,
        out_dir / CROSS_HEATMAP_FILENAME,
    )

    summary = summarize(matrix, reps_a, reps_b,
                        members_a, members_b, tag_a, tag_b)
    (out_dir / CROSS_SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[cross-ted] wrote summary to {out_dir / CROSS_SUMMARY_FILENAME}")
    print(f"[cross-ted] done. min={summary['min']:.3f} "
          f"mean={summary['mean']:.3f} max={summary['max']:.3f}")


if __name__ == "__main__":
    main()
