#!/usr/bin/env python3
"""Batch-render CFG pattern trees for every kernel_*.cu under <run_dir>.

Output mirrors the `kernels/exp_*/` layout under `<run_dir>/pattern_trees/`
(or `--output-dir`), and an `index.csv` is written alongside for
cross-reference against `pattern_analysis/kernels.jsonl`.

Usage:
    python draw_pattern_trees_batch.py <run_dir> [--output-dir DIR]
        [--format png|svg|pdf] [--jobs N] [--no-collapse-seq] [--force]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pattern import extract_filtered_kernel_text  # noqa: E402
from analyze_patterns_batch import discover_kernels  # noqa: E402
from draw_pattern_tree import render_kernel_tree  # noqa: E402


def _output_path(kernel: Path, kernels_root: Path, out_dir: Path,
                 fmt: str) -> Path:
    rel = kernel.relative_to(kernels_root)
    return (out_dir / rel).with_suffix("." + fmt)


def _artifact_state(kernel: Path, out_path: Path) -> str:
    """Return 'fresh' (both artifacts up to date), 'source-only' (image is
    current but .cu is missing/stale), or 'render' (image missing/stale)."""
    source_path = out_path.with_suffix(".cu")
    try:
        kernel_mtime = kernel.stat().st_mtime
    except OSError:
        return "render"

    def _fresh(p: Path) -> bool:
        if not p.exists():
            return False
        try:
            return p.stat().st_mtime >= kernel_mtime
        except OSError:
            return False

    if not _fresh(out_path):
        return "render"
    if not _fresh(source_path):
        return "source-only"
    return "fresh"


def _emit_source_companion(kernel: Path, image_path: Path) -> str:
    """Write the filtered-kernel text next to `image_path` as `<stem>.cu`."""
    src = kernel.read_text()
    source_path = image_path.with_suffix(".cu")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(extract_filtered_kernel_text(src))
    return str(source_path)


def _render_worker(args: tuple) -> dict:
    kernel, out_path, collapse, fmt = args
    out_path = Path(out_path)
    try:
        result = render_kernel_tree(
            Path(kernel), out_path, collapse=collapse, fmt=fmt,
        )
        return {"status": "ok", "kernel": str(kernel),
                "image": result["image"],
                "node_count": result["node_count"],
                "depth": result["depth"],
                "branches": result["branches"],
                "loops": result["loops"]}
    except Exception as e:  # noqa: BLE001 — batch: one bad file shouldn't kill run
        return {"status": "fail", "kernel": str(kernel),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] if __doc__ else "")
    ap.add_argument("run_dir", type=Path,
                    help="Directory containing kernels/exp_*/kernel_*.cu")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Default: <run_dir>/pattern_trees")
    ap.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--no-collapse-seq", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if output is newer than input")
    ap.add_argument("--correct-only", action="store_true",
                    help="Render only kernels that passed evaluation "
                         "(walks <run_root>/success/ instead of kernels/). "
                         "Default output dir becomes pattern_trees_correct/.")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"run_dir not found: {run_dir}")

    if run_dir.name == "kernels":
        run_root = run_dir.parent
    elif (run_dir / "kernels").is_dir():
        run_root = run_dir
    else:
        run_root = run_dir.parent

    # discover_kernels accepts any dir and rglob's kernel_*.cu. Feed it the
    # kernels/ subdir (or success/) so paths we turn into relative outputs
    # have a stable root.
    if args.correct_only:
        kernels_root = run_root / "success"
        if not kernels_root.is_dir():
            sys.exit(
                f"--correct-only: cannot locate sibling success/ at {kernels_root}"
            )
        default_out_name = "pattern_trees_correct"
    else:
        kernels_root = (run_dir / "kernels") if (run_dir / "kernels").is_dir() else run_dir
        default_out_name = "pattern_trees"

    out_dir: Path = (args.output_dir or (run_root / default_out_name)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels = discover_kernels(kernels_root)
    if not kernels:
        sys.exit(f"no kernel_*.cu under {kernels_root}")

    mode_tag = " (correct-only)" if args.correct_only else ""
    print(f"[draw_pattern_trees_batch]{mode_tag} {len(kernels)} kernels → {out_dir}")

    tasks: list[tuple] = []
    skipped: list[Path] = []
    source_only: list[tuple[Path, Path]] = []
    for k in kernels:
        op = _output_path(k, kernels_root, out_dir, args.format)
        op.parent.mkdir(parents=True, exist_ok=True)
        state = "render" if args.force else _artifact_state(k, op)
        if state == "fresh":
            skipped.append(k)
            continue
        if state == "source-only":
            source_only.append((k, op))
            continue
        tasks.append((k, op, not args.no_collapse_seq, args.format))

    # Fast path: for kernels whose PNG is current but .cu is missing, just
    # emit the source companion in-process (no graphviz round-trip).
    for k, op in source_only:
        try:
            _emit_source_companion(k, op)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL (source-only) {k}: {type(e).__name__}: {e}", file=sys.stderr)

    t0 = time.time()
    if args.jobs > 1 and len(tasks) > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(_render_worker, tasks)
    else:
        results = [_render_worker(t) for t in tasks]
    dt = time.time() - t0

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "fail"]

    for r in failed:
        print(f"  FAIL {r['kernel']}: {r['error']}", file=sys.stderr)

    # index.csv: one row per kernel (including skipped — so the index is
    # complete). Re-read previously-rendered outputs' stats by including
    # only fields we have; skipped rows leave stats blank unless we already
    # rendered in this pass.
    index_rows: list[dict] = []
    for r in ok:
        kp = Path(r["kernel"])
        rel = kp.relative_to(kernels_root)
        index_rows.append({
            "kernel_id": f"{rel.parent.name}/{rel.name}",
            "path": str(kp),
            "image": r["image"],
            "source": r.get("source") or str(Path(r["image"]).with_suffix(".cu")),
            "node_count": r["node_count"],
            "depth": r["depth"],
            "branches": r["branches"],
            "loops": r["loops"],
        })
    for k, op in source_only:
        rel = k.relative_to(kernels_root)
        index_rows.append({
            "kernel_id": f"{rel.parent.name}/{rel.name}",
            "path": str(k),
            "image": str(op),
            "source": str(op.with_suffix(".cu")),
            "node_count": "",
            "depth": "",
            "branches": "",
            "loops": "",
        })
    for k in skipped:
        rel = k.relative_to(kernels_root)
        op = _output_path(k, kernels_root, out_dir, args.format)
        index_rows.append({
            "kernel_id": f"{rel.parent.name}/{rel.name}",
            "path": str(k),
            "image": str(op),
            "source": str(op.with_suffix(".cu")),
            "node_count": "",
            "depth": "",
            "branches": "",
            "loops": "",
        })

    index_rows.sort(key=lambda r: r["kernel_id"])
    index_path = out_dir / "index.csv"
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)

    print(
        f"[draw_pattern_trees_batch] rendered={len(ok)} "
        f"source-only={len(source_only)} skipped={len(skipped)} "
        f"failed={len(failed)} in {dt:.1f}s → {index_path}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
