#!/usr/bin/env python3
"""Merge a base eval run directory with a rerun directory into a combined view.

The rerun launcher (`rerun_failed_experiments.py`) resumes container-killed
experiments: `run_v2.py` is invoked with `--resume-trajectory`, producing a
trajectory file that already contains the original prefix, and the rerun's
success dir is pre-seeded with the base run's `kernel_v*.cu` / `record.json`
files before new versions are appended. So for any exp present in the rerun,
the rerun artifacts are the complete supersets; for exps not rerun, only the
base has data.

This script copies each exp's artifacts into `<output>/{trajectories,success}`,
preferring rerun when present, falling back to base otherwise.

Example:
    python merge_rerun_results.py \\
        --base   /home/ubuntu/AccRL-exps/eval_runs/2026-0422-2352 \\
        --rerun  /home/ubuntu/AccRL-exps/eval_runs/2026-0422-2352_rerun \\
        --output /home/ubuntu/AccRL-exps/eval_runs/2026-0422-2352_complete
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a base eval run with its rerun directory into an output dir.",
    )
    parser.add_argument("--base", required=True,
                        help="Path to the original eval run directory.")
    parser.add_argument("--rerun", required=True,
                        help="Path to the rerun directory produced by rerun_failed_experiments.py.")
    parser.add_argument("--output", required=True,
                        help="Path for the merged output directory (must not exist).")
    return parser.parse_args()


def merge_files(base_dir: Path, rerun_dir: Path, out_dir: Path) -> tuple[int, int]:
    """Copy one file per exp_id into out_dir; prefer rerun_dir when present.

    Returns (num_from_rerun, num_from_base).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_files = {p.name: p for p in base_dir.iterdir() if p.is_file()} if base_dir.is_dir() else {}
    rerun_files = {p.name: p for p in rerun_dir.iterdir() if p.is_file()} if rerun_dir.is_dir() else {}
    from_rerun = 0
    from_base = 0
    for name in sorted(set(base_files) | set(rerun_files)):
        src = rerun_files.get(name) or base_files[name]
        shutil.copy2(src, out_dir / name)
        if name in rerun_files:
            from_rerun += 1
        else:
            from_base += 1
    return from_rerun, from_base


def has_success_kernel(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() for p in path.glob("kernel_v*.cu"))


def merge_subdirs(base_dir: Path, rerun_dir: Path, out_dir: Path) -> tuple[int, int]:
    """Copy one subdir per exp_id into out_dir; prefer rerun_dir when present.

    Returns (num_from_rerun, num_from_base).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_subs = (
        {p.name: p for p in base_dir.iterdir() if has_success_kernel(p)}
        if base_dir.is_dir() else {}
    )
    rerun_subs = (
        {p.name: p for p in rerun_dir.iterdir() if has_success_kernel(p)}
        if rerun_dir.is_dir() else {}
    )
    from_rerun = 0
    from_base = 0
    for name in sorted(set(base_subs) | set(rerun_subs)):
        src = rerun_subs.get(name) or base_subs[name]
        shutil.copytree(src, out_dir / name)
        if name in rerun_subs:
            from_rerun += 1
        else:
            from_base += 1
    return from_rerun, from_base


def main() -> None:
    args = parse_args()

    base = Path(args.base).resolve()
    rerun = Path(args.rerun).resolve()
    output = Path(args.output)

    if not base.is_dir():
        print(f"Error: --base {base} is not a directory.")
        sys.exit(1)
    if not rerun.is_dir():
        print(f"Error: --rerun {rerun} is not a directory.")
        sys.exit(1)
    if output.exists():
        print(f"Error: --output {output} already exists; refusing to overwrite.")
        sys.exit(1)

    output = output.resolve()
    output.mkdir(parents=True)

    print(f"Base:   {base}")
    print(f"Rerun:  {rerun}")
    print(f"Output: {output}")
    print()

    traj_rerun, traj_base = merge_files(
        base / "trajectories", rerun / "trajectories", output / "trajectories",
    )
    print(f"trajectories: {traj_rerun + traj_base} files "
          f"({traj_rerun} from rerun, {traj_base} from base)")

    succ_rerun, succ_base = merge_subdirs(
        base / "success", rerun / "success", output / "success",
    )
    print(f"success:      {succ_rerun + succ_base} dirs  "
          f"({succ_rerun} from rerun, {succ_base} from base)")


if __name__ == "__main__":
    main()
