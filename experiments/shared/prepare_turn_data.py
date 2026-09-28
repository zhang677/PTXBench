#!/usr/bin/env python3
"""Export per-turn correctness and kernels for an experiment manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.shared.run_manifest import path, read_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--profile-url", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    exporter = ROOT / "experiments/shared/export_turn_correctness_arch.py"
    cmd = [sys.executable, str(exporter), "--experiments-csv", str(args.manifest), "--base-url", args.profile_url]
    if args.force:
        cmd.append("--force")
    subprocess.run(cmd, cwd=ROOT, check=True)
    extractor = ROOT / "experiments/shared/extract_turn_kernels.py"
    for row in rows:
        subprocess.run([sys.executable, str(extractor), "--run-dir", str(path(row["exp_dir"]))], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
