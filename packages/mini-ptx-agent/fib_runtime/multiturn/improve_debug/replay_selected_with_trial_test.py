#!/usr/bin/env python3
"""Replay selected kernels through trial_2026_0427_mha_bwd_d128/test.py.

For each selected row this copies the row's kernel to the trial directory as
kernel.cu, runs test.py, then preserves traces.json and process logs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SELECTED = SCRIPT_DIR / "selected_error_kernels.json"
DEFAULT_TRIAL_DIR = SCRIPT_DIR / "trial_2026_0427_mha_bwd_d128"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "trial_test_replay_outputs"


def safe_name(row: dict, index: int) -> str:
    category = str(row.get("category", "unknown")).lower().replace(" ", "_")
    exp = str(row.get("exp", f"row_{index:03d}"))
    turn = row.get("turn", "unknown")
    return f"{index:03d}_{category}_{exp}_t{turn}"


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    rows = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        if "kernel" not in row:
            raise ValueError(f"row {index} is missing kernel")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected_json", type=Path, nargs="?", default=DEFAULT_SELECTED)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base-url", default="http://localhost:10000")
    parser.add_argument("--category", action="append", help="Only run rows with this exact category")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    selected_json = args.selected_json.expanduser().resolve()
    trial_dir = args.trial_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    test_py = trial_dir / "test.py"
    traces_path = trial_dir / "traces.json"
    kernel_dst = trial_dir / "kernel.cu"

    if not selected_json.exists():
        print(f"selected JSON does not exist: {selected_json}", file=sys.stderr)
        return 2
    if not test_py.exists():
        print(f"trial test.py does not exist: {test_py}", file=sys.stderr)
        return 2

    rows = load_rows(selected_json)
    categories = set(args.category or [])
    indexed_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if index >= args.start_index
        and (not categories or str(row.get("category")) in categories)
        and str(row.get("category")) != "Correct"
    ]
    if args.limit is not None:
        indexed_rows = indexed_rows[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"
    failures = 0

    with summary_path.open("a") as summary:
        for index, row in indexed_rows:
            stem = safe_name(row, index)
            trace_out = output_dir / f"{stem}.traces.json"
            meta_out = output_dir / f"{stem}.json"
            stdout_out = output_dir / f"{stem}.stdout"
            stderr_out = output_dir / f"{stem}.stderr"
            if args.resume and trace_out.exists():
                print(f"[{index}] skip existing {trace_out}")
                continue

            kernel_src = Path(str(row["kernel"])).expanduser().resolve()
            if not kernel_src.exists():
                failures += 1
                record = {
                    "index": index,
                    "status": "missing_kernel",
                    "kernel": str(kernel_src),
                    "category": row.get("category"),
                    "exp": row.get("exp"),
                    "turn": row.get("turn"),
                }
                meta_out.write_text(json.dumps(record, indent=2) + "\n")
                summary.write(json.dumps(record) + "\n")
                summary.flush()
                print(f"[{index}] missing kernel: {kernel_src}", file=sys.stderr)
                continue

            try:
                traces_path.unlink()
            except FileNotFoundError:
                pass
            shutil.copy2(kernel_src, kernel_dst)
            env = os.environ.copy()
            env["PROFILE_BASE_URL"] = args.base_url

            print(f"[{index}] {row.get('category')} {kernel_src}", flush=True)
            started = time.time()
            proc = subprocess.run(
                [args.python, str(test_py)],
                cwd=trial_dir,
                env=env,
                text=True,
                capture_output=True,
            )
            elapsed = time.time() - started
            stdout_out.write_text(proc.stdout or "")
            stderr_out.write_text(proc.stderr or "")

            trace_saved = False
            if traces_path.exists():
                shutil.copy2(traces_path, trace_out)
                trace_saved = True

            status = "ok" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                failures += 1
            record = {
                "index": index,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "kernel": str(kernel_src),
                "category": row.get("category"),
                "exp": row.get("exp"),
                "turn": row.get("turn"),
                "original_message": row.get("message"),
                "trace_path": str(trace_out) if trace_saved else None,
                "stdout_path": str(stdout_out),
                "stderr_path": str(stderr_out),
            }
            meta_out.write_text(json.dumps(record, indent=2) + "\n")
            summary.write(json.dumps(record) + "\n")
            summary.flush()
            print(f"  -> {status} in {elapsed:.1f}s traces={trace_saved}", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
