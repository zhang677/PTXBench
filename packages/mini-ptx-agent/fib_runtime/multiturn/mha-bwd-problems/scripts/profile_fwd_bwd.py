"""Submit a fwd+bwd attention solution (cuDNN by default, FA selectable) to the
flashinfer-bench profiling service (default localhost:10000) and report
PASS/FAIL + speedup vs the cuDNN-backward reference for every
mha_bwd_h48_d128 workload.

Usage:
    PYTHONPATH=/home/ubuntu/AccRL python scripts/profile_fwd_bwd.py
    PYTHONPATH=/home/ubuntu/AccRL python scripts/profile_fwd_bwd.py --s 2048
    PYTHONPATH=/home/ubuntu/AccRL python scripts/profile_fwd_bwd.py --solution fa
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from accrl.utils.solution_utils import build_solution


DEFINITION_NAME = "mha_bwd_h48_d128"
BASE = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")
SOLUTIONS_DIR = Path(__file__).resolve().parent.parent / "solutions"
SOLUTIONS = {
    "cudnn": SOLUTIONS_DIR / "cudnn_fwd_bwd_main.py",
    "fa": SOLUTIONS_DIR / "fa_fwd_bwd_main.py",
}


def fetch_workloads(name: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{BASE}/definitions/{name}/workloads", timeout=30)
    r.raise_for_status()
    return r.json()


def submit_and_poll(payload: Dict[str, Any], deadline_s: int) -> Dict[str, Any]:
    resp = requests.post(f"{BASE}/evaluate", json=payload, timeout=30)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"  task_id={task_id}", flush=True)

    deadline = time.time() + deadline_s
    while time.time() < deadline:
        r = requests.get(f"{BASE}/tasks/{task_id}?timeout=30", timeout=60)
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("completed", "failed"):
            return data
        print(f"  status={data['status']}, polling...", flush=True)
        time.sleep(2)
    raise TimeoutError(f"evaluate task {task_id} timed out after {deadline_s}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-s", type=int, default=900)
    parser.add_argument("--s", type=int, default=None,
                        help="Restrict to a single S value; default runs all six workloads.")
    parser.add_argument("--solution", choices=sorted(SOLUTIONS.keys()), default="cudnn",
                        help="Which fwd+bwd solution to submit (default: cudnn).")
    args = parser.parse_args()

    solution_file = SOLUTIONS[args.solution]
    if not solution_file.is_file() or solution_file.stat().st_size == 0:
        print(f"solution file is missing or empty: {solution_file}", file=sys.stderr)
        return 1

    print(f"profiling service: {BASE}")
    print(f"definition: {DEFINITION_NAME}")
    print(f"solution source: {solution_file}")

    code = solution_file.read_text()
    sol = build_solution(code=code, definition_name=DEFINITION_NAME, language="python")
    sol_dict = sol.model_dump(mode="json")
    print(f"built solution: {sol_dict['name']}")

    workloads = fetch_workloads(DEFINITION_NAME)
    if args.s is not None:
        workloads = [w for w in workloads if int(w["axes"].get("S", -1)) == args.s]
        if not workloads:
            print(f"no workload with S={args.s}", file=sys.stderr)
            return 1
    workloads.sort(key=lambda w: int(w["axes"]["S"]))
    uuids = [w["uuid"] for w in workloads]
    print(f"submitting {len(uuids)} workloads: S={[int(w['axes']['S']) for w in workloads]}")

    data = submit_and_poll({"solution": sol_dict, "workload_uuids": uuids}, args.deadline_s)
    if data["status"] == "failed":
        print(f"  TASK FAILED: {data.get('error', 'unknown')}", flush=True)
        return 1

    traces = data.get("traces") or []
    if not traces:
        print("  no traces returned", file=sys.stderr)
        return 1

    all_pass = True
    for t in traces:
        ev = t.get("evaluation") or {}
        st = ev.get("status", "MISSING")
        wl_axes = (t.get("workload") or {}).get("axes", {})
        s_val = wl_axes.get("S", "?")
        corr = ev.get("correctness") or {}
        perf = ev.get("performance") or {}
        marker = "PASS" if st == "PASSED" else f"FAIL[{st}]"
        line = f"  [S={s_val}] {marker}"
        if corr:
            line += (
                f" | max_abs={corr.get('max_absolute_error', '?')}"
                f" max_rel={corr.get('max_relative_error', '?')}"
            )
        if perf:
            line += (
                f" | latency={perf.get('latency_ms')}ms"
                f" ref={perf.get('reference_latency_ms')}ms"
                f" speedup={perf.get('speedup_factor')}x"
            )
        if st != "PASSED":
            log = ev.get("log") or ""
            if log:
                line += f"\n      log[:600]: {log[:600]}"
            all_pass = False
        print(line, flush=True)

    if not all_pass:
        print("\nSERVICE EVALUATION HAD FAILURES", file=sys.stderr)
        return 2  # 2 = had failures but task itself completed
    print("\nSERVICE EVALUATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
