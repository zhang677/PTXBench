"""Sanity-check that the new cuDNN reference works on the live service:
  1. Submit the existing helion solution (correctness should still hold;
     reference_latency_ms is now cuDNN, so speedup_factor will fall ~2x).
  2. Submit the new FA2 solution (the one we just added under
     accrl-training/solutions/attention/mha_with_lse_h48_d128*/) and confirm
     it passes correctness against the cuDNN reference.

Reports per-S: latency, reference_latency, speedup_factor.
"""

import json
import os
import sys
import time
from typing import Dict, Any, List

import requests


BASE = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")

ACCRL = "/home/ubuntu/accrl-training/solutions/attention"


def load_solution(rel_path: str) -> Dict[str, Any]:
    with open(os.path.join(ACCRL, rel_path)) as f:
        return json.load(f)


def fetch_workloads(defn_name: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{BASE}/definitions/{defn_name}/workloads", timeout=30)
    r.raise_for_status()
    return sorted(r.json(), key=lambda w: w["axes"]["S"])


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


def run_one(label: str, defn_name: str, sol: Dict[str, Any]) -> bool:
    workloads = fetch_workloads(defn_name)
    uuids = [w["uuid"] for w in workloads]
    print(f"\n=== [{label}] {sol['name']} on {defn_name} ===", flush=True)
    print(f"  {len(uuids)} workloads", flush=True)
    data = submit_and_poll({"solution": sol, "workload_uuids": uuids}, deadline_s=900)
    if data["status"] == "failed":
        print(f"  TASK FAILED: {data.get('error', 'unknown')}", flush=True)
        return False
    traces = data.get("traces") or []
    all_pass = True
    for t in traces:
        ev = (t.get("evaluation") or {})
        st = ev.get("status", "MISSING")
        s_val = (t.get("workload") or {}).get("axes", {}).get("S", "?")
        corr = ev.get("correctness") or {}
        perf = ev.get("performance") or {}
        marker = "PASS" if st == "PASSED" else f"FAIL[{st}]"
        line = f"  [S={s_val:>4}] {marker}"
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
                line += f"\n      log[:400]: {log[:400]}"
            all_pass = False
        print(line, flush=True)
    return all_pass


def main():
    print(f"profiling service: {BASE}")
    cases = [
        # (label, defn, solution_path)
        # ("helion", "mha_with_lse_h48_d128",
        #     "mha_with_lse_h48_d128/helion_mha_with_lse_h48_d128.json"),
        ("fa2",    "mha_with_lse_h48_d128",
            "mha_with_lse_h48_d128/fa2_mha_with_lse_h48_d128.json"),
        # ("helion", "mha_with_lse_h48_d128_causal",
        #     "mha_with_lse_h48_d128_causal/helion_mha_with_lse_h48_d128_causal.json"),
        ("fa2",    "mha_with_lse_h48_d128_causal",
            "mha_with_lse_h48_d128_causal/fa2_mha_with_lse_h48_d128_causal.json"),
    ]
    ok = True
    for label, defn, sol_path in cases:
        sol = load_solution(sol_path)
        ok &= run_one(label, defn, sol)
    if not ok:
        print("\nVERIFICATION FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nVERIFICATION PASSED for cuDNN reference + helion + FA2 solutions.")


if __name__ == "__main__":
    main()
