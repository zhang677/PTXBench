"""Profile the locally-built FA2 solutions for mha_h48_d128 and
mha_h48_d128_causal against the live profiling service. Reports per-S
correctness + latency + speedup against the F.sdpa reference.

Mirrors the pattern in 2026-0426-1410/verify_cudnn_reference.py but loads
solution JSONs from this directory (not from accrl-training).
"""

import json
import os
import sys
import time
from typing import Dict, Any, List

import requests


BASE = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_solution(filename: str) -> Dict[str, Any]:
    with open(os.path.join(HERE, filename)) as f:
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
        ("fa2", "mha_h48_d128",        "fa2_mha_h48_d128.json"),
        ("fa2", "mha_h48_d128_causal", "fa2_mha_h48_d128_causal.json"),
    ]
    ok = True
    for label, defn, sol_file in cases:
        sol = load_solution(sol_file)
        ok &= run_one(label, defn, sol)
    if not ok:
        print("\nVERIFICATION FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nVERIFICATION PASSED for FA2 solutions on mha_h48_d128 and mha_h48_d128_causal.")


if __name__ == "__main__":
    main()
