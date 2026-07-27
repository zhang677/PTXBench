"""Verify the helion solutions match the F.sdpa reference using the
flashinfer-bench profiling service at $PROFILE_BASE_URL (default localhost:10000).

For each Definition:
  - Build the helion Solution in-process via make_attention_problem.build_solution
  - GET /definitions/{name}/workloads to fetch all workloads from the service
  - POST /evaluate with the Solution + workload UUIDs
  - Poll /tasks/{id} until completed
  - Assert every trace's evaluation.status == "PASSED"
"""

import os
import sys
import time
from typing import Dict, Any, List

import requests


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_attention_problem import build_solution  # noqa: E402

BASE = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")


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


def verify_pair(defn_name: str) -> bool:
    causal = defn_name.endswith("_causal")
    sol = build_solution(causal)
    workloads = fetch_workloads(defn_name)
    uuids = [w["uuid"] for w in workloads]
    s_vals = ", ".join(str(w["axes"]["S"]) for w in workloads)
    print(f"\n=== {sol['name']} on {defn_name} ===", flush=True)
    print(f"  {len(uuids)} workloads: S = {s_vals}", flush=True)

    data = submit_and_poll({"solution": sol, "workload_uuids": uuids}, deadline_s=600)
    if data["status"] == "failed":
        print(f"  TASK FAILED: {data.get('error', 'unknown')}", flush=True)
        return False

    traces = data.get("traces") or []
    all_pass = True
    for t in traces:
        ev = (t.get("evaluation") or {})
        st = ev.get("status", "MISSING")
        wl_axes = (t.get("workload") or {}).get("axes", {})
        s_val = wl_axes.get("S", "?")
        # correctness + perf
        corr = ev.get("correctness") or {}
        perf = ev.get("performance") or {}
        marker = "PASS" if st == "PASSED" else f"FAIL[{st}]"
        speedup = perf.get("speedup_factor")
        latency = perf.get("latency_ms")
        ref_latency = perf.get("reference_latency_ms")
        line = f"  [S={s_val:>4}] {marker}"
        if corr:
            line += (
                f" | max_abs={corr.get('max_absolute_error', '?')}"
                f" max_rel={corr.get('max_relative_error', '?')}"
            )
        if perf:
            line += (
                f" | latency={latency}ms ref={ref_latency}ms"
                f" speedup={speedup}x"
            )
        if st != "PASSED":
            log = ev.get("log") or ""
            if log:
                line += f"\n      log[:400]: {log[:400]}"
            all_pass = False
        print(line, flush=True)
    return all_pass


def main():
    defn_names = ["mha_h48_d128", "mha_h48_d128_causal"]
    print(f"profiling service: {BASE}")
    ok = True
    for defn_name in defn_names:
        ok &= verify_pair(defn_name)
    if not ok:
        print("\nSERVICE VERIFICATION FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nSERVICE VERIFICATION PASSED for both solutions on all 6 workloads each.")


if __name__ == "__main__":
    main()
