"""Verify the reference compiles and runs as a Solution via
the flashinfer-bench profiling service at $PROFILE_BASE_URL (default
localhost:10000).

The Solution source code IS the Definition's reference code,
so this is a ref-vs-ref smoke test: correctness should PASS and
speedup_factor should be ~1x.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List

import requests

# Add "/home/ubuntu/AccRL" to sys.path to import from accrl.utils
sys.path.append("/home/ubuntu/AccRL")
from accrl.utils.solution_utils import build_solution


DEFINITION_NAMES = [
    "fp8_gemm_nt_1d2d_n24576_k1536", "fp8_gemm_nt_1d2d_n32768_k512", "fp8_gemm_nt_1d2d_n7168_k16384", "fp8_gemm_nt_1d2d_n4096_k7168", "fp8_gemm_nt_1d2d_n7168_k2048",
    "fp8_gemm_nt_1d1d_n24576_k1536", "fp8_gemm_nt_1d1d_n32768_k512", "fp8_gemm_nt_1d1d_n7168_k16384", "fp8_gemm_nt_1d1d_n4096_k7168", "fp8_gemm_nt_1d1d_n7168_k2048"
]
m_LIST = [4096]
BASE = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")


def get_flops(m_val: int, n_val: int, k_val: int) -> int:
    return 2 * m_val * n_val * k_val


def fetch_definition(name: str) -> Dict[str, Any]:
    r = requests.get(f"{BASE}/definitions/{name}", timeout=30)
    r.raise_for_status()
    return r.json()


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
    parser.add_argument("--deadline-s", type=int, default=600)
    parser.add_argument("--perf-csv", default="perf.csv")
    args = parser.parse_args()

    print(f"profiling service: {BASE}")
    print(f"perf csv: {args.perf_csv}")
    with open(args.perf_csv, "w", newline="") as perf_file:
        perf_writer = csv.DictWriter(
            perf_file,
            fieldnames=[
                "definition_name",
                "m",
                "n",
                "k",
                "latency_ms",
                "tflops",
                "workload_uuid",
            ],
        )
        perf_writer.writeheader()
        perf_file.flush()

        for def_name in DEFINITION_NAMES:
            for target_m in m_LIST:
                print(f"definition: {def_name}, target M={target_m}")

                defn = fetch_definition(def_name)
                ref_code = defn.get("reference")
                if not ref_code:
                    print(f"definition {def_name!r} has no reference source", file=sys.stderr)
                    return 1

                sol = build_solution(code=ref_code, definition_name=def_name, language="python")
                sol_dict = sol.model_dump(mode="json")
                print(f"built solution: {sol_dict['name']}")

                workloads = fetch_workloads(def_name)
                matched = [w for w in workloads if int(w["axes"].get("m", -1)) == target_m]
                if len(matched) != 1:
                    print(
                        f"expected exactly one workload with m={target_m}, got {len(matched)} "
                        f"(out of {len(workloads)} total)",
                        file=sys.stderr,
                    )
                    return 1
                workload = matched[0]
                uuids = [workload["uuid"]]
                print(f"selected workload: uuid={workload['uuid']} axes={workload['axes']}")

                payload = {
                    "solution": sol_dict,
                    "workload_uuids": uuids,
                    "run_baseline": False,
                    "profile_baseline": False,
                }
                data = submit_and_poll(payload, args.deadline_s)
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
                    wl = t.get("workload") or {}
                    wl_axes = wl.get("axes", {})
                    workload_uuid = wl.get("uuid") or workload["uuid"]
                    m_val = wl_axes.get("m", "?")
                    corr = ev.get("correctness") or {}
                    perf = ev.get("performance") or {}
                    marker = "PASS" if st == "PASSED" else f"FAIL[{st}]"
                    line = f"  [M={m_val}] {marker}"
                    if corr:
                        line += (
                            f" | max_abs={corr.get('max_absolute_error', '?')}"
                            f" max_rel={corr.get('max_relative_error', '?')}"
                        )
                    if perf:
                        latency_ms = perf.get("latency_ms")
                        tflops = (
                            get_flops(
                                int(wl_axes.get("m", 1)),
                                int(wl_axes.get("n", 1)),
                                int(wl_axes.get("k", 1)),
                            )
                            * 1e-9
                            / latency_ms
                            if latency_ms
                            else "?"
                        )
                        perf_writer.writerow(
                            {
                                "definition_name": def_name,
                                "workload_uuid": workload_uuid,
                                "m": m_val,
                                "n": wl_axes.get("n", "?"),
                                "k": wl_axes.get("k", "?"),
                                "latency_ms": latency_ms,
                                "tflops": tflops,
                            }
                        )
                        perf_file.flush()
                        line += (
                            f" | latency={latency_ms}ms"
                            f" ref={perf.get('reference_latency_ms')}ms"
                            f" tflops={tflops}"
                            f" speedup={perf.get('speedup_factor')}x"
                        )
                    if st != "PASSED":
                        log = ev.get("log") or ""
                        if log:
                            line += f"\n      log: {log}"
                        all_pass = False
                    print(line, flush=True)

                if not all_pass:
                    print("\nSERVICE VERIFICATION FAILED", file=sys.stderr)
                    return 1
    print("\nSERVICE VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
