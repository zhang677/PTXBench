"""Verify the mha_bwd_h48_d128 reference compiles and runs as a Solution via
the flashinfer-bench profiling service at $PROFILE_BASE_URL (default
localhost:10000).

The Solution source code IS the Definition's reference code (cuDNN attention),
so this is a cuDNN-vs-cuDNN smoke test: correctness should PASS and
speedup_factor should be ~1x. By default all standard sequence lengths are
exercised; pass --seq-lens to target a subset.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List
import csv

import requests
# Add "/home/ubuntu/AccRL" to sys.path to import from accrl.utils
sys.path.append("/home/ubuntu/AccRL")
from accrl.utils.solution_utils import build_solution

DEFINITION_NAMES = [
    'mha_with_lse_d64', 'mha_with_lse_d96', 'mha_with_lse_d128', 'mha_with_lse_d256', 'mha_with_lse_d64_causal', 'mha_with_lse_d96_causal', 'mha_with_lse_d128_causal', 'mha_with_lse_d256_causal',
]
seq_var = "S"
d_var = "D"
S_LIST = [512, 1024, 2048, 4096, 8192, 16384]
BASE = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")


def parse_int_list(value: str) -> List[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_str_list(value: str) -> List[str]:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one definition name")
    return parsed

def get_flops(B: int, H: int, D: int, S: int, is_fwd: bool, is_causal: bool) -> int:
    if is_fwd and is_causal:
        return 2 * S * S * B * H * D
    elif is_fwd and not is_causal:
        return 4 * S * S * B * H * D
    elif not is_fwd and is_causal:
        return 5 * S * S * B * H * D
    else:
        return 10 * S * S * B * H * D


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
    parser.add_argument("--seq-lens", type=parse_int_list, default=S_LIST,
                        help="comma-separated sequence lengths to verify")
    parser.add_argument("--definitions", type=parse_str_list, default=DEFINITION_NAMES,
                        help="comma-separated definition names to verify")
    args = parser.parse_args()

    print(f"profiling service: {BASE}")
    print(f"perf csv: {args.perf_csv}")
    with open(args.perf_csv, "w", newline="") as perf_file:
        perf_writer = csv.DictWriter(
            perf_file,
            fieldnames=["definition_name", "seq_len", "latency_ms", "tflops", "workload_uuid",],
        )
        perf_writer.writeheader()
        perf_file.flush()

        for def_name in args.definitions:
            is_fwd = False if "bwd" in def_name else True
            is_causal = True if "causal" in def_name else False
            for target_s in args.seq_lens:
                print(f"definition: {def_name}, target {seq_var}={target_s}")

                defn = fetch_definition(def_name)
                ref_code = defn.get("reference")
                if not ref_code:
                    print(f"definition {def_name!r} has no reference source", file=sys.stderr)
                    return 1

                sol = build_solution(code=ref_code, definition_name=def_name, language="python")
                sol_dict = sol.model_dump(mode="json")
                print(f"built solution: {sol_dict['name']}")

                workloads = fetch_workloads(def_name)
                matched = [w for w in workloads if int(w["axes"].get(seq_var, -1)) == target_s]
                if len(matched) != 1:
                    print(
                        f"expected exactly one workload with {seq_var}={target_s}, got {len(matched)} "
                        f"(out of {len(workloads)} total)",
                        file=sys.stderr,
                    )
                    return 1
                workload = matched[0]
                uuids = [workload["uuid"]]
                print(f"selected workload: uuid={workload['uuid']} axes={workload['axes']}")

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
                    wl = t.get("workload") or {}
                    wl_axes = wl.get("axes", {})
                    workload_uuid = wl.get("uuid") or workload["uuid"]
                    s_val = wl_axes.get(seq_var, "?")
                    corr = ev.get("correctness") or {}
                    perf = ev.get("performance") or {}
                    marker = "PASS" if st == "PASSED" else f"FAIL[{st}]"
                    line = f"  [{seq_var}={s_val}] {marker}"
                    if corr:
                        line += (
                            f" | max_abs={corr.get('max_absolute_error', '?')}"
                            f" max_rel={corr.get('max_relative_error', '?')}"
                        )
                    if perf:
                        ref_flops = get_flops(
                            B=int(wl_axes.get("B", 1)),
                            H=int(wl_axes.get("H", 1)),
                            D=int(wl_axes.get(d_var, 1)),
                            S=int(s_val),
                            is_fwd=is_fwd,
                            is_causal=is_causal,
                        )
                        ref_latency = perf.get("reference_latency_ms")
                        tflops = (
                            (ref_flops * 1e-9 / ref_latency)
                            if ref_latency
                            else "?"
                        )
                        perf_writer.writerow(
                            {
                                "definition_name": def_name,
                                "workload_uuid": workload_uuid,
                                "seq_len": s_val,
                                "latency_ms": perf.get("latency_ms"),
                                "tflops": tflops,
                            }
                        )
                        perf_file.flush()
                        line += (
                            f" | latency={perf.get('latency_ms')}ms"
                            f" ref={ref_latency}ms, tflops={tflops},"
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
