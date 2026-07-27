#!/usr/bin/env python3
"""Evaluate every bf16 host-wrapped kernel against the full workload sweep.

Emits a CSV at the project root whose schema matches `results_fp16_h100.csv`:
    kernel,M,N,K,status,latency_ms,reference_latency_ms,speedup

N and K are left empty because they are constants in the definition
`gemm_n7168_k5120` (N=7168, K=5120).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "examples_with_host_sm90_bf16"
OUT_CSV = ROOT / "results_bf16_h100.csv"
DEFINITION_NAME = "gemm_n7168_k5120"
BASE_URL = "http://localhost:10000"
EVALUATE_TIMEOUT_S = 600


def build_solution(name: str, kernel_source: str) -> dict[str, Any]:
    return {
        "name": name,
        "definition": DEFINITION_NAME,
        "spec": {
            "language": "cuda",
            "binding": "tvm-ffi",
            "target_hardware": ["H100"],
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "sweep",
        "sources": [{"path": "kernel.cu", "content": kernel_source}],
    }


def submit_and_poll(endpoint: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/{endpoint}", json=payload)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = requests.get(f"{BASE_URL}/tasks/{task_id}?timeout=30")
        result.raise_for_status()
        data = result.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(5)
    raise TimeoutError(f"{endpoint} task {task_id} timed out")


def load_workloads() -> list[dict[str, Any]]:
    resp = requests.get(f"{BASE_URL}/definitions/{DEFINITION_NAME}/workloads")
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    workloads = load_workloads()
    uuid_to_m = {w["uuid"]: w["axes"]["M"] for w in workloads}
    workload_uuids = [w["uuid"] for w in workloads]
    print(f"{len(workload_uuids)} workloads for {DEFINITION_NAME}", file=sys.stderr)

    kernel_files = sorted(
        KERNEL_DIR.glob("gemm_v*.cu"),
        key=lambda p: int(p.stem.removeprefix("gemm_v")),
    )
    if not kernel_files:
        print(f"no kernels found in {KERNEL_DIR}", file=sys.stderr)
        return 1

    rows: list[tuple[str, int, str, str, str, str, str, str]] = []
    for kernel_path in kernel_files:
        name = kernel_path.stem  # gemm_vN
        print(f"evaluating {name}...", file=sys.stderr)
        solution = build_solution(name, kernel_path.read_text())
        payload = {"solution": solution, "workload_uuids": workload_uuids}
        try:
            data = submit_and_poll("evaluate", payload, EVALUATE_TIMEOUT_S)
        except Exception as exc:
            print(f"  {name} failed to submit/poll: {exc}", file=sys.stderr)
            continue
        if data.get("status") == "failed":
            print(f"  {name} task failed: {data.get('error')}", file=sys.stderr)
            continue
        traces = data.get("traces") or []
        for trace in traces:
            uuid = trace.get("workload", {}).get("uuid")
            M = uuid_to_m.get(uuid, trace.get("workload", {}).get("axes", {}).get("M", ""))
            ev = trace.get("evaluation") or {}
            status = ev.get("status", "?")
            perf = ev.get("performance") or {}
            latency = perf.get("latency_ms", "")
            ref = perf.get("reference_latency_ms", "")
            speedup = perf.get("speedup_factor", "")
            rows.append((name, M, "", "", status, str(latency), str(ref), str(speedup)))

    rows.sort(key=lambda r: (r[0], r[1] if isinstance(r[1], int) else 0))

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["kernel", "M", "N", "K", "status", "latency_ms", "reference_latency_ms", "speedup"]
        )
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
