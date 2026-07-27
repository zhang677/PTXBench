#!/usr/bin/env python3
"""Benchmark SM90 FP16 GEMM kernels via the profiling server and write results to CSV.

Submits each kernel .cu file to the evaluation server, collects latency and
speedup across workloads, and writes a CSV with one row per (kernel, M) pair.

Usage:
    # Benchmark all kernels in a directory against the default definition
    python bench_kernels.py -k ../examples_sm90_fp16/

    # Benchmark a single kernel
    python bench_kernels.py -k ../examples_sm90_fp16/gemm_v1.cu

    # Custom server / definition / output
    python bench_kernels.py -k ../examples_sm90_fp16/ \\
        --base-url http://localhost:10000 \\
        --definition gemm_n6144_k4096 \\
        -o results.csv
"""
import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import requests

DEFAULT_BASE_URL = "http://localhost:10000"
DEFAULT_DEFINITION = "gemm_n6144_k4096"
DEFAULT_TIMEOUT = 300  # seconds


def build_solution(kernel_source: str, definition: str, name: str) -> dict:
    return {
        "name": name,
        "definition": definition,
        "spec": {
            "language": "cuda",
            "binding": "tvm-ffi",
            "target_hardware": ["H100"],
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "bench",
        "sources": [
            {"path": "kernel.cu", "content": kernel_source},
        ],
    }


def fetch_workloads(base_url: str, definition: str) -> list[dict]:
    resp = requests.get(f"{base_url}/definitions/{definition}/workloads")
    resp.raise_for_status()
    return resp.json()


def submit_and_poll(base_url: str, solution: dict,
                    workload_uuids: list[str],
                    timeout: int) -> dict | None:
    resp = requests.post(
        f"{base_url}/evaluate",
        json={"solution": solution, "workload_uuids": workload_uuids},
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"    task_id={task_id[:16]}...", end="", file=sys.stderr)

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{base_url}/tasks/{task_id}?timeout=30")
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("completed", "failed", "error"):
            return data
        time.sleep(3)
    return None


def parse_traces(traces: list[dict]) -> list[dict]:
    """Extract per-workload rows from evaluation traces."""
    rows = []
    for t in traces:
        ev = t.get("evaluation", {})
        wl = t.get("workload", {})
        axes = wl.get("axes", {}) if isinstance(wl, dict) else {}
        status = ev.get("status", "UNKNOWN")

        row = {
            "M": axes.get("M", ""),
            "N": axes.get("N", ""),
            "K": axes.get("K", ""),
            "status": status,
            "latency_ms": "",
            "reference_latency_ms": "",
            "speedup": "",
        }

        if status == "PASSED":
            perf = ev.get("performance", {})
            row["latency_ms"] = perf.get("latency_ms", "")
            row["reference_latency_ms"] = perf.get("reference_latency_ms", "")
            row["speedup"] = perf.get("speedup_factor", "")

        rows.append(row)
    return rows


def collect_kernel_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("gemm_v*.cu"))
    # glob pattern
    return sorted(Path(f) for f in glob.glob(path))


def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM kernels via profiling server")
    parser.add_argument("-k", "--kernels", required=True,
                        help="Path to a .cu file or directory of .cu files")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Profiling server URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--definition", default=DEFAULT_DEFINITION,
                        help=f"Task definition name (default: {DEFAULT_DEFINITION})")
    parser.add_argument("-o", "--output", default="bench_results.csv",
                        help="Output CSV path (default: bench_results.csv)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Per-kernel poll timeout in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    kernel_files = collect_kernel_files(args.kernels)
    if not kernel_files:
        print(f"No kernel files found at {args.kernels}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(kernel_files)} kernel(s)", file=sys.stderr)

    # Fetch available workloads
    print(f"Fetching workloads for '{args.definition}'...", file=sys.stderr)
    workloads = fetch_workloads(args.base_url, args.definition)
    uuids = [w["uuid"] for w in workloads]
    m_vals = sorted(set(w.get("axes", {}).get("M", "?") for w in workloads))
    print(f"  {len(workloads)} workload(s), M values: {m_vals}", file=sys.stderr)

    all_rows: list[dict] = []

    for kf in kernel_files:
        kname = kf.stem
        print(f"\n  {kname}:", end="", file=sys.stderr)
        source = kf.read_text()
        solution = build_solution(source, args.definition, kname)

        data = submit_and_poll(args.base_url, solution, uuids, args.timeout)
        if data is None:
            print(" TIMEOUT", file=sys.stderr)
            all_rows.append({"kernel": kname, "status": "TIMEOUT"})
            continue

        if data["status"] in ("error", "failed"):
            err = data.get("error", "unknown")
            print(f" {data['status']}: {err}", file=sys.stderr)
            all_rows.append({"kernel": kname, "status": data["status"]})
            continue

        traces = data.get("traces", [])
        rows = parse_traces(traces)
        passed = sum(1 for r in rows if r["status"] == "PASSED")
        print(f" {passed}/{len(rows)} PASSED", file=sys.stderr)

        for r in rows:
            r["kernel"] = kname
            all_rows.append(r)

    # Write CSV
    fieldnames = ["kernel", "M", "N", "K", "status", "latency_ms",
                   "reference_latency_ms", "speedup"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n\nResults written to {args.output}", file=sys.stderr)

    # Print summary table
    print(f"\n{'kernel':<16} {'M':>6} {'latency_ms':>12} {'ref_ms':>12} {'speedup':>10} {'status'}")
    print("-" * 72)
    for r in all_rows:
        lat = f"{float(r['latency_ms']):.4f}" if r.get("latency_ms") else ""
        ref = f"{float(r['reference_latency_ms']):.4f}" if r.get("reference_latency_ms") else ""
        spd = f"{float(r['speedup']):.3f}x" if r.get("speedup") else ""
        print(f"{r.get('kernel',''):<16} {str(r.get('M','')):>6} {lat:>12} {ref:>12} {spd:>10} {r.get('status','')}")


if __name__ == "__main__":
    main()
