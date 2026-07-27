"""Verify copied GDN baseline solutions through the flashinfer-bench profiling
service at $PROFILE_BASE_URL (default: http://localhost:10000).

The script reads ../gdn_problems.csv, loads one copied solution JSON per
definition from accrl-training, and submits each selected workload UUID to
/evaluate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROBLEMS_CSV = SCRIPT_DIR.parent / "gdn_problems.csv"
BASE = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")

SOLUTION_FILES = {
    "gdn_decode_qk4_v8_d128_k_last": "flashinfer_wrapper_9b7f1e.json",
    "gdn_decode_qk8_v16_d128_k_last": "flashinfer_wrapper_a5e9d2.json",
    "gdn_mtp_qk4_v8_d128_k_last": "flashinfer_wrapper_a3d7c2.json",
    "gdn_mtp_qk8_v16_d128_k_last": "flashinfer_wrapper_b5e9f1.json",
    "gdn_prefill_qk4_v8_d128_k_last": "flashinfer_wrapper_c3f8a1.json",
    "gdn_prefill_qk8_v16_d128_k_last": "flashinfer_wrapper_b7d4e2.json",
}


def default_accrl_training_root() -> Path:
    env_root = os.environ.get("ACCRL_TRAINING_ROOT")
    if env_root:
        return Path(env_root)
    for candidate in (Path("/home/ubuntu/accrl-training"), Path("/workspace/accrl-training")):
        if candidate.exists():
            return candidate
    return Path("/home/ubuntu/accrl-training")


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


def load_solution(solutions_root: Path, definition_name: str) -> Dict[str, Any]:
    file_name = SOLUTION_FILES.get(definition_name)
    if not file_name:
        raise KeyError(f"no solution configured for {definition_name}")
    path = solutions_root / definition_name / file_name
    with path.open() as f:
        solution = json.load(f)
    if solution.get("definition") != definition_name:
        raise ValueError(
            f"{path} has definition={solution.get('definition')!r}, expected {definition_name!r}"
        )
    return solution


def load_problem_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"definition_name", "workload_uuid", "seq_len"}
    if not rows:
        raise ValueError(f"{path} has no problem rows")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-s", type=int, default=900)
    parser.add_argument("--problems-csv", type=Path, default=DEFAULT_PROBLEMS_CSV)
    parser.add_argument(
        "--solutions-root",
        type=Path,
        default=default_accrl_training_root() / "solutions" / "gdn",
    )
    parser.add_argument("--perf-csv", type=Path, default=SCRIPT_DIR.parent / "gdn_verify_perf.csv")
    args = parser.parse_args()

    rows = load_problem_rows(args.problems_csv)
    print(f"profiling service: {BASE}")
    print(f"problems csv: {args.problems_csv}")
    print(f"solutions root: {args.solutions_root}")
    print(f"perf csv: {args.perf_csv}")

    workload_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    solution_cache: Dict[str, Dict[str, Any]] = {}

    args.perf_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.perf_csv.open("w", newline="") as perf_file:
        perf_writer = csv.DictWriter(
            perf_file,
            fieldnames=[
                "definition_name",
                "workload_uuid",
                "seq_len",
                "status",
                "latency_ms",
                "reference_latency_ms",
                "speedup_factor",
                "max_absolute_error",
                "max_relative_error",
            ],
        )
        perf_writer.writeheader()
        perf_file.flush()

        all_pass = True
        for row in rows:
            definition_name = row["definition_name"]
            workload_uuid = row["workload_uuid"]
            seq_len = row["seq_len"]
            print(f"definition: {definition_name}, workload={workload_uuid}, seq_len={seq_len}")

            if definition_name not in solution_cache:
                solution_cache[definition_name] = load_solution(args.solutions_root, definition_name)
                print(f"  loaded solution: {solution_cache[definition_name]['name']}")
            if definition_name not in workload_cache:
                workloads = fetch_workloads(definition_name)
                workload_cache[definition_name] = {w["uuid"]: w for w in workloads}

            workload = workload_cache[definition_name].get(workload_uuid)
            if workload is None:
                print(
                    f"  workload {workload_uuid} not found by service for {definition_name}",
                    file=sys.stderr,
                )
                return 1
            print(f"  axes={workload.get('axes', {})}")

            payload = {
                "solution": solution_cache[definition_name],
                "workload_uuids": [workload_uuid],
            }
            data = submit_and_poll(payload, args.deadline_s)
            if data["status"] == "failed":
                print(f"  TASK FAILED: {data.get('error', 'unknown')}", flush=True)
                return 1

            traces = data.get("traces") or []
            if not traces:
                print("  no traces returned", file=sys.stderr)
                return 1

            for trace in traces:
                ev = trace.get("evaluation") or {}
                status = ev.get("status", "MISSING")
                corr = ev.get("correctness") or {}
                perf = ev.get("performance") or {}
                marker = "PASS" if status == "PASSED" else f"FAIL[{status}]"
                line = f"  {marker}"
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
                print(line, flush=True)

                perf_writer.writerow(
                    {
                        "definition_name": definition_name,
                        "workload_uuid": workload_uuid,
                        "seq_len": seq_len,
                        "status": status,
                        "latency_ms": perf.get("latency_ms"),
                        "reference_latency_ms": perf.get("reference_latency_ms"),
                        "speedup_factor": perf.get("speedup_factor"),
                        "max_absolute_error": corr.get("max_absolute_error"),
                        "max_relative_error": corr.get("max_relative_error"),
                    }
                )
                perf_file.flush()

                if status != "PASSED":
                    log = ev.get("log") or ""
                    if log:
                        print(f"      log: {log}", flush=True)
                    all_pass = False

            if not all_pass:
                print("\nSERVICE VERIFICATION FAILED", file=sys.stderr)
                return 1

    print("\nSERVICE VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
