#!/usr/bin/env python3
"""Check all mounted definitions on each backend and run reference evaluation."""
import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"


def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    with urlopen(Request(url, data=data, headers=headers), timeout=timeout) as response:
        return json.load(response)


def check_metadata(base, expected_backends, startup_wait_s):
    deadline = time.monotonic() + startup_wait_s
    while True:
        health = request_json(f"{base}/health")
        backends = health.get("backends", [])
        if (health.get("status") == "ok" and len(backends) == expected_backends
                and all(backend.get("healthy") for backend in backends)):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"backends did not become healthy: {health}")
        time.sleep(2)
    if health.get("queue_size") != 0:
        raise RuntimeError(f"nonempty queue before smoke: {health}")

    cases = []
    for definition_path in sorted((DATASET_ROOT / "definitions").rglob("*.json")):
        definition = json.loads(definition_path.read_text())
        name = definition["name"]
        workload_path = next((DATASET_ROOT / "workloads").rglob(f"{name}.jsonl"))
        workloads = [json.loads(line)["workload"] for line in workload_path.read_text().splitlines() if line.strip()]
        cases.append((name, definition, workloads))

    if not cases:
        raise RuntimeError(f"no definitions in {DATASET_ROOT}")
    for backend in backends:
        url = backend["url"]
        for name, _, workloads in cases:
            remote = request_json(f"{url}/definitions/{name}")
            if remote["name"] != name:
                raise RuntimeError(f"{url} returned wrong definition for {name}")
            remote_workloads = request_json(f"{url}/definitions/{name}/workloads")
            expected = {w["uuid"] for w in workloads}
            if {w["uuid"] for w in remote_workloads} != expected:
                raise RuntimeError(f"{url} has wrong workloads for {name}")
    print(f"Metadata passed: {len(backends)} healthy backends, {len(cases)} definitions")
    return cases, backends


def evaluate_reference(base, case, deadline_s):
    name, definition, workloads = case
    solution = {
        "name": f"submission_smoke_{name}",
        "definition": name,
        "spec": {
            "language": "python",
            "target_hardware": ["Hopper"],
            "entry_point": "reference.py::run",
            "dependencies": [],
            "destination_passing_style": False,
        },
        "author": "submission-smoke",
        "sources": [{"path": "reference.py", "content": definition["reference"]}],
    }
    submission = request_json(
        f"{base}/evaluate",
        {"solution": solution, "workload_uuids": [w["uuid"] for w in workloads]},
    )
    task_id = submission["task_id"]
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        result = request_json(f"{base}/tasks/{task_id}?timeout=30", timeout=45)
        if result["status"] == "completed":
            traces = result.get("traces") or []
            statuses = [(trace.get("evaluation") or {}).get("status") for trace in traces]
            if len(statuses) != len(workloads) or any(status != "PASSED" for status in statuses):
                raise RuntimeError(f"{name} task {task_id} completed without passing: {statuses}")
            print(f"Evaluation passed: {name}, task {task_id}, endpoint {base}")
            return
        if result["status"] == "failed":
            raise RuntimeError(f"{name} task {task_id} failed: {result.get('error')}")
    raise TimeoutError(f"{name} task {task_id} did not finish within {deadline_s}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("PROFILE_BASE_URL", "http://127.0.0.1:10000"))
    parser.add_argument("--backends", type=int, default=4)
    parser.add_argument("--deadline-s", type=int, default=600)
    parser.add_argument("--startup-wait-s", type=int, default=180)
    parser.add_argument("--all", action="store_true", help="evaluate every included workload")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    cases, backends = check_metadata(base, args.backends, args.startup_wait_s)
    if args.all:
        for case in cases:
            evaluate_reference(base, case, args.deadline_s)
    else:
        gemm = next(c for c in cases if c[0] == "gemm_n7168_k5120")
        for backend in backends:
            evaluate_reference(backend["url"], gemm, args.deadline_s)
        evaluate_reference(base, gemm, args.deadline_s)


if __name__ == "__main__":
    main()
