#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path

import requests


BASE_URL = "http://localhost:10003"
DEFINITION = "mha_bwd_d128"
WORKLOAD_UUID = "38c3b07c-f006-5f5e-9860-ba214c805a6b"
KERNEL_PATH = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/ubuntu/AccRL-exps/deprecated-runs/"
    "fixit-v5-qwen36-linfo-gemini/exp_001/kernel.cu"
)
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "replay_memcheck_timeout_exp001")


def build_solution(kernel_source: str) -> dict:
    return {
        "name": "exp001_timeout_replay",
        "definition": DEFINITION,
        "spec": {
            "language": "cuda",
            "binding": "tvm-ffi",
            "target_hardware": ["H100"],
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "debug-replay",
        "sources": [{"path": "kernel.cu", "content": kernel_source}],
    }


def submit_and_poll(endpoint: str, payload: dict, wait_s: int) -> dict:
    response = requests.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=(5, 30))
    response.raise_for_status()
    submit_data = response.json()
    task_id = submit_data["task_id"]
    (OUT_DIR / f"{endpoint}_submit.json").write_text(json.dumps(submit_data, indent=2) + "\n")

    deadline = time.time() + wait_s
    last = None
    while time.time() < deadline:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}?timeout=30", timeout=(5, 40))
        response.raise_for_status()
        last = response.json()
        (OUT_DIR / f"{endpoint}_latest.json").write_text(json.dumps(last, indent=2) + "\n")
        if last.get("status") in {"completed", "failed"}:
            return last
        time.sleep(5)
    raise TimeoutError(f"{endpoint} task {task_id} did not finish within {wait_s}s")


def write_log_artifacts(prefix: str, task_data: dict) -> None:
    (OUT_DIR / f"{prefix}_task.json").write_text(json.dumps(task_data, indent=2) + "\n")
    logs = task_data.get("logs") or []
    if logs and isinstance(logs[0], dict):
        log = logs[0].get("log")
        if isinstance(log, str):
            (OUT_DIR / f"{prefix}_logs0.log").write_text(log)
        metadata = logs[0].get("metadata")
        if isinstance(metadata, dict):
            (OUT_DIR / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def copy_debug_raw(task_data: dict) -> None:
    logs = task_data.get("logs") or []
    if not logs or not isinstance(logs[0], dict):
        return
    metadata = logs[0].get("metadata")
    if not isinstance(metadata, dict):
        return
    report = metadata.get("FlashInfer CUDA debug report") or {}
    msg = report.get("msg") if isinstance(report, dict) else None
    if not isinstance(msg, str):
        return
    match = re.search(r"debug_dir:\s*(\S+)", msg)
    if not match:
        return
    debug_dir = Path(match.group(1))
    (OUT_DIR / "debug_dir.txt").write_text(str(debug_dir) + "\n")
    raw = debug_dir / "debug_raw.log"
    if raw.exists():
        (OUT_DIR / "debug_raw.log").write_text(raw.read_text(errors="replace"))
    else:
        (OUT_DIR / "debug_raw_missing.txt").write_text(f"{raw} does not exist or is not host-visible\n")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    kernel_source = KERNEL_PATH.read_text()
    (OUT_DIR / "kernel.cu").write_text(kernel_source)
    solution = build_solution(kernel_source)
    (OUT_DIR / "solution.json").write_text(json.dumps(solution, indent=2) + "\n")

    sanitize_payload = {
        "solution": solution,
        "workload_uuids": [WORKLOAD_UUID],
        "sanitizer_types": ["memcheck"],
        "timeout": 120,
        "max_lines": None,
        "print_limit": None,
    }
    (OUT_DIR / "sanitize_payload.json").write_text(json.dumps(sanitize_payload, indent=2) + "\n")
    sanitize = submit_and_poll("sanitize", sanitize_payload, wait_s=420)
    write_log_artifacts("sanitize", sanitize)

    debug_payload = {
        "solution": solution,
        "workload_uuids": [WORKLOAD_UUID],
        "sanitizer_types": ["memcheck"],
        "timeout": 120,
        "evaluation_timeout": 120,
        "max_lines": None,
        "print_limit": None,
        "source_context_lines": 4,
        "enable_coredump": True,
        "coredump_grace_seconds": 30,
    }
    (OUT_DIR / "debug_payload.json").write_text(json.dumps(debug_payload, indent=2) + "\n")
    debug = submit_and_poll("debug", debug_payload, wait_s=420)
    write_log_artifacts("debug", debug)
    copy_debug_raw(debug)


if __name__ == "__main__":
    main()
