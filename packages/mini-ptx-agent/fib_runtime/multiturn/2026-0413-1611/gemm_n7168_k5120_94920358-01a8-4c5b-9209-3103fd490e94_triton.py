import json
import os
import sys
import time

import requests


DEFINITION_NAME = "gemm_n7168_k5120"
WORKLOAD_UUID = "94920358-01a8-4c5b-9209-3103fd490e94"
queue_timeout = 240
sanitizer_timeout = 120
max_check_runs = 2


def target_hardware_from_triton_arch(triton_arch: str) -> list[str]:
    if triton_arch == "hopper":
        return ["Hopper"]
    if triton_arch == "blackwell":
        return ["Blackwell"]
    raise ValueError(f"Unsupported TRITON_GPU_ARCH: {triton_arch}")


def build_solution(kernel_source: str) -> dict:
    triton_arch = os.environ.get("TRITON_GPU_ARCH", "hopper")
    return {
        "name": "eval_kernel",
        "definition": DEFINITION_NAME,
        "spec": {
            "language": "triton",
            "target_hardware": target_hardware_from_triton_arch(triton_arch),
            "entry_point": "kernel.py::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "eval",
        "sources": [{"path": "kernel.py", "content": kernel_source}],
    }


def submit_and_poll(base_url: str, endpoint: str, payload: dict, timeout: int) -> dict:
    """Submit to an endpoint and poll its task until terminal."""
    try:
        response = requests.post(
            f"{base_url}/{endpoint}",
            json=payload,
            timeout=(5, 30),
        )
        response.raise_for_status()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        print(
            f"INFRA_TIMEOUT: {endpoint} request to {base_url} timed out: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    task_id = response.json()["task_id"]
    print(f"Submitted {endpoint} task: {task_id}", file=sys.stderr)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = requests.get(
                f"{base_url}/tasks/{task_id}?timeout=30",
                timeout=(5, 40),
            )
            result.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            print(
                f"INFRA_TIMEOUT: {endpoint} task {task_id} poll to {base_url} timed out: {exc}",
                file=sys.stderr,
            )
            sys.exit(2)
        data = result.json()
        if data["status"] in ("completed", "failed"):
            return data
        print(f"Status: {data['status']}, waiting...", file=sys.stderr)
        time.sleep(5)

    print(f"Timed out after {timeout}s waiting for {endpoint}", file=sys.stderr)
    sys.exit(1)


def has_failure(traces) -> bool:
    if not traces:
        return False
    return any(
        ((trace.get("evaluation") or {}).get("status"))
        in ("RUNTIME_ERROR", "TIMEOUT")
        for trace in traces
        if isinstance(trace, dict)
    )


def _strip_cu_get_proc_address_noise(log: str) -> str:
    output = []
    skipping = False
    for line in log.split("\n"):
        if "Program hit" in line and "cuGetProcAddress_v2" in line:
            skipping = True
            continue
        if skipping:
            if line.strip() == "=========":
                skipping = False
            continue
        output.append(line)
    return "\n".join(output)


def sanitizer_failure_status(logs) -> str | None:
    if not logs:
        return None
    for trace in logs:
        log = trace.get("log", "") if isinstance(trace, dict) else ""
        clean_log = _strip_cu_get_proc_address_noise(log or "")
        if "timed out after" in clean_log.lower():
            return "TIMEOUT"
        if "Program hit" in clean_log:
            return "RUNTIME_ERROR"
    return None


def run_sanitize(base_url: str, solution: dict) -> dict:
    payload = {
        "solution": solution,
        "workload_uuids": [WORKLOAD_UUID],
        "sanitizer_types": ["memcheck"],
        "print_limit": 1,
        "max_lines": None,
        "timeout": sanitizer_timeout,
    }
    return submit_and_poll(
        base_url,
        "sanitize",
        payload,
        sanitizer_timeout + queue_timeout,
    )


def render_debug_metadata(metadata: dict) -> str:
    rendered = []
    for title, section in metadata.items():
        if title == "FlashInfer CUDA debug report":
            continue
        if isinstance(section, dict) and not section.get("exist"):
            continue
        rendered.append(str(title))
        rendered.append("-" * 80)
        if isinstance(section, dict) and section.get("msg"):
            rendered.append(str(section["msg"]))
        rendered.append("")
    report = "\n".join(rendered).rstrip()
    return report + "\n" if report else ""


def run_debug(base_url: str, solution: dict, status: str) -> str:
    payload = {
        "solution": solution,
        "workload_uuids": [WORKLOAD_UUID],
        "sanitizer_types": ["memcheck"],
        "timeout": sanitizer_timeout,
        "max_lines": None,
        "print_limit": 100,
        "source_context_lines": 4,
        "enable_coredump": status == "TIMEOUT",
        "coredump_grace_seconds": 30,
    }
    try:
        data = submit_and_poll(
            base_url,
            "debug",
            payload,
            sanitizer_timeout + queue_timeout,
        )
    except Exception as exc:
        print(f"Debug task failed; keeping original failure log: {exc}", file=sys.stderr)
        return ""
    if data["status"] == "failed":
        return ""
    logs = data.get("logs") or []
    if not logs:
        return ""
    metadata = logs[0].get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return render_debug_metadata(metadata)


def replace_failure_logs_with_debug(traces, debug_report: str):
    if not debug_report or not traces:
        return traces
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        evaluation = trace.get("evaluation") or {}
        if evaluation.get("status") in ("RUNTIME_ERROR", "TIMEOUT"):
            evaluation["log"] = debug_report
            trace["evaluation"] = evaluation
    return traces


def sanitizer_logs_to_traces(logs, status: str):
    traces = logs or []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        evaluation = trace.get("evaluation") or {}
        evaluation["status"] = status
        if "log" not in evaluation and isinstance(trace.get("log"), str):
            evaluation["log"] = trace["log"]
        trace["evaluation"] = evaluation
    return traces


def write_traces(traces) -> None:
    with open("traces.json", "w") as output:
        json.dump(traces, output, indent=2)


def main() -> None:
    base_url = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")
    with open("kernel.py") as kernel_file:
        solution = build_solution(kernel_file.read())

    failing_logs = None
    sanitizer_status = None
    for check_index in range(max_check_runs):
        print(
            f"Running memcheck {check_index + 1}/{max_check_runs}...",
            file=sys.stderr,
        )
        sanitizer_data = run_sanitize(base_url, solution)
        if sanitizer_data["status"] == "failed":
            print(
                "Sanitize task failed:",
                sanitizer_data.get("error", "Unknown error"),
                file=sys.stderr,
            )
            sys.exit(1)
        sanitizer_status = sanitizer_failure_status(sanitizer_data.get("logs"))
        if sanitizer_status is not None:
            failing_logs = sanitizer_data.get("logs")
            break

    if failing_logs is not None:
        print(
            f"memcheck detected {sanitizer_status}; skipping /evaluate",
            file=sys.stderr,
        )
        traces = sanitizer_logs_to_traces(failing_logs, sanitizer_status)
        debug_report = run_debug(base_url, solution, sanitizer_status)
        replace_failure_logs_with_debug(traces, debug_report)
        write_traces(traces)
        return

    print("memcheck clean; running /evaluate...", file=sys.stderr)
    data = submit_and_poll(
        base_url,
        "evaluate",
        {"solution": solution, "workload_uuids": [WORKLOAD_UUID]},
        queue_timeout,
    )
    traces = data.get("traces")
    if has_failure(traces):
        statuses = {
            (trace.get("evaluation") or {}).get("status")
            for trace in traces or []
            if isinstance(trace, dict)
        }
        debug_report = run_debug(
            base_url,
            solution,
            "TIMEOUT" if "TIMEOUT" in statuses else "RUNTIME_ERROR",
        )
        replace_failure_logs_with_debug(traces, debug_report)
    write_traces(traces)
    if data["status"] == "failed":
        print(
            "Task failed at the profiling server. Error:",
            data.get("error", "Unknown error"),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
