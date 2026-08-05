"""Task-bound CUDA kernel evaluation for agent environments."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from mini_ptx_agent.ptx import extract_architecture_usage

SCHEMA_VERSION = "ptxbench.eval.v1"
SUPPORTED_GENCODES = {
    "arch=compute_90a,code=sm_90a": ["H100"],
    "arch=compute_100a,code=sm_100a": ["Blackwell"],
}
_CUBLAS_PATTERN = re.compile(
    r"""
    (?:^\s*\#\s*include\s*[<"]\s*cublas(?:_v2)?\.h\s*[>"])
    |
    (?:\bcublas[A-Za-z0-9_]*\s*\()
    |
    (?:\bcublasHandle_t\b)
    """,
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)
_CUDNN_PATTERN = re.compile(r"cudnn|cudnnCreate|cudnn\.h|cudnnConvolution|cudnnSetTensor", re.IGNORECASE)
_LIBRARY_BANNED_MSG = (
    "ERROR: Your kernel uses {library} library calls instead of a hand-written kernel. "
    "Please implement the kernel using CUDA directly — "
    "cuBLAS, cuDNN, and other library shortcuts are not allowed."
)


class EvaluationInfrastructureError(RuntimeError):
    """The evaluator could not produce a candidate result."""


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    definition: str
    workload_uuids: tuple[str, ...]
    target_hardware: tuple[str, ...]
    nvcc_gencode: str
    tvm_ffi_dir: Path
    max_check_runs: int = 2
    queue_timeout_sec: int = 240
    sanitizer_timeout_sec: int = 120
    target_speedup: float = 1.0


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task config field {key!r} must be a non-empty string")
    return value


def _required_strings(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"task config field {key!r} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"task config field {key!r} must contain non-empty strings")
    return tuple(value)


def load_task(path: Path) -> EvaluationTask:
    """Load the immutable definition/workload binding for one benchmark task."""
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read task config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in task config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"task config {path} must contain a JSON object")

    nvcc_gencode = _required_string(data, "nvcc_gencode")
    target_hardware = _required_strings(data, "target_hardware")
    expected_hardware = SUPPORTED_GENCODES.get(nvcc_gencode)
    if expected_hardware is None:
        raise ValueError(f"unsupported nvcc_gencode: {nvcc_gencode}")
    if list(target_hardware) != expected_hardware:
        raise ValueError(
            f"target_hardware does not match nvcc_gencode: expected {expected_hardware}, got {list(target_hardware)}"
        )

    max_check_runs = int(data.get("max_check_runs", 2))
    queue_timeout_sec = int(data.get("queue_timeout_sec", 240))
    sanitizer_timeout_sec = int(data.get("sanitizer_timeout_sec", 120))
    target_speedup = float(data.get("target_speedup", 1.0))
    if max_check_runs < 0:
        raise ValueError("max_check_runs must be non-negative")
    if queue_timeout_sec <= 0 or sanitizer_timeout_sec <= 0:
        raise ValueError("evaluation timeouts must be positive")
    if target_speedup < 0:
        raise ValueError("target_speedup must be non-negative")

    return EvaluationTask(
        task_id=_required_string(data, "task_id"),
        definition=_required_string(data, "definition"),
        workload_uuids=_required_strings(data, "workload_uuids"),
        target_hardware=target_hardware,
        nvcc_gencode=nvcc_gencode,
        tvm_ffi_dir=Path(_required_string(data, "tvm_ffi_dir")),
        max_check_runs=max_check_runs,
        queue_timeout_sec=queue_timeout_sec,
        sanitizer_timeout_sec=sanitizer_timeout_sec,
        target_speedup=target_speedup,
    )


def build_solution(task: EvaluationTask, kernel_source: str) -> dict[str, Any]:
    """Build the FlashInfer-Bench solution submitted to FIBServe."""
    return {
        "name": "eval_kernel",
        "definition": task.definition,
        "spec": {
            "language": "cuda",
            "binding": "tvm-ffi",
            "target_hardware": list(task.target_hardware),
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "eval",
        "sources": [{"path": "kernel.cu", "content": kernel_source}],
    }


def _strip_cpp_comments(source: str) -> str:
    """Remove C/C++ comments while preserving quoted strings and line boundaries."""
    output: list[str] = []
    index = 0
    while index < len(source):
        character = source[index]
        if character in {'"', "'"}:
            quote = character
            end = index + 1
            while end < len(source):
                if source[end] == "\\" and end + 1 < len(source):
                    end += 2
                    continue
                if source[end] == quote:
                    end += 1
                    break
                end += 1
            output.append(source[index:end])
            index = end
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline == -1 else newline
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            comment_end = len(source) if end == -1 else end + 2
            output.extend("\n" for character in source[index:comment_end] if character == "\n")
            index = comment_end
            continue
        output.append(character)
        index += 1
    return "".join(output)


def _banned_library_error(kernel_source: str) -> str | None:
    source_without_comments = _strip_cpp_comments(kernel_source)
    for library, pattern in (("cuBLAS", _CUBLAS_PATTERN), ("cuDNN", _CUDNN_PATTERN)):
        if pattern.search(source_without_comments):
            return _LIBRARY_BANNED_MSG.format(library=library)
    return None


def _ptx_gencode(nvcc_gencode: str) -> tuple[str, str]:
    architecture = nvcc_gencode.removeprefix("arch=").split(",", 1)[0]
    return f"arch={architecture},code={architecture}", architecture


def compile_candidate(task: EvaluationTask, kernel_path: Path) -> tuple[bool, str, dict[str, Any] | None]:
    """Compile the candidate and inspect a separately emitted PTX artifact."""
    include_dir = task.tvm_ffi_dir / "include"
    library_dir = task.tvm_ffi_dir / "lib"
    if not include_dir.is_dir() or not library_dir.is_dir():
        raise EvaluationInfrastructureError(f"tvm_ffi include/lib directories not found under {task.tvm_ffi_dir}")

    with tempfile.TemporaryDirectory(prefix="ptxbench-eval-") as temp_dir:
        output_path = Path(temp_dir) / "kernel.so"
        ptx_path = Path(temp_dir) / "kernel.ptx"
        command = [
            "nvcc",
            "-shared",
            "-O3",
            "-gencode",
            task.nvcc_gencode,
            str(kernel_path),
            "-lineinfo",
            "--ptxas-options=-v",
            "-Xcompiler",
            "-fPIC,-fvisibility=hidden",
            "-lcuda",
            f"-I{include_dir}",
            f"-I{kernel_path.parent}",
            "-std=c++17",
            f"-L{library_dir}",
            "-ltvm_ffi",
            "-o",
            str(output_path),
        ]
        cuda_stub_dir = Path("/usr/local/cuda/lib64/stubs")
        if cuda_stub_dir.is_dir():
            command.insert(command.index("-ltvm_ffi"), f"-L{cuda_stub_dir}")
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise EvaluationInfrastructureError(f"cannot run nvcc: {exc}") from exc
        compile_log = completed.stdout or ""
        if completed.returncode != 0:
            return False, compile_log, None

        ptx_gencode, virtual_arch = _ptx_gencode(task.nvcc_gencode)
        ptx_command = [
            "nvcc",
            "--ptx",
            "-O3",
            "-gencode",
            ptx_gencode,
            str(kernel_path),
            "-lineinfo",
            f"-I{include_dir}",
            f"-I{kernel_path.parent}",
            "-std=c++17",
            "-o",
            str(ptx_path),
        ]
        try:
            ptx_completed = subprocess.run(
                ptx_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise EvaluationInfrastructureError(f"cannot emit PTX artifact: {exc}") from exc
        ptx_log = ptx_completed.stdout or ""
        if ptx_completed.returncode != 0:
            combined_log = compile_log + ("\nPTX artifact compilation failed:\n" + ptx_log)
            return False, combined_log, None

        try:
            ptx_bytes = ptx_path.read_bytes()
        except OSError as exc:
            raise EvaluationInfrastructureError(f"cannot read emitted PTX artifact: {exc}") from exc
        ptx_text = ptx_bytes.decode(errors="replace")
        ptx_metadata = {
            "sha256": hashlib.sha256(ptx_bytes).hexdigest(),
            "size_bytes": len(ptx_bytes),
            **extract_architecture_usage(ptx_text, virtual_arch),
        }
    return True, compile_log, ptx_metadata


def submit_and_poll(
    service_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_sec: int,
) -> dict[str, Any]:
    """Submit an asynchronous FIBServe request and wait for its terminal result."""
    base_url = service_url.rstrip("/")
    try:
        response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=(5, 30))
        response.raise_for_status()
        task_id = response.json()["task_id"]
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise EvaluationInfrastructureError(f"failed to submit {endpoint} request to {base_url}: {exc}") from exc

    print(f"Submitted {endpoint} task: {task_id}", file=sys.stderr)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/tasks/{task_id}?timeout=30", timeout=(5, 40))
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, TypeError, ValueError) as exc:
            raise EvaluationInfrastructureError(f"failed to poll {endpoint} task {task_id}: {exc}") from exc
        if data.get("status") in {"completed", "failed"}:
            return data
        time.sleep(1)

    raise EvaluationInfrastructureError(f"timed out after {timeout_sec}s waiting for {endpoint} task {task_id}")


def _strip_driver_probe_noise(log: str) -> str:
    lines: list[str] = []
    skipping = False
    for line in log.splitlines():
        if "Program hit" in line and "cuGetProcAddress_v2" in line:
            skipping = True
            continue
        if skipping:
            if line.strip() == "=========":
                skipping = False
            continue
        lines.append(line)
    return "\n".join(lines)


def sanitizer_failure_status(logs: list[dict[str, Any]]) -> str | None:
    """Return the first real memcheck failure, excluding CUDA driver probe noise."""
    for item in logs:
        log = item.get("log")
        if not isinstance(log, str):
            continue
        cleaned = _strip_driver_probe_noise(log)
        if "timed out after" in cleaned.lower():
            return "TIMEOUT"
        if "Program hit" in cleaned:
            return "RUNTIME_ERROR"
    return None


def _synthetic_failure_traces(
    task: EvaluationTask, status: str, logs: list[dict[str, Any]], compile_log: str
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for index, workload_uuid in enumerate(task.workload_uuids):
        log = ""
        if index < len(logs) and isinstance(logs[index].get("log"), str):
            log = logs[index]["log"]
        if compile_log:
            log = compile_log + ("\n" + log if log else "")
        traces.append(
            {
                "definition": task.definition,
                "workload": {"uuid": workload_uuid},
                "solution": "eval_kernel",
                "evaluation": {"status": status, "log": log},
            }
        )
    return traces


def _trace_status(trace: dict[str, Any]) -> str | None:
    evaluation = trace.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    status = evaluation.get("status")
    return status if isinstance(status, str) else None


def _trace_speedup(trace: dict[str, Any]) -> float | None:
    evaluation = trace.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    performance = evaluation.get("performance")
    if not isinstance(performance, dict):
        return None
    speedup = performance.get("speedup_factor")
    if isinstance(speedup, int | float):
        return float(speedup)
    return None


def _result_envelope(
    task: EvaluationTask,
    kernel_sha256: str,
    *,
    status: str,
    traces: list[dict[str, Any]],
    sanitizer_runs: int,
    sanitizer_clean: bool,
    compile_log: str,
    ptx: dict[str, Any] | None,
) -> dict[str, Any]:
    statuses = [_trace_status(trace) for trace in traces]
    all_passed = bool(statuses) and all(item == "PASSED" for item in statuses)
    speedups = [speedup for trace in traces if (speedup := _trace_speedup(trace)) is not None]
    min_speedup = min(speedups) if speedups else None
    return {
        "schema": SCHEMA_VERSION,
        "evaluation_id": f"ev_{uuid.uuid4().hex}",
        "task_id": task.task_id,
        "definition": task.definition,
        "workload_uuids": list(task.workload_uuids),
        "kernel_sha256": kernel_sha256,
        "status": status,
        "all_passed": all_passed,
        "min_speedup": min_speedup,
        "target_met": bool(all_passed and min_speedup is not None and min_speedup >= task.target_speedup),
        "sanitizer": {
            "requested_runs": task.max_check_runs,
            "completed_runs": sanitizer_runs,
            "clean": sanitizer_clean,
        },
        "compile_log": compile_log,
        "ptx": ptx,
        "traces": traces,
    }


def evaluate_kernel(kernel_path: Path, task: EvaluationTask, service_url: str) -> dict[str, Any]:
    """Compile, sanitize, and evaluate one candidate kernel."""
    try:
        kernel_source = kernel_path.read_text()
    except OSError as exc:
        raise ValueError(f"cannot read kernel {kernel_path}: {exc}") from exc
    kernel_sha256 = hashlib.sha256(kernel_source.encode()).hexdigest()
    if library_error := _banned_library_error(kernel_source):
        traces = _synthetic_failure_traces(task, "COMPILE_ERROR", [], library_error)
        return _result_envelope(
            task,
            kernel_sha256,
            status="COMPILE_ERROR",
            traces=traces,
            sanitizer_runs=0,
            sanitizer_clean=False,
            compile_log=library_error,
            ptx=None,
        )

    compiled, compile_log, ptx = compile_candidate(task, kernel_path)
    if not compiled:
        traces = _synthetic_failure_traces(task, "COMPILE_ERROR", [], compile_log)
        return _result_envelope(
            task,
            kernel_sha256,
            status="COMPILE_ERROR",
            traces=traces,
            sanitizer_runs=0,
            sanitizer_clean=False,
            compile_log=compile_log,
            ptx=ptx,
        )

    solution = build_solution(task, kernel_source)
    completed_sanitizer_runs = 0
    for index in range(task.max_check_runs):
        print(
            f"Running memcheck {index + 1}/{task.max_check_runs}...",
            file=sys.stderr,
        )
        data = submit_and_poll(
            service_url,
            "sanitize",
            {
                "solution": solution,
                "workload_uuids": list(task.workload_uuids),
                "sanitizer_types": ["memcheck"],
                "print_limit": 1,
                "max_lines": None,
                "timeout": task.sanitizer_timeout_sec,
            },
            task.sanitizer_timeout_sec + task.queue_timeout_sec,
        )
        if data.get("status") == "failed":
            raise EvaluationInfrastructureError(f"sanitize task failed: {data.get('error', 'unknown error')}")
        completed_sanitizer_runs += 1
        logs = data.get("logs") or []
        if not isinstance(logs, list):
            raise EvaluationInfrastructureError("sanitize task returned invalid logs")
        failure_status = sanitizer_failure_status(logs)
        if failure_status is not None:
            traces = _synthetic_failure_traces(task, failure_status, logs, compile_log)
            return _result_envelope(
                task,
                kernel_sha256,
                status=failure_status,
                traces=traces,
                sanitizer_runs=completed_sanitizer_runs,
                sanitizer_clean=False,
                compile_log=compile_log,
                ptx=ptx,
            )

    print("Memcheck clean; running evaluate...", file=sys.stderr)
    data = submit_and_poll(
        service_url,
        "evaluate",
        {
            "solution": solution,
            "workload_uuids": list(task.workload_uuids),
        },
        task.queue_timeout_sec,
    )
    if data.get("status") == "failed":
        raise EvaluationInfrastructureError(f"evaluate task failed: {data.get('error', 'unknown error')}")
    traces = data.get("traces") or []
    if not isinstance(traces, list) or not traces:
        raise EvaluationInfrastructureError("evaluate task returned no traces")
    for trace in traces:
        if not isinstance(trace, dict):
            raise EvaluationInfrastructureError("evaluate task returned an invalid trace")
        evaluation = trace.get("evaluation")
        if isinstance(evaluation, dict) and compile_log:
            existing_log = evaluation.get("log") or ""
            evaluation["log"] = compile_log + ("\n" + existing_log if existing_log else "")

    statuses = [_trace_status(trace) for trace in traces]
    status = (
        "PASSED"
        if all(item == "PASSED" for item in statuses)
        else next((item for item in statuses if item and item != "PASSED"), "FAILED")
    )
    return _result_envelope(
        task,
        kernel_sha256,
        status=status,
        traces=traces,
        sanitizer_runs=completed_sanitizer_runs,
        sanitizer_clean=True,
        compile_log=compile_log,
        ptx=ptx,
    )


def infrastructure_error_result(message: str) -> dict[str, Any]:
    """Return a parseable result even when no candidate evaluation was produced."""
    return {
        "schema": SCHEMA_VERSION,
        "evaluation_id": f"ev_{uuid.uuid4().hex}",
        "status": "INFRA_ERROR",
        "all_passed": False,
        "min_speedup": None,
        "target_met": False,
        "error": message,
        "traces": [],
    }
