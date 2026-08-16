import json
import os
import subprocess
import sys
import time
import requests

COMPILE_LOG = ""
EXTRA_PAYLOAD = {"run_baseline": False, "profile_baseline": False, "atol": 1e-3, "rtol": 1e-3}

def prepend_compile_log(traces, compile_log: str):
    """Merge compile_log into each trace's evaluation.log so ptxas messages
    (e.g. 'Potential Performance Loss') are visible downstream."""
    if not compile_log or not traces:
        return traces
    for t in traces:
        if not isinstance(t, dict):
            continue
        evaluation = t.get("evaluation") or {}
        existing = evaluation.get("log", "") or ""
        evaluation["log"] = compile_log + ("\n" + existing if existing else "")
        t["evaluation"] = evaluation
    return traces

COMPILE_SCRIPT_CONTENT = """
TVM_FFI_DIR=/usr/local/lib/python3.12/dist-packages/tvm_ffi
GENCODE="${NVCC_GENCODE:-arch=compute_90a,code=sm_90a}"
nvcc -shared -O3 -gencode "$GENCODE" kernel.cu \
    -lineinfo --ptxas-options=-v \
     -Xcompiler -fPIC,-fvisibility=hidden -lcuda \
     -I${TVM_FFI_DIR}/include -std=c++17 \
     -L${TVM_FFI_DIR}/lib -ltvm_ffi \
     -o kernel.so
"""

DEFINITION_NAME = "fp8_gemm_nt_1d2d_n4096_k7168"
WORKLOAD_UUID = "e52b28ef-1525-4e15-a81d-454e48209fcd"
REF_LATENCY_MS = [0.1744550323486328]


def build_solution(kernel_source: str) -> dict:
    nvcc_gencode = os.environ.get("NVCC_GENCODE", "arch=compute_90a,code=sm_90a")
    if nvcc_gencode == "arch=compute_90a,code=sm_90a":
        target_hardware = ["H100"]
    elif nvcc_gencode == "arch=compute_100a,code=sm_100a":
        target_hardware = ["Blackwell"]
    else:
        raise ValueError(f"Unsupported NVCC_GENCODE: {nvcc_gencode}")
    return {
        "name": "eval_kernel",
        "definition": DEFINITION_NAME,
        "spec": {
            "language": "cuda",
            "binding": "tvm-ffi",
            "target_hardware": target_hardware,
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "eval",
        "sources": [
            {"path": "kernel.cu", "content": kernel_source},
        ],
    }


def submit_and_poll(base_url: str, endpoint: str, payload: dict, timeout: int) -> dict:
    """Submit to /<endpoint> then poll /tasks/{id} until terminal. Returns final task data."""
    resp = requests.post(f"{base_url}/{endpoint}", json=payload)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"Submitted {endpoint} task: {task_id}", file=sys.stderr)

    deadline = time.time() + timeout
    while time.time() < deadline:
        result = requests.get(f"{base_url}/tasks/{task_id}?timeout=30")
        result.raise_for_status()
        data = result.json()
        if data["status"] in ("completed", "failed"):
            return data
        print(f"Status: {data['status']}, waiting...", file=sys.stderr)
        time.sleep(5)

    print(f"Timed out after {timeout}s waiting for {endpoint}", file=sys.stderr)
    sys.exit(1)


def has_failure(traces) -> bool:
    """True if any trace failed with a runtime fault (RUNTIME_ERROR or TIMEOUT).

    Both classes are worth sanitizing: RUNTIME_ERROR for direct faults (illegal
    memory access, illegal instruction, invalid argument, etc.) and TIMEOUT for
    hangs -- which can be races *or* the side-effect of a bad access that
    wedged the CUDA context.
    """
    if not traces:
        return False
    return any(
        ((t.get("evaluation") or {}).get("status")) in ("RUNTIME_ERROR", "TIMEOUT")
        for t in traces
    )


# compute-sanitizer summary markers that indicate the tool saw no issues.
_CLEAN_MARKERS = {
    "memcheck": "ERROR SUMMARY: 0 errors",
    "initcheck": "ERROR SUMMARY: 0 errors",
    "synccheck": "ERROR SUMMARY: 0 errors",
    "racecheck": "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
}


def sanitizer_found_issue(traces, tool: str) -> bool:
    """True when the sanitizer reported an issue on any trace.

    Must be called on *raw* traces (before `clean_sanitizer_log`, which strips
    the `ERROR SUMMARY` line). If the clean-run marker is absent on any trace
    we assume the tool surfaced something (either an error, or the program
    crashed before emitting the summary -- both cases warrant attention).
    """
    if not traces:
        return False
    marker = _CLEAN_MARKERS.get(tool, "ERROR SUMMARY: 0 errors")
    for t in traces:
        log = t.get("log", "") or ""
        if marker not in log:
            return True
    return False


def run_sanitize(base_url: str, solution: dict, tool: str, sanitize_timeout: int) -> dict:
    payload = {
        "solution": solution,
        "workload_uuids": [WORKLOAD_UUID],
        "sanitizer_types": [tool],
        "print_limit": 1,
        "max_lines": 200,
    }
    return submit_and_poll(base_url, "sanitize", payload, sanitize_timeout)


def filter_device_frames(log: str) -> str:
    """Keep only the Device Frames of the STDOUT: drop Host Frame lines and the
    'Saved host backtrace' preamble."""
    kept = []
    for line in log.split("\n"):
        if "Host Frame" in line:
            continue
        if "Saved host backtrace up to driver entry point" in line:
            continue
        kept.append(line)
    return "\n".join(kept)


# Substrings whose presence anywhere in the line makes it noise.
_NOISE_SUBSTRINGS = (
    "========= COMPUTE-SANITIZER",
    "========= Target application returned an error",
    "ERROR SUMMARY",
    "Sanitizer checks complete",
    "WARNING:",
    "passed successfully",
)


def strip_scaffolding(log: str) -> str:
    """Remove wrapper scaffolding around the error body: banner bars, 'Running <TYPE>'
    headers, STDOUT label, 'Return code' line, 'ERROR SUMMARY' counts, and redundant
    'target errored' / 'WARNING' chatter. STDERR content is kept."""
    kept = []
    for line in log.split("\n"):
        stripped = line.strip()
        if stripped and set(stripped) == {"="}:
            continue
        if stripped.startswith("Running ") and stripped.split()[-1] in {
            "MEMCHECK",
            "RACECHECK",
            "INITCHECK",
            "SYNCCHECK",
        }:
            continue
        if line.startswith("STDOUT:"):
            continue
        if line.startswith("Return code:"):
            continue
        if any(s in line for s in _NOISE_SUBSTRINGS):
            continue
        kept.append(line)
    # collapse consecutive blank lines
    out = []
    prev_blank = False
    for l in kept:
        is_blank = (l.strip() == "")
        if is_blank and prev_blank:
            continue
        out.append(l)
        prev_blank = is_blank
    return "\n".join(out).strip()


def clean_sanitizer_log(log: str) -> str:
    return strip_scaffolding(filter_device_frames(log))


def strip_host_frames_from_traces(traces):
    if not traces:
        return traces
    for t in traces:
        if t and isinstance(t.get("log"), str):
            t["log"] = clean_sanitizer_log(t["log"])
    return traces

def update_ref_latency(traces, ref_latency_ms):
    for (t, ref_lat) in zip(traces, ref_latency_ms):
        if not isinstance(t, dict):
            continue
        if t.get("evaluation"):
            if t["evaluation"]["status"] == "PASSED":
                t["evaluation"]["performance"]["reference_latency_ms"] = ref_lat
                t["evaluation"]["performance"]["speedup_factor"] = ref_lat / t["evaluation"]["performance"]["latency_ms"]

def main():
    base_url = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")
    kernel_path = "kernel.cu"
    timeout = 40
    sanitize_timeout = 100
    with open(kernel_path) as f:
        kernel_source = f.read()

    solution = build_solution(kernel_source)

    # Memcheck gate: multiple runs, stop at first error. Only run /evaluate if
    # every run is clean; on any detected issue, skip /evaluate and report
    # RUNTIME_ERROR with the sanitizer log.
    max_check_runs = 3
    check_type = "memcheck"
    failing_logs = None
    for i in range(max_check_runs):
        print(f"Running {check_type} {i + 1}/{max_check_runs}...", file=sys.stderr)
        san_data = run_sanitize(base_url, solution, check_type, sanitize_timeout)
        if san_data["status"] == "failed":
            print("Sanitize task failed:", san_data.get("error", "Unknown error"), file=sys.stderr)
            sys.exit(1)
        if sanitizer_found_issue(san_data.get("logs"), check_type):
            failing_logs = san_data.get("logs")
            break

    if failing_logs is not None:
        print(f"{check_type} detected an issue; marking RUNTIME_ERROR and skipping /evaluate", file=sys.stderr)
        san_logs = strip_host_frames_from_traces(failing_logs)
        for t in san_logs or []:
            if isinstance(t, dict):
                evaluation = t.get("evaluation") or {}
                evaluation["status"] = "RUNTIME_ERROR"
                # common.py's format_traces_feedback reads the log from
                # trace["evaluation"]["log"]; sanitize responses put it at
                # the top level, so surface it where the consumer looks.
                if "log" not in evaluation and isinstance(t.get("log"), str):
                    evaluation["log"] = t["log"]
                t["evaluation"] = evaluation
        prepend_compile_log(san_logs, COMPILE_LOG)
        with open("traces.json", "w") as f:
            json.dump(san_logs, f, indent=2)
        return

    print(f"{check_type} clean; running /evaluate...", file=sys.stderr)
    payload = {"solution": solution, "workload_uuids": [WORKLOAD_UUID], **EXTRA_PAYLOAD}
    data = submit_and_poll(base_url, "evaluate", payload, timeout)
    traces = data.get("traces")
    prepend_compile_log(traces, COMPILE_LOG)
    update_ref_latency(traces, REF_LATENCY_MS)
    with open("traces.json", "w") as f:
        json.dump(traces, f, indent=2)
    if data["status"] == "failed":
        print("Task failed at the profiling server. Error:", data.get("error", "Unknown error"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Write compile.sh
    with open("compile.sh", "w") as f:
        f.write(COMPILE_SCRIPT_CONTENT)
    # Remove stale artifact, then run compile capturing stdout+stderr so ptxas
    # diagnostics (e.g. 'Potential Performance Loss') can be surfaced downstream.
    try:
        os.remove("kernel.so")
    except FileNotFoundError:
        pass
    proc = subprocess.run(
        ["bash", "compile.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    COMPILE_LOG = proc.stdout or ""
    # Mirror to stderr so interactive runs still see the compiler output.
    if COMPILE_LOG:
        sys.stderr.write(COMPILE_LOG)
        sys.stderr.flush()
    if proc.returncode != 0:
        print("Failed to compile kernel", file=sys.stderr)
        sys.exit(1)
    main()
