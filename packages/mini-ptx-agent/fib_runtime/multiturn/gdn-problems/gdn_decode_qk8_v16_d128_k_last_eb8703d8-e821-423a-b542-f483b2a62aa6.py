import json
import os
import subprocess
import sys
import time
import requests

DEFINITION_NAME = "gdn_decode_qk8_v16_d128_k_last"
WORKLOAD_UUID = "eb8703d8-e821-423a-b542-f483b2a62aa6"
TVM_FFI_DIR = "/usr/local/lib/python3.12/dist-packages/tvm_ffi"
queue_timeout = 240
sanitizer_timeout = 120
max_check_runs = 2


REQUIRED_INSTS = {
    "H": {
        "cp.async.bulk.tensor",
        "wgmma."
    },
    "B": {
        "cp.async.bulk.tensor",
        "tcgen05"
    }
}

COMPILE_LOG = ""

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


def prepend_log_warning(traces, warning: str):
    if not warning or not traces:
        return traces
    for t in traces:
        if not isinstance(t, dict):
            continue
        evaluation = t.get("evaluation") or {}
        existing = evaluation.get("log", "") or ""
        evaluation["log"] = warning + ("\n" + existing if existing else "")
        t["evaluation"] = evaluation
    return traces


COMPILE_SCRIPT_CONTENT = f"""
TVM_FFI_DIR={TVM_FFI_DIR}
GENCODE="${{NVCC_GENCODE:-arch=compute_90a,code=sm_90a}}"
nvcc -shared -O3 -gencode "$GENCODE" kernel.cu \
    -lineinfo --ptxas-options=-v \
     -Xcompiler -fPIC,-fvisibility=hidden -lcuda \
     -I${{TVM_FFI_DIR}}/include -std=c++17 \
     -L${{TVM_FFI_DIR}}/lib -ltvm_ffi \
     -o kernel.so
"""


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


def ptx_gencode_from_nvcc_gencode(nvcc_gencode: str) -> str:
    if nvcc_gencode == "arch=compute_90a,code=sm_90a":
        return "arch=compute_90a,code=compute_90a"
    if nvcc_gencode == "arch=compute_100a,code=sm_100a":
        return "arch=compute_100a,code=compute_100a"
    raise ValueError(f"Unsupported NVCC_GENCODE: {nvcc_gencode}")


def required_arch_tag_from_nvcc_gencode(nvcc_gencode: str) -> str:
    if nvcc_gencode == "arch=compute_90a,code=sm_90a":
        return "H"
    if nvcc_gencode == "arch=compute_100a,code=sm_100a":
        return "B"
    raise ValueError(f"Unsupported NVCC_GENCODE: {nvcc_gencode}")


def compile_to_ptx(kernel_path: str, nvcc_gencode: str) -> tuple[str | None, str]:
    ptx_gencode = ptx_gencode_from_nvcc_gencode(nvcc_gencode)
    result = subprocess.run(
        [
            "nvcc",
            "--ptx",
            "-O3",
            "-gencode",
            ptx_gencode,
            kernel_path,
            "-lineinfo",
            f"-I{TVM_FFI_DIR}/include",
            "-std=c++17",
            "-o",
            "kernel.ptx",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout or ""
    if output:
        sys.stderr.write(output)
        sys.stderr.flush()
    if result.returncode != 0:
        return None, output
    with open("kernel.ptx", errors="replace") as f:
        return f.read(), output



def instruction_warning_message(ptx_text: str, required_tag: str) -> str:
    required_insts = REQUIRED_INSTS[required_tag]
    # Suggestions are those required instructions that are not found in the ptx.
    must_have = [inst for inst in required_insts if inst not in ptx_text]
    if not must_have:
        return ""
    return (
        f"INSTRUCTION WARNING: the kernel is not fully exercising the specialized hardware units. "
        f"You must try to use these instructions: {must_have}."
    )


def check_required_arch_instruction(kernel_path: str) -> str:
    nvcc_gencode = os.environ.get("NVCC_GENCODE", "arch=compute_90a,code=sm_90a")
    required_tag = required_arch_tag_from_nvcc_gencode(nvcc_gencode)
    ptx_text, _ = compile_to_ptx(kernel_path, nvcc_gencode)
    if ptx_text is None:
        print("Failed to compile kernel.ptx", file=sys.stderr)
        return ""
    return instruction_warning_message(ptx_text, required_tag)


def submit_and_poll(base_url: str, endpoint: str, payload: dict, timeout: int) -> dict:
    """Submit to /<endpoint> then poll /tasks/{id} until terminal. Returns final task data."""
    try:
        resp = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=(5, 30))
        resp.raise_for_status()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        print(f"INFRA_TIMEOUT: {endpoint} request to {base_url} timed out: {e}", file=sys.stderr)
        sys.exit(2)
    task_id = resp.json()["task_id"]
    print(f"Submitted {endpoint} task: {task_id}", file=sys.stderr)

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = requests.get(f"{base_url}/tasks/{task_id}?timeout=30", timeout=(5, 40))
            result.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"INFRA_TIMEOUT: {endpoint} task {task_id} poll to {base_url} timed out: {e}", file=sys.stderr)
            sys.exit(2)
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


def _strip_cu_get_proc_address_noise(log: str) -> str:
    """Drop ``Program hit ... cuGetProcAddress_v2`` blocks.

    The CUDA driver probes function pointers at startup via
    cuGetProcAddress_v2; compute-sanitizer faithfully reports each probe
    miss as CUDA_ERROR_INVALID_VALUE, but they are unrelated to the kernel
    under test. Each block runs from the ``Program hit ... cuGetProcAddress_v2``
    line through the next bare ``=========`` separator.
    """
    out = []
    skipping = False
    for line in log.split("\n"):
        if "Program hit" in line and "cuGetProcAddress_v2" in line:
            skipping = True
            continue
        if skipping:
            if line.strip() == "=========":
                skipping = False
            continue
        out.append(line)
    return "\n".join(out)


def sanitizer_found_issue(traces, tool: str) -> bool:
    """True when the sanitizer reported a real issue on any trace.

    Must be called on *raw* traces (before `clean_sanitizer_log`, which strips
    structural markers). We rule out cuGetProcAddress_v2 driver-probe noise by
    removing those blocks structurally, then look for any remaining
    ``Program hit`` line as a real sanitizer event. The ``ERROR SUMMARY``
    count itself is unreliable because the sanitizer attributes the filtered
    probe events to it, so we don't gate on it for memcheck/initcheck/synccheck.
    racecheck output is unaffected by cuGetProcAddress noise, so we keep the
    original marker-based check there.
    """
    if not traces:
        return False
    for t in traces:
        log = t.get("log", "") or ""

        if tool == "racecheck":
            marker = _CLEAN_MARKERS["racecheck"]
            if marker not in log:
                return True
            continue

        clean_log = _strip_cu_get_proc_address_noise(log)
        if "Program hit" in clean_log:
            return True
    return False


def run_sanitize(base_url: str, solution: dict, tool: str, sanitizer_timeout: int, queue_timeout: int) -> dict:
    payload = {
        "solution": solution,
        "workload_uuids": [WORKLOAD_UUID],
        "sanitizer_types": [tool],
        "print_limit": 1,
        # cuGetProcAddress_v2 driver-probe noise can run several hundred lines
        # before the ERROR SUMMARY appears; request the full log so the
        # post-strip marker check can see it.
        "max_lines": None,
        "sanitizer_timeout": sanitizer_timeout,
    }
    return submit_and_poll(base_url, "sanitize", payload, sanitizer_timeout + queue_timeout)


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


def main(isa_warning: str = ""):
    base_url = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")
    kernel_path = "kernel.cu"
    with open(kernel_path) as f:
        kernel_source = f.read()

    solution = build_solution(kernel_source)

    # Memcheck gate: multiple runs, stop at first error. Only run /evaluate if
    # every run is clean; on any detected issue, skip /evaluate and report
    # RUNTIME_ERROR with the sanitizer log.
    
    check_type = "memcheck"
    failing_logs = None
    for i in range(max_check_runs):
        print(f"Running {check_type} {i + 1}/{max_check_runs}...", file=sys.stderr)
        san_data = run_sanitize(base_url, solution, check_type, sanitizer_timeout, queue_timeout)
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
        prepend_log_warning(san_logs, isa_warning)
        with open("traces.json", "w") as f:
            json.dump(san_logs, f, indent=2)
        return

    print(f"{check_type} clean; running /evaluate...", file=sys.stderr)
    payload = {"solution": solution, "workload_uuids": [WORKLOAD_UUID]}
    data = submit_and_poll(base_url, "evaluate", payload, queue_timeout)
    traces = data.get("traces")
    prepend_compile_log(traces, COMPILE_LOG)
    prepend_log_warning(traces, isa_warning)
    with open("traces.json", "w") as f:
        json.dump(traces, f, indent=2)
    if data["status"] == "failed":
        print("Task failed at the profiling server. Error:", data.get("error", "Unknown error"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Write compile.sh
    with open("compile.sh", "w") as f:
        f.write(COMPILE_SCRIPT_CONTENT)
    # Remove stale artifacts, then run compile capturing stdout+stderr so ptxas
    # diagnostics (e.g. 'Potential Performance Loss') can be surfaced downstream.
    for artifact in ("kernel.so", "kernel.ptx"):
        try:
            os.remove(artifact)
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
    isa_warning = check_required_arch_instruction("kernel.cu")
    main(isa_warning)
