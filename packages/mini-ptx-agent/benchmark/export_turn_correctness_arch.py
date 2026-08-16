#!/usr/bin/env python3
"""Export per-turn correctness, speedup, and verified SASS architecture tags.

For every experiment in a manifest, this script reads the multiturn trajectory
files and writes one canonical CSV:

    <exp_dir>/figures/turn_correctness_arch.csv

Correct CUDA turns are verified in two steps. First, the candidate is compiled
for the experiment architecture and its embedded cubin is inspected with
``cuobjdump --dump-sass``. Candidates with a selected architecture-specific
SASS family are then profiled through FIBServe. ``sass_arch_tag`` is set only
when Nsight Compute reports positive predicate-true execution for a matching
instruction. No intermediate correctness CSV or CSV merge is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from accrl.utils.code_utils import extract_code_block

DEFAULT_OUTPUT_NAME = "turn_correctness_arch.csv"
DEFAULT_STATIC_CACHE_DIR_NAME = "sass_cubin_cache_v1"
DEFAULT_PROFILE_CACHE_DIR_NAME = "sass_profile_cache_v2"
STATIC_CACHE_SCHEMA_VERSION = 1
PROFILE_CACHE_SCHEMA_VERSION = 4
TERMINAL_TASK_STATUSES = {"completed", "failed"}
TMA_TRANSFER_OPCODES = frozenset({"UTMALDG", "UTMASTG", "UTMAREDG"})
BLACKWELL_TCGEN_PREFIXES = ("UTC", "LDTM", "STTM")


@dataclass(frozen=True)
class ArchitectureConfig:
    tag: str
    target_hardware: tuple[str, ...]
    gencode: str


ARCHITECTURES = {
    "hopper": ArchitectureConfig(
        tag="H",
        target_hardware=("H100",),
        gencode="arch=compute_90a,code=sm_90a",
    ),
    "blackwell": ArchitectureConfig(
        tag="B",
        target_hardware=("Blackwell",),
        gencode="arch=compute_100a,code=sm_100a",
    ),
}

BASE_FIELDS = ["trajectory_id", "turn", "correctness", "speedup"]
SASS_FIELDS = [
    "cubin_sass_arch_tag",
    "cubin_gmma_instruction_count",
    "cubin_tma_instruction_count",
    "cubin_tcgen_instruction_count",
    "cubin_container_sha256",
    "sass_arch_tag",
    "sass_gmma_count",
    "sass_tma_count",
    "sass_tcgen_count",
    "sass_gmma_thread_inst_executed_true",
    "sass_tma_thread_inst_executed_true",
    "sass_tcgen_thread_inst_executed_true",
    "sass_profile_task_id",
    "sass_verification_status",
]

# NCU's source page ends each instruction row with the requested warp-level
# instruction count and predicate-true thread count, in request order.
SASS_LINE_RE = re.compile(
    r"^\s*0x[0-9a-f]+\s+.*?\b(?P<opcode>"
    r"[A-Z0-9]*GMMA(?:\.[A-Z0-9]+)*|UTMA[A-Z0-9.]*|"
    r"UTC[A-Z0-9.]*|LDTM(?:\.[A-Z0-9]+)*|STTM(?:\.[A-Z0-9]+)*)\b"
    r".*?\s(?P<count>[0-9][0-9,]*)\s+(?P<pred_on_count>[0-9][0-9,]*)\s*$",
    re.IGNORECASE,
)

# Executable cuobjdump rows begin with an address comment. Requiring the
# address prevents opcode names in metadata from becoming static positives.
STATIC_SASS_LINE_RE = re.compile(
    r"^\s*/\*[0-9a-f]+\*/.*?\b(?P<opcode>"
    r"[A-Z0-9]*GMMA(?:\.[A-Z0-9]+)*|UTMA[A-Z0-9.]*|"
    r"UTC[A-Z0-9.]*|LDTM(?:\.[A-Z0-9]+)*|STTM(?:\.[A-Z0-9]+)*)\b",
    re.IGNORECASE,
)


class ProfileError(RuntimeError):
    """Raised when dynamic SASS evidence cannot be collected reliably."""


class CubinInspectionError(RuntimeError):
    """Raised when a candidate cannot be compiled or disassembled."""


@dataclass(frozen=True)
class Candidate:
    trajectory_id: str
    turn: int
    source: str
    source_sha256: str


@dataclass(frozen=True)
class StaticSassEvidence:
    cubin_sass_arch_tag: str
    cubin_gmma_instruction_count: int
    cubin_tma_instruction_count: int
    cubin_tcgen_instruction_count: int
    cubin_container_sha256: str
    matched_lines: tuple[str, ...]
    disassembly_line_count: int


@dataclass(frozen=True)
class DynamicSassEvidence:
    sass_arch_tag: str
    gmma_count: int
    tma_count: int
    gmma_pred_on_thread_count: int
    tma_pred_on_thread_count: int
    matched_lines: tuple[str, ...]
    profile_line_count: int
    task_id: str = ""
    tcgen_count: int = 0
    tcgen_pred_on_thread_count: int = 0


def nested_get(value: object, path: tuple[str, ...]) -> object | None:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def as_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_turn(content: str) -> str:
    if "PASSED" in content:
        return "Correct"
    if "INCORRECT_NUMERICAL" in content or "Result is incorrect" in content:
        return "Numerical error"
    if "Failed to compile kernel" in content:
        return "Compilation error"
    if re.search(r"Timed out after \d+(?:\.\d+)?s waiting for sanitize", content):
        return "Sanitize Timeout"
    if "returncode 137" in content:
        return "Profiling Service Timeout"
    if (
        "Kernel execution timed out" in content
        or "Evaluation timeout after" in content
        or "memcheck timed out" in content
        or re.search(r"\]\s+TIMEOUT\b", content)
    ):
        return "Kernel Execution Timeout"
    if "Could not extract" in content:
        return "Extraction error"
    if "RUNTIME_ERROR" in content or "CUDA error" in content or "CU error" in content:
        return "Runtime error"
    return "Other error"


def extract_turn_sequence(traj: dict) -> list[str]:
    seq = []
    first_user = True
    for index, message in enumerate(traj.get("messages", [])):
        if message.get("role") != "user" or index == 0:
            continue
        if first_user:
            first_user = False
            continue
        seq.append(classify_turn(message.get("content", "")))
    return seq


def speedup_from_eval_message(eval_message: dict) -> float | None:
    extra = eval_message.get("extra") or {}
    speedups = []
    for trace in extra.get("traces") or []:
        speedup = nested_get(trace, ("evaluation", "performance", "speedup_factor"))
        value = as_number(speedup)
        if value is not None:
            speedups.append(value)
    if speedups:
        return max(speedups)
    return as_number(extra.get("min_speedup"))


def extract_turn_speedups(traj: dict) -> dict[int, float]:
    speedups = {}
    first_user = True
    fallback_turn = -1
    for index, message in enumerate(traj.get("messages", [])):
        if message.get("role") != "user" or index == 0:
            continue
        if first_user:
            first_user = False
            continue
        fallback_turn += 1
        turn = as_number(nested_get(message, ("extra", "rollout", "turn_idx")))
        turn = int(turn) if turn is not None else fallback_turn
        speedup = speedup_from_eval_message(message)
        if speedup is not None:
            speedups[turn] = speedup
    return speedups


def assistant_eval_turns(traj: dict) -> list[tuple[int, dict, dict]]:
    """Pair each assistant kernel response with its following evaluation message."""

    turns = []
    saw_initial_user = False
    pending_assistant = None
    for message in traj.get("messages", []):
        role = message.get("role")
        if role == "user":
            if not saw_initial_user:
                saw_initial_user = True
                continue
            if pending_assistant is not None:
                turns.append((len(turns), pending_assistant, message))
                pending_assistant = None
        elif role == "assistant" and saw_initial_user:
            pending_assistant = message
    return turns


def normalize_architecture(value: str) -> str:
    key = value.strip().lower()
    aliases = {
        "hopper": "hopper",
        "compute_90a": "hopper",
        "sm_90a": "hopper",
        "blackwell": "blackwell",
        "compute_100a": "blackwell",
        "comput_100a": "blackwell",
        "sm_100a": "blackwell",
    }
    if key not in aliases:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unsupported architecture {value!r}; expected one of: {supported}")
    return aliases[key]


def parse_static_sass(
    text: str,
    *,
    architecture: str,
    cubin_container_sha256: str = "",
) -> StaticSassEvidence:
    """Count selected executable instruction addresses in cubin SASS."""

    architecture = normalize_architecture(architecture)
    config = ARCHITECTURES[architecture]
    gmma_count = 0
    tma_count = 0
    tcgen_count = 0
    matched_lines = []
    lines = text.splitlines()
    for line in lines:
        match = STATIC_SASS_LINE_RE.match(line)
        if match is None:
            continue
        opcode = match.group("opcode").upper()
        matched_lines.append(line.rstrip())
        if "GMMA" in opcode:
            gmma_count += 1
        elif opcode.split(".", 1)[0] in TMA_TRANSFER_OPCODES:
            tma_count += 1
        if opcode.startswith(BLACKWELL_TCGEN_PREFIXES):
            tcgen_count += 1

    has_target = (gmma_count > 0 or tma_count > 0) if architecture == "hopper" else tcgen_count > 0
    return StaticSassEvidence(
        cubin_sass_arch_tag=config.tag if has_target else "",
        cubin_gmma_instruction_count=gmma_count,
        cubin_tma_instruction_count=tma_count,
        cubin_tcgen_instruction_count=tcgen_count,
        cubin_container_sha256=cubin_container_sha256,
        matched_lines=tuple(matched_lines),
        disassembly_line_count=len(lines),
    )


def parse_dynamic_sass(
    text: str,
    *,
    architecture: str,
    task_id: str = "",
) -> DynamicSassEvidence:
    """Parse selected SASS rows and require positive predicate-true execution."""

    architecture = normalize_architecture(architecture)
    config = ARCHITECTURES[architecture]
    gmma_count = 0
    tma_count = 0
    tcgen_count = 0
    gmma_pred_on_thread_count = 0
    tma_pred_on_thread_count = 0
    tcgen_pred_on_thread_count = 0
    matched_lines = []
    lines = text.splitlines()
    for line in lines:
        match = SASS_LINE_RE.match(line)
        if match is None:
            continue
        opcode = match.group("opcode").upper()
        count = int(match.group("count").replace(",", ""))
        pred_on_count = int(match.group("pred_on_count").replace(",", ""))
        matched_lines.append(line.rstrip())
        if "GMMA" in opcode:
            gmma_pred_on_thread_count += pred_on_count
            if pred_on_count > 0:
                gmma_count += count
        elif opcode.split(".", 1)[0] in TMA_TRANSFER_OPCODES:
            tma_pred_on_thread_count += pred_on_count
            if pred_on_count > 0:
                tma_count += count
        if opcode.startswith(BLACKWELL_TCGEN_PREFIXES):
            tcgen_pred_on_thread_count += pred_on_count
            if pred_on_count > 0:
                tcgen_count += count

    has_target = (gmma_count > 0 or tma_count > 0) if architecture == "hopper" else tcgen_count > 0
    return DynamicSassEvidence(
        sass_arch_tag=config.tag if has_target else "",
        gmma_count=gmma_count,
        tma_count=tma_count,
        gmma_pred_on_thread_count=gmma_pred_on_thread_count,
        tma_pred_on_thread_count=tma_pred_on_thread_count,
        matched_lines=tuple(matched_lines),
        profile_line_count=len(lines),
        task_id=task_id,
        tcgen_count=tcgen_count,
        tcgen_pred_on_thread_count=tcgen_pred_on_thread_count,
    )


def find_tvm_ffi_dir(explicit: Path | None = None) -> Path:
    """Resolve the TVM-FFI package root used by the CUDA evaluator."""

    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    configured = os.environ.get("TVM_FFI_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    spec = importlib.util.find_spec("tvm_ffi")
    if spec is not None and spec.submodule_search_locations:
        candidates.append(Path(next(iter(spec.submodule_search_locations))))
    candidates.append(Path("/usr/local/lib/python3.12/dist-packages/tvm_ffi"))

    for candidate in candidates:
        if (candidate / "include").is_dir() and (candidate / "lib").is_dir():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates) or "(none)"
    raise FileNotFoundError(f"could not locate TVM-FFI with include/ and lib/ directories; checked: {checked}")


def inspect_candidate_cubin(
    candidate: Candidate,
    *,
    architecture: str,
    tvm_ffi_dir: Path,
    nvcc_path: str = "nvcc",
    cuobjdump_path: str = "cuobjdump",
    compile_timeout: int = 180,
) -> StaticSassEvidence:
    """Build a candidate like the evaluator and inspect its embedded cubin."""

    architecture = normalize_architecture(architecture)
    config = ARCHITECTURES[architecture]
    if shutil.which(nvcc_path) is None:
        raise CubinInspectionError(f"nvcc executable not found: {nvcc_path}")
    if shutil.which(cuobjdump_path) is None:
        raise CubinInspectionError(f"cuobjdump executable not found: {cuobjdump_path}")

    with tempfile.TemporaryDirectory(prefix="ptxbench_sass_") as temp_dir:
        build_dir = Path(temp_dir)
        source_path = build_dir / "kernel.cu"
        binary_path = build_dir / "kernel.so"
        source_path.write_text(candidate.source)
        compile_command = [
            nvcc_path,
            "-shared",
            "-O3",
            "-gencode",
            config.gencode,
            str(source_path),
            "-lineinfo",
            "--ptxas-options=-v",
            "-Xcompiler",
            "-fPIC,-fvisibility=hidden",
            "-lcuda",
            f"-I{tvm_ffi_dir / 'include'}",
            "-std=c++17",
            f"-L{tvm_ffi_dir / 'lib'}",
            "-ltvm_ffi",
            "-o",
            str(binary_path),
        ]
        try:
            compiled = subprocess.run(
                compile_command,
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=compile_timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CubinInspectionError(f"nvcc failed: {exc}") from exc
        if compiled.returncode != 0 or not binary_path.is_file():
            detail = (compiled.stdout or "").strip()[-4000:]
            raise CubinInspectionError(f"nvcc exited with code {compiled.returncode}: {detail}")

        container_sha256 = hashlib.sha256(binary_path.read_bytes()).hexdigest()
        try:
            disassembled = subprocess.run(
                [cuobjdump_path, "--dump-sass", str(binary_path)],
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=compile_timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CubinInspectionError(f"cuobjdump failed: {exc}") from exc
        if disassembled.returncode != 0:
            detail = (disassembled.stdout or "").strip()[-4000:]
            raise CubinInspectionError(f"cuobjdump exited with code {disassembled.returncode}: {detail}")
        return parse_static_sass(
            disassembled.stdout or "",
            architecture=architecture,
            cubin_container_sha256=container_sha256,
        )


def build_solution(source: str, definition: str, architecture: str) -> dict:
    config = ARCHITECTURES[normalize_architecture(architecture)]
    return {
        "name": "ptxbench_sass_verification",
        "definition": definition,
        "spec": {
            "language": "cuda",
            "binding": "tvm-ffi",
            "target_hardware": list(config.target_hardware),
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "ptxbench",
        "sources": [{"path": "kernel.cu", "content": source}],
    }


def extract_profile_text(task: dict) -> str:
    reports = []
    for entry in task.get("logs") or []:
        if isinstance(entry, dict) and isinstance(entry.get("log"), str):
            reports.append(entry["log"])
    if reports:
        return "\n".join(reports)
    for trace in task.get("traces") or []:
        if not isinstance(trace, dict):
            continue
        evaluation = trace.get("evaluation") or {}
        if isinstance(evaluation, dict) and isinstance(evaluation.get("log"), str):
            reports.append(evaluation["log"])
    return "\n".join(reports)


def submit_and_poll_profile(
    *,
    base_url: str,
    source: str,
    definition: str,
    workload: str,
    architecture: str,
    server_timeout: int,
    poll_timeout: int,
    poll_interval: float,
    request_timeout: float,
    kernel_name: str | None,
) -> DynamicSassEvidence:
    """Profile a candidate and return dynamically executed SASS evidence."""

    payload: dict[str, object] = {
        "solution": build_solution(source, definition, architecture),
        "workload_uuids": [workload],
        "set": None,
        "sections": None,
        "metrics": ["inst_executed", "thread_inst_executed_true"],
        "page": "source",
        "timeout": server_timeout,
        "max_lines": None,
    }
    if kernel_name:
        payload["kernel_name"] = kernel_name

    url = base_url.rstrip("/")
    response: requests.Response | None = None
    try:
        response = requests.post(f"{url}/profile", json=payload, timeout=request_timeout)
        response.raise_for_status()
        task_id = response.json()["task_id"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        detail = f"; response={response.text.strip()[:1000]}" if response is not None else ""
        raise ProfileError(f"could not submit /profile request: {exc}{detail}") from exc

    print(f"  submitted profile task {task_id}", flush=True)
    deadline = time.monotonic() + poll_timeout
    task = None
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            response = requests.get(
                f"{url}/tasks/{task_id}?timeout=30",
                timeout=min(max(remaining, 1), max(request_timeout, 45)),
            )
            response.raise_for_status()
            task = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProfileError(f"could not poll profile task {task_id}: {exc}") from exc
        if task.get("status") in TERMINAL_TASK_STATUSES:
            break
        time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0)))
    else:
        raise ProfileError(f"profile task {task_id} did not finish within {poll_timeout} seconds")

    if task is None or task.get("status") != "completed":
        error = (task or {}).get("error") or "unknown server error"
        raise ProfileError(f"profile task {task_id} failed: {error}")
    report = extract_profile_text(task)
    if not report:
        raise ProfileError(f"profile task {task_id} completed without an NCU report")
    if report.lstrip().startswith("ERROR:"):
        raise ProfileError(f"profile task {task_id} returned:\n{report.strip()}")
    return parse_dynamic_sass(report, architecture=architecture, task_id=task_id)


def static_source_hash(source: str, architecture: str) -> str:
    material = (f"sass-cubin-v{STATIC_CACHE_SCHEMA_VERSION}\0{normalize_architecture(architecture)}\0{source}").encode()
    return hashlib.sha256(material).hexdigest()


def profile_source_hash(source: str, definition: str, workload: str, architecture: str) -> str:
    material = (
        f"sass-profile-v{PROFILE_CACHE_SCHEMA_VERSION}\0"
        f"{normalize_architecture(architecture)}\0{definition}\0{workload}\0{source}"
    ).encode()
    return hashlib.sha256(material).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temp_path.write_text(json.dumps(payload, indent=2) + "\n")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*BASE_FIELDS, *SASS_FIELDS], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_cached_static_evidence(path: Path) -> StaticSassEvidence:
    data = json.loads(path.read_text())
    if data.get("schema_version") != STATIC_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported static cache schema in {path}")
    evidence = data["evidence"]
    return StaticSassEvidence(
        cubin_sass_arch_tag=evidence["cubin_sass_arch_tag"],
        cubin_gmma_instruction_count=int(evidence["cubin_gmma_instruction_count"]),
        cubin_tma_instruction_count=int(evidence["cubin_tma_instruction_count"]),
        cubin_tcgen_instruction_count=int(evidence["cubin_tcgen_instruction_count"]),
        cubin_container_sha256=evidence.get("cubin_container_sha256", ""),
        matched_lines=tuple(evidence["matched_lines"]),
        disassembly_line_count=int(evidence["disassembly_line_count"]),
    )


def load_cached_dynamic_evidence(path: Path) -> DynamicSassEvidence:
    data = json.loads(path.read_text())
    if data.get("schema_version") != PROFILE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported profile cache schema in {path}")
    evidence = data["evidence"]
    return DynamicSassEvidence(
        sass_arch_tag=evidence["sass_arch_tag"],
        gmma_count=int(evidence["gmma_count"]),
        tma_count=int(evidence["tma_count"]),
        gmma_pred_on_thread_count=int(evidence["gmma_pred_on_thread_count"]),
        tma_pred_on_thread_count=int(evidence["tma_pred_on_thread_count"]),
        matched_lines=tuple(evidence["matched_lines"]),
        profile_line_count=int(evidence["profile_line_count"]),
        task_id=evidence.get("task_id", ""),
        tcgen_count=int(evidence.get("tcgen_count", 0)),
        tcgen_pred_on_thread_count=int(evidence.get("tcgen_pred_on_thread_count", 0)),
    )


def cache_static_evidence(path: Path, candidate: Candidate, architecture: str, evidence: StaticSassEvidence) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": STATIC_CACHE_SCHEMA_VERSION,
            "architecture": normalize_architecture(architecture),
            "source_sha256": candidate.source_sha256,
            "first_seen_at": {"trajectory_id": candidate.trajectory_id, "turn": candidate.turn},
            "evidence": asdict(evidence),
        },
    )


def cache_dynamic_evidence(
    path: Path,
    candidate: Candidate,
    architecture: str,
    definition: str,
    workload: str,
    evidence: DynamicSassEvidence,
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "architecture": normalize_architecture(architecture),
            "definition": definition,
            "workload": workload,
            "source_sha256": candidate.source_sha256,
            "first_seen_at": {"trajectory_id": candidate.trajectory_id, "turn": candidate.turn},
            "evidence": asdict(evidence),
        },
    )


def blank_sass_fields() -> dict[str, str]:
    return {field: "" for field in SASS_FIELDS}


def candidate_from_message(trajectory_id: str, turn: int, message: dict) -> Candidate | None:
    content = message.get("content", "")
    source = extract_code_block(
        content if isinstance(content, str) else "",
        languages=["cpp"],
        keep_separators=False,
    )
    if not source:
        return None
    return Candidate(
        trajectory_id=trajectory_id,
        turn=turn,
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def export_run(
    *,
    run_dir: Path,
    architecture: str,
    definition: str,
    workload: str,
    out_name: str,
    verify_sass: bool,
    num_parallel: int,
    num_compile_parallel: int,
    force_static: bool,
    force_profile: bool,
    continue_on_static_error: bool,
    continue_on_profile_error: bool,
    inspect_one: Callable[[Candidate], StaticSassEvidence],
    profile_one: Callable[[Candidate], DynamicSassEvidence],
) -> tuple[int, int]:
    """Build the canonical per-turn rows and attach native SASS evidence."""

    architecture = normalize_architecture(architecture)
    config = ARCHITECTURES[architecture]
    if num_parallel < 1 or num_compile_parallel < 1:
        raise ValueError("parallelism values must be positive")
    trajectory_dir = run_dir / "trajectories"
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"missing trajectories directory: {trajectory_dir}")

    rows: list[dict[str, object]] = []
    candidate_by_row: dict[tuple[str, int], Candidate] = {}
    unique_static_candidates: dict[str, Candidate] = {}
    correct_rows = 0
    for trajectory_path in sorted(trajectory_dir.glob("*.json")):
        trajectory_id = trajectory_path.stem
        trajectory = json.loads(trajectory_path.read_text())
        sequence = extract_turn_sequence(trajectory)
        speedups = extract_turn_speedups(trajectory)
        assistant_by_turn = {turn: assistant for turn, assistant, _evaluation in assistant_eval_turns(trajectory)}
        for turn, correctness in enumerate(sequence):
            row: dict[str, object] = {
                "trajectory_id": trajectory_id,
                "turn": turn,
                "correctness": correctness,
                "speedup": speedups.get(turn),
                **blank_sass_fields(),
            }
            rows.append(row)
            if correctness != "Correct":
                continue
            correct_rows += 1
            if not verify_sass:
                row["sass_verification_status"] = "not_requested"
                continue
            assistant = assistant_by_turn.get(turn)
            candidate = candidate_from_message(trajectory_id, turn, assistant or {})
            if candidate is None:
                row["sass_verification_status"] = "missing_source"
                continue
            row_key = (trajectory_id, turn)
            candidate_by_row[row_key] = candidate
            unique_static_candidates.setdefault(static_source_hash(candidate.source, architecture), candidate)

    if not verify_sass:
        write_csv_atomic(run_dir / "figures" / out_name, rows)
        return len(rows), correct_rows

    static_cache_dir = run_dir / "figures" / DEFAULT_STATIC_CACHE_DIR_NAME
    profile_cache_dir = run_dir / "figures" / DEFAULT_PROFILE_CACHE_DIR_NAME
    static_evidence: dict[str, StaticSassEvidence] = {}
    static_errors: set[str] = set()
    pending_static: dict[str, Candidate] = {}
    for key, candidate in unique_static_candidates.items():
        cache_path = static_cache_dir / f"{key}.json"
        if cache_path.is_file() and not force_static:
            static_evidence[key] = load_cached_static_evidence(cache_path)
        else:
            pending_static[key] = candidate

    def inspect_and_cache(key: str, candidate: Candidate) -> tuple[str, StaticSassEvidence]:
        evidence = inspect_one(candidate)
        cache_static_evidence(static_cache_dir / f"{key}.json", candidate, architecture, evidence)
        return key, evidence

    if pending_static:
        workers = min(num_compile_parallel, len(pending_static))
        print(
            f"{run_dir}: inspecting {len(pending_static)} unique cubins with num_compile_parallel={workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(inspect_and_cache, key, candidate): (key, candidate)
                for key, candidate in pending_static.items()
            }
            for future in as_completed(futures):
                key, candidate = futures[future]
                try:
                    result_key, evidence = future.result()
                except CubinInspectionError as exc:
                    if not continue_on_static_error:
                        raise
                    static_errors.add(key)
                    print(
                        f"  WARNING: {candidate.trajectory_id} turn {candidate.turn} cubin inspection failed: {exc}",
                        flush=True,
                    )
                    continue
                static_evidence[result_key] = evidence

    dynamic_evidence: dict[str, DynamicSassEvidence] = {}
    dynamic_errors: set[str] = set()
    pending_dynamic: dict[str, Candidate] = {}
    for static_key, candidate in unique_static_candidates.items():
        if static_key in static_errors:
            continue
        if static_evidence[static_key].cubin_sass_arch_tag != config.tag:
            continue
        profile_key = profile_source_hash(candidate.source, definition, workload, architecture)
        cache_path = profile_cache_dir / f"{profile_key}.json"
        if cache_path.is_file() and not force_profile:
            dynamic_evidence[profile_key] = load_cached_dynamic_evidence(cache_path)
        else:
            pending_dynamic[profile_key] = candidate

    def profile_and_cache(key: str, candidate: Candidate) -> tuple[str, DynamicSassEvidence]:
        evidence = profile_one(candidate)
        cache_dynamic_evidence(
            profile_cache_dir / f"{key}.json",
            candidate,
            architecture,
            definition,
            workload,
            evidence,
        )
        return key, evidence

    if pending_dynamic:
        workers = min(num_parallel, len(pending_dynamic))
        print(
            f"{run_dir}: profiling {len(pending_dynamic)} static-positive kernels with num_parallel={workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(profile_and_cache, key, candidate): (key, candidate)
                for key, candidate in pending_dynamic.items()
            }
            for future in as_completed(futures):
                key, candidate = futures[future]
                try:
                    result_key, evidence = future.result()
                except ProfileError as exc:
                    if not continue_on_profile_error:
                        raise
                    dynamic_errors.add(key)
                    print(
                        f"  WARNING: {candidate.trajectory_id} turn {candidate.turn} profiling failed: {exc}",
                        flush=True,
                    )
                    continue
                dynamic_evidence[result_key] = evidence

    rows_by_key = {(str(row["trajectory_id"]), int(row["turn"])): row for row in rows}
    for row_key, candidate in candidate_by_row.items():
        row = rows_by_key[row_key]
        static_key = static_source_hash(candidate.source, architecture)
        if static_key in static_errors:
            row["sass_verification_status"] = "cubin_inspection_error"
            continue
        cubin = static_evidence[static_key]
        row.update(
            {
                "cubin_sass_arch_tag": cubin.cubin_sass_arch_tag,
                "cubin_gmma_instruction_count": cubin.cubin_gmma_instruction_count,
                "cubin_tma_instruction_count": cubin.cubin_tma_instruction_count,
                "cubin_tcgen_instruction_count": cubin.cubin_tcgen_instruction_count,
                "cubin_container_sha256": cubin.cubin_container_sha256,
            }
        )
        if cubin.cubin_sass_arch_tag != config.tag:
            row["sass_verification_status"] = "cubin_sass_absent"
            continue
        profile_key = profile_source_hash(candidate.source, definition, workload, architecture)
        if profile_key in dynamic_errors:
            row["sass_verification_status"] = "profile_error"
            continue
        dynamic = dynamic_evidence[profile_key]
        row.update(
            {
                "sass_arch_tag": dynamic.sass_arch_tag,
                "sass_gmma_count": dynamic.gmma_count,
                "sass_tma_count": dynamic.tma_count,
                "sass_tcgen_count": dynamic.tcgen_count,
                "sass_gmma_thread_inst_executed_true": dynamic.gmma_pred_on_thread_count,
                "sass_tma_thread_inst_executed_true": dynamic.tma_pred_on_thread_count,
                "sass_tcgen_thread_inst_executed_true": dynamic.tcgen_pred_on_thread_count,
                "sass_profile_task_id": dynamic.task_id,
                "sass_verification_status": (
                    "dynamic_present" if dynamic.sass_arch_tag == config.tag else "dynamic_not_executed"
                ),
            }
        )

    write_csv_atomic(run_dir / "figures" / out_name, rows)
    return len(rows), correct_rows


def load_experiment_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    required = {"arch", "exp_dir"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return rows


def load_workload_migrations(path: Path | None) -> dict[tuple[str, str], tuple[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"old_definition", "old_workload", "new_definition", "new_workload"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        migrations = {}
        for line_number, row in enumerate(reader, start=2):
            old = (row["old_definition"].strip(), row["old_workload"].strip())
            new = (row["new_definition"].strip(), row["new_workload"].strip())
            if not all((*old, *new)):
                raise ValueError(f"incomplete migration at {path}:{line_number}")
            previous = migrations.get(old)
            if previous is not None and previous != new:
                raise ValueError(f"conflicting migrations for {old} in {path}")
            migrations[old] = new
    return migrations


def resolve_workload(
    definition: str,
    workload: str,
    migrations: dict[tuple[str, str], tuple[str, str]],
) -> tuple[str, str]:
    pair = (definition, workload)
    seen = set()
    while pair in migrations:
        if pair in seen:
            raise ValueError(f"workload migration cycle at {pair}")
        seen.add(pair)
        pair = migrations[pair]
    return pair


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        required=True,
        help="CSV with arch, definition, workload, and exp_dir columns",
    )
    parser.add_argument(
        "--base-url",
        help="FIBServe profiling base URL, required unless --skip-sass-verification is used",
    )
    parser.add_argument(
        "--out-name",
        default=DEFAULT_OUTPUT_NAME,
        help="output filename inside each exp_dir/figures directory",
    )
    parser.add_argument(
        "--migration-csv",
        type=Path,
        help="optional workload migration CSV",
    )
    parser.add_argument(
        "--skip-sass-verification",
        action="store_true",
        help="export correctness and speedup with blank SASS tags",
    )
    parser.add_argument("--force", action="store_true", help="regenerate existing output CSVs")
    parser.add_argument("--force-static", action="store_true", help="ignore cached cubin scans")
    parser.add_argument("--force-profile", action="store_true", help="ignore cached NCU profiles")
    parser.add_argument("--num-parallel", type=int, default=1)
    parser.add_argument("--num-compile-parallel", type=int, default=4)
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--poll-timeout", type=int, default=720)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--profile-retries", type=int, default=1)
    parser.add_argument("--tvm-ffi-dir", type=Path)
    parser.add_argument("--nvcc-path", default="nvcc")
    parser.add_argument("--cuobjdump-path", default="cuobjdump")
    parser.add_argument("--kernel-name")
    parser.add_argument("--continue-on-static-error", action="store_true")
    parser.add_argument("--continue-on-profile-error", action="store_true")
    args = parser.parse_args()
    if not args.skip_sass_verification and not args.base_url:
        parser.error("--base-url is required unless --skip-sass-verification is used")
    if args.num_parallel < 1 or args.num_compile_parallel < 1:
        parser.error("parallelism values must be positive")
    if args.profile_retries < 0:
        parser.error("--profile-retries must be non-negative")
    if args.compile_timeout < 1:
        parser.error("--compile-timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    migrations = load_workload_migrations(args.migration_csv)
    tvm_ffi_dir = None
    if not args.skip_sass_verification:
        tvm_ffi_dir = find_tvm_ffi_dir(args.tvm_ffi_dir)

    for experiment in load_experiment_rows(args.experiments_csv):
        run_dir = expand_path(experiment["exp_dir"])
        output_path = run_dir / "figures" / args.out_name
        if output_path.exists() and not args.force:
            with output_path.open(newline="") as handle:
                fieldnames = set(csv.DictReader(handle).fieldnames or [])
            required_output_fields = {"sass_arch_tag", "sass_verification_status"}
            if not required_output_fields.issubset(fieldnames):
                raise ValueError(f"{output_path} does not use the native SASS schema; rerun with --force")
            print(f"{run_dir}: skipped existing figures/{args.out_name}")
            continue

        architecture = normalize_architecture(experiment["arch"])
        definition = experiment.get("definition", "").strip()
        workload = experiment.get("workload", "").strip()
        if not args.skip_sass_verification and (not definition or not workload):
            raise ValueError(
                f"{args.experiments_csv}: SASS verification requires non-empty definition and workload columns"
            )
        definition, workload = resolve_workload(definition, workload, migrations)

        def inspect_one(
            candidate: Candidate,
            *,
            architecture: str = architecture,
        ) -> StaticSassEvidence:
            if tvm_ffi_dir is None:
                raise AssertionError("SASS inspection requested without TVM-FFI")
            return inspect_candidate_cubin(
                candidate,
                architecture=architecture,
                tvm_ffi_dir=tvm_ffi_dir,
                nvcc_path=args.nvcc_path,
                cuobjdump_path=args.cuobjdump_path,
                compile_timeout=args.compile_timeout,
            )

        def profile_one(
            candidate: Candidate,
            *,
            architecture: str = architecture,
            definition: str = definition,
            workload: str = workload,
        ) -> DynamicSassEvidence:
            for attempt in range(args.profile_retries + 1):
                try:
                    return submit_and_poll_profile(
                        base_url=args.base_url,
                        source=candidate.source,
                        definition=definition,
                        workload=workload,
                        architecture=architecture,
                        server_timeout=args.server_timeout,
                        poll_timeout=args.poll_timeout,
                        poll_interval=args.poll_interval,
                        request_timeout=args.request_timeout,
                        kernel_name=args.kernel_name,
                    )
                except ProfileError as exc:
                    if attempt >= args.profile_retries:
                        raise
                    print(
                        f"  WARNING: {candidate.trajectory_id} turn {candidate.turn} "
                        f"profile attempt {attempt + 1} failed: {exc}; retrying",
                        flush=True,
                    )
            raise AssertionError("unreachable")

        row_count, correct_count = export_run(
            run_dir=run_dir,
            architecture=architecture,
            definition=definition,
            workload=workload,
            out_name=args.out_name,
            verify_sass=not args.skip_sass_verification,
            num_parallel=args.num_parallel,
            num_compile_parallel=args.num_compile_parallel,
            force_static=args.force_static,
            force_profile=args.force_profile,
            continue_on_static_error=args.continue_on_static_error,
            continue_on_profile_error=args.continue_on_profile_error,
            inspect_one=inspect_one,
            profile_one=profile_one,
        )
        print(
            f"{run_dir}: wrote {row_count} rows ({correct_count} Correct) to figures/{args.out_name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
