"""CUDA debug helper for generated kernels."""

from __future__ import annotations

import errno
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Union

from flashinfer_bench.agents.sanitizer import (
    VALID_SANITIZER_TYPES,
    SanitizerType,
    _build_sanitizer_command,
)
from flashinfer_bench.data import Solution, TraceSet, Workload
from flashinfer_bench.utils import run_managed_subprocess

logger = logging.getLogger(__name__)

DEFAULT_COREDUMP_FLAGS = (
    "skip_nonrelocated_elf_images,"
    "skip_global_memory,"
    "skip_shared_memory,"
    "skip_local_memory,"
    "skip_constbank_memory"
)

_CUDA_SOURCE_LINE_PATTERNS = (
    re.compile(r"(?:^|\s)(?P<path>[\w./:-]*kernel\.cu)[:(](?P<line>\d+)\)?"),
    re.compile(r'\bat\s+(?P<path>[^:\s"]+\.cu):(?P<line>\d+)\b'),
    re.compile(r'File\s+"(?P<path>[^"]+)",\s+line\s+(?P<line>\d+)'),
)

_DIAGNOSTIC_PATTERNS = (
    re.compile(r"CUDA error[^\n]*", re.IGNORECASE),
    re.compile(r"CUDA Exception:[^\n]*", re.IGNORECASE),
    re.compile(
        r"Invalid (?!argument\" on CUDA API call to cuGetProcAddress_v2)[^\n]*", re.IGNORECASE
    ),
    re.compile(r"Race reported[^\n]*", re.IGNORECASE),
    re.compile(r"ERROR:[^\n]*", re.IGNORECASE),
    re.compile(r"WARNING:[^\n]*detected issues[^\n]*", re.IGNORECASE),
    re.compile(r".*timed out after \d+ seconds\.", re.IGNORECASE),
)

_IGNORED_DIAGNOSTIC_SNIPPETS = ("cuGetProcAddress_v2", "Variable environment CUDA_")

_SANITIZER_FAULT_HEADER_RE = re.compile(
    r"^=+\s+(?P<kind>(?:Invalid|Misaligned|Race|Uninitialized|Barrier|Warp|"
    r"Out-of-range|Stack overflow)[^\n]*)$",
    re.IGNORECASE,
)
_SANITIZER_LOCATION_RE = re.compile(r"\bin\s+(?P<path>[^:\s]+\.cu):(?P<line>\d+)\b")
_SANITIZER_THREAD_RE = re.compile(
    r"by thread\s+(?P<thread>\([^)]+\))\s+in block\s+(?P<block>\([^)]+\))"
)
_CUDA_ERROR_AT_LINE_RE = re.compile(
    r"CUDA error (?P<message>.*?) at (?P<path>[\w./:-]*kernel\.cu):(?P<line>\d+)", re.IGNORECASE
)
_CUDA_GDB_FRAME_RE = re.compile(
    r"^#(?P<index>\d+)\s+.*?\bin\s+(?P<function>.*?)\s+at\s+" r"(?P<path>[^:\s]+\.cu):(?P<line>\d+)"
)
_CUDA_GDB_INLINE_RE = re.compile(r"\binlined from (?P<path>[\w./:-]*kernel\.cu):(?P<line>\d+)")
_CUDA_GDB_FOCUS_RE = re.compile(r"Current focus set to (?P<focus>.*)", re.IGNORECASE)
_CUDA_GDB_DIAGNOSTIC_RE = re.compile(
    r"(CUDA Exception:[^\n]*|Program (?:received|terminated with) signal[^\n]*|"
    r"ERROR: cuda-gdb[^\n]*)",
    re.IGNORECASE,
)

DebugSection = dict[str, Union[bool, str]]
DebugMetadata = dict[str, DebugSection]
DebugResponse = dict[str, DebugMetadata]


def _metadata_response(message: str, *, exist: bool = False) -> DebugResponse:
    return {"metadata": {"FlashInfer CUDA debug report": {"exist": exist, "msg": message}}}


def _section(lines: list[str], *, exist: bool) -> DebugSection:
    return {"exist": exist, "msg": "\n".join(lines).rstrip()}


def _load_solution(solution: Union[Solution, str]) -> Union[Solution, str]:
    if not isinstance(solution, str):
        return solution
    path = Path(solution)
    if not path.exists():
        return f"ERROR: Solution file not found: {solution}"
    try:
        return Solution.model_validate_json(path.read_text())
    except Exception as e:
        return f"ERROR: Failed to parse solution file: {e}"


def _load_workload(workload: Union[Workload, str]) -> Union[Workload, str]:
    if not isinstance(workload, str):
        return workload
    path = Path(workload)
    if not path.exists():
        return f"ERROR: Workload file not found: {workload}"
    try:
        return Workload.model_validate_json(path.read_text())
    except Exception as e:
        return f"ERROR: Failed to parse workload file: {e}"


def extract_cuda_source_lines(log: str) -> list[tuple[str, int]]:
    """Return unique ``(path, line)`` CUDA source locations found in a debug log."""
    seen: set[tuple[str, int]] = set()
    locations: list[tuple[str, int]] = []
    for pattern in _CUDA_SOURCE_LINE_PATTERNS:
        for match in pattern.finditer(log):
            line = int(match.group("line"))
            path = match.group("path")
            key = (path, line)
            if key not in seen:
                seen.add(key)
                locations.append(key)
    return locations


def extract_sanitizer_faults(log: str) -> list[dict[str, object]]:
    """Extract high-signal compute-sanitizer fault blocks."""
    lines = log.splitlines()
    faults: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for index, line in enumerate(lines):
        match = _SANITIZER_FAULT_HEADER_RE.match(line.strip())
        if not match:
            continue

        fault: dict[str, object] = {"kind": match.group("kind").strip(), "frames": []}
        for detail in lines[index + 1 : index + 16]:
            stripped = detail.strip()
            if stripped.startswith("========= ") and _SANITIZER_FAULT_HEADER_RE.match(stripped):
                break

            if " by thread " in stripped or stripped.startswith("=========     by thread"):
                thread_match = _SANITIZER_THREAD_RE.search(stripped)
                if thread_match:
                    fault["thread"] = thread_match.group("thread")
                    fault["block"] = thread_match.group("block")
                continue

            if "Address " in stripped:
                fault["address"] = stripped.replace("========= ", "").strip()
                continue

            loc_match = _SANITIZER_LOCATION_RE.search(stripped)
            if loc_match:
                frame = {
                    "path": loc_match.group("path"),
                    "line": int(loc_match.group("line")),
                    "text": stripped.replace("========= ", "").strip(),
                    "is_device_frame": "Device Frame:" in stripped,
                }
                frames = fault["frames"]
                assert isinstance(frames, list)
                frames.append(frame)

        frames = fault["frames"]
        assert isinstance(frames, list)
        if not frames:
            continue
        key = (
            fault.get("kind"),
            tuple((frame["path"], frame["line"], frame["text"]) for frame in frames[:2]),
        )
        if key in seen:
            continue
        seen.add(key)
        faults.append(fault)
        if len(faults) >= 8:
            break
    return faults


def _condense_raw_output(raw_log: str, max_lines: int = 80) -> str:
    """Keep the raw section readable while preserving high-signal tool lines."""
    lines = raw_log.splitlines()
    keep: list[str] = []
    index = 0
    kept_fault_block = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        fault_match = _SANITIZER_FAULT_HEADER_RE.match(stripped)
        if fault_match:
            if kept_fault_block:
                index += 1
                continue
            kept_fault_block = True
            keep.append(line)
            kept_thread = False
            kept_device_frame = False
            for detail in lines[index + 1 : index + 16]:
                if _SANITIZER_FAULT_HEADER_RE.match(detail.strip()):
                    break
                if " in kernel.cu:" in detail and "Device Frame:" not in detail:
                    keep.append(detail)
                    continue
                if " by thread " in detail and not kept_thread:
                    keep.append(detail)
                    kept_thread = True
                    continue
                if (
                    "Device Frame:" in detail
                    and " in kernel.cu:" in detail
                    and not kept_device_frame
                ):
                    keep.append(detail)
                    kept_device_frame = True
                    continue
                if "Address " in detail:
                    keep.append(detail)
            index += 1
            continue

        should_keep = (
            "COMPUTE-SANITIZER" in line
            or "CUDA coredump backtrace" in line
            or "Opening GPU coredump" in line
            or "Current focus set to CUDA kernel" in line
            or "Running " in line
            or "Return code:" in line
            or "ERROR SUMMARY" in line
            or "WARNING:" in line
            or "ERROR:" in line
            or "CUDA error" in line
            or line.startswith("#")
        )
        if should_keep:
            keep.append(line)
            index += 1
            continue
        index += 1

    condensed: list[str] = []
    seen = set()
    for line in keep:
        if line in seen:
            continue
        seen.add(line)
        condensed.append(line)
        if len(condensed) >= max_lines:
            condensed.append(f"[condensed raw output truncated at {max_lines} lines]")
            break

    return "\n".join(condensed) if condensed else "[no high-signal raw output lines found]"


def _entry_source(solution: Solution) -> tuple[str, list[str]]:
    source = solution.get_entry_source()
    if source is None:
        return solution.spec.entry_point.split("::")[0], []
    return source.path, source.content.splitlines()


def infer_cuda_api_faults(log: str, source_lines: list[str]) -> list[dict[str, object]]:
    """Infer the likely launch line for CUDA API errors reported at check sites."""
    faults: list[dict[str, object]] = []
    seen: set[tuple[int, Optional[int], str]] = set()
    for match in _CUDA_ERROR_AT_LINE_RE.finditer(log):
        line_no = int(match.group("line"))
        message = match.group("message").strip()
        launch_line: Optional[int] = None
        if 1 <= line_no <= len(source_lines):
            check_line = source_lines[line_no - 1]
            if "cudaGetLastError" in check_line or "cudaStreamSynchronize" in check_line:
                for candidate in range(line_no - 1, max(0, line_no - 12), -1):
                    text = source_lines[candidate - 1].strip()
                    if not text:
                        continue
                    if "<<<" in text or "cudaLaunch" in text:
                        launch_line = candidate
                        break
        key = (line_no, launch_line, message)
        if key in seen:
            continue
        seen.add(key)
        faults.append({"message": message, "reported_line": line_no, "launch_line": launch_line})
    return faults


def extract_cuda_gdb_insights(log: str) -> dict[str, object]:
    """Extract a compact source-oriented summary from a cuda-gdb coredump backtrace."""
    diagnostics: list[str] = []
    focuses: list[str] = []
    frames: list[dict[str, object]] = []
    seen_frames: set[tuple[int, str, int]] = set()

    for line in log.splitlines():
        stripped = line.strip()
        diagnostic_match = _CUDA_GDB_DIAGNOSTIC_RE.search(stripped)
        if diagnostic_match:
            diagnostic = diagnostic_match.group(1).strip()
            if diagnostic not in diagnostics:
                diagnostics.append(diagnostic)

        focus_match = _CUDA_GDB_FOCUS_RE.search(stripped)
        if focus_match:
            focus = focus_match.group("focus").strip()
            if focus and focus not in focuses:
                focuses.append(focus)

        frame_match = _CUDA_GDB_FRAME_RE.match(stripped)
        if not frame_match:
            continue
        index = int(frame_match.group("index"))
        path = frame_match.group("path")
        line_no = int(frame_match.group("line"))
        key = (index, path, line_no)
        if key in seen_frames:
            continue
        seen_frames.add(key)
        frames.append(
            {
                "index": index,
                "function": re.sub(r"\s+", " ", frame_match.group("function")).strip(),
                "path": path,
                "line": line_no,
                "text": stripped,
            }
        )
        inline_match = _CUDA_GDB_INLINE_RE.search(stripped)
        if inline_match:
            frames[-1]["inline_path"] = inline_match.group("path")
            frames[-1]["inline_line"] = int(inline_match.group("line"))
        if len(frames) >= 8:
            break

    return {"diagnostics": diagnostics, "focuses": focuses, "frames": frames}


def _source_excerpt(lines: list[str], line_no: int, context_lines: int) -> str:
    if line_no < 1 or line_no > len(lines):
        return f"  line {line_no}: outside available source range 1-{len(lines)}"

    start = max(1, line_no - context_lines)
    end = min(len(lines), line_no + context_lines)
    width = len(str(end))
    rendered = []
    for current in range(start, end + 1):
        marker = ">" if current == line_no else " "
        rendered.append(f"{marker} {current:{width}d}: {lines[current - 1]}")
    return "\n".join(rendered)


def _line_text(lines: list[str], line_no: int) -> str:
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def format_debug_metadata(
    *,
    solution: Solution,
    workload: Workload,
    device: str,
    raw_log: str,
    debug_dir: Path,
    coredumps: Iterable[Path],
    source_context_lines: int = 4,
) -> DebugMetadata:
    """Create structured debug metadata from sanitizer/coredump output."""
    entry_path, entry_lines = _entry_source(solution)
    faults = extract_sanitizer_faults(raw_log)
    api_faults = infer_cuda_api_faults(raw_log, entry_lines)
    coredump_insights = extract_cuda_gdb_insights(raw_log)
    locations = extract_cuda_source_lines(raw_log)
    diagnostics: list[str] = []
    for pattern in _DIAGNOSTIC_PATTERNS:
        for match in pattern.finditer(raw_log):
            text = match.group(0).strip()
            if any(snippet in text for snippet in _IGNORED_DIAGNOSTIC_SNIPPETS):
                continue
            if text and text not in diagnostics:
                diagnostics.append(text)

    metadata: DebugMetadata = {}
    metadata["FlashInfer CUDA debug report"] = _section(
        [
            f"solution: {solution.name}",
            f"definition: {solution.definition}",
            f"workload: {workload.uuid} axes={workload.axes}",
            f"device: {device}",
            f"debug_dir: {debug_dir}",
        ],
        exist=True,
    )

    section: list[str] = []
    if faults:
        for fault in faults[:3]:
            section.append(f"- {fault['kind']}")
            if fault.get("thread") and fault.get("block"):
                section.append(f"  thread {fault['thread']} in block {fault['block']}")
            if fault.get("address"):
                section.append(f"  {fault['address']}")
            frames = fault["frames"]
            assert isinstance(frames, list)
            for frame in frames[:3]:
                label = "device frame" if frame["is_device_frame"] else "faulting instruction"
                section.append(f"  {label}: {entry_path}:{frame['line']}")
                section.append(f"    {frame['text']}")
            if len(frames) >= 2:
                first = frames[0]
                second = frames[1]
                section.append(
                    "  interpretation: the wrapper/synchronize line may only report the "
                    f"asynchronous CUDA failure; inspect {entry_path}:{second['line']} "
                    f"and the helper at {entry_path}:{first['line']} first."
                )
    elif api_faults:
        for fault in api_faults[:3]:
            section.append(f"- CUDA API reported: {fault['message']}")
            if fault.get("launch_line"):
                section.append(f"  likely failing launch: {entry_path}:{fault['launch_line']}")
                section.append(
                    "    CUDA reports many kernel faults asynchronously at the next "
                    "cudaGetLastError/cudaStreamSynchronize; inspect this launch first."
                )
            section.append(f"  reported at check site: {entry_path}:{fault['reported_line']}")
    else:
        section.append("- No compute-sanitizer device fault block was found.")
    metadata["Most precise CUDA fault"] = _section(section, exist=bool(faults or api_faults))

    section = []
    coredump_frames = coredump_insights["frames"]
    assert isinstance(coredump_frames, list)
    coredump_diagnostics = coredump_insights["diagnostics"]
    assert isinstance(coredump_diagnostics, list)
    coredump_focuses = coredump_insights["focuses"]
    assert isinstance(coredump_focuses, list)
    if coredump_diagnostics:
        section.extend(f"- {line}" for line in coredump_diagnostics[:4])
    if coredump_focuses:
        section.extend(f"- focus: {line}" for line in coredump_focuses[:2])
    if coredump_frames:
        wait_like_frames: list[dict[str, object]] = []
        for frame in coredump_frames[:4]:
            line_no = int(frame["line"])
            inline_line = frame.get("inline_line")
            function = str(frame["function"])
            source_line = _line_text(entry_lines, line_no)
            section.append(f"- frame #{frame['index']}: {entry_path}:{line_no}")
            if function:
                section.append(f"  function: {function}")
            if source_line:
                section.append(f"  source: {source_line}")
            else:
                section.append(f"  raw: {frame['text']}")
            if isinstance(inline_line, int) and inline_line != line_no:
                inline_source_line = _line_text(entry_lines, inline_line)
                section.append(f"  inlined call site: {entry_path}:{inline_line}")
                if inline_source_line:
                    section.append(f"  call source: {inline_source_line}")
            haystack = f"{function} {source_line} {frame['text']}".lower()
            if any(
                token in haystack
                for token in ("wait", "barrier", "mbarrier", "syncthreads", "spin", "poll")
            ):
                wait_like_frames.append(frame)
        if wait_like_frames:
            frame = wait_like_frames[0]
            inspect_line = frame.get("inline_line", frame["line"])
            section.append(
                "  interpretation: the coredump caught the GPU in a wait/barrier path. "
                f"For timeout failures, inspect {entry_path}:{inspect_line} and its caller "
                "before treating this as a generic CUDA core dump."
            )
        elif "timed out after" in raw_log:
            first = coredump_frames[0]
            section.append(
                "  interpretation: this is a timeout coredump; the top CUDA frames show "
                f"where the kernel was executing when it was interrupted. Start at "
                f"{entry_path}:{first['line']}."
            )
    metadata["CUDA core dump analysis"] = _section(
        section if section else ["- cuda-gdb produced no source-level backtrace."],
        exist=bool(section),
    )

    section = []
    if diagnostics:
        section.extend(f"- {line}" for line in diagnostics[:12])
    else:
        section.append("- No high-signal CUDA diagnostic was found in the tool output.")
    metadata["Primary diagnostics"] = _section(section, exist=bool(diagnostics))

    section = []
    if locations and entry_lines:
        for raw_path, line_no in locations[:8]:
            section.append(f"{entry_path}:{line_no}")
            section.append(_source_excerpt(entry_lines, line_no, source_context_lines))
            section.append("")
    elif locations:
        for raw_path, line_no in locations[:8]:
            section.append(f"- {raw_path}:{line_no} (entry source {entry_path!r} unavailable)")
    else:
        section.append(
            "- No source line was reported. For hangs, inspect any CUDA core dump below."
        )
    metadata["Source locations"] = _section(section, exist=bool(locations))

    coredump_paths = [p for p in coredumps if p.exists()]
    section = []
    if coredump_paths:
        for path in coredump_paths:
            section.append(f"- {path}")
        if coredump_frames:
            section.append("- Source-level coredump analysis is reported above.")
        elif section:
            section.append(
                "Automatic cuda-gdb analysis did not produce a source-level frame; "
                "the dump file is retained for manual follow-up."
            )
    else:
        section.append("- No CUDA core dump file was produced.")
    metadata["CUDA core dumps"] = _section(
        section, exist=bool(coredump_paths and not coredump_frames)
    )
    return metadata


def format_debug_report(
    *,
    solution: Solution,
    workload: Workload,
    device: str,
    raw_log: str,
    debug_dir: Path,
    coredumps: Iterable[Path],
    source_context_lines: int = 4,
) -> str:
    """Create the user-facing debug message from sanitizer/coredump output."""
    metadata = format_debug_metadata(
        solution=solution,
        workload=workload,
        device=device,
        raw_log=raw_log,
        debug_dir=debug_dir,
        coredumps=coredumps,
        source_context_lines=source_context_lines,
    )
    report: list[str] = []
    for title, section in metadata.items():
        report.append(title)
        report.append("=" * 80 if title == "FlashInfer CUDA debug report" else "-" * 80)
        msg = str(section["msg"])
        if msg:
            report.append(msg)
        report.append("")
    return "\n".join(report) + "\n"


def _truncate_metadata(metadata: DebugMetadata, max_lines: int) -> DebugMetadata:
    remaining = max_lines
    truncated = False
    result: DebugMetadata = {}
    for title, section in metadata.items():
        msg_lines = str(section["msg"]).splitlines()
        if remaining <= 0:
            result[title] = {"exist": section["exist"], "msg": "[Output truncated]"}
            truncated = True
            continue
        if len(msg_lines) > remaining:
            omitted = len(msg_lines) - remaining
            result[title] = {
                "exist": section["exist"],
                "msg": "\n".join(msg_lines[:remaining])
                + f"\n\n[Output truncated: {omitted} more lines]",
            }
            remaining = 0
            truncated = True
            continue
        result[title] = section
        remaining -= len(msg_lines)
    return result if truncated else metadata


def _trigger_coredump(coredump_dir: Path) -> None:
    pipes = sorted(coredump_dir.glob("cuda_coredump_pipe_*"))
    if not pipes:
        logger.warning("No CUDA coredump pipe found in %s", coredump_dir)
        return
    payload = b"\0" * (1024 * 1024)
    for pipe in pipes:
        try:
            fd = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            if e.errno not in (errno.ENXIO, errno.ENOENT):
                logger.warning("Failed to open CUDA coredump pipe %s: %s", pipe, e)
            continue
        try:
            os.write(fd, payload)
            logger.info("Triggered CUDA core dump through %s", pipe)
        except OSError as e:
            logger.warning("Failed to write CUDA coredump pipe %s: %s", pipe, e)
        finally:
            os.close(fd)


def _find_coredumps(debug_dir: Path) -> list[Path]:
    return sorted(p for p in debug_dir.glob("cuda_coredump_*") if "pipe" not in p.name)


def _build_direct_runner_command(
    data_dir: Path, device: str, trace_set_path: Optional[Path]
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "flashinfer_bench.agents._solution_runner",
        "--data-dir",
        str(data_dir),
        "--device",
        device,
    ]
    if trace_set_path:
        cmd.extend(["--trace-set-path", str(trace_set_path)])
    return cmd


def _run_cuda_gdb_backtrace(coredump: Path, timeout: int = 30) -> str:
    cuda_gdb = shutil.which("cuda-gdb")
    if cuda_gdb is None:
        return "ERROR: cuda-gdb not found; cannot inspect CUDA coredump automatically.\n"
    try:
        result = subprocess.run(
            [cuda_gdb, "-batch", "-ex", f"target cudacore {coredump}", "-ex", "bt"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = e.stderr if isinstance(e.stderr, str) else ""
        return (
            f"ERROR: cuda-gdb timed out after {timeout} seconds while reading {coredump}.\n"
            f"{stdout}{stderr}"
        )
    return result.stdout + result.stderr


def flashinfer_bench_debug_solution(
    solution: Union[Solution, str],
    workload: Union[Workload, str],
    *,
    device: str = "cuda:0",
    trace_set_path: Optional[str] = None,
    sanitizer_types: Optional[List[SanitizerType]] = None,
    sanitizer_path: str = "compute-sanitizer",
    timeout: int = 120,
    evaluation_timeout: Optional[int] = None,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = None,
    print_limit: Optional[int] = 100,
    source_context_lines: int = 4,
    enable_coredump: bool = True,
    coredump_grace_seconds: float = 30,
) -> DebugResponse:
    """Run a generated kernel under CUDA debug tooling and return structured metadata.

    The fast path uses compute-sanitizer with line-info-enabled builds and maps reported
    ``kernel.cu:<line>`` locations back to the submitted source. For hangs/timeouts,
    it enables CUDA user-triggered core dumps and writes to the generated pipe before
    killing the process group, leaving a dump path in the report for cuda-gdb/nvdisasm.
    """
    loaded_solution = _load_solution(solution)
    if isinstance(loaded_solution, str):
        return _metadata_response(loaded_solution)
    solution = loaded_solution

    loaded_workload = _load_workload(workload)
    if isinstance(loaded_workload, str):
        return _metadata_response(loaded_workload)
    workload = loaded_workload

    if sanitizer_types is None:
        sanitizer_types = ["memcheck"]

    if isinstance(sanitizer_types, str):
        return _metadata_response("ERROR: sanitizer_types must be a list, not a string")
    invalid_types = [st for st in sanitizer_types if st not in VALID_SANITIZER_TYPES]
    if invalid_types:
        return _metadata_response(
            f"ERROR: Invalid sanitizer type(s) {invalid_types}. "
            f"Must be one of: {sorted(VALID_SANITIZER_TYPES)}"
        )

    try:
        trace_set = TraceSet.from_path(trace_set_path)
    except Exception as e:
        return _metadata_response(f"ERROR: Failed to load trace set: {e}")

    if solution.definition not in trace_set.definitions:
        return _metadata_response(
            f"ERROR: Definition '{solution.definition}' not found in trace database. "
            f"Available definitions: {list(trace_set.definitions.keys())}"
        )
    definition = trace_set.definitions[solution.definition]

    if shutil.which(sanitizer_path) is None:
        return _metadata_response(
            f"ERROR: compute-sanitizer executable not found at '{sanitizer_path}'. "
            "Please install NVIDIA CUDA toolkit."
        )

    debug_parent = Path(tmpdir or os.environ.get("FIB_DEBUG_DIR", "/tmp/fib_debug"))
    debug_parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(tempfile.mkdtemp(prefix="run_", dir=debug_parent))

    (debug_dir / "definition.json").write_text(definition.model_dump_json())
    (debug_dir / "solution.json").write_text(solution.model_dump_json())
    (debug_dir / "workload.json").write_text(workload.model_dump_json())
    entry_source = solution.get_entry_source()
    if entry_source is not None:
        entry_path = debug_dir / entry_source.path
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(entry_source.content)

    env = os.environ.copy()
    env["TMPDIR"] = str(debug_dir)
    # Keep NVCC line information even if a future builder path misses its own flags.
    env["NVCC_PREPEND_FLAGS"] = f"{env.get('NVCC_PREPEND_FLAGS', '')} -lineinfo".strip()
    if enable_coredump:
        env.update(
            {
                "CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1",
                "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                "CUDA_COREDUMP_SHOW_PROGRESS": "1",
                "CUDA_COREDUMP_GENERATION_FLAGS": DEFAULT_COREDUMP_FLAGS,
                "CUDA_COREDUMP_FILE": str(debug_dir / "cuda_coredump_%h.%p.%t"),
                "CUDA_COREDUMP_PIPE": str(debug_dir / "cuda_coredump_pipe_%h.%p.%t"),
            }
        )

    raw_output = ""
    sanitizer_timed_out = False
    for sanitizer_type in sanitizer_types:
        raw_output += f"\n{'=' * 60}\n"
        raw_output += f"Running {sanitizer_type.upper()} debug pass\n"
        raw_output += f"{'=' * 60}\n\n"
        cmd = _build_sanitizer_command(
            sanitizer_type,
            debug_dir,
            device,
            Path(trace_set_path) if trace_set_path else None,
            sanitizer_path,
            print_limit=print_limit,
        )
        logger.info("FIBServe debug: Running Command: %s", " ".join(cmd))
        try:
            result = run_managed_subprocess(
                cmd,
                timeout=timeout,
                env=env,
                on_timeout=lambda _proc: _trigger_coredump(debug_dir) if enable_coredump else None,
                timeout_grace_seconds=coredump_grace_seconds if enable_coredump else 0,
            )
            raw_output += f"STDOUT:\n{result.stdout}\n\n"
            if result.stderr:
                raw_output += f"STDERR:\n{result.stderr}\n\n"
            raw_output += f"Return code: {result.returncode}\n"
            if result.returncode != 0:
                raw_output += f"\nWARNING: {sanitizer_type} detected issues!\n"
                break
            else:
                raw_output += f"\n{sanitizer_type} passed successfully.\n"
        except subprocess.TimeoutExpired as e:
            sanitizer_timed_out = True
            stdout = e.stdout if isinstance(e.stdout, str) else ""
            stderr = e.stderr if isinstance(e.stderr, str) else ""
            raw_output += f"STDOUT:\n{stdout}\n\n"
            if stderr:
                raw_output += f"STDERR:\n{stderr}\n\n"
            raw_output += f"ERROR: {sanitizer_type} timed out after {timeout} seconds.\n"
        except Exception as e:
            raw_output += f"ERROR: Failed to run {sanitizer_type}: {e}\n"

    if sanitizer_timed_out and enable_coredump and not _find_coredumps(debug_dir):
        direct_timeout = evaluation_timeout if evaluation_timeout is not None else timeout
        raw_output += f"\n{'=' * 60}\n"
        raw_output += "Running DIRECT COREDUMP timeout pass\n"
        raw_output += f"{'=' * 60}\n\n"
        cmd = _build_direct_runner_command(
            debug_dir, device, Path(trace_set_path) if trace_set_path else None
        )
        logger.info("FIBServe debug: Running direct coredump command: %s", " ".join(cmd))
        try:
            result = run_managed_subprocess(
                cmd,
                timeout=direct_timeout,
                env=env,
                on_timeout=lambda _proc: _trigger_coredump(debug_dir),
                timeout_grace_seconds=coredump_grace_seconds,
            )
            raw_output += f"STDOUT:\n{result.stdout}\n\n"
            if result.stderr:
                raw_output += f"STDERR:\n{result.stderr}\n\n"
            raw_output += f"Return code: {result.returncode}\n"
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout if isinstance(e.stdout, str) else ""
            stderr = e.stderr if isinstance(e.stderr, str) else ""
            raw_output += f"STDOUT:\n{stdout}\n\n"
            if stderr:
                raw_output += f"STDERR:\n{stderr}\n\n"
            raw_output += (
                f"ERROR: direct runner timed out after {direct_timeout} seconds; "
                "requested CUDA user-triggered core dump before terminating it.\n"
            )
        except Exception as e:
            raw_output += f"ERROR: Failed to run direct coredump pass: {e}\n"

    coredumps = _find_coredumps(debug_dir)
    if coredumps:
        raw_output += f"\n{'=' * 60}\n"
        raw_output += "CUDA coredump backtrace\n"
        raw_output += f"{'=' * 60}\n\n"
        raw_output += _run_cuda_gdb_backtrace(coredumps[0])
    (debug_dir / "debug_raw.log").write_text(raw_output)
    metadata = format_debug_metadata(
        solution=solution,
        workload=workload,
        device=device,
        raw_log=raw_output,
        debug_dir=debug_dir,
        coredumps=coredumps,
        source_context_lines=source_context_lines,
    )
    if max_lines is not None:
        metadata = _truncate_metadata(metadata, max_lines)
    return {"metadata": metadata}


def concise_debug_message(report: str, width: int = 1000) -> str:
    """Return a compact single-string preview for logs/UIs."""
    return textwrap.shorten(" ".join(report.split()), width=width, placeholder=" ...")
