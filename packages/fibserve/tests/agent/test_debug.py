"""Unit tests for CUDA debug report formatting."""

import subprocess
from pathlib import Path

import flashinfer_bench.agents.debug as debug_module
from flashinfer_bench.agents.debug import (
    extract_cuda_gdb_insights,
    extract_cuda_source_lines,
    extract_sanitizer_faults,
    flashinfer_bench_debug_solution,
    format_debug_metadata,
    format_debug_report,
    infer_cuda_api_faults,
)
from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages, Workload


def _solution() -> Solution:
    source = "\n".join(
        [
            "#include <cuda_runtime.h>",
            "namespace test {",
            "__global__ void kernel(float* x) {",
            "  x[threadIdx.x] = 1.0f;",
            "}",
            "void run() {}",
            "}",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, test::run);",
        ]
    )
    return Solution(
        name="bad",
        definition="test_def",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cuda"],
            entry_point="kernel.cu::run",
            binding="tvm-ffi",
        ),
        sources=[SourceFile(path="kernel.cu", content=source)],
    )


def _launch_error_solution() -> Solution:
    source = "\n".join(
        [
            "#include <cuda_runtime.h>",
            "#define CUDA_CHECK(x) do { auto e = (x); if (e) return; } while(0)",
            "__global__ void bad_kernel(float* x) { x[threadIdx.x] = 1.0f; }",
            "void run(float* x) {",
            "  bad_kernel<<<1, 128>>>(x);",
            "  CUDA_CHECK(cudaGetLastError());",
            "}",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);",
        ]
    )
    return Solution(
        name="bad_launch",
        definition="test_def",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cuda"],
            entry_point="kernel.cu::run",
            binding="tvm-ffi",
        ),
        sources=[SourceFile(path="kernel.cu", content=source)],
    )


def _barrier_timeout_solution() -> Solution:
    source = "\n".join(
        [
            "#include <cuda_runtime.h>",
            "__device__ void mbarrier_try_wait_parity() {",
            '  asm volatile("WAIT_%=:\\n");',
            "}",
            "__global__ void timeout_kernel(float* x) {",
            "  mbarrier_try_wait_parity();",
            "}",
            "void run(float* x) { timeout_kernel<<<1, 128>>>(x); }",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);",
        ]
    )
    return Solution(
        name="timeout",
        definition="test_def",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=["cuda"],
            entry_point="kernel.cu::run",
            binding="tvm-ffi",
        ),
        sources=[SourceFile(path="kernel.cu", content=source)],
    )


def _patch_debug_runtime(monkeypatch, solution: Solution):
    definition = type("DefinitionStub", (), {"model_dump_json": lambda self: "{}"})()
    trace_set = type("TraceSetStub", (), {"definitions": {solution.definition: definition}})()
    monkeypatch.setattr(debug_module.TraceSet, "from_path", staticmethod(lambda _path: trace_set))
    monkeypatch.setattr(debug_module.shutil, "which", lambda _path: "/usr/bin/tool")
    monkeypatch.setattr(debug_module, "_run_cuda_gdb_backtrace", lambda _path: "fake cuda-gdb bt\n")


def test_debug_reuses_sanitizer_timeout_coredump_without_direct_replay(monkeypatch, tmp_path):
    solution = _barrier_timeout_solution()
    _patch_debug_runtime(monkeypatch, solution)
    calls = []

    def fake_trigger_coredump(debug_dir: Path) -> None:
        (debug_dir / "cuda_coredump_test").write_text("fake")

    def fake_run_managed_subprocess(cmd, *, timeout, env, on_timeout=None, timeout_grace_seconds=0):
        calls.append((cmd, timeout))
        if on_timeout is not None:
            on_timeout(None)
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(debug_module, "_trigger_coredump", fake_trigger_coredump)
    monkeypatch.setattr(debug_module, "run_managed_subprocess", fake_run_managed_subprocess)

    result = flashinfer_bench_debug_solution(
        solution,
        Workload(uuid="abc", axes={"N": 1}, inputs={}),
        sanitizer_types=["memcheck"],
        timeout=12,
        evaluation_timeout=99,
        tmpdir=str(tmp_path),
    )

    assert len(calls) == 1
    assert calls[0][1] == 12
    raw_log = next(tmp_path.glob("run_*/debug_raw.log")).read_text()
    assert "memcheck timed out after 12 seconds" in raw_log
    assert "DIRECT COREDUMP timeout pass" not in raw_log


def test_debug_direct_coredump_fallback_uses_evaluation_timeout(monkeypatch, tmp_path):
    solution = _barrier_timeout_solution()
    _patch_debug_runtime(monkeypatch, solution)
    calls = []

    def fake_run_managed_subprocess(cmd, *, timeout, env, on_timeout=None, timeout_grace_seconds=0):
        calls.append((cmd, timeout))
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(debug_module, "run_managed_subprocess", fake_run_managed_subprocess)

    result = flashinfer_bench_debug_solution(
        solution,
        Workload(uuid="abc", axes={"N": 1}, inputs={}),
        sanitizer_types=["memcheck"],
        timeout=12,
        evaluation_timeout=99,
        tmpdir=str(tmp_path),
    )

    assert [timeout for _, timeout in calls] == [12, 99]
    raw_log = next(tmp_path.glob("run_*/debug_raw.log")).read_text()
    assert "memcheck timed out after 12 seconds" in raw_log
    assert "DIRECT COREDUMP timeout pass" in raw_log
    assert "direct runner timed out after 99 seconds" in raw_log


def test_debug_stops_after_first_sanitizer_fault(monkeypatch, tmp_path):
    solution = _barrier_timeout_solution()
    _patch_debug_runtime(monkeypatch, solution)
    calls = []

    def fake_run_managed_subprocess(cmd, *, timeout, env, on_timeout=None, timeout_grace_seconds=0):
        calls.append((cmd, timeout))
        return subprocess.CompletedProcess(cmd, 1, stdout="fault", stderr="")

    monkeypatch.setattr(debug_module, "run_managed_subprocess", fake_run_managed_subprocess)

    flashinfer_bench_debug_solution(
        solution,
        Workload(uuid="abc", axes={"N": 1}, inputs={}),
        sanitizer_types=["memcheck", "racecheck"],
        timeout=12,
        evaluation_timeout=99,
        tmpdir=str(tmp_path),
    )

    assert len(calls) == 1
    raw_log = next(tmp_path.glob("run_*/debug_raw.log")).read_text()
    assert "memcheck detected issues" in raw_log
    assert "Running RACECHECK" not in raw_log
    assert "DIRECT COREDUMP timeout pass" not in raw_log


def test_extract_cuda_source_lines_deduplicates_locations():
    log = (
        "CUDA error an illegal memory access was encountered at "
        "/tmp/cache/kernel.cu:4\n"
        "========= at /tmp/cache/kernel.cu:4 in kernel()\n"
        "kernel.cu(6): warning #550-D\n"
    )
    assert extract_cuda_source_lines(log) == [("/tmp/cache/kernel.cu", 4), ("kernel.cu", 6)]


def test_extract_cuda_gdb_insights_finds_frames_and_focus():
    log = "\n".join(
        [
            "CUDA coredump backtrace",
            "[Current focus set to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0)]",
            "#0  0x00007f in timeout_kernel<<<(1,1,1),(128,1,1)>>> () at /tmp/run/kernel.cu:3 in mbarrier_try_wait_parity inlined from kernel.cu:6",
            "#1  0x000080 in timeout_kernel(float*) at /tmp/run/kernel.cu:6",
        ]
    )

    insights = extract_cuda_gdb_insights(log)

    assert insights["focuses"] == ["CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0)]"]
    assert insights["frames"][0]["function"] == "timeout_kernel<<<(1,1,1),(128,1,1)>>> ()"
    assert insights["frames"][0]["line"] == 3
    assert insights["frames"][0]["inline_line"] == 6


def test_format_debug_report_includes_source_excerpt():
    solution = _solution()
    workload = Workload(uuid="abc", axes={"N": 1}, inputs={})
    log = "CUDA error an illegal memory access was encountered at /tmp/cache/kernel.cu:4\n"

    report = format_debug_report(
        solution=solution,
        workload=workload,
        device="cuda:0",
        raw_log=log,
        debug_dir=Path("/tmp/fib_debug/run_test"),
        coredumps=[],
        source_context_lines=1,
    )

    assert "Primary diagnostics" in report
    assert "kernel.cu:4" in report
    assert "> 4:   x[threadIdx.x] = 1.0f;" in report


def test_format_debug_metadata_uses_section_keys():
    solution = _solution()
    workload = Workload(uuid="abc", axes={"N": 1}, inputs={})
    log = "CUDA error an illegal memory access was encountered at /tmp/cache/kernel.cu:4\n"

    metadata = format_debug_metadata(
        solution=solution,
        workload=workload,
        device="cuda:0",
        raw_log=log,
        debug_dir=Path("/tmp/fib_debug/run_test"),
        coredumps=[],
        source_context_lines=1,
    )

    assert metadata["FlashInfer CUDA debug report"]["exist"] is True
    assert metadata["Primary diagnostics"]["exist"] is True
    assert "CUDA error an illegal memory access" in metadata["Primary diagnostics"]["msg"]
    assert metadata["CUDA core dumps"] == {
        "exist": False,
        "msg": "- No CUDA core dump file was produced.",
    }


def test_format_debug_metadata_summarizes_cuda_coredump_backtrace(tmp_path):
    solution = _barrier_timeout_solution()
    workload = Workload(uuid="abc", axes={"N": 1}, inputs={})
    coredump = tmp_path / "cuda_coredump_test"
    coredump.write_text("fake")
    log = "\n".join(
        [
            "ERROR: direct runner timed out after 45 seconds; requested CUDA user-triggered core dump before terminating it.",
            "CUDA coredump backtrace",
            "[Current focus set to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0)]",
            "#0  0x00007f in timeout_kernel<<<(1,1,1),(128,1,1)>>> () at /tmp/run/kernel.cu:3 in mbarrier_try_wait_parity inlined from kernel.cu:6",
            "#1  0x000080 in timeout_kernel(float*) at /tmp/run/kernel.cu:6",
        ]
    )

    metadata = format_debug_metadata(
        solution=solution,
        workload=workload,
        device="cuda:0",
        raw_log=log,
        debug_dir=tmp_path,
        coredumps=[coredump],
        source_context_lines=1,
    )

    section = metadata["CUDA core dump analysis"]
    assert section["exist"] is True
    assert "frame #0: kernel.cu:3" in section["msg"]
    assert "source: asm volatile" in section["msg"]
    assert "inlined call site: kernel.cu:6" in section["msg"]
    assert "call source: mbarrier_try_wait_parity();" in section["msg"]
    assert "inspect kernel.cu:6" in section["msg"]
    assert "wait/barrier path" in section["msg"]

    coredump_section = metadata["CUDA core dumps"]
    assert coredump_section["exist"] is False
    assert "Source-level coredump analysis is reported above." in coredump_section["msg"]
    assert "Open a dump with:" not in coredump_section["msg"]


def test_format_debug_report_prioritizes_sanitizer_fault_block():
    solution = _solution()
    workload = Workload(uuid="abc", axes={"N": 1}, inputs={})
    log = "\n".join(
        [
            "========= Misaligned shared or local address",
            "=========     at tma_store_2d_fn()+0xce80 in kernel.cu:4",
            "=========     by thread (0,0,0) in block (26,0,0)",
            "=========         Device Frame: parent()+0xcdc0 in kernel.cu:6",
            "CUDA error misaligned address at /tmp/cache/kernel.cu:6",
            '========= Program hit CUDA_ERROR_INVALID_VALUE due to "invalid argument" on CUDA API call to cuGetProcAddress_v2.',
        ]
    )

    faults = extract_sanitizer_faults(log)
    assert faults[0]["kind"] == "Misaligned shared or local address"
    assert faults[0]["thread"] == "(0,0,0)"
    assert faults[0]["block"] == "(26,0,0)"

    report = format_debug_report(
        solution=solution,
        workload=workload,
        device="cuda:0",
        raw_log=log,
        debug_dir=Path("/tmp/fib_debug/run_test"),
        coredumps=[],
        source_context_lines=1,
    )

    assert "Most precise CUDA fault" in report
    assert "faulting instruction: kernel.cu:4" in report
    assert "device frame: kernel.cu:6" in report
    assert (
        "cuGetProcAddress_v2"
        not in report.split("Primary diagnostics", 1)[1].split("Source locations", 1)[0]
    )


def test_format_debug_report_inferrs_preceding_launch_for_api_error():
    solution = _launch_error_solution()
    workload = Workload(uuid="abc", axes={"N": 1}, inputs={})
    log = "CUDA error an illegal memory access was encountered at /tmp/cache/kernel.cu:6\n"

    _, lines = solution.get_entry_source().path, solution.get_entry_source().content.splitlines()
    assert infer_cuda_api_faults(log, lines)[0]["launch_line"] == 5

    report = format_debug_report(
        solution=solution,
        workload=workload,
        device="cuda:0",
        raw_log=log,
        debug_dir=Path("/tmp/fib_debug/run_test"),
        coredumps=[],
        source_context_lines=1,
    )

    assert "likely failing launch: kernel.cu:5" in report
    assert "reported at check site: kernel.cu:6" in report


def test_debug_solution_validation_error_returns_metadata_field():
    result = flashinfer_bench_debug_solution(
        _solution(),
        Workload(uuid="abc", axes={"N": 1}, inputs={}),
        sanitizer_types="memcheck",  # type: ignore[arg-type]
    )

    assert set(result) == {"metadata"}
    section = result["metadata"]["FlashInfer CUDA debug report"]
    assert section["exist"] is False
    assert section["msg"] == "ERROR: sanitizer_types must be a list, not a string"
