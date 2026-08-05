from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = ROOT / "packages" / "mini-ptx-agent"
sys.path.insert(0, str(MINI_ROOT))

from mini_ptx_agent import evaluation
from mini_ptx_agent.ptx import extract_architecture_usage


def write_task(path: Path) -> evaluation.EvaluationTask:
    path.write_text(
        json.dumps(
            {
                "task_id": "gemm-test",
                "definition": "gemm_n7168_k5120",
                "workload_uuids": ["workload-1"],
                "target_hardware": ["H100"],
                "nvcc_gencode": "arch=compute_90a,code=sm_90a",
                "tvm_ffi_dir": "/opt/tvm_ffi",
                "max_check_runs": 2,
                "target_speedup": 1.0,
            }
        )
    )
    return evaluation.load_task(path)


def test_task_manifest_binds_definition_and_workload(tmp_path: Path) -> None:
    task = write_task(tmp_path / "task.json")
    solution = evaluation.build_solution(task, "// candidate")

    assert task.definition == "gemm_n7168_k5120"
    assert task.workload_uuids == ("workload-1",)
    assert solution["definition"] == task.definition
    assert solution["sources"] == [{"path": "kernel.cu", "content": "// candidate"}]


def test_compile_error_is_a_structured_candidate_result(tmp_path: Path, monkeypatch) -> None:
    task = write_task(tmp_path / "task.json")
    kernel = tmp_path / "kernel.cu"
    kernel.write_text("broken")
    monkeypatch.setattr(evaluation, "compile_candidate", lambda task, path: (False, "nvcc failed", None))

    result = evaluation.evaluate_kernel(kernel, task, "http://unused")

    assert result["schema"] == "ptxbench.eval.v1"
    assert result["status"] == "COMPILE_ERROR"
    assert result["all_passed"] is False
    assert result["sanitizer"]["completed_runs"] == 0
    assert result["traces"][0]["evaluation"]["status"] == "COMPILE_ERROR"


def test_cublas_and_cudnn_are_rejected_before_compilation(tmp_path: Path, monkeypatch) -> None:
    task = write_task(tmp_path / "task.json")
    kernel = tmp_path / "kernel.cu"

    def unexpected_compile(task, path):
        raise AssertionError("banned library source reached nvcc")

    monkeypatch.setattr(evaluation, "compile_candidate", unexpected_compile)
    cases = (
        ("#include <cublas_v2.h>\nvoid f() { cublasGemmEx(); }", "cuBLAS"),
        ("#include <cudnn.h>\nvoid f() { cudnnCreate(nullptr); }", "cuDNN"),
    )
    for source, library in cases:
        kernel.write_text(source)
        result = evaluation.evaluate_kernel(kernel, task, "http://unused")

        assert result["status"] == "COMPILE_ERROR"
        assert result["all_passed"] is False
        assert result["sanitizer"]["completed_runs"] == 0
        assert result["ptx"] is None
        assert f"uses {library} library calls" in result["compile_log"]
        assert result["traces"][0]["evaluation"]["log"] == result["compile_log"]


def test_library_names_in_comments_and_cutlass_are_allowed(tmp_path: Path, monkeypatch) -> None:
    task = write_task(tmp_path / "task.json")
    kernel = tmp_path / "kernel.cu"
    source = """
// cublasGemmEx() is not actually called.
/* cudnnCreate(nullptr); */
#include <cutlass/gemm/device/gemm.h>
"""
    kernel.write_text(source)
    compiled_sources: list[str] = []

    def fake_compile(task, path):
        compiled_sources.append(path.read_text())
        return False, "nvcc reached", None

    monkeypatch.setattr(evaluation, "compile_candidate", fake_compile)
    result = evaluation.evaluate_kernel(kernel, task, "http://unused")

    assert compiled_sources == [source]
    assert result["compile_log"] == "nvcc reached"


def test_success_runs_two_sanitizers_then_evaluate(tmp_path: Path, monkeypatch) -> None:
    task = write_task(tmp_path / "task.json")
    kernel = tmp_path / "kernel.cu"
    kernel.write_text("valid")
    ptx = {"source": "compiled_ptx_artifact", "instruction_counts": {"wgmma.": 2}}
    monkeypatch.setattr(evaluation, "compile_candidate", lambda task, path: (True, "ptxas info", ptx))
    calls: list[str] = []

    def fake_submit(service_url, endpoint, payload, timeout_sec):
        calls.append(endpoint)
        assert payload["workload_uuids"] == ["workload-1"]
        if endpoint == "sanitize":
            return {"status": "completed", "logs": [{"log": "clean"}]}
        return {
            "status": "completed",
            "traces": [
                {
                    "definition": task.definition,
                    "workload": {"uuid": "workload-1"},
                    "evaluation": {
                        "status": "PASSED",
                        "log": "",
                        "performance": {"speedup_factor": 1.25},
                    },
                }
            ],
        }

    monkeypatch.setattr(evaluation, "submit_and_poll", fake_submit)
    result = evaluation.evaluate_kernel(kernel, task, "http://service")

    assert calls == ["sanitize", "sanitize", "evaluate"]
    assert result["status"] == "PASSED"
    assert result["all_passed"] is True
    assert result["min_speedup"] == 1.25
    assert result["target_met"] is True
    assert result["sanitizer"] == {
        "requested_runs": 2,
        "completed_runs": 2,
        "clean": True,
    }
    assert result["traces"][0]["evaluation"]["log"].startswith("ptxas info")
    assert result["ptx"] == ptx


def test_sanitizer_failure_skips_evaluate(tmp_path: Path, monkeypatch) -> None:
    task = write_task(tmp_path / "task.json")
    kernel = tmp_path / "kernel.cu"
    kernel.write_text("unsafe")
    monkeypatch.setattr(evaluation, "compile_candidate", lambda task, path: (True, "", {}))
    calls: list[str] = []

    def fake_submit(service_url, endpoint, payload, timeout_sec):
        calls.append(endpoint)
        return {
            "status": "completed",
            "logs": [{"log": "Program hit CUDA_ERROR_ILLEGAL_ADDRESS"}],
        }

    monkeypatch.setattr(evaluation, "submit_and_poll", fake_submit)
    result = evaluation.evaluate_kernel(kernel, task, "http://service")

    assert calls == ["sanitize"]
    assert result["status"] == "RUNTIME_ERROR"
    assert result["all_passed"] is False
    assert result["sanitizer"]["completed_runs"] == 1


def test_architecture_usage_is_extracted_from_ptx_opcodes_only() -> None:
    ptx = """
    // Source mentions tcgen05.mma, but comments are not usage.
    .visible .entry kernel() {
      wgmma.fence.sync.aligned;
      @%p0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes;
      /* mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32; */
      ret;
    }
    """

    usage = extract_architecture_usage(ptx, "compute_90a")

    assert usage["source"] == "compiled_ptx_artifact"
    assert usage["instruction_counts"] == {"cp.async.bulk.": 1, "wgmma.": 1}
    assert usage["architecture_counts"] == {"hopper": 2}


def test_harbor_gemm_task_is_bound_to_quickstart_workload() -> None:
    task_root = ROOT / "integrations" / "harbor" / "tasks" / "gemm_n7168_k5120"
    harbor_readme = (ROOT / "integrations" / "harbor" / "README.md").read_text()
    manifest_text = (task_root / "environment" / "task.json").read_text()
    manifest = json.loads(manifest_text)
    instruction = (task_root / "instruction.md").read_text()
    environment = task_root / "environment"
    dockerfile = (environment / "Dockerfile").read_text()
    reference = (environment / "reference" / "README.md").read_text()

    assert json.loads((task_root / "tests" / "task.json").read_text()) == manifest
    assert manifest["definition"] == "gemm_n7168_k5120"
    assert manifest["workload_uuids"] == ["94920358-01a8-4c5b-9209-3103fd490e94"]
    assert not (environment / "kernel.cu").exists()
    assert "kernel.cu" not in dockerfile
    assert "does not contain a starter kernel" in instruction
    assert "no more than" not in instruction
    assert "ptxbench eval /workspace/kernel.cu --json" in instruction
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run)" in reference
    assert "-m openai/gpt-5.4-mini" in harbor_readme
    assert '"step_limit":30' in harbor_readme
