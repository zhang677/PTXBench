import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from ptxbench_eval.evaluator import Evaluator
from ptxbench_eval.feedback import append_runtime_diagnostics, render_debug_metadata
from ptxbench_eval.registry import Registry
from ptxbench_eval.models import WorkloadManifest


FIXTURES = json.loads((Path(__file__).parent / "fixtures/diagnostic_errors.json").read_text())


@pytest.mark.parametrize("language", ["triton", "cutedsl", "cuda"])
@pytest.mark.parametrize("feedback_style", ["default", "multiturn"])
@pytest.mark.parametrize("fixture_name", ["triton_log2e", "triton_c", "python_type_error"])
def test_gateway_retains_complete_errors_and_replays(tmp_path, monkeypatch, language, feedback_style, fixture_name):
    fixture = FIXTURES[fixture_name]
    manifest = {
        "schema_version": 1, "task_id": "test", "definition": "test", "workload_uuid": "uuid",
        "language": language, "target_hardware": ["H100"],
        "definition_payload": {"name": "test", "reference": "def run(x): return x"},
    }
    if language == "cuda":
        manifest.update(binding="tvm-ffi", nvcc_gencode="arch=compute_90a,code=sm_90a")
    monkeypatch.setattr("ptxbench_eval.evaluator.compile_source", lambda *a, **k: {"ok": True, "output": ""})
    registry = Registry(tmp_path)
    token = registry.create_run(
        run_id="run", manifest=manifest, agent_prompt="test", token_budget=100,
        harness="codex", model="test", prompt_tag="test", feedback_style=feedback_style,
    )
    registry.update_usage("run", {"total_tokens": 1}, source_event=1)
    calls = []

    class Client:
        def submit_and_poll(self, endpoint, payload, timeout, emit):
            calls.append(endpoint)
            assert payload["solution"]["spec"]["language"] == ("python" if language == "cutedsl" else language)
            if endpoint == "sanitize":
                return {"status": "completed", "logs": []}
            if endpoint == "debug":
                return {"status": "completed", "logs": [{"metadata": fixture["metadata"]}]}
            assert endpoint == "evaluate"
            return {"status": "completed", "traces": [{
                "definition": "test", "workload": {"uuid": "uuid"},
                "evaluation": {"status": "RUNTIME_ERROR", "log": fixture["log"]},
            }]}

    evaluator = Evaluator(registry=registry, profile_base_url="http://unused", client=Client(),
                          feedback_style=feedback_style, usage_wait_timeout=1)
    payload = {"run_id": "run", "action_id": "error", "workload": manifest,
               "source": "def run(x, out): pass" if language != "cuda" else "// CUDA test source"}
    result = evaluator.evaluate(payload, token)
    assert not result["all_passed"]
    assert fixture["log"] in result["feedback"]
    assert calls == ["sanitize", "sanitize", "evaluate", "debug"]
    if language != "cuda":
        assert "Primary diagnostics" not in result["feedback"]
        assert "kernel.py:196" not in result["feedback"]
    if feedback_style == "multiturn":
        assert result["traces"] is None
    else:
        assert result["traces"][0]["evaluation"]["log"].startswith(fixture["log"])
    replay = evaluator.evaluate(payload, token)
    assert replay["idempotent_replay"]
    assert replay["feedback"] == result["feedback"]
    assert len(calls) == 4


@pytest.mark.parametrize("language", ["cuda", "python", "triton"])
@pytest.mark.parametrize("status", ["RUNTIME_ERROR", "TIMEOUT"])
def test_device_fault_diagnostics_supplement_original_error(language, status):
    metadata = {
        "Most precise CUDA fault": {"exist": True, "msg": "Invalid __global__ write of size 4 in kernel.cu:2"},
        "Source locations": {"exist": True, "msg": "kernel.cu:2\n> 2: output[index] = 1;"},
    }
    traces = [{"evaluation": {"status": status, "log": "Original exception\n"}},
              {"evaluation": {"status": "PASSED", "log": "unchanged"}}]
    report = render_debug_metadata(metadata, language=language)
    append_runtime_diagnostics(traces, report)
    assert traces[0]["evaluation"]["log"].startswith("Original exception\n")
    assert "Additional CUDA diagnostics:" in traces[0]["evaluation"]["log"]
    assert "Invalid __global__ write" in traces[0]["evaluation"]["log"]
    assert traces[1]["evaluation"]["log"] == "unchanged"


@pytest.fixture(scope="module")
def backend_debug():
    # Load the actual formatter without importing the CUDA runtime. The stubbed
    # dependencies only support imports; these tests never execute GPU work.
    stub_modules = {}
    for name, attrs in {
        "flashinfer_bench.agents.sanitizer": {
            "VALID_SANITIZER_TYPES": [], "SanitizerType": str, "_build_sanitizer_command": None},
        "flashinfer_bench.data": {"Solution": object, "TraceSet": object, "Workload": object},
        "flashinfer_bench.utils": {"run_managed_subprocess": None},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        stub_modules[name] = module
    path = (Path(__file__).resolve().parents[2] / "fib-profile/flashinfer-bench-private/"
            "flashinfer_bench/agents/debug.py")
    spec = importlib.util.spec_from_file_location("backend_debug_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


def debug_metadata(module, tmp_path, raw_log, sources):
    items = [SimpleNamespace(path=path, content=content) for path, content in sources.items()]
    solution = SimpleNamespace(name="test", definition="test", sources=items,
                               get_entry_source=lambda: items[0])
    return module.format_debug_metadata(
        solution=solution, workload=SimpleNamespace(uuid="uuid", axes={}), device="cuda:0",
        raw_log=raw_log, debug_dir=tmp_path, coredumps=[],
    )


def test_python_library_frames_do_not_become_kernel_lines(backend_debug, tmp_path):
    raw_log = '''Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
  File "/root/.cache/fib_triton_test/pkg/kernel.py", line 2, in run
  File "/workspace/acc/lib/python3.10/site-packages/triton/runtime/jit.py", line 370, in run
NameError("Cannot access global variable LOG2E")
'''
    metadata = debug_metadata(backend_debug, tmp_path, raw_log,
                              {"kernel.py": "def run():\n    launch()\n"})
    locations = metadata["Source locations"]["msg"]
    assert metadata["Source locations"]["exist"]
    assert "kernel.py:2" in locations and "launch()" in locations
    assert "196" not in locations and "370" not in locations
    assert "outside available source" not in locations


def test_library_only_trace_has_no_submitted_source_excerpt(backend_debug, tmp_path):
    metadata = debug_metadata(backend_debug, tmp_path,
                              'File "/usr/lib/python3.10/runpy.py", line 196',
                              {"kernel.py": "def run(): pass"})
    assert not metadata["Source locations"]["exist"]


def test_cuda_fault_and_coredump_keep_source_identity(backend_debug, tmp_path):
    raw_log = '''========= Invalid __global__ write of size 4
=========     at kernel.cu:2 in launch
=========     by thread (0,0,0) in block (0,0,0)
=========     Device Frame: in helpers/write.cu:2
#0 0x1 in wait_for_kernel() at /tmp/cache/helpers/write.cu:2
#1 0x2 in internal() at /usr/local/cuda/internal.cu:196
timed out after 120 seconds.
'''
    metadata = debug_metadata(backend_debug, tmp_path, raw_log, {
        "kernel.cu": "void launch() {\n  write();\n}",
        "helpers/write.cu": "void write() {\n  output[index] = 1;\n}",
    })
    assert metadata["Most precise CUDA fault"]["exist"]
    assert "helpers/write.cu:2" in metadata["Most precise CUDA fault"]["msg"]
    locations = metadata["Source locations"]["msg"]
    assert "helpers/write.cu:2" in locations and "output[index]" in locations
    assert "kernel.cu:196" not in locations
    coredump = metadata["CUDA core dump analysis"]["msg"]
    assert "/usr/local/cuda/internal.cu:196" in coredump
    assert "kernel.cu:196" not in coredump


@pytest.mark.parametrize("language", ["python", "triton"])
def test_cuda_api_failure_without_source_frame_is_not_hidden(language):
    metadata = {"Primary diagnostics": {
        "exist": True, "msg": "CUDA error: an illegal memory access was encountered"}}
    assert "illegal memory access" in render_debug_metadata(metadata, language=language)


@pytest.mark.parametrize("language", ["cuda", "cutedsl", "triton"])
@pytest.mark.parametrize("timeout", [False, True])
def test_sanitizer_failure_preserves_original_and_adds_device_report(tmp_path, language, timeout):
    original = ("The kernel timed out after 120 seconds." if timeout else
                "Program hit CUDA_EXCEPTION_14: Warp Illegal Address\noriginal kernel traceback")
    calls = []

    class Client:
        def submit_and_poll(self, endpoint, payload, deadline, emit):
            calls.append(endpoint)
            if endpoint == "sanitize":
                return {"status": "completed", "logs": [{"log": original}]}
            assert endpoint == "debug"
            return {"status": "completed", "logs": [{"metadata": {
                "CUDA core dump analysis": {"exist": True, "msg": "kernel stopped at kernel.cu:2"}
            }}]}

    manifest = WorkloadManifest({"workload_uuid": "uuid", "language": language,
                                 "definition": "test", "target_hardware": ["H100"]})
    evaluator = Evaluator(registry=Registry(tmp_path), profile_base_url="http://unused", client=Client())
    traces = evaluator._profile_once(manifest.solution("def run(): pass"), manifest,
                                     SimpleNamespace(emit=lambda *a, **k: None))
    result = traces[0]["evaluation"]
    assert result["status"] == ("TIMEOUT" if timeout else "RUNTIME_ERROR")
    assert result["log"].startswith(original)
    assert "kernel stopped at kernel.cu:2" in result["log"]
    assert calls == ["sanitize", "debug"]
