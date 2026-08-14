import ast
import json
import queue
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = REPO_ROOT / "packages" / "mini-ptx-agent"
MULTITURN_DIR = MINI_ROOT / "fib_runtime" / "multiturn"
sys.path.insert(0, str(MULTITURN_DIR))
sys.path.insert(0, str(MINI_ROOT))

import common  # noqa: E402
import run_parallel_v2  # noqa: E402
import run_v2  # noqa: E402
from create_triton_test import render  # noqa: E402
from mini_ptx_agent.quickstart import build_result  # noqa: E402


VALID_TRITON = """
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(x, y, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(y + offsets, values, mask=mask)

def run(x, y):
    torch.cuda.set_device(x.device)
    n_elements = y.numel()
    _kernel[(triton.cdiv(n_elements, 256),)](
        x, y, n_elements, BLOCK=256
    )
"""


@pytest.mark.parametrize(
    ("addition", "expected"),
    [
        ("\nimport cutlass.cute as cute\n", "CuTeDSL/CUTLASS imports"),
        ("\nfrom torch import matmul\n", "from torch"),
        ("\ndef bad(a, b):\n    return torch.matmul(a, b)\n", "PyTorch call"),
        ("\ndef bad(a):\n    return torch.empty_like(a)\n", "PyTorch call"),
        ("\ndef bad(a, b):\n    return a.mm(b)\n", "compute method"),
        ("\ndef bad(a, b):\n    return a @ b\n", "matrix multiplication operator"),
    ],
)
def test_triton_validator_rejects_fallbacks(addition, expected):
    error = common.validate_triton_source(VALID_TRITON + addition)
    assert error is not None
    assert expected in error


def test_triton_validator_accepts_runtime_glue_and_descriptor_allocator():
    with_allocator = VALID_TRITON + """

def _descriptor_allocator(size, alignment, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_descriptor_allocator)
"""
    assert common.validate_triton_source(with_allocator) is None


def test_triton_validator_requires_jit_kernel_and_run():
    no_jit = "import triton\n\ndef run(x, y):\n    return None\n"
    assert "@triton.jit" in common.validate_triton_source(no_jit)
    no_run = "import triton\n\n@triton.jit\ndef kernel(x):\n    return\n"
    assert "run(...)" in common.validate_triton_source(no_run)


def _runner_args(language: str):
    return SimpleNamespace(
        timeout=60,
        image="ptxbench-eval:dev",
        model="test-model",
        service_url="http://localhost:10000",
        gpu_arch="hopper",
        turn_timeout=30,
        llm_context_policy="full",
        without_local_gpu=True,
        verbose=False,
        max_profiles=None,
        language=language,
    )


def test_parallel_runner_forwards_triton_language(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(run_parallel_v2.subprocess, "run", fake_run)
    for name in ("logs", "trajectories", "success"):
        (tmp_path / name).mkdir()
    item = {
        "definition": "gemm_n7168_k5120",
        "test_path": "/tmp/test.py",
        "num_turns": 1,
        "target_speedup": 1.0,
        "prompt_tag": "triton-hopper",
    }
    gpu_queue = queue.Queue()
    gpu_queue.put("none")

    result = run_parallel_v2.run_single_experiment(
        0, item, _runner_args("triton"), gpu_queue, tmp_path
    )
    assert result["status"] == "success"
    position = captured["cmd"].index("--language")
    assert captured["cmd"][position + 1] == "triton"


def test_environment_uses_triton_python_artifacts(tmp_path, monkeypatch):
    traces = [
        {
            "workload": {"axes": {"M": 1}},
            "evaluation": {
                "status": "PASSED",
                "performance": {
                    "speedup_factor": 1.1,
                    "latency_ms": 1.0,
                    "reference_latency_ms": 1.1,
                },
            },
        }
    ]
    environment = object.__new__(common.KernelDockerEnvironment)
    environment._language = "triton"
    environment._turn = -1
    environment._success_dir = tmp_path / "success"
    environment._success_version = 0
    environment._target_speedup = 0.0
    environment._definition = "gemm_n7168_k5120"
    environment._last_traces = None
    environment._best_speedup = 0.0
    environment.config = SimpleNamespace(timeout=30, run_args=["-v", f"{tmp_path}:/workspace"])
    monkeypatch.setattr(
        common.DockerEnvironment,
        "execute",
        lambda self, action, **kwargs: {
            "output": "",
            "returncode": 0,
            "exception_info": "",
        },
    )
    monkeypatch.setattr(environment, "_read_traces_file", lambda: traces)

    result = environment.evaluate_kernel(VALID_TRITON)
    assert result["returncode"] == 0
    assert (tmp_path / "kernel.py").read_text() == VALID_TRITON
    assert (tmp_path / "success" / "kernel_v0.py").read_text() == VALID_TRITON


def test_triton_prompt_is_language_specific_and_clean():
    prompt = run_v2.build_triton_system_prompt("triton-hopper", "hopper")
    assert "Hopper SM90/SM90a" in prompt
    assert "destination-passing" in prompt
    assert "@triton.jit" in prompt
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC" not in prompt
    assert "cutlass.cute" not in prompt
    assert "AccRL" not in prompt
    assert "/home/ubuntu" not in prompt


def test_triton_template_renders_native_language_payload():
    rendered = render("gemm_n7168_k5120", "workload-uuid")
    ast.parse(rendered)
    assert '"language": "triton"' in rendered
    assert '"entry_point": "kernel.py::run"' in rendered
    assert '"dependencies": []' in rendered
    assert "cutedsl" not in rendered.lower()
    assert "<definition_name>" not in rendered
    assert "<workload_uuid>" not in rendered


def test_hub_exposes_architecture_specific_triton_roots():
    hub = json.loads((MULTITURN_DIR / "prompt_configs" / "hub.json").read_text())
    assert hub["triton-hopper"] == [
        "structural_doc/document/triton_knowledge_sm90_plus.md"
    ]
    assert hub["triton-blackwell"] == [
        "structural_doc/document/triton_knowledge_sm100_plus.md"
    ]


def test_quickstart_report_finds_python_candidates(tmp_path):
    output_root = tmp_path / "quickstart"
    (output_root / "exp_000").mkdir(parents=True)
    (output_root / "exp_000" / "kernel.py").write_text(VALID_TRITON)
    success = output_root / "success" / "exp_000"
    success.mkdir(parents=True)
    (success / "kernel_v0.py").write_text(VALID_TRITON)

    result = build_result(output_root)
    assert result["outcome"]["generated_candidate_count"] == 1
    assert result["outcome"]["correct_kernel_count"] == 1
    assert result["experiments"][0]["candidate_kernel"] == "exp_000/kernel.py"
    assert result["experiments"][0]["correct_kernels"] == [
        "success/exp_000/kernel_v0.py"
    ]
