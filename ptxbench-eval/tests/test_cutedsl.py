import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from ptxbench_eval.cutedsl import source_snapshot
from ptxbench_eval.python_kernel import validation_error
from ptxbench_eval.codex_runner import _ptxbench_mcp_server_config
from ptxbench_eval.evaluator import Evaluator
from ptxbench_eval.eval_mcp_server import _evaluate_kernel, _handle_request
from ptxbench_eval.launcher import (
    _cutedsl_mount_args, _evaluator_container_args, _registry_for_launch, prepare,
)
from ptxbench_eval.models import ContractError, WorkloadManifest, validate_language_contract
from ptxbench_eval.registry import Registry


MANIFEST = {
    "schema_version": 1,
    "task_id": "task-a",
    "definition": "gemm",
    "workload_uuid": "uuid-a",
    "language": "cuda",
    "binding": "tvm-ffi",
    "target_hardware": ["H100"],
    "nvcc_gencode": "arch=compute_90a,code=sm_90a",
    "definition_payload": {
        "name": "gemm",
        "reference": "import torch\n\ndef run(A, B):\n    return A @ B.T",
    },
}

class FakeClient:
    def __init__(self):
        self.calls = []

    def submit_and_poll(self, endpoint, payload, timeout, emit):
        self.calls.append(endpoint)
        emit("fake_call", endpoint=endpoint, request=payload)
        if endpoint == "sanitize":
            return {"status": "completed", "logs": []}
        return {
            "status": "completed",
            "traces": [
                {
                    "definition": "gemm",
                    "workload": {"uuid": "uuid-a", "axes": {"M": 8192}},
                    "evaluation": {
                        "status": "PASSED",
                        "performance": {
                            "speedup_factor": 1.25,
                            "latency_ms": 1.0,
                            "reference_latency_ms": 1.25,
                        },
                    },
                }
            ],
        }

DEFINITION_PAYLOAD = {
    "name": "gemm",
    "inputs": {"A": {"shape": ["M", 5120]}},
    "outputs": {"C": {"shape": ["M", 7168]}},
    "reference": "import torch\n\ndef run(A, B):\n    return A @ B.T",
}

WORKLOAD_PAYLOAD = {
    "uuid": "uuid-a",
    "axes": {"M": 8192},
    "inputs": {"A": "sha256:input-a"},
}

@pytest.fixture(autouse=True)
def fake_profile_service(monkeypatch):
    responses = {
        "http://profile.test/health": {
            "status": "ok",
            "backends": [{"healthy": True}],
            "queue_size": 0,
        },
        "http://profile.test/definitions/gemm": DEFINITION_PAYLOAD,
        "http://profile.test/definitions/gemm/workloads": [WORKLOAD_PAYLOAD],
    }

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    monkeypatch.setattr(
        "ptxbench_eval.launcher.requests.get",
        lambda url, timeout: FakeResponse(responses[url]),
    )

def write_inputs(tmp_path):
    manifest = {
        "schema_version": 1,
        "task_id": "task-a",
        "definition": "gemm",
        "workload_uuid": "uuid-a",
        "language": "cuda",
        "binding": "tvm-ffi",
        "target_hardware": ["H100"],
        "nvcc_gencode": "arch=compute_90a,code=sm_90a",
        "definition_payload": DEFINITION_PAYLOAD,
    }
    (tmp_path / "workload.json").write_text(json.dumps(manifest))
    (tmp_path / "example.cu").write_text("// exact ffi example\n")
    (tmp_path / "context" / "docs").mkdir(parents=True)
    (tmp_path / "context" / "headers").mkdir()
    (tmp_path / "context" / "patterns").mkdir()
    (tmp_path / "context" / "docs" / "base.md").write_text("base knowledge\n")
    (tmp_path / "context" / "headers" / "helper.h").write_text(
        "// exact helper header\n"
    )
    (tmp_path / "context" / "patterns" / "hint.md").write_text(
        "optimization hint\n"
    )
    (tmp_path / "context" / "patterns" / "extra.md").write_text(
        "extra user markdown\n"
    )
    (tmp_path / "context" / "headers" / "extra.h").write_text(
        "// extra user header\n"
    )
    (tmp_path / "hub.json").write_text(
        json.dumps(
            {
                "hopper-no-hint": ["docs/base.md", "headers/helper.h"],
                "hopper-05": ["hopper-no-hint", "patterns/hint.md"],
                "user-extra": ["patterns/extra.md", "headers/extra.h"],
            }
        )
    )
    return manifest

def config(tmp_path):
    return {
        "_config_dir": str(tmp_path),
        "experiment_root": str(tmp_path / "experiment"),
        "profile_base_url": "http://profile.test",
        "task_id": "task-a",
        "definition": "gemm",
        "workload_uuid": "uuid-a",
        "language": "cuda",
        "binding": "tvm-ffi",
        "target_hardware": ["H100"],
        "nvcc_gencode": "arch=compute_90a,code=sm_90a",
        "prompt_hub": "hub.json",
        "prompt_files_root": "context",
        "tvm_ffi_example": "example.cu",
        "gateway_url": "http://127.0.0.1:18082",
        "agent_image": "ptxbench-antigravity:dev",
        "model": "gemini-3.7-flash-medium",
        "token_budget": 200000,
        "runs": [
            {
                "run_id": "exp_000",
                "prompt_tag": "hopper-no-hint",
            },
            {
                "run_id": "exp_001",
                "prompt_tag": "hopper-05",
            },
        ],
    }

def python_manifest(language="cutedsl"):
    manifest = {k: v for k, v in MANIFEST.items() if k not in {"binding", "nvcc_gencode"}}
    manifest["language"] = language
    return manifest


def snapshot(tmp_path):
    root = tmp_path / "source"
    (root / "cutlass/cute").mkdir(parents=True)
    content = b"# service version source\n"
    (root / "cutlass/cute/__init__.py").write_bytes(content)
    manifest = {
        "schema_version": 1, "version": "4.6.0",
        "packages": {"nvidia-cutlass-dsl": "4.6.0"},
        "files": {"cutlass/cute/__init__.py": hashlib.sha256(content).hexdigest()},
    }
    (root / "source_manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize("language", ["cutedsl", "triton"])
def test_python_service_contract(language):
    manifest = WorkloadManifest.parse(python_manifest(language))
    solution = manifest.solution("def run(A, B, C): pass")
    assert solution["spec"] == {
        "language": "python" if language == "cutedsl" else "triton",
        "target_hardware": ["H100"], "entry_point": "kernel.py::run",
        "dependencies": ["nvidia-cutlass-dsl", "cuda-python"] if language == "cutedsl" else ["triton"],
        "destination_passing_style": True,
    }
    assert solution["sources"][0]["path"] == "kernel.py"
    with pytest.raises(ContractError, match="omit binding"):
        WorkloadManifest.parse({**python_manifest(language), "binding": "tvm-ffi"})


@pytest.mark.parametrize("language", [None, "python", [], {}])
def test_unsupported_language_rejected(language):
    with pytest.raises(ContractError, match="language must be cuda, cutedsl, or triton"):
        validate_language_contract({"language": language})


@pytest.mark.parametrize("corruption", ["changed", "missing", "extra", "symlink"])
def test_source_snapshot_rejects_drift(tmp_path, corruption):
    root = snapshot(tmp_path)
    assert source_snapshot(root)["version"] == "4.6.0"
    source = root / "cutlass/cute/__init__.py"
    if corruption == "changed":
        source.write_text("changed")
    elif corruption == "missing":
        source.unlink()
    elif corruption == "extra":
        (root / "extra.py").write_text("extra")
    else:
        (root / "link").symlink_to(source)
    with pytest.raises(ValueError, match="CuTeDSL source"):
        source_snapshot(root)


def test_prepare_cutedsl_context_prompt_and_readonly_mount(tmp_path):
    write_inputs(tmp_path)
    root = snapshot(tmp_path)
    settings = config(tmp_path)
    settings.update(language="cutedsl", cutedsl_source=str(root), agent="codex", disable_web=True,
                    task_view="multiturn", feedback_style="multiturn", cutedsl_example="example.py")
    example = "def run(x, scale, shift, y):\n    pass\n"
    (tmp_path / "example.py").write_text(example)
    for key in ["binding", "nvcc_gencode", "tvm_ffi_example"]:
        settings.pop(key)
    settings["runs"] = [{"run_id": "exp_000", "prompt_tag": "hopper-no-header"}]
    (tmp_path / "hub.json").write_text(json.dumps({"hopper-no-header": ["docs/base.md"]}))
    registry = prepare(settings)
    workspace = registry.root / "workspaces/exp_000"
    knowledge = (workspace / "knowledge_pack.md").read_text()
    assert knowledge.startswith("base knowledge\n")
    assert "## CuTeDSL integration example" in knowledge
    assert f"```python\n{example}```" in knowledge
    assert "TVM-FFI" not in knowledge and "```cpp" not in knowledge
    provenance = registry.get_run("exp_000")["artifact_provenance"]
    assert provenance["cutedsl_example_sha256"] == hashlib.sha256(example.encode()).hexdigest()
    assert provenance["cutedsl_example_source"] == str(tmp_path / "example.py")
    assert "tvm_ffi_example_sha256" not in provenance
    assert {p.name for p in workspace.iterdir()} == {"knowledge_pack.md", "cutedsl"}
    prompt = registry.prompt_path("exp_000").read_text()
    assert "CuTeDSL" in prompt and "4.6.0" in prompt and "read-only" in prompt
    assert "ptxbench-eval /workspace/kernel.py /opt/ptxbench/workload.json" in prompt
    assert "kernel.cu" not in prompt and "TVM-FFI" not in prompt
    assert _cutedsl_mount_args(settings) == ["-v", f"{root}:/workspace/cutedsl:ro"]
    assert "PTXBENCH_KERNEL_LANGUAGE=cutedsl" in _evaluator_container_args(settings, "exp_000", tmp_path / "secret")
    prepare(settings)
    _registry_for_launch(settings)
    (tmp_path / "example.py").write_text(example + "# changed\n")
    for check in [prepare, _registry_for_launch]:
        with pytest.raises(ValueError, match="different CuTeDSL example"):
            check(settings)
    (tmp_path / "example.py").write_text(example)
    (root / "cutlass/cute/__init__.py").write_text("drift")
    with pytest.raises(ValueError, match="hash mismatch"):
        _registry_for_launch(settings)


@pytest.mark.parametrize("example", [None, "", "missing.py"])
@pytest.mark.parametrize("language", ["cutedsl", "triton"])
def test_prepare_requires_python_example(tmp_path, example, language):
    write_inputs(tmp_path)
    settings = config(tmp_path)
    settings.update(language=language)
    if language == "cutedsl":
        settings["cutedsl_source"] = str(snapshot(tmp_path))
    settings.pop("binding")
    if example is not None:
        settings[f"{language}_example"] = example
    if example == "missing.py":
        with pytest.raises(FileNotFoundError):
            prepare(settings)
    else:
        with pytest.raises(ValueError, match=f"{language}_example is required"):
            prepare(settings)


def test_python_preflight_never_executes_source_and_rejects_shortcuts():
    assert validation_error("raise RuntimeError('must not execute')\ndef run(A, B, C): pass") is None
    assert "Invalid Python" in validation_error("def run(:")
    assert "Invalid Python" in validation_error("return\ndef run(): pass")
    assert "run function" in validation_error("x = 1")
    assert "matrix multiplication" in validation_error("def run(A, B, C): C = A @ B")
    assert "torch.mm" in validation_error("from torch import mm as mat\ndef run(A, B, C): mat(A, B, out=C)")


@pytest.mark.parametrize("source,passes", [("def run(A, B, C): pass", True), ("def run(:", False)])
@pytest.mark.parametrize("language", ["cutedsl", "triton"])
def test_python_gateway_bypasses_nvcc_and_preserves_feedback(tmp_path, monkeypatch, source, passes, language):
    monkeypatch.setattr("ptxbench_eval.evaluator.compile_source", lambda *a, **k: pytest.fail("NVCC called for Python"))
    registry = Registry(tmp_path)
    manifest = python_manifest(language)
    token = registry.create_run(run_id="exp_000", manifest=manifest, agent_prompt="task", token_budget=100,
                                harness="codex", model="test", prompt_tag="hopper-no-header", feedback_style="multiturn")
    registry.update_usage("exp_000", {"total_tokens": 1}, source_event=1)
    client = FakeClient()
    evaluator = Evaluator(registry=registry, profile_base_url="http://unused", client=client,
                          feedback_style="multiturn", usage_wait_timeout=1)
    result = evaluator.evaluate({"run_id": "exp_000", "action_id": "test", "workload": manifest, "source": source}, token)
    assert result["all_passed"] is passes and result["traces"] is None
    assert client.calls == (["sanitize", "sanitize", "evaluate"] if passes else [])
    assert registry.canonical_turn_count("exp_000") == 1
    events = [json.loads(line) for line in registry.raw_events_path("exp_000").read_text().splitlines()]
    submissions = [e for e in events if e["event"] == "fake_call"]
    if passes:
        expected_language = "python" if language == "cutedsl" else "triton"
        assert all(e["request"]["solution"]["spec"]["language"] == expected_language for e in submissions)


@pytest.mark.parametrize("language", ["cutedsl", "triton"])
def test_python_mcp_selects_kernel_py(monkeypatch, language):
    monkeypatch.setenv("PTXBENCH_KERNEL_LANGUAGE", language)
    assert _ptxbench_mcp_server_config()["env"] == {"PTXBENCH_KERNEL_LANGUAGE": language}
    monkeypatch.setenv("PTXBENCH_RUN_ID", "exp_000")
    monkeypatch.setenv("PTXBENCH_EVAL_EXCHANGE_DIR", "/workspace/exchange")
    observed = []
    def run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="passed")
    monkeypatch.setattr("ptxbench_eval.eval_mcp_server.subprocess.run", run)
    assert _evaluate_kernel()["isError"] is False
    assert observed[0][1] == "/workspace/kernel.py"
    response = _handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "kernel.py" in response["result"]["tools"][0]["description"]


@pytest.mark.parametrize("agent,use_mcp", [("codex", False), ("codex", True), ("agy", False)])
def test_prepare_triton_agent_and_example_drift(tmp_path, agent, use_mcp):
    write_inputs(tmp_path)
    settings = config(tmp_path)
    settings.update(language="triton", agent=agent, disable_web=True,
                    task_view="multiturn", feedback_style="multiturn",
                    triton_example="example.py", codex_use_mcp=use_mcp)
    for key in ["binding", "nvcc_gencode", "tvm_ffi_example"]:
        settings.pop(key)
    example_path = Path(__file__).resolve().parents[2] / "multiturn/prompts/examples/triton.py"
    example = example_path.read_text()
    (tmp_path / "example.py").write_text(example)
    settings["runs"] = [{"run_id": "exp_000", "prompt_tag": "triton-hopper"}]
    (tmp_path / "hub.json").write_text(json.dumps({"triton-hopper": ["docs/base.md"]}))
    registry = prepare(settings)
    workspace = registry.root / "workspaces/exp_000"
    assert {p.name for p in workspace.iterdir()} == {"knowledge_pack.md"}
    knowledge = (workspace / "knowledge_pack.md").read_text()
    assert "## Triton integration example" in knowledge
    assert f"```python\n{example.rstrip()}\n```" in knowledge
    assert "TVM-FFI" not in knowledge and "CuTeDSL" not in knowledge
    prompt = registry.prompt_path("exp_000").read_text()
    assert "Triton" in prompt and "preallocated" in prompt
    assert "kernel.cu" not in prompt and "CuTeDSL" not in prompt
    if use_mcp or agent == "agy":
        assert "`evaluate_kernel` tool" in prompt
    else:
        assert "ptxbench-eval /workspace/kernel.py /opt/ptxbench/workload.json" in prompt
    assert _cutedsl_mount_args(settings) == []
    assert "PTXBENCH_KERNEL_LANGUAGE=triton" in _evaluator_container_args(settings, "exp_000", tmp_path / "secret")
    provenance = registry.get_run("exp_000")["artifact_provenance"]
    assert provenance["triton_example_sha256"] == hashlib.sha256(example.encode()).hexdigest()
    assert "cutedsl_source" not in provenance and "tvm_ffi_example_sha256" not in provenance
    if agent == "agy":
        definitions = list((registry.root / "antigravity_home").rglob("agent.md"))
        # The no-web AGY agent must use the same fixed Python evaluator path.
        assert len(definitions) == 1
        assert "/workspace/kernel.py" in definitions[0].read_text()
        assert "kernel.cu" not in definitions[0].read_text()
    prepare(settings)
    _registry_for_launch(settings)
    (tmp_path / "example.py").write_text(example + "# changed\n")
    for check in [prepare, _registry_for_launch]:
        with pytest.raises(ValueError, match="different Triton example"):
            check(settings)


def test_triton_preflight_accepts_example_and_labels_shortcuts():
    example = Path(__file__).resolve().parents[2] / "multiturn/prompts/examples/triton.py"
    assert validation_error(example.read_text(), language="Triton") is None
    feedback = validation_error("import torch\ndef run(A, B, C): torch.mm(A, B, out=C)", language="Triton")
    assert "Triton" in feedback and "torch.mm" in feedback
