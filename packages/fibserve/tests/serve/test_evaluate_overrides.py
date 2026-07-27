"""Tests for per-request overrides on /evaluate."""

from pathlib import Path

import pytest
import pytest_asyncio
import safetensors.torch as st
import torch

from flashinfer_bench.bench import BenchmarkConfig
from flashinfer_bench.data import (
    AxisConst,
    Definition,
    SafetensorsInput,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)
from flashinfer_bench.serve.app import app, init_app
from flashinfer_bench.serve.scheduler import Scheduler
from tests.serve.conftest import (
    make_test_definition,
    make_test_workload,
    solution_correct,
    solution_slow,
    solution_wrong_value,
)

try:
    from httpx import ASGITransport, AsyncClient
except ImportError:
    pytest.skip("httpx not installed", allow_module_level=True)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(torch.cuda.device_count() == 0, reason="No CUDA devices available"),
]

DEFINITION = "test_scale"


# ── Fixtures with stored-output workload ──


def make_stored_definition() -> Definition:
    """Same as test_scale but with N=8 to make the safetensors small."""
    return Definition(
        name="stored_scale",
        op_type="test",
        axes={"N": AxisConst(value=8)},
        inputs={"X": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"Y": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(X):\n    return X * 2\n",
    )


def make_stored_workload(tmp_path: Path) -> Workload:
    """Workload with safetensors inputs AND stored reference outputs."""
    x = torch.arange(8, dtype=torch.float32)
    y = x * 2.0

    in_path = tmp_path / "x.safetensors"
    out_path = tmp_path / "y.safetensors"
    st.save_file({"X": x.contiguous()}, str(in_path))
    st.save_file({"Y": y.contiguous()}, str(out_path))

    return Workload(
        axes={"N": 8},
        inputs={"X": SafetensorsInput(path=str(in_path), tensor_key="X")},
        uuid="stored_workload_001",
        outputs={"Y": SafetensorsInput(path=str(out_path), tensor_key="Y")},
    )


@pytest.fixture
def stored_trace_set(tmp_path) -> TraceSet:
    """TraceSet with both a regular workload and a stored-outputs workload."""
    test_def = make_test_definition()
    stored_def = make_stored_definition()
    regular_wl = make_test_workload()
    stored_wl = make_stored_workload(tmp_path)

    trace_set = TraceSet()
    trace_set.definitions[test_def.name] = test_def
    trace_set.definitions[stored_def.name] = stored_def
    trace_set.workloads[test_def.name] = [
        Trace(definition=test_def.name, workload=regular_wl, solution=None, evaluation=None)
    ]
    trace_set.workloads[stored_def.name] = [
        Trace(definition=stored_def.name, workload=stored_wl, solution=None, evaluation=None)
    ]
    return trace_set


@pytest.fixture
def short_config() -> BenchmarkConfig:
    return BenchmarkConfig(warmup_runs=1, iterations=2, num_trials=1, timeout_seconds=15)


@pytest.fixture
def stored_scheduler(stored_trace_set, short_config) -> Scheduler:
    sched = Scheduler(trace_set=stored_trace_set, config=short_config, devices=["cuda:0"])
    init_app(sched)
    yield sched
    sched.shutdown()


@pytest_asyncio.fixture
async def stored_client(stored_scheduler) -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Helpers ──


async def _submit_and_wait(client, payload: dict, timeout: float = 30) -> dict:
    resp = await client.post("/evaluate", json=payload)
    if resp.status_code != 200:
        return {"_status_code": resp.status_code, "_detail": resp.json().get("detail", resp.text)}
    task_id = resp.json()["task_id"]
    resp = await client.get(f"/tasks/{task_id}", params={"timeout": timeout})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── profile_baseline override ──


async def test_profile_baseline_false_zeros_reference_latency(stored_client):
    """profile_baseline=False per-request -> reference_latency_ms == 0.0, baseline still ran."""
    sol = solution_correct("stored_scale")
    result = await _submit_and_wait(
        stored_client, {"solution": sol.model_dump(mode="json"), "profile_baseline": False}
    )
    assert result["status"] == "completed"
    eval_block = result["traces"][0]["evaluation"]
    assert eval_block["status"] == "PASSED"
    perf = eval_block["performance"]
    assert perf["reference_latency_ms"] == 0.0
    # Solution latency still measured.
    assert perf["latency_ms"] > 0.0


async def test_profile_baseline_default_measures_reference(stored_client):
    """Without override, server default profiles the baseline -> nonzero ref latency."""
    sol = solution_correct("stored_scale")
    result = await _submit_and_wait(stored_client, {"solution": sol.model_dump(mode="json")})
    assert result["status"] == "completed"
    perf = result["traces"][0]["evaluation"]["performance"]
    assert perf["reference_latency_ms"] > 0.0


# ── timeout override ──


async def test_timeout_override_marks_slow_evaluate_as_timeout(stored_client):
    """timeout per-request overrides the server default BenchmarkConfig timeout."""
    sol = solution_slow(DEFINITION)
    result = await _submit_and_wait(
        stored_client, {"solution": sol.model_dump(mode="json"), "timeout": 1}
    )
    assert result["status"] == "completed"
    eval_block = result["traces"][0]["evaluation"]
    assert eval_block["status"] == "TIMEOUT"
    assert "Evaluation timeout after 1 seconds" in eval_block["log"]


# ── run_baseline override (validation) ──


async def test_run_baseline_false_rejects_workload_without_outputs(stored_client):
    """test_scale workload has no `outputs` field -> 400."""
    sol = solution_correct(DEFINITION)
    resp = await stored_client.post(
        "/evaluate", json={"solution": sol.model_dump(mode="json"), "run_baseline": False}
    )
    assert resp.status_code == 400
    assert "no `outputs`" in resp.json()["detail"]


async def test_run_baseline_false_rejects_random_inputs(stored_client, stored_trace_set, tmp_path):
    """Workload with stored outputs but a RandomInput -> 400."""
    from flashinfer_bench.data import RandomInput

    # Mutate one workload to have a random input alongside outputs.
    bad_wl = Workload(
        axes={"N": 8},
        inputs={"X": RandomInput()},
        uuid="bad_wl",
        outputs={"Y": SafetensorsInput(path=str(tmp_path / "y.safetensors"), tensor_key="Y")},
    )
    stored_trace_set.workloads["stored_scale"].append(
        Trace(definition="stored_scale", workload=bad_wl, solution=None, evaluation=None)
    )

    sol = solution_correct("stored_scale")
    resp = await stored_client.post(
        "/evaluate",
        json={
            "solution": sol.model_dump(mode="json"),
            "workload_uuids": ["bad_wl"],
            "run_baseline": False,
        },
    )
    assert resp.status_code == 400
    assert "non-deterministic" in resp.json()["detail"]


# ── run_baseline override (happy path) ──


async def test_run_baseline_false_passes_against_stored_outputs(stored_client):
    """run_baseline=False with stored safetensors outputs -> correctness PASSED, ref latency 0."""
    sol = solution_correct("stored_scale")
    result = await _submit_and_wait(
        stored_client, {"solution": sol.model_dump(mode="json"), "run_baseline": False}
    )
    assert result["status"] == "completed"
    eval_block = result["traces"][0]["evaluation"]
    assert eval_block["status"] == "PASSED"
    perf = eval_block["performance"]
    assert perf["reference_latency_ms"] == 0.0
    assert perf["latency_ms"] > 0.0


async def test_run_baseline_false_detects_incorrect_solution(stored_client):
    """run_baseline=False still checks correctness — wrong solution must be flagged."""
    sol = solution_wrong_value("stored_scale")
    result = await _submit_and_wait(
        stored_client, {"solution": sol.model_dump(mode="json"), "run_baseline": False}
    )
    assert result["status"] == "completed"
    assert result["traces"][0]["evaluation"]["status"] == "INCORRECT_NUMERICAL"


# ── baseline cache override ──


async def test_invalid_baseline_cache_mode_is_rejected(stored_client):
    sol = solution_correct("stored_scale")
    resp = await stored_client.post(
        "/evaluate",
        json={"solution": sol.model_dump(mode="json"), "baseline_cache_mode": "invalid"},
    )
    assert resp.status_code == 422


async def test_baseline_cache_bypass_does_not_populate_cache(stored_client, stored_scheduler):
    sol = solution_correct("stored_scale")
    result = await _submit_and_wait(
        stored_client, {"solution": sol.model_dump(mode="json"), "baseline_cache_mode": "bypass"}
    )

    assert result["status"] == "completed"
    assert result["traces"][0]["evaluation"]["status"] == "PASSED"
    worker = stored_scheduler.workers[0]
    assert worker._baseline_cache == {}
    assert worker._gpu_worker._baselines == {}
