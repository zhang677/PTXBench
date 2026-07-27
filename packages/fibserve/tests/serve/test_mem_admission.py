"""Unit tests for per-GPU-worker baseline cache memory admission.

These tests do not require a GPU: the device memory queries and
``torch.cuda.empty_cache`` calls on ``_GPUWorkerThread`` are overridden with
in-memory fakes so the cache-admission logic can be exercised on CPU.
"""

import queue
import threading
from types import SimpleNamespace

import pytest

from flashinfer_bench.bench import BenchmarkConfig
from flashinfer_bench.bench.runner.runner import BaselineHandle
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    TensorSpec,
    Workload,
)
from flashinfer_bench.serve.scheduler import (
    EVAL_CHECK_FACTOR,
    _GPUWorkerThread,
    estimate_baseline_bytes,
    estimate_eval_reserve_bytes,
)
from flashinfer_bench.serve.task_store import BaselineCacheMode

# ── Fakes ──


class _FakeGPUWorker:
    """Mimics the slice of PersistentSubprocessWorker the scheduler touches."""

    def __init__(self) -> None:
        self._baselines: dict = {}
        self.released: list = []
        self._counter = 0

    def run_ref(self, definition, workload, cfg, root, *, profile_baseline=None, run_baseline=None):
        handle = BaselineHandle(f"h{self._counter}")
        self._counter += 1
        self._baselines[handle] = object()
        return handle

    def release(self, handle) -> None:
        self._baselines.pop(handle, None)
        self.released.append(handle)

    def run_solution(self, solution, baseline, cfg):
        assert baseline in self._baselines
        return Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="test-gpu"),
            timestamp="2026-01-01T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(),
        )

    def clear_baselines(self) -> int:
        count = len(self._baselines)
        self.released.extend(self._baselines)
        self._baselines.clear()
        return count


def _make_worker(max_mem_ratio: float = 0.85) -> _GPUWorkerThread:
    config = BenchmarkConfig(max_mem_ratio=max_mem_ratio)
    worker = _GPUWorkerThread(
        device="cuda:0",
        task_queue=queue.Queue(),
        task_store=None,
        trace_set=SimpleNamespace(root=None),
        config=config,
        shutdown_event=threading.Event(),
    )
    worker._gpu_worker = _FakeGPUWorker()
    return worker


def _seed(worker: _GPUWorkerThread, key, *, run_ref_args=None):
    """Insert a cached baseline into both layers and return its handle."""
    handle = worker._gpu_worker.run_ref(None, None, None, None)
    worker._baseline_cache[key] = handle
    return handle


# ── Test definitions/workloads ──


def _estimator_definition() -> Definition:
    """input X[N, H] bf16 + scalar s; output Y[N, H] float32."""
    return Definition(
        name="est_def",
        op_type="test",
        axes={"N": AxisVar(), "H": AxisConst(value=4)},
        inputs={
            "X": TensorSpec(shape=["N", "H"], dtype="bfloat16"),
            "s": TensorSpec(shape=None, dtype="float32"),
        },
        outputs={"Y": TensorSpec(shape=["N", "H"], dtype="float32")},
        reference="import torch\n\ndef run(X, s):\n    return X.float()\n",
    )


def _estimator_workload() -> Workload:
    return Workload(
        axes={"N": 10, "H": 4}, inputs={"X": RandomInput(), "s": RandomInput()}, uuid="est_wl"
    )


def _tiny_definition() -> Definition:
    return Definition(
        name="d",
        op_type="test",
        axes={"N": AxisConst(value=1)},
        inputs={"X": TensorSpec(shape=["N"], dtype="float32")},
        outputs={"Y": TensorSpec(shape=["N"], dtype="float32")},
        reference="import torch\n\ndef run(X):\n    return X\n",
    )


def _tiny_workload(uuid: str) -> Workload:
    return Workload(axes={"N": 1}, inputs={"X": RandomInput()}, uuid=uuid)


# ── Estimator ──


def test_estimate_counts_tensor_payload_per_trial_and_ignores_scalars():
    definition = _estimator_definition()
    workload = _estimator_workload()

    # input X: 10*4 elems * 2 bytes (bf16) = 80; scalar s contributes 0
    # output Y: 10*4 elems * 4 bytes (f32) = 160
    # per-trial payload = 240; with num_trials=2 -> 480
    assert estimate_baseline_bytes(definition, workload, num_trials=2) == 480
    assert estimate_baseline_bytes(definition, workload, num_trials=1) == 240


def test_eval_reserve_models_subprocess_copy_plus_float32_check():
    definition = _estimator_definition()
    workload = _estimator_workload()

    # live eval working set = subprocess copy of the baseline (== baseline bytes)
    # + EVAL_CHECK_FACTOR * largest output's float32 size.
    # output Y: 10*4 = 40 elems -> 40*4 = 160 bytes in float32 (regardless of native dtype)
    # num_trials=2 -> baseline=480, reserve = 480 + EVAL_CHECK_FACTOR * 160
    expected = 480 + EVAL_CHECK_FACTOR * 160
    assert estimate_eval_reserve_bytes(definition, workload, num_trials=2) == expected


# ── Cache pressure clearing ──


def test_admit_clears_cached_baselines_from_both_layers():
    worker = _make_worker(max_mem_ratio=0.8)
    fake = worker._gpu_worker

    h0 = _seed(worker, ("d", "w0", None, None))
    h1 = _seed(worker, ("d", "w1", None, None))

    # total=1000 -> budget=800; used = 100 + 300 * (#cached baselines)
    worker._get_device_total_bytes = lambda: 1000
    worker._get_device_used_bytes = lambda: 100 + 300 * len(fake._baselines)
    worker._empty_device_cache = lambda: None

    # estimate 500: with 2 cached (used=700) 500+700=1200 >= 800 -> clear both
    # after clearing used=100, 500+100=600 < 800 -> stop
    worker._admit_baseline(_tiny_definition(), _tiny_workload("w_new"), 500, 0)

    assert h0 not in fake._baselines
    assert h1 not in fake._baselines
    assert ("d", "w0", None, None) not in worker._baseline_cache
    assert ("d", "w1", None, None) not in worker._baseline_cache
    assert fake.released == [h0, h1]


def test_admit_no_clear_when_within_budget():
    worker = _make_worker(max_mem_ratio=0.9)
    fake = worker._gpu_worker
    h0 = _seed(worker, ("d", "w0", None, None))

    worker._get_device_total_bytes = lambda: 1000
    worker._get_device_used_bytes = lambda: 100
    worker._empty_device_cache = lambda: None

    worker._admit_baseline(_tiny_definition(), _tiny_workload("w_new"), 100, 0)

    assert h0 in fake._baselines
    assert fake.released == []


def test_admit_reserves_eval_working_set():
    """A baseline whose cache cost alone fits the budget but whose live-eval
    reserve does not must still trigger a cache clear."""
    worker = _make_worker(max_mem_ratio=0.8)
    fake = worker._gpu_worker
    h0 = _seed(worker, ("d", "w0", None, None))

    worker._get_device_total_bytes = lambda: 1000
    worker._get_device_used_bytes = lambda: 100 + 300 * len(fake._baselines)
    worker._empty_device_cache = lambda: None

    # cache-only: 100 + used(400) = 500 < 800 -> would NOT clear
    # with eval reserve 350: 100 + 350 + 400 = 850 >= 800 -> clear h0
    # after clearing used=100: 100 + 350 + 100 = 550 < 800 -> stop
    worker._admit_baseline(_tiny_definition(), _tiny_workload("w_new"), 100, 350)

    assert h0 not in fake._baselines
    assert fake.released == [h0]


# ── Oversized failure ──


def test_admit_oversized_fails_clearly_after_clearing_all():
    worker = _make_worker(max_mem_ratio=0.85)
    fake = worker._gpu_worker
    h0 = _seed(worker, ("d", "w0", None, None))

    worker._get_device_total_bytes = lambda: 1000
    worker._get_device_used_bytes = lambda: 50 + 300 * len(fake._baselines)
    worker._empty_device_cache = lambda: None

    with pytest.raises(RuntimeError) as excinfo:
        worker._admit_baseline(_tiny_definition(), _tiny_workload("huge"), 2000, 0)

    msg = str(excinfo.value)
    assert "d" in msg and "huge" in msg
    assert "max_mem_ratio" in msg
    # all cached baselines were cleared in the attempt
    assert fake._baselines == {}
    assert fake.released == [h0]


def test_admit_clears_runner_baselines_even_if_scheduler_cache_is_empty():
    worker = _make_worker(max_mem_ratio=0.8)
    fake = worker._gpu_worker
    stale = fake.run_ref(None, None, None, None)

    worker._get_device_total_bytes = lambda: 1000
    worker._get_device_used_bytes = lambda: 100 + 300 * len(fake._baselines)
    worker._empty_device_cache = lambda: None

    # Scheduler has no tracked handle, but the runner still owns a baseline tensor.
    assert worker._baseline_cache == {}
    worker._admit_baseline(_tiny_definition(), _tiny_workload("w_new"), 500, 0)

    assert stale not in fake._baselines
    assert fake.released == [stale]


# ── Cache hits ──


def test_cache_hit_reuses_baseline_and_skips_build():
    worker = _make_worker(max_mem_ratio=0.99)
    fake = worker._gpu_worker

    # never clear
    worker._get_device_total_bytes = lambda: 10**12
    worker._get_device_used_bytes = lambda: 0
    worker._empty_device_cache = lambda: None

    definition = _tiny_definition()
    wl0 = _tiny_workload("w0")
    wl1 = _tiny_workload("w1")

    h0 = worker._get_or_build_ref(definition, wl0)
    h1 = worker._get_or_build_ref(definition, wl1)
    assert set(worker._baseline_cache.values()) == {h0, h1}

    # Hit on w0 returns cached handle and does not build a new baseline.
    again = worker._get_or_build_ref(definition, wl0)
    assert again == h0
    assert len(fake._baselines) == 2
    assert set(worker._baseline_cache.values()) == {h0, h1}


def test_build_ref_bypasses_cache():
    worker = _make_worker(max_mem_ratio=0.99)
    fake = worker._gpu_worker
    worker._get_device_total_bytes = lambda: 10**12
    worker._get_device_used_bytes = lambda: 0
    worker._empty_device_cache = lambda: None

    handle = worker._build_ref(_tiny_definition(), _tiny_workload("w0"))

    assert handle in fake._baselines
    assert worker._baseline_cache == {}


def test_evaluate_bypass_releases_ephemeral_baseline():
    worker = _make_worker(max_mem_ratio=0.99)
    fake = worker._gpu_worker
    definition = _tiny_definition()
    workload = _tiny_workload("w0")
    worker._trace_set = SimpleNamespace(
        definitions={definition.name: definition}, workloads={definition.name: []}, root=None
    )
    worker._resolve_workloads = lambda task: [workload]
    worker._get_device_total_bytes = lambda: 10**12
    worker._get_device_used_bytes = lambda: 0
    worker._empty_device_cache = lambda: None

    task = SimpleNamespace(
        definition_name=definition.name,
        solution=SimpleNamespace(name="solution"),
        atol=None,
        rtol=None,
        timeout=None,
        profile_baseline=None,
        run_baseline=None,
        baseline_cache_mode=BaselineCacheMode.BYPASS,
    )

    traces = worker._evaluate_task(task)

    assert len(traces) == 1
    assert fake._baselines == {}
    assert fake.released == [BaselineHandle("h0")]
    assert worker._baseline_cache == {}
