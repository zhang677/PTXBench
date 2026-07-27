"""GPU worker scheduling for the benchmark server."""

import logging
import os
import queue
import threading
from math import prod
from typing import List, Optional

import torch

from flashinfer_bench.agents.debug import flashinfer_bench_debug_solution
from flashinfer_bench.agents.ncu import flashinfer_bench_run_ncu
from flashinfer_bench.agents.sanitizer import flashinfer_bench_run_sanitizer
from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.runner.persistent_runner import PersistentSubprocessWorker
from flashinfer_bench.bench.runner.runner import BaselineHandle
from flashinfer_bench.data import Definition, EvaluationStatus, Solution, Trace, TraceSet, Workload
from flashinfer_bench.serve.task_store import BaselineCacheMode, RunLog, Task, TaskKind, TaskStore
from flashinfer_bench.utils import kill_all_tracked_subprocesses

logger = logging.getLogger(__name__)

# Multiplier for the float32 correctness-check working set. The default evaluator's
# ``compute_error_stats`` converts each output (and the reference) to float32 and holds
# several full-size float32/bool temporaries simultaneously (``x``, ``y``, ``abs_error``,
# ``rel_error`` and comparison masks). This bounds that transient peak as a multiple of
# the largest output's float32 size. It is a generic schema-derived upper bound, not an
# evaluator-specific hook; heavier-than-default evaluators are absorbed by the headroom
# left below ``max_mem_ratio``.
EVAL_CHECK_FACTOR = 8


def estimate_baseline_bytes(definition: Definition, workload: Workload, num_trials: int) -> int:
    """Estimate the steady-state GPU bytes a cached baseline will occupy.

    The baseline caches ``num_trials`` copies of the generated inputs and the
    reference outputs (see the evaluators' ``build_baseline``). This models that
    payload from the schema shapes and dtypes alone, before any GPU tensor is
    materialized::

        estimated_bytes = num_trials * (sum(input tensor bytes) + sum(output tensor bytes))

    Scalars (``shape is None``) count as zero because they are cached as plain
    Python values, not GPU tensors. The estimate intentionally ignores temporary
    peak allocations, CUDA context memory, allocator fragmentation, and compiled
    runnable memory; ``max_mem_ratio`` headroom is expected to absorb those.
    """
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)

    def _payload(shapes, dtypes) -> int:
        total = 0
        for shape, dtype in zip(shapes, dtypes):
            if shape is None:  # scalar -> Python value, not a cached GPU tensor
                continue
            total += prod(shape) * dtype.itemsize
        return total

    per_trial = _payload(input_shapes, definition.torch_input_dtypes) + _payload(
        output_shapes, definition.torch_output_dtypes
    )
    return num_trials * per_trial


def estimate_eval_reserve_bytes(definition: Definition, workload: Workload, num_trials: int) -> int:
    """Estimate the transient GPU bytes the *live evaluation* of this workload needs.

    Unlike the cached baseline (which lives in the scheduler process), the evaluation
    runs in the worker subprocess, which re-materializes its own copy of the inputs and
    reference outputs and produces solution outputs, then runs the float32 correctness
    check. Peak device usage is therefore the cached baseline PLUS this live working set,
    so admission must reserve room for it or a heavy evaluation OOMs even while the cache
    is within budget.

    Modeled generically from the schema as::

        baseline_bytes (subprocess re-materializes a comparable working set)
        + EVAL_CHECK_FACTOR * max(output float32 bytes)   (compute_error_stats temporaries)

    Outputs are sized in float32 regardless of their native dtype because the correctness
    check upcasts them.
    """
    output_shapes = definition.get_output_shapes(workload.axes)
    max_output_f32 = 0
    for shape in output_shapes:
        if shape is None:  # scalar
            continue
        max_output_f32 = max(max_output_f32, prod(shape) * 4)

    return estimate_baseline_bytes(definition, workload, num_trials) + EVAL_CHECK_FACTOR * (
        max_output_f32
    )


class Scheduler:
    """Manages GPU workers and dispatches evaluate/profile/sanitize tasks."""

    def __init__(self, trace_set: TraceSet, config: BenchmarkConfig, devices: List[str]):
        self._trace_set = trace_set
        self._config = config
        self._task_store = TaskStore()
        # Memory-awareness is implemented per-GPU-worker via baseline-cache admission
        # (see _GPUWorkerThread._admit_baseline), NOT at this top-level scheduler.
        # Workers self-select tasks from one shared FIFO queue; memory-aware top-level
        # assignment (route a task to the worker with the most free HBM) would require
        # replacing this shared queue with per-worker queues plus a dispatcher, which is
        # out of scope for the cache-admission goal.
        self._queue: queue.Queue[str] = queue.Queue()
        self._shutdown = threading.Event()

        self._workers: List[_GPUWorkerThread] = []
        for device in devices:
            worker = _GPUWorkerThread(
                device=device,
                task_queue=self._queue,
                task_store=self._task_store,
                trace_set=trace_set,
                config=config,
                shutdown_event=self._shutdown,
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Scheduler started with {len(devices)} GPU workers: {devices}")

    @property
    def trace_set(self) -> TraceSet:
        return self._trace_set

    @property
    def task_store(self) -> TaskStore:
        return self._task_store

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def workers(self) -> List["_GPUWorkerThread"]:
        return self._workers

    def submit_evaluate(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        *,
        baseline_cache_mode: BaselineCacheMode = BaselineCacheMode.REUSE,
        profile_baseline: Optional[bool] = None,
        run_baseline: Optional[bool] = None,
        timeout: Optional[int] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> str:
        """Submit a solution for evaluation. Returns task_id.

        ``profile_baseline`` / ``run_baseline`` / ``atol`` / ``rtol`` override the
        server-global ``BenchmarkConfig`` for this task only. ``None`` inherits the
        global value. When ``run_baseline=False``, every selected workload must
        declare reference outputs in safetensors and use only deterministic
        (safetensors/scalar) inputs.
        """
        if run_baseline is False:
            self._validate_run_baseline_false(solution.definition, workload_uuids)
        task_id = self._task_store.create_task(
            solution,
            workload_uuids,
            kind=TaskKind.EVALUATE,
            baseline_cache_mode=baseline_cache_mode,
            profile_baseline=profile_baseline,
            run_baseline=run_baseline,
            timeout=timeout,
            atol=atol,
            rtol=rtol,
        )
        self._queue.put(task_id)
        return task_id

    def _validate_run_baseline_false(
        self, definition_name: str, workload_uuids: Optional[List[str]]
    ) -> None:
        """Raise ValueError if any selected workload is incompatible with run_baseline=False."""
        traces = self._trace_set.workloads.get(definition_name, [])
        if workload_uuids:
            uuid_set = set(workload_uuids)
            traces = [t for t in traces if t.workload.uuid in uuid_set]
        if not traces:
            return  # _evaluate_task will surface the "no workloads" failure

        for tr in traces:
            wl = tr.workload
            if not wl.outputs:
                raise ValueError(
                    f"Workload '{wl.uuid}' has no `outputs`; required when run_baseline=False"
                )
            bad_inputs = [
                name
                for name, spec in wl.inputs.items()
                if spec.type not in ("safetensors", "scalar")
            ]
            if bad_inputs:
                raise ValueError(
                    f"Workload '{wl.uuid}' has non-deterministic inputs {bad_inputs}; "
                    "run_baseline=False requires all inputs to be safetensors or scalar"
                )

    def submit_profile(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        *,
        ncu_set: str = "detailed",
        ncu_sections: Optional[List[str]] = None,
        ncu_kernel_name: Optional[str] = None,
        ncu_page: str = "details",
        ncu_path: str = "ncu",
        ncu_timeout: int = 60,
        ncu_max_lines: Optional[int] = None,
    ) -> str:
        """Submit a solution for NCU profiling. Returns task_id."""
        task_id = self._task_store.create_task(
            solution,
            workload_uuids,
            kind=TaskKind.PROFILE,
            ncu_set=ncu_set,
            ncu_sections=ncu_sections,
            ncu_kernel_name=ncu_kernel_name,
            ncu_page=ncu_page,
            ncu_path=ncu_path,
            ncu_timeout=ncu_timeout,
            ncu_max_lines=ncu_max_lines,
        )
        self._queue.put(task_id)
        return task_id

    def submit_sanitize(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        *,
        sanitizer_types: Optional[List[str]] = None,
        sanitizer_path: str = "compute-sanitizer",
        sanitizer_timeout: int = 120,
        sanitizer_max_lines: Optional[int] = None,
        sanitizer_print_limit: Optional[int] = None,
    ) -> str:
        """Submit a solution for compute-sanitizer checks. Returns task_id."""
        task_id = self._task_store.create_task(
            solution,
            workload_uuids,
            kind=TaskKind.SANITIZE,
            sanitizer_types=sanitizer_types,
            sanitizer_path=sanitizer_path,
            sanitizer_timeout=sanitizer_timeout,
            sanitizer_max_lines=sanitizer_max_lines,
            sanitizer_print_limit=sanitizer_print_limit,
        )
        self._queue.put(task_id)
        return task_id

    def submit_debug(
        self,
        solution: Solution,
        workload_uuids: Optional[List[str]] = None,
        *,
        sanitizer_types: Optional[List[str]] = None,
        sanitizer_path: str = "compute-sanitizer",
        sanitizer_timeout: int = 120,
        sanitizer_max_lines: Optional[int] = None,
        sanitizer_print_limit: Optional[int] = 100,
        evaluation_timeout: Optional[int] = None,
        source_context_lines: int = 4,
        enable_coredump: bool = True,
        coredump_grace_seconds: float = 30,
    ) -> str:
        """Submit a solution for source-line-focused CUDA debugging. Returns task_id."""
        task_id = self._task_store.create_task(
            solution,
            workload_uuids,
            kind=TaskKind.DEBUG,
            sanitizer_types=sanitizer_types or ["memcheck"],
            sanitizer_path=sanitizer_path,
            sanitizer_timeout=sanitizer_timeout,
            sanitizer_max_lines=sanitizer_max_lines,
            sanitizer_print_limit=sanitizer_print_limit,
            timeout=evaluation_timeout,
            debug_source_context_lines=source_context_lines,
            debug_enable_coredump=enable_coredump,
            debug_coredump_grace_seconds=coredump_grace_seconds,
        )
        self._queue.put(task_id)
        return task_id

    def shutdown(self) -> None:
        self._shutdown.set()
        # Worker threads may be blocked in subprocess.run for compute-sanitizer
        # or ncu; killing the tracked process groups unblocks communicate() so
        # the threads can observe _shutdown and exit promptly. Without this,
        # each such thread would sit in the join timeout and the subprocess
        # (plus its _solution_runner grandchild) would keep running on the GPU
        # after the serve process exits.
        killed = kill_all_tracked_subprocesses()
        if killed:
            logger.info("Terminated %d in-flight managed subprocess group(s) on shutdown", killed)
        for worker in self._workers:
            worker.join(timeout=10)
        for worker in self._workers:
            worker.close()
        logger.info("Scheduler shut down")


class _GPUWorkerThread(threading.Thread):
    """Background thread owning a PersistentSubprocessWorker, processing tasks from the queue."""

    def __init__(
        self,
        device: str,
        task_queue: queue.Queue,
        task_store: TaskStore,
        trace_set: TraceSet,
        config: BenchmarkConfig,
        shutdown_event: threading.Event,
    ):
        super().__init__(daemon=True, name=f"gpu-worker-{device}")
        self._device = device
        self._queue = task_queue
        self._store = task_store
        self._trace_set = trace_set
        self._config = config
        self._shutdown = shutdown_event
        self._gpu_worker: Optional[PersistentSubprocessWorker] = None
        # Cache key -> baseline handle owned by the persistent worker.
        self._baseline_cache: dict[
            tuple[str, str, Optional[bool], Optional[bool]], BaselineHandle
        ] = {}
        self._worker_lock = threading.RLock()
        self._processing_task = False
        self._worker_starting = True

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_healthy(self) -> bool:
        return self._gpu_worker is not None and self._gpu_worker.is_healthy()

    @property
    def is_available(self) -> bool:
        """Return non-invasive worker liveness for HTTP health checks."""
        if self._shutdown.is_set() or not self.is_alive():
            return False
        with self._worker_lock:
            if self._gpu_worker is not None and self._gpu_worker.process_alive:
                return True
            if self._processing_task or self._worker_starting:
                return False
        return self._restart_worker("dead idle worker detected by health check")

    def close(self) -> None:
        with self._worker_lock:
            if self._gpu_worker:
                self._gpu_worker.close()
                self._gpu_worker = None

    def _exit_backend(self, reason: str) -> None:
        logger.error("%s; exiting backend process for supervisor restart", reason)
        os._exit(1)

    def _restart_worker(self, reason: str, *, force: bool = False) -> bool:
        with self._worker_lock:
            if self._shutdown.is_set():
                return False
            if not force and self._gpu_worker is not None and self._gpu_worker.process_alive:
                return True

            logger.warning("Worker on %s unhealthy: %s; restarting", self._device, reason)
            try:
                self._worker_starting = True
                if self._gpu_worker is None:
                    self._gpu_worker = PersistentSubprocessWorker(self._device)
                elif not self._gpu_worker.restart():
                    self._exit_backend(f"Failed to restart worker on {self._device}")
                self._baseline_cache.clear()
                if not self._gpu_worker.process_alive:
                    self._exit_backend(f"Worker on {self._device} is still dead after restart")
                return True
            except Exception as e:
                self._exit_backend(f"Failed to restart worker on {self._device}: {e}")
            finally:
                self._worker_starting = False
        return False

    def run(self) -> None:
        try:
            worker = PersistentSubprocessWorker(self._device)
            with self._worker_lock:
                self._gpu_worker = worker
                self._worker_starting = False
        except Exception as e:
            with self._worker_lock:
                self._worker_starting = False
            logger.error(f"Failed to start GPU worker on {self._device}: {e}")
            self._exit_backend(f"Failed to start GPU worker on {self._device}")

        while not self._shutdown.is_set():
            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task = self._store.get_task(task_id)
            if task is None:
                continue

            self._store.mark_running(task_id)
            with self._worker_lock:
                self._processing_task = True
            try:
                if task.kind == TaskKind.EVALUATE:
                    traces = self._evaluate_task(task)
                    self._store.complete_task(task_id, traces=traces)
                elif task.kind == TaskKind.PROFILE:
                    logs = self._profile_task(task)
                    self._store.complete_task(task_id, logs=logs)
                elif task.kind == TaskKind.SANITIZE:
                    logs = self._sanitize_task(task)
                    self._store.complete_task(task_id, logs=logs)
                elif task.kind == TaskKind.DEBUG:
                    logs = self._debug_task(task)
                    self._store.complete_task(task_id, logs=logs)
                else:
                    raise ValueError(f"Unknown task kind: {task.kind}")
            except Exception as e:
                logger.error(f"Task {task_id} failed on {self._device}: {e}")
                self._store.fail_task(task_id, str(e))
                if not self._gpu_worker.is_healthy():
                    self._restart_worker("task failure left worker unhealthy", force=True)
            finally:
                with self._worker_lock:
                    self._processing_task = False

    def _resolve_workloads(self, task: Task) -> List[Workload]:
        """Return the list of Workload objects for this task's definition + uuid filter."""
        workload_traces = self._trace_set.workloads.get(task.definition_name, [])
        if task.workload_uuids:
            uuid_set = set(task.workload_uuids)
            workload_traces = [t for t in workload_traces if t.workload.uuid in uuid_set]

        if not workload_traces:
            raise ValueError(f"No workloads found for definition: {task.definition_name}")

        return [t.workload for t in workload_traces]

    def _evaluate_task(self, task: Task) -> List[Trace]:
        definition = self._trace_set.definitions.get(task.definition_name)
        if definition is None:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workloads = self._resolve_workloads(task)

        cfg_overrides = {}
        if task.atol is not None:
            cfg_overrides["atol"] = task.atol
        if task.rtol is not None:
            cfg_overrides["rtol"] = task.rtol
        if task.timeout is not None:
            cfg_overrides["timeout_seconds"] = task.timeout
        task_cfg = self._config.model_copy(update=cfg_overrides) if cfg_overrides else self._config

        traces = []
        for workload in workloads:
            bypass_cache = task.baseline_cache_mode == BaselineCacheMode.BYPASS
            ref_handle = None
            try:
                if bypass_cache:
                    ref_handle = self._build_ref(
                        definition,
                        workload,
                        cfg=task_cfg,
                        profile_baseline=task.profile_baseline,
                        run_baseline=task.run_baseline,
                    )
                else:
                    ref_handle = self._get_or_build_ref(
                        definition,
                        workload,
                        cfg=task_cfg,
                        profile_baseline=task.profile_baseline,
                        run_baseline=task.run_baseline,
                    )
                evaluation = self._gpu_worker.run_solution(task.solution, ref_handle, task_cfg)
                trace = Trace(
                    definition=task.definition_name,
                    workload=workload,
                    solution=task.solution.name,
                    evaluation=evaluation,
                )
                traces.append(trace)

                # A timed-out worker may still be running the previous CUDA call; kill it
                # before the backend accepts another task. Other failures get a CUDA
                # health check so illegal accesses do not poison later workloads.
                if evaluation.status == EvaluationStatus.TIMEOUT:
                    self._restart_worker("TIMEOUT left worker state unknown", force=True)
                elif evaluation.status != EvaluationStatus.PASSED:
                    if not self._gpu_worker.is_healthy():
                        self._restart_worker(
                            f"{evaluation.status} left worker unhealthy", force=True
                        )
            finally:
                if bypass_cache and ref_handle is not None:
                    self._gpu_worker.release(ref_handle)

        return traces

    def _profile_task(self, task: Task) -> List[RunLog]:
        """Run NCU profiling per workload and return logs."""
        if task.definition_name not in self._trace_set.definitions:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workloads = self._resolve_workloads(task)
        trace_set_path = str(self._trace_set.root) if self._trace_set.root else None

        logs: List[RunLog] = []
        for workload in workloads:
            log = flashinfer_bench_run_ncu(
                task.solution,
                workload,
                device=self._device,
                trace_set_path=trace_set_path,
                set=task.ncu_set,
                sections=task.ncu_sections,
                kernel_name=task.ncu_kernel_name,
                page=task.ncu_page,
                ncu_path=task.ncu_path,
                timeout=task.ncu_timeout,
                max_lines=task.ncu_max_lines,
            )
            logs.append(
                RunLog(
                    definition=task.definition_name,
                    workload=workload.model_dump(mode="json"),
                    solution=task.solution.name,
                    log=log,
                )
            )
        return logs

    def _sanitize_task(self, task: Task) -> List[RunLog]:
        """Run compute-sanitizer per workload and return logs."""
        if task.definition_name not in self._trace_set.definitions:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workloads = self._resolve_workloads(task)
        trace_set_path = str(self._trace_set.root) if self._trace_set.root else None

        logs: List[RunLog] = []
        for workload in workloads:
            log = flashinfer_bench_run_sanitizer(
                task.solution,
                workload,
                device=self._device,
                trace_set_path=trace_set_path,
                sanitizer_types=task.sanitizer_types,  # type: ignore[arg-type]
                sanitizer_path=task.sanitizer_path,
                timeout=task.sanitizer_timeout,
                max_lines=task.sanitizer_max_lines,
                print_limit=task.sanitizer_print_limit,
            )
            logs.append(
                RunLog(
                    definition=task.definition_name,
                    workload=workload.model_dump(mode="json"),
                    solution=task.solution.name,
                    log=log,
                )
            )

            # Sanitizer runs kernels that may have crashed; check worker health.
            if not self._gpu_worker.is_healthy():
                self._restart_worker("sanitize left worker unhealthy", force=True)

        return logs

    def _debug_task(self, task: Task) -> List[RunLog]:
        """Run CUDA debug tooling per workload and return annotated logs."""
        if task.definition_name not in self._trace_set.definitions:
            raise ValueError(f"Definition not found: {task.definition_name}")

        workloads = self._resolve_workloads(task)
        trace_set_path = str(self._trace_set.root) if self._trace_set.root else None

        logs: List[RunLog] = []
        for workload in workloads:
            evaluation_timeout = (
                task.timeout if task.timeout is not None else self._config.timeout_seconds
            )
            debug_result = flashinfer_bench_debug_solution(
                task.solution,
                workload,
                device=self._device,
                trace_set_path=trace_set_path,
                sanitizer_types=task.sanitizer_types,  # type: ignore[arg-type]
                sanitizer_path=task.sanitizer_path,
                timeout=task.sanitizer_timeout,
                evaluation_timeout=evaluation_timeout,
                max_lines=task.sanitizer_max_lines,
                print_limit=task.sanitizer_print_limit,
                source_context_lines=task.debug_source_context_lines,
                enable_coredump=task.debug_enable_coredump,
                coredump_grace_seconds=task.debug_coredump_grace_seconds,
            )
            logs.append(
                RunLog(
                    definition=task.definition_name,
                    workload=workload.model_dump(mode="json"),
                    solution=task.solution.name,
                    metadata=debug_result["metadata"],
                )
            )

            if not self._gpu_worker.is_healthy():
                self._restart_worker("debug left worker unhealthy", force=True)

        return logs

    def _get_or_build_ref(
        self,
        definition: Definition,
        workload: Workload,
        *,
        cfg: Optional[BenchmarkConfig] = None,
        profile_baseline: Optional[bool] = None,
        run_baseline: Optional[bool] = None,
    ) -> BaselineHandle:
        """Get cached reference or build a new one.

        Cache key includes the per-request overrides so requests with different
        ``run_baseline`` / ``profile_baseline`` values do not share baselines.
        ``atol`` / ``rtol`` are not part of the cache key because they are not
        used during baseline construction.
        """
        key = (definition.name, workload.uuid, profile_baseline, run_baseline)
        if key in self._baseline_cache:
            return self._baseline_cache[key]

        handle = self._build_ref(
            definition,
            workload,
            cfg=cfg,
            profile_baseline=profile_baseline,
            run_baseline=run_baseline,
        )
        self._baseline_cache[key] = handle
        return handle

    def _build_ref(
        self,
        definition: Definition,
        workload: Workload,
        *,
        cfg: Optional[BenchmarkConfig] = None,
        profile_baseline: Optional[bool] = None,
        run_baseline: Optional[bool] = None,
    ) -> BaselineHandle:
        """Build and return a baseline without reading or populating the cache."""
        eff_cfg = cfg if cfg is not None else self._config
        resolved = eff_cfg.resolve_eval_config(
            definition, profile_baseline=profile_baseline, run_baseline=run_baseline
        )
        estimated_bytes = estimate_baseline_bytes(definition, workload, resolved.num_trials)
        eval_reserve_bytes = estimate_eval_reserve_bytes(definition, workload, resolved.num_trials)
        # Clear cached baselines under memory pressure so both the new cached baseline
        # and the live evaluation working set fit the memory budget.
        self._admit_baseline(definition, workload, estimated_bytes, eval_reserve_bytes)

        handle = self._gpu_worker.run_ref(
            definition,
            workload,
            eff_cfg,
            self._trace_set.root,
            profile_baseline=profile_baseline,
            run_baseline=run_baseline,
        )
        return handle

    # ── Memory-aware baseline cache admission ──

    def _device_index(self) -> int:
        return int(self._device.split(":")[1])

    def _get_device_total_bytes(self) -> int:
        """Total HBM on this worker's device, in bytes."""
        _free, total = torch.cuda.mem_get_info(self._device_index())
        return total

    def _get_device_used_bytes(self) -> int:
        """Real used bytes on this worker's device (total - free), in bytes.

        Reads actual device memory so the estimate is checked against true
        occupancy, including the worker subprocess and any other process.
        """
        free, total = torch.cuda.mem_get_info(self._device_index())
        return total - free

    def _empty_device_cache(self) -> None:
        """Release cached allocator blocks for this worker's device, best-effort."""
        try:
            with torch.cuda.device(self._device_index()):
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"empty_cache failed on {self._device}: {e}")

    def _clear_cached_baselines(self) -> int:
        """Release cached baselines for this worker and return the tracked count."""
        tracked_count = len(self._baseline_cache)
        if hasattr(self._gpu_worker, "clear_baselines"):
            cleared_count = self._gpu_worker.clear_baselines()
        else:
            for handle in list(self._baseline_cache.values()):
                self._gpu_worker.release(handle)
            cleared_count = tracked_count
        self._baseline_cache.clear()
        self._empty_device_cache()
        logger.info(
            "Cleared %d cached baseline(s) on %s to free GPU memory (%d tracked by scheduler)",
            cleared_count,
            self._device,
            tracked_count,
        )
        return cleared_count

    def _admit_baseline(
        self,
        definition: Definition,
        workload: Workload,
        estimated_bytes: int,
        eval_reserve_bytes: int,
    ) -> None:
        """Make room for a new baseline and its live evaluation within the budget.

        If the next workload does not fit, clears the worker's cached baselines and
        then re-reads real device memory before deciding whether to fail.
        ``estimated_bytes`` is the cached baseline payload (held in this process);
        ``eval_reserve_bytes`` reserves the transient working set the worker
        subprocess needs to evaluate this workload (see ``estimate_eval_reserve_bytes``),
        without which a heavy evaluation can OOM even while the cache is within budget.
        If the budget cannot be satisfied even after clearing cached baselines on
        this worker, the task fails early with a clear error.
        """
        total_bytes = self._get_device_total_bytes()
        budget_bytes = self._config.max_mem_ratio * total_bytes
        required_bytes = estimated_bytes + eval_reserve_bytes

        used_bytes = self._get_device_used_bytes()
        if required_bytes + used_bytes < budget_bytes:
            return

        self._clear_cached_baselines()
        used_bytes = self._get_device_used_bytes()
        if required_bytes + used_bytes < budget_bytes:
            return

        raise RuntimeError(
            f"Baseline for definition '{definition.name}' workload '{workload.uuid}' "
            f"needs ~{estimated_bytes} cache bytes + ~{eval_reserve_bytes} eval bytes "
            f"but only ~{max(0.0, budget_bytes - used_bytes):.0f} bytes fit within the "
            f"memory budget (max_mem_ratio={self._config.max_mem_ratio} * {total_bytes} "
            f"total HBM bytes) on {self._device} after clearing cached baselines."
        )
