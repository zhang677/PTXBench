"""Persistent runner that reuses worker processes across solutions."""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import multiprocessing as mp

import flashinfer_bench.utils as fib_utils
from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.utils import make_eval
from flashinfer_bench.compile import BuilderRegistry, BuildError
from flashinfer_bench.data import Definition, Evaluation, EvaluationStatus, Solution, Workload
from flashinfer_bench.utils import redirect_stdio_to_tempfile, set_parent_death_signal

from .runner import BaselineHandle, DeviceBaseline, Runner, RunnerError, RunnerFatalError
from .gpu_memory import GpuMemoryMonitor

logger = logging.getLogger(__name__)


class WorkerCommand(Enum):
    BUILD_BASELINE = "build_baseline"
    RELEASE_BASELINE = "release_baseline"
    RUN_SOLUTION = "run_solution"
    HEALTH_CHECK = "health_check"
    CLEAR_BASELINES = "clear_baselines"
    SHUTDOWN = "shutdown"


class WorkerResponse(Enum):
    BASELINE_READY = "baseline_ready"
    READY = "ready"
    EVALUATION = "evaluation"
    ERROR = "error"
    HEALTHY = "healthy"
    CORRUPTED = "corrupted"
    BASELINES_CLEARED = "baselines_cleared"


@dataclass
class SolutionFailureRecord:
    """Track failures for a solution."""

    solution_name: str
    failure_count: int
    last_error: str
    last_status: EvaluationStatus
    last_failure_time: float


class PersistentSubprocessWorker:
    def __init__(self, device: str) -> None:
        """Per device persistent subprocess worker

        Parameters
        ----------
        device : str
            Device string (e.g. "cuda:0").
        """
        self._device = device
        self._baselines: Dict[BaselineHandle, Definition] = {}
        self._memory = GpuMemoryMonitor(device)
        self._environment = None

        # Solution failure tracking
        self._failure_records: Dict[str, SolutionFailureRecord] = {}
        self._max_failures = 3  # if a solution fails for more than 3 times, it will be skipped

        self._worker_proc: Optional[mp.Process] = None
        self._parent_conn: Optional[mp.connection.Connection] = None
        self._worker_pgid: Optional[int] = None

        self._start_worker()

    def _start_worker(self) -> None:
        if self._worker_proc is not None and self._worker_proc.is_alive():
            self._shutdown_worker()

        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe(duplex=True)

        self._worker_proc = ctx.Process(
            target=_persistent_worker_main, args=(child_conn, self._device), daemon=True
        )
        self._worker_proc.start()
        child_conn.close()

        try:
            if not self._parent_conn.poll(timeout=120):
                raise RunnerFatalError("Worker startup timed out")
            msg = self._parent_conn.recv()
            if msg.get("cmd") == WorkerResponse.READY.value:
                self._environment = msg["environment"]
                self._worker_pgid = self._get_worker_pgid()
                if self._worker_pgid is None:
                    raise RunnerFatalError("Worker has no isolated process group")
                logger.info(f"Persistent worker started for device {self._device}")
            else:
                raise RunnerFatalError(f"Worker failed to start: {msg}")
        except Exception as e:
            self._shutdown_worker()
            raise RunnerFatalError(f"Failed to start worker: {e}")

    def _get_worker_pgid(self) -> Optional[int]:
        if self._worker_proc is None:
            return None
        try:
            pgid = os.getpgid(self._worker_proc.pid)
        except (ProcessLookupError, OSError):
            return None
        if pgid == os.getpgrp():
            logger.warning("Worker for device %s did not enter its own process group", self._device)
            return None
        return pgid

    def _kill_worker_process_group(self) -> None:
        if self._worker_pgid is None:
            return
        try:
            os.killpg(self._worker_pgid, 9)
        except ProcessLookupError:
            pass
        except PermissionError as e:
            logger.warning(
                "Failed to kill worker process group %s on %s: %s",
                self._worker_pgid,
                self._device,
                e,
            )

    def _shutdown_worker(self) -> None:
        proc = self._worker_proc
        if self._parent_conn is not None:
            try:
                self._parent_conn.send({"cmd": WorkerCommand.SHUTDOWN.value})
                self._parent_conn.close()
            except (OSError, EOFError):
                pass
            self._parent_conn = None
        if proc is not None:
            proc.join(timeout=5)
            # Signal the group even when its leader exited: descendants can
            # retain CUDA contexts after the Python worker has gone away.
            self._kill_worker_process_group()
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=10)
            if proc.is_alive():
                raise RunnerFatalError(f"GPU worker {proc.pid} did not exit")
            self._worker_proc = None
            self._worker_pgid = None
        self._baselines.clear()
        reclaimed = self._memory.wait_until_reclaimed()
        logger.info(
            "GPU state reclaimed on %s: used=%d idle=%d",
            self._device,
            reclaimed.used,
            self._memory.idle.used,
        )

    def memory_snapshot(self):
        return self._memory.snapshot()

    def _make_eval(self, **kwargs) -> Evaluation:
        return make_eval(**kwargs, environment=self._environment)

    def _baseline_command(self, message, timeout=300):
        if not self.process_alive:
            raise RunnerFatalError("GPU worker is not running")
        self._parent_conn.send(message)
        if not self._parent_conn.poll(timeout):
            raise RunnerFatalError("GPU baseline command timed out")
        response = self._parent_conn.recv()
        if response.get("cmd") == WorkerResponse.ERROR.value:
            raise RunnerError(response.get("error", "GPU baseline command failed"))
        return response

    def is_healthy(self) -> bool:
        if (
            self._parent_conn is None
            or self._worker_proc is None
            or not self._worker_proc.is_alive()
        ):
            return False

        # Check if connection is closed
        if self._parent_conn.closed:
            logger.warning(f"Connection is closed for device {self._device}")
            return False

        try:
            self._parent_conn.send({"cmd": WorkerCommand.HEALTH_CHECK.value})

            if self._parent_conn.poll(timeout=5.0):
                try:
                    msg = self._parent_conn.recv()

                    if msg.get("cmd") == WorkerResponse.HEALTHY.value:
                        return True
                    elif msg.get("cmd") == WorkerResponse.CORRUPTED.value:
                        logger.warning(f"GPU context corrupted on device {self._device}")
                        return False
                    else:
                        logger.warning(
                            f"Unexpected health check response on device {self._device}: {msg}"
                        )
                        return False

                except (EOFError, ConnectionResetError, OSError) as e:
                    logger.warning(
                        f"Connection error during health check on device {self._device}: {e}"
                    )
                    return False
                except Exception as e:
                    error_str = str(e).lower()
                    if (
                        "ran out of input" in error_str
                        or "pickle" in error_str
                        or "unpickling" in error_str
                    ):
                        logger.warning(
                            f"Connection closed or corrupted during health check on device {self._device}: {e}"
                        )
                    else:
                        logger.warning(
                            f"Failed to decode health check response on device {self._device}: {e}"
                        )
                    return False
            else:
                logger.warning(f"Health check timeout on device {self._device}")
                return False

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Connection broken during health check on device {self._device}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Health check failed on device {self._device}: {e}")
            return False

    @property
    def process_alive(self) -> bool:
        """Return worker process liveness without touching the command pipe."""
        return (
            self._parent_conn is not None
            and not self._parent_conn.closed
            and self._worker_proc is not None
            and self._worker_proc.is_alive()
        )

    def restart(self) -> bool:
        """Restart the worker process.

        Returns
        -------
        bool
            True if restart was successful, False otherwise.
        """
        try:
            logger.info(f"Restarting worker for device {self._device}")

            self._baselines.clear()
            self._failure_records.clear()
            self._shutdown_worker()
            self._start_worker()

            logger.info(f"Successfully restarted worker for device {self._device}")
            return True

        except Exception as e:
            logger.error(f"Failed to restart worker for device {self._device}: {e}")
            return False

    def _should_skip_solution(self, solution_name: str) -> Optional[SolutionFailureRecord]:
        if solution_name in self._failure_records:
            record = self._failure_records[solution_name]
            if record.failure_count >= self._max_failures:
                return record
        return None

    def _record_failure(self, solution_name: str, error: str, status: EvaluationStatus) -> None:
        if solution_name in self._failure_records:
            record = self._failure_records[solution_name]
            if status in (EvaluationStatus.COMPILE_ERROR, EvaluationStatus.TIMEOUT):
                record.failure_count = self._max_failures
            else:
                record.failure_count += 1
            record.last_error = error
            record.last_status = status
            record.last_failure_time = time.time()
        else:
            failure_count = (
                self._max_failures
                if status in (EvaluationStatus.COMPILE_ERROR, EvaluationStatus.TIMEOUT)
                else 1
            )
            self._failure_records[solution_name] = SolutionFailureRecord(
                solution_name=solution_name,
                failure_count=failure_count,
                last_error=error,
                last_status=status,
                last_failure_time=time.time(),
            )

    def _clear_failure_record(self, solution_name: str) -> None:
        self._failure_records.pop(solution_name, None)

    def run_ref(
        self,
        definition: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        trace_set_root: Optional[Path] = None,
        *,
        profile_baseline: Optional[bool] = None,
        run_baseline: Optional[bool] = None,
    ) -> BaselineHandle:
        response = self._baseline_command(
            {
                "cmd": WorkerCommand.BUILD_BASELINE.value,
                "definition": definition,
                "workload": workload,
                "config": cfg,
                "trace_set_root": trace_set_root,
                "profile_baseline": profile_baseline,
                "run_baseline": run_baseline,
            },
            timeout=max(300, cfg.timeout_seconds),
        )
        if response.get("cmd") != WorkerResponse.BASELINE_READY.value:
            raise RunnerFatalError(f"Invalid baseline response: {response}")
        handle = response["handle"]
        self._baselines[handle] = definition
        return handle

    def run_solution(
        self, solution: Solution, baseline: BaselineHandle, cfg: BenchmarkConfig
    ) -> Evaluation:
        """Run solution using cached compilation."""
        if baseline not in self._baselines:
            raise RunnerError(f"Baseline handle not found: {baseline}")

        solution_name = solution.name
        failure_record = self._should_skip_solution(solution_name)
        if failure_record is not None:
            logger.info(
                f"Skipping solution {solution.name} due to {failure_record.failure_count} consecutive failures"
            )
            return self._make_eval(
                status=failure_record.last_status,
                device=self._device,
                extra_msg=f"Solution skipped after {failure_record.failure_count} failures. Last error: {failure_record.last_error}",
            )

        # Pre-allocate a log file path so the parent can read subprocess
        # output even if the worker crashes before sending a response.
        log_fd, worker_log_path = tempfile.mkstemp(suffix=".log", prefix="fib_")
        os.close(log_fd)

        eval_msg = {
            "cmd": WorkerCommand.RUN_SOLUTION.value,
            "baseline": baseline,
            "solution": solution,
            "config": cfg,
            "solution_name": solution.name,
            "log_path": worker_log_path,
        }

        if self._parent_conn is None or self._parent_conn.closed:
            error_msg = "Connection is closed or invalid"
            try:
                os.unlink(worker_log_path)
            except OSError:
                pass
            return self._make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=self._device, extra_msg=error_msg
            )

        try:
            self._parent_conn.send(eval_msg)

            if self._parent_conn.poll(timeout=cfg.timeout_seconds):
                try:
                    response = self._parent_conn.recv()

                    if response.get("cmd") == WorkerResponse.EVALUATION.value:
                        evaluation = response["evaluation"]
                        if evaluation.status == EvaluationStatus.PASSED:
                            self._clear_failure_record(solution.name)
                        elif evaluation.status in (
                            EvaluationStatus.RUNTIME_ERROR,
                            EvaluationStatus.INCORRECT_SHAPE,
                            EvaluationStatus.INCORRECT_DTYPE,
                            EvaluationStatus.COMPILE_ERROR,
                        ):
                            error_text = (evaluation.log or "").strip() or "Evaluation failed"
                            self._record_failure(solution.name, error_text, evaluation.status)
                        # Worker already consumed the log file via make_eval;
                        # clean up defensively in case it still exists.
                        try:
                            os.unlink(worker_log_path)
                        except OSError:
                            pass
                        return evaluation
                    elif response.get("cmd") == WorkerResponse.ERROR.value:
                        error_msg = response.get("error", "Unknown evaluation error")
                        self._record_failure(
                            solution.name, error_msg, EvaluationStatus.RUNTIME_ERROR
                        )
                        return self._make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=self._device,
                            log_path=worker_log_path,
                            extra_msg=error_msg,
                        )
                    else:
                        error_msg = f"Unexpected evaluation response: {response}"
                        self._record_failure(
                            solution.name, error_msg, EvaluationStatus.RUNTIME_ERROR
                        )
                        return self._make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=self._device,
                            log_path=worker_log_path,
                            extra_msg=error_msg,
                        )

                except (EOFError, ConnectionResetError, OSError) as e:
                    error_msg = f"Runtime error during evaluation ({type(e).__name__}): {e!r}"
                    return self._make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=self._device,
                        log_path=worker_log_path,
                        extra_msg=error_msg,
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if (
                        "ran out of input" in error_str
                        or "pickle" in error_str
                        or "unpickling" in error_str
                    ):
                        error_msg = f"Connection closed or corrupted during evaluation: {e}"
                    else:
                        error_msg = f"Failed to decode evaluation response: {e}"
                    return self._make_eval(
                        status=EvaluationStatus.RUNTIME_ERROR,
                        device=self._device,
                        log_path=worker_log_path,
                        extra_msg=error_msg,
                    )
            else:
                error_msg = f"Evaluation timeout after {cfg.timeout_seconds} seconds for solution {solution.name}"
                self._record_failure(solution.name, error_msg, EvaluationStatus.TIMEOUT)
                return self._make_eval(
                    status=EvaluationStatus.TIMEOUT,
                    device=self._device,
                    log_path=worker_log_path,
                    extra_msg=error_msg,
                )

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            error_msg = f"Runtime error during evaluation ({type(e).__name__}): {e!r}"
            return self._make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_path=worker_log_path,
                extra_msg=error_msg,
            )
        except Exception as e:
            error_msg = f"Failed to communicate with worker: {e}"
            return self._make_eval(
                status=EvaluationStatus.RUNTIME_ERROR,
                device=self._device,
                log_path=worker_log_path,
                extra_msg=error_msg,
            )

    def release(self, baseline: BaselineHandle) -> None:
        if baseline not in self._baselines:
            return
        response = self._baseline_command(
            {"cmd": WorkerCommand.RELEASE_BASELINE.value, "baseline": baseline}
        )
        if response.get("cmd") != WorkerResponse.BASELINES_CLEARED.value:
            raise RunnerFatalError(f"Invalid release response: {response}")
        self._baselines.pop(baseline, None)

    def clear_baselines(self) -> int:
        response = self._baseline_command({"cmd": WorkerCommand.CLEAR_BASELINES.value})
        if response.get("cmd") != WorkerResponse.BASELINES_CLEARED.value:
            raise RunnerFatalError(f"Invalid clear response: {response}")
        self._baselines.clear()
        return response["count"]

    def close(self) -> None:
        self._shutdown_worker()
        self._baselines.clear()
        self._failure_records.clear()


class PersistentRunner(Runner):
    def __init__(self) -> None:
        """Initialize the persistent runner with multiple workers."""
        # Track retry attempts for each device
        self._device_retry_counts: Dict[str, int] = {}
        self._worker_max_retries = 3

        self._available_devices = fib_utils.list_cuda_devices()
        self._workers = [PersistentSubprocessWorker(d) for d in self._available_devices]

        self._curr_worker_idx = 0

        if len(self._workers) == 0:
            raise RuntimeError("No CUDA devices available")

        logger.info(
            f"Initialized benchmark persistent runner on {len(self._available_devices)} CUDA devices "
            f"and {len(self._workers)} workers"
        )

    def _pick_workers(self, K: int) -> list[PersistentSubprocessWorker]:
        """Pick K workers in round-robin fashion."""
        if K <= 0 or not self._workers:
            return []
        D = len(self._workers)
        start = self._curr_worker_idx
        sel = [self._workers[(start + i) % D] for i in range(min(K, D))]
        self._curr_worker_idx = (start + K) % D
        return sel

    def _handle_failed_workers(
        self, failed_workers: List[PersistentSubprocessWorker], increment_retries: bool = True
    ) -> None:
        """Handle failed workers by attempting to restart them or removing them.

        Parameters
        ----------
        failed_workers : List[PersistentSubprocessWorker]
            List of workers that have failed.
        increment_retries : bool, optional
            Whether to increment retry count (True for health failures, False for solution failures), by default True.
        """
        workers_to_remove = []

        for failed_worker in failed_workers:
            device = failed_worker._device
            retry_count = self._device_retry_counts.get(device, 0)

            if retry_count < self._worker_max_retries:
                if increment_retries:
                    self._device_retry_counts[device] = retry_count + 1
                    new_retry_count = retry_count + 1
                else:
                    new_retry_count = retry_count

                if failed_worker.restart():
                    logger.info(f"Successfully restarted persistent worker for device {device}")
                else:
                    logger.error(f"Failed to restart persistent worker for device {device}")
                    if new_retry_count >= self._worker_max_retries:
                        workers_to_remove.append(failed_worker)
                        logger.warning(
                            f"Removing device {device} after {self._worker_max_retries} failed attempts"
                        )
            else:
                workers_to_remove.append(failed_worker)
                logger.warning(
                    f"Removing device {device} after {self._worker_max_retries} failed attempts"
                )

        if workers_to_remove:
            for worker in workers_to_remove:
                try:
                    worker.close()
                except Exception:
                    pass
            self._workers = [r for r in self._workers if r not in workers_to_remove]

        if self._workers:
            self._curr_worker_idx %= len(self._workers)

    def _has_healthy_workers(self) -> bool:
        return bool(self._workers)

    def run_workload(
        self,
        definition: Definition,
        workload: Workload,
        solutions: List[Solution],
        config: BenchmarkConfig,
        root: Path,
    ) -> Dict[str, Evaluation]:
        """Run a workload with the given solutions and return evaluation results.

        Parameters
        ----------
        definition : Definition
            Operation definition.
        workload : Workload
            Workload specification.
        solutions : List[Solution]
            List of solutions to evaluate.
        config : BenchmarkConfig
            Benchmark configuration.
        root : Path
            Root path for the trace set.

        Returns
        -------
        Dict[str, Evaluation]
            Dictionary mapping solution names to their evaluations.
        """
        if not solutions:
            return {}

        K = min(len(self._workers), len(solutions))
        selected = self._pick_workers(K)
        if not selected:
            raise RuntimeError("No healthy persistent workers available")

        # Build baselines on each worker
        baselines: dict[PersistentSubprocessWorker, BaselineHandle] = {}
        failed_workers: list[PersistentSubprocessWorker] = []

        with ThreadPoolExecutor(max_workers=K) as pool:
            baseline_futs = {
                pool.submit(r.run_ref, definition, workload, config, root): r for r in selected
            }
            for fut, r in baseline_futs.items():
                try:
                    h = fut.result()
                    baselines[r] = h
                except Exception as e:
                    failed_workers.append(r)
                    logger.error(
                        f"Persistent worker {r._device} failed while running reference for "
                        f"definition={definition.name} workload={workload.uuid}: {e}"
                    )

        if failed_workers:
            self._handle_failed_workers(failed_workers, increment_retries=True)
            if not self._has_healthy_workers():
                raise RuntimeError("No healthy persistent workers available")

        # Filter out workers that failed to build baselines
        selected = [r for r in selected if r in baselines]
        if not selected:
            raise RuntimeError("No healthy persistent workers available after baseline setup")

        def run_solution_with_health_check(
            worker: PersistentSubprocessWorker, solution: Solution, baseline_handle: BaselineHandle
        ) -> Evaluation:
            try:
                if not worker.is_healthy():
                    logger.warning(
                        f"Worker on device {worker._device} is unhealthy, attempting restart"
                    )
                    if worker.restart():
                        try:
                            new_baseline = worker.run_ref(definition, workload, config, root)
                            worker.release(baseline_handle)
                            baseline_handle = new_baseline
                            logger.info(f"Rebuilt baseline for worker on device {worker._device}")
                        except Exception as e:
                            logger.error(
                                f"Failed to rebuild baseline after restart for device {worker._device}: {e}"
                            )
                            return worker._make_eval(
                                status=EvaluationStatus.RUNTIME_ERROR,
                                device=worker._device,
                                extra_msg=f"Failed to rebuild baseline after restart: {e}",
                            )
                    else:
                        logger.error(f"Failed to restart worker on device {worker._device}")
                        return worker._make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR,
                            device=worker._device,
                            extra_msg="Worker restart failed",
                        )

                # Run the solution
                eval_start_time = time.perf_counter()
                result = worker.run_solution(solution, baseline_handle, config)
                eval_time = time.perf_counter() - eval_start_time
                logger.info(
                    f"Solution '{solution.name}' workload={workload.uuid}: "
                    f"{result.status.value} evaluation time={eval_time:.1f}s"
                )
                return result

            except Exception as e:
                logger.error(f"Unexpected error in solution execution for {solution.name}: {e}")
                return worker._make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    device=worker._device,
                    extra_msg=f"Unexpected error: {e}",
                )

        try:
            with ThreadPoolExecutor(max_workers=len(selected)) as pool:
                sol_futs: Dict[str, any] = {}

                for i, solution in enumerate(solutions):
                    worker = selected[i % len(selected)]
                    baseline_handle = baselines[worker]

                    sol_futs[solution.name] = pool.submit(
                        run_solution_with_health_check, worker, solution, baseline_handle
                    )

                results: Dict[str, Evaluation] = {
                    name: fut.result() for name, fut in sol_futs.items()
                }
        finally:
            # Clean up baselines
            for r in selected:
                if r in baselines:
                    try:
                        r.release(baselines[r])
                    except Exception as e:
                        logger.warning(f"Failed to release baseline for device {r._device}: {e}")

        return results

    def close(self) -> None:
        """Release all resources and terminate all worker processes."""
        for worker in self._workers:
            try:
                worker.close()
            except Exception as e:
                logger.warning(f"Failed to close worker for device {worker._device}: {e}")
        self._workers.clear()


def _persistent_worker_main(conn: mp.connection.Connection, device: str) -> None:
    """Long-lived worker process that handles solution evaluations.

    Caches compiled solutions to avoid recompilation (handled in builder registry).

    Parameters
    ----------
    conn : mp.connection.Connection
        Multiprocessing connection for communication with parent process.
    device : str
        Device string (e.g. "cuda:0").
    """
    try:
        os.setsid()
        set_parent_death_signal()

        torch.cuda.set_device(int(device.split(":")[1]))
        registry = BuilderRegistry.get_instance()
        baselines: Dict[BaselineHandle, DeviceBaseline] = {}
        conn.send(
            {"cmd": WorkerResponse.READY.value, "environment": fib_utils.env_snapshot(device)}
        )

        while True:
            try:
                msg = conn.recv()
                cmd = msg.get("cmd")

                if cmd == WorkerCommand.SHUTDOWN.value:
                    print("Shutting down worker")
                    break

                elif cmd == WorkerCommand.HEALTH_CHECK.value:
                    try:
                        # GPU health check
                        test_tensor = torch.zeros(1, device=device)
                        test_tensor += 1
                        torch.cuda.synchronize(device=device)
                        del test_tensor
                        conn.send({"cmd": WorkerResponse.HEALTHY.value})
                    except Exception:
                        print("Worker failed health check")
                        conn.send({"cmd": WorkerResponse.CORRUPTED.value})
                        break

                elif cmd == WorkerCommand.BUILD_BASELINE.value:
                    try:
                        definition = msg["definition"]
                        cfg = msg["config"]
                        eval_cfg = cfg.resolve_eval_config(
                            definition,
                            profile_baseline=msg["profile_baseline"],
                            run_baseline=msg["run_baseline"],
                        )
                        built = resolve_evaluator(definition).build_baseline(
                            definition=definition,
                            workload=msg["workload"],
                            cfg=eval_cfg,
                            device=device,
                            trace_set_root=msg["trace_set_root"],
                        )
                        baselines[built.handle] = built
                        handle = built.handle
                        built = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        conn.send({"cmd": WorkerResponse.BASELINE_READY.value, "handle": handle})
                    except Exception:
                        import traceback

                        conn.send(
                            {"cmd": WorkerResponse.ERROR.value, "error": traceback.format_exc()}
                        )
                    finally:
                        built = msg = None

                elif cmd in (
                    WorkerCommand.CLEAR_BASELINES.value,
                    WorkerCommand.RELEASE_BASELINE.value,
                ):
                    try:
                        count = len(baselines)
                        if cmd == WorkerCommand.CLEAR_BASELINES.value:
                            baselines.clear()
                        else:
                            baselines.pop(msg["baseline"], None)
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(device=device)
                        conn.send(
                            {
                                "cmd": WorkerResponse.BASELINES_CLEARED.value,
                                "count": count - len(baselines),
                            }
                        )
                    except Exception as e:
                        conn.send({"cmd": WorkerResponse.ERROR.value, "error": str(e)})

                elif cmd == WorkerCommand.RUN_SOLUTION.value:
                    baseline = baselines[msg["baseline"]]
                    definition = baseline.definition
                    solution = msg["solution"]
                    inputs_bl = baseline.inputs
                    ref_outputs_bl = baseline.outputs
                    ref_mean_latency_ms = baseline.mean_latency_ms
                    cfg = msg["config"]

                    # Use parent-provided log path so the parent can read
                    # captured output even if this process crashes.
                    log_path = redirect_stdio_to_tempfile(msg.get("log_path"))

                    try:
                        # Use registry to build/get cached solution
                        runnable_sol = registry.build(definition, solution)

                        inputs: List[List[Any]] = [
                            [v.clone() if isinstance(v, torch.Tensor) else v for v in inp]
                            for inp in inputs_bl
                        ]

                        evaluator_cls = resolve_evaluator(definition)
                        eval_cfg = cfg.resolve_eval_config(definition)
                        evaluation = evaluator_cls.evaluate(
                            definition=definition,
                            sol_runnable=runnable_sol,
                            inputs=inputs,
                            ref_outputs=ref_outputs_bl,
                            ref_mean_latency_ms=ref_mean_latency_ms,
                            cfg=eval_cfg,
                            log_path=log_path,
                            device=device,
                        )

                        conn.send(
                            {"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation}
                        )

                    except BuildError as e:
                        import traceback

                        print(f"BuildError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")

                        evaluation = make_eval(
                            status=EvaluationStatus.COMPILE_ERROR, device=device, log_path=log_path
                        )
                        conn.send(
                            {"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation}
                        )
                    except Exception as e:
                        import traceback

                        print(
                            f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        )

                        evaluation = make_eval(
                            status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                        )
                        conn.send(
                            {"cmd": WorkerResponse.EVALUATION.value, "evaluation": evaluation}
                        )
                    finally:
                        # Release this evaluation's working-set tensors and return the
                        # cached allocator blocks to the driver. Without this the
                        # subprocess retains a large pool across tasks, which inflates
                        # device "used" memory and starves the scheduler's baseline-cache
                        # admission of headroom for the next (possibly heavier) workload.
                        # Drop every local (and the request msg) still referencing the
                        # cloned/received GPU tensors before emptying the allocator.
                        inputs = inputs_bl = ref_outputs_bl = baseline = None
                        msg = None
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                else:
                    conn.send(
                        {"cmd": WorkerResponse.ERROR.value, "error": f"Unknown command: {cmd}"}
                    )

            except EOFError:
                # parent closed connection
                break
            except Exception as e:
                import traceback

                print(f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
                try:
                    conn.send({"cmd": WorkerResponse.ERROR.value, "error": str(e)})
                except Exception:
                    break

    except Exception as e:
        import traceback

        print(f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
        try:
            conn.send({"cmd": WorkerResponse.ERROR.value, "error": f"Worker startup failed: {e}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
