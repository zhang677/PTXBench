"""CPU regression tests for GPU admission and verified worker replacement."""

import logging
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from flashinfer_bench.bench.runner.gpu_memory import GpuMemoryMonitor, MemorySnapshot
from flashinfer_bench.bench.runner.persistent_runner import PersistentSubprocessWorker
from flashinfer_bench.serve.scheduler import _GPUWorkerThread, _is_cuda_oom
from flashinfer_bench.data import EvaluationStatus
from flashinfer_bench.serve.task_store import BaselineCacheMode


class MemoryRecoveryTest(unittest.TestCase):
    def thread(self, used=20, clean=5, required=30):
        obj = _GPUWorkerThread.__new__(_GPUWorkerThread)
        obj._config = NS(max_mem_ratio=0.85)
        obj._device = "cuda:0"
        obj._gpu_worker = Mock()
        obj._gpu_worker.memory_snapshot.side_effect = [
            MemorySnapshot(100, used, frozenset()),
            MemorySnapshot(100, clean, frozenset()),
        ]
        obj._restart_worker = Mock(return_value=True)
        obj._baseline_cache = {}
        return obj

    def test_normal_admission_does_not_restart(self):
        obj = self.thread()
        self.assertFalse(obj._ensure_memory(30, "task"))
        obj._restart_worker.assert_not_called()

    def test_retained_memory_restarts_before_admission(self):
        obj = self.thread(used=60)
        self.assertTrue(obj._ensure_memory(30, "task"))
        obj._restart_worker.assert_called_once()
        self.assertTrue(obj._restart_worker.call_args.kwargs["force"])

    def test_task_too_large_after_cleanup_is_rejected_once(self):
        obj = self.thread(used=90, clean=60)
        with self.assertRaisesRegex(RuntimeError, "refusing task"):
            obj._ensure_memory(30, "task")
        obj._restart_worker.assert_called_once()

    def test_failed_restart_does_not_admit(self):
        obj = self.thread(used=90)
        obj._restart_worker.return_value = False
        with self.assertRaisesRegex(RuntimeError, "refusing task"):
            obj._ensure_memory(30, "task")

    def test_cached_baseline_is_checked_and_rebuilt_after_restart(self):
        obj = self.thread()
        definition, workload = NS(name="toy"), NS(uuid="case")
        obj._config = Mock()
        obj._config.resolve_eval_config.return_value.num_trials = 3
        obj._baseline_cache[("toy", "case", None, None)] = "old"
        obj._ensure_memory = Mock(return_value=True)
        obj._build_ref = Mock(return_value="new")
        with patch("flashinfer_bench.serve.scheduler.estimate_eval_reserve_bytes", return_value=30):
            self.assertEqual(obj._get_or_build_ref(definition, workload), "new")
        obj._ensure_memory.assert_called_once_with(30, "cached-baseline evaluation")
        self.assertEqual(obj._baseline_cache[("toy", "case", None, None)], "new")

    def test_cached_baseline_reused_with_sufficient_memory(self):
        obj = self.thread()
        definition, workload = NS(name="toy"), NS(uuid="case")
        obj._config = Mock()
        obj._baseline_cache[("toy", "case", None, None)] = "old"
        obj._ensure_memory = Mock(return_value=False)
        obj._build_ref = Mock()
        with patch("flashinfer_bench.serve.scheduler.estimate_eval_reserve_bytes", return_value=30):
            self.assertEqual(obj._get_or_build_ref(definition, workload), "old")
        obj._build_ref.assert_not_called()

    def test_oom_restarts_even_if_tiny_health_check_would_pass(self):
        obj = self.thread()
        obj._trace_set = NS(definitions={"toy": NS()})
        obj._resolve_workloads = Mock(return_value=[NS()])
        obj._get_or_build_ref = Mock(return_value="baseline")
        obj._gpu_worker.run_solution.return_value = NS(
            status=EvaluationStatus.RUNTIME_ERROR, log="CUDA out of memory"
        )
        obj._gpu_worker.is_healthy.return_value = True
        task = NS(
            definition_name="toy",
            atol=None,
            rtol=None,
            timeout=None,
            baseline_cache_mode=BaselineCacheMode.REUSE,
            profile_baseline=None,
            run_baseline=None,
            solution=NS(name="candidate"),
        )
        with patch("flashinfer_bench.serve.scheduler.Trace", side_effect=lambda **kw: NS(**kw)):
            obj._evaluate_task(task)
        obj._restart_worker.assert_called_once()
        obj._gpu_worker.is_healthy.assert_not_called()

    def test_oom_markers(self):
        for message in (
            "CUDA out of memory",
            "OutOfMemoryError",
            "cudaErrorMemoryAllocation",
            "CUDA_ERROR_OUT_OF_MEMORY",
            "CUBLAS_STATUS_ALLOC_FAILED",
        ):
            self.assertTrue(_is_cuda_oom(message))
        self.assertFalse(_is_cuda_oom("numerical mismatch"))

    def monitor(self, snapshot):
        monitor = GpuMemoryMonitor.__new__(GpuMemoryMonitor)
        monitor.uuid = "GPU-test"
        monitor.idle = MemorySnapshot(100, 0, frozenset())
        monitor.snapshot = Mock(return_value=snapshot)
        return monitor

    def test_live_pid_blocks_restart_even_with_small_memory(self):
        monitor = self.monitor(MemorySnapshot(100, 0, frozenset({123})))
        with self.assertRaisesRegex(RuntimeError, "refusing replacement"):
            monitor.wait_until_reclaimed(timeout=0)

    def test_retained_memory_blocks_restart_even_without_visible_pid(self):
        monitor = self.monitor(MemorySnapshot(100, 32 * 1024**2, frozenset()))
        with self.assertRaisesRegex(RuntimeError, "refusing replacement"):
            monitor.wait_until_reclaimed(timeout=0)

    def test_reclamation_observed_before_success(self):
        monitor = self.monitor(MemorySnapshot(100, 0, frozenset()))
        self.assertEqual(monitor.wait_until_reclaimed(timeout=0).used, 0)

    def test_torch_bare_uuid_matches_nvml(self):
        uuid = "c73e6995-bf09-8af1-96e3-17da9531399f"
        with patch.object(
            GpuMemoryMonitor, "_query", side_effect=[[["GPU-" + uuid, "80000", "0"]], []]
        ), patch(
            "flashinfer_bench.bench.runner.gpu_memory.subprocess.run",
            return_value=NS(stdout=uuid + "\n"),
        ), patch.object(
            GpuMemoryMonitor, "wait_until_reclaimed"
        ):
            monitor = GpuMemoryMonitor("cuda:0")
        self.assertEqual(monitor.uuid, "GPU-" + uuid)

    def test_supervisor_cannot_adopt_orphan_as_idle(self):
        uuid = "GPU-test"
        with patch.object(
            GpuMemoryMonitor, "_query", side_effect=[[[uuid, "80000", "8192"]], [[uuid, "123"]]]
        ), patch(
            "flashinfer_bench.bench.runner.gpu_memory.subprocess.run",
            return_value=NS(stdout=uuid + "\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "already has compute clients"):
                GpuMemoryMonitor("cuda:0")

    def test_telemetry_failure_never_means_free_memory(self):
        monitor = self.monitor(MemorySnapshot(100, 0, frozenset()))
        monitor.snapshot.side_effect = RuntimeError("NVML unavailable")
        with self.assertRaisesRegex(RuntimeError, "NVML unavailable"):
            monitor.wait_until_reclaimed(timeout=0)

    def test_no_replacement_process_if_cleanup_fails(self):
        worker = PersistentSubprocessWorker.__new__(PersistentSubprocessWorker)
        worker._device = "cuda:0"
        worker._baselines = {"old": object()}
        worker._failure_records = {}
        worker._shutdown_worker = Mock(side_effect=RuntimeError("memory remains"))
        worker._start_worker = Mock()
        self.assertFalse(worker.restart())
        worker._start_worker.assert_not_called()

    def test_kill_descendants_even_if_leader_already_exited(self):
        worker = PersistentSubprocessWorker.__new__(PersistentSubprocessWorker)
        worker._device = "cuda:0"
        worker._baselines = {}
        worker._parent_conn = Mock()
        worker._worker_proc = Mock()
        worker._worker_proc.is_alive.return_value = False
        worker._worker_pgid = 42
        worker._kill_worker_process_group = Mock()
        worker._memory = self.monitor(MemorySnapshot(100, 0, frozenset()))
        worker._shutdown_worker()
        worker._kill_worker_process_group.assert_called_once()
        worker._memory.snapshot.assert_called_once()
        self.assertIsNone(worker._worker_proc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
