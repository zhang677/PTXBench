"""Observe GPU memory without creating a CUDA context in the controller."""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    total: int
    used: int
    compute_pids: frozenset[int]


class GpuMemoryMonitor:
    def __init__(self, device: str):
        self.uuid = None
        initial_devices = self._query("--query-gpu", "uuid,memory.total,memory.used")
        initial_processes = self._query("--query-compute-apps", "gpu_uuid,pid")
        # Resolve the CUDA ordinal in a short-lived process. This respects CUDA's
        # visible-device ordering without initializing CUDA in the controller.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys,torch; print(torch.cuda.get_device_properties(sys.argv[1]).uuid)",
                device,
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.uuid = probe.stdout.strip().splitlines()[-1]
        if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", self.uuid):
            self.uuid = "GPU-" + self.uuid
        if not self.uuid.startswith(("GPU-", "MIG-")):
            raise RuntimeError(f"Cannot resolve GPU UUID for {device}: {self.uuid!r}")
        matches = [(total, used) for uuid, total, used in initial_devices if uuid == self.uuid]
        if len(matches) != 1:
            raise RuntimeError(f"No initial memory snapshot for {self.uuid}")
        total, used = (int(x) * 1024**2 for x in matches[0])
        self.idle = MemorySnapshot(
            total, used, frozenset(int(pid) for uuid, pid in initial_processes if uuid == self.uuid)
        )
        # A supervisor must not adopt an orphaned previous worker as its new
        # idle baseline. Profiling workers require exclusive use of a device.
        if self.idle.compute_pids:
            raise RuntimeError(
                f"GPU {self.uuid} already has compute clients {sorted(self.idle.compute_pids)}; "
                "refusing to start until the device is clean"
            )
        self.wait_until_reclaimed()

    def _query(self, option: str, fields: str) -> list[list[str]]:
        result = subprocess.run(
            [
                "nvidia-smi",
                *(["--id=" + self.uuid] if self.uuid else []),
                option + "=" + fields,
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))

    def snapshot(self) -> MemorySnapshot:
        rows = self._query("--query-gpu", "memory.total,memory.used")
        if len(rows) != 1:
            raise RuntimeError(f"Expected one GPU for {self.uuid}, got {rows!r}")
        total, used = (int(x) * 1024**2 for x in rows[0])
        processes = self._query("--query-compute-apps", "gpu_uuid,pid")
        pids = frozenset(int(pid) for uuid, pid in processes if uuid == self.uuid)
        return MemorySnapshot(total, used, pids)

    def wait_until_reclaimed(self, timeout: float = 15.0) -> MemorySnapshot:
        """Fail closed unless all new GPU clients and their allocations disappear.

        The small byte allowance accommodates NVML rounding/driver bookkeeping;
        it never permits an old compute PID to survive. Unexpected clients
        are never killed by this monitor; they prevent reuse of the device.
        """
        deadline = time.monotonic() + timeout
        while True:
            current = self.snapshot()
            remaining = current.compute_pids - self.idle.compute_pids
            if not remaining and current.used <= self.idle.used + 16 * 1024**2:
                return current
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"GPU cleanup incomplete on {self.uuid}: compute PIDs {sorted(remaining)}, "
                    f"used={current.used}, idle={self.idle.used}; refusing replacement worker"
                )
            time.sleep(0.1)
