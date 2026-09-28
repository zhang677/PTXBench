from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .recovery import ProfileRecoveryError, ProfileRequirement, ProfileServiceReadiness


class ProfileSupervisorError(RuntimeError):
    pass


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


@dataclass(frozen=True)
class ProfileSupervisorConfig:
    transport: str
    container: str
    host: str | None = None
    ssh_options: tuple[str, ...] = ()
    restart_script: str = "/workspace/scripts/restart_profiling.sh"
    torch_python: str = "/workspace/acc/bin/python"
    min_gpus: int = 1
    fib_devices: str | None = None
    profile_max_gpus: int | None = None
    command_timeout_seconds: float = 600.0
    cuda_ready_timeout_seconds: float = 360.0
    poll_interval_seconds: float = 5.0
    gpu_cooldown_sleep_s: int | None = None

    @classmethod
    def from_recovery_config(cls, recovery: dict[str, Any]) -> "ProfileSupervisorConfig":
        value = recovery.get("supervisor")
        if not isinstance(value, dict):
            raise ValueError("profile_recovery.supervisor must be an object")
        transport = _non_empty(value.get("transport", "ssh"), "supervisor.transport")
        if transport not in {"local", "ssh"}:
            raise ValueError("supervisor.transport must be local or ssh")
        host_value = value.get("host")
        host = None if host_value is None else _non_empty(host_value, "supervisor.host")
        if transport == "ssh" and host is None:
            raise ValueError("supervisor.host is required for ssh transport")
        options = value.get("ssh_options", [])
        if not isinstance(options, list) or not all(
            isinstance(item, str) and item for item in options
        ):
            raise ValueError("supervisor.ssh_options must be a string array")
        min_gpus = value.get("min_gpus", 1)
        if isinstance(min_gpus, bool) or not isinstance(min_gpus, int) or min_gpus <= 0:
            raise ValueError("supervisor.min_gpus must be a positive integer")
        profile_max_gpus = value.get("profile_max_gpus")
        if profile_max_gpus is not None and (
            isinstance(profile_max_gpus, bool)
            or not isinstance(profile_max_gpus, int)
            or profile_max_gpus <= 0
        ):
            raise ValueError("supervisor.profile_max_gpus must be a positive integer")
        fib_devices = value.get("fib_devices")
        if fib_devices is not None:
            fib_devices = _non_empty(fib_devices, "supervisor.fib_devices")
        gpu_cooldown_sleep_s = value.get("gpu_cooldown_sleep_s")
        if gpu_cooldown_sleep_s is not None and (
            isinstance(gpu_cooldown_sleep_s, bool)
            or not isinstance(gpu_cooldown_sleep_s, int)
            or gpu_cooldown_sleep_s < 0
        ):
            raise ValueError("supervisor.gpu_cooldown_sleep_s must be a non-negative integer")
        return cls(
            transport=transport,
            host=host,
            ssh_options=tuple(options),
            container=_non_empty(value.get("container"), "supervisor.container"),
            restart_script=_non_empty(
                value.get(
                    "restart_script",
                    "/workspace/scripts/restart_profiling.sh",
                ),
                "supervisor.restart_script",
            ),
            torch_python=_non_empty(
                value.get("torch_python", "/workspace/acc/bin/python"),
                "supervisor.torch_python",
            ),
            min_gpus=min_gpus,
            fib_devices=fib_devices,
            profile_max_gpus=profile_max_gpus,
            gpu_cooldown_sleep_s=gpu_cooldown_sleep_s,
            command_timeout_seconds=_positive_number(
                value.get(
                    "command_timeout_seconds",
                    recovery.get("restart_timeout_seconds", 600),
                ),
                "supervisor.command_timeout_seconds",
            ),
            cuda_ready_timeout_seconds=_positive_number(
                value.get("cuda_ready_timeout_seconds", 360),
                "supervisor.cuda_ready_timeout_seconds",
            ),
            poll_interval_seconds=_positive_number(
                value.get("poll_interval_seconds", 5),
                "supervisor.poll_interval_seconds",
            ),
        )


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


class CommandExecutor:
    def __init__(self):
        self._lock = threading.RLock()
        self._active: subprocess.Popen[str] | None = None

    def __call__(self, command: list[str], timeout: float) -> CommandResult:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        with self._lock:
            self._active = process
        try:
            try:
                output, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._terminate(process)
                try:
                    output, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    output, _ = process.communicate()
                raise ProfileSupervisorError(
                    f"command timed out after {timeout:g}s: {shlex.join(command)}\n"
                    f"{output or ''}"
                )
        finally:
            with self._lock:
                self._active = None
        return CommandResult(process.returncode, (output or "")[-(1 << 20) :])

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass

    def stop(self) -> None:
        with self._lock:
            process = self._active
        if process is not None:
            self._terminate(process)


class ProfileSupervisor:
    """Recover a local or remote Docker-hosted profiling service."""

    def __init__(
        self,
        *,
        config: ProfileSupervisorConfig,
        profile_base_url: str,
        readiness_timeout: float = 720.0,
        readiness_poll_interval: float = 1.0,
        session: Any = None,
        executor: Callable[[list[str], float], CommandResult] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.profile_base_url = profile_base_url.rstrip("/")
        self.readiness_timeout = readiness_timeout
        self.readiness_poll_interval = readiness_poll_interval
        self.executor = executor or CommandExecutor()
        self.sleep = sleep
        self.readiness = ProfileServiceReadiness(
            profile_base_url=self.profile_base_url,
            session=session,
        )

    @classmethod
    def from_recovery_config(
        cls,
        recovery: dict[str, Any],
        *,
        profile_base_url: str,
        session: Any = None,
        executor: Callable[[list[str], float], CommandResult] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "ProfileSupervisor":
        return cls(
            config=ProfileSupervisorConfig.from_recovery_config(recovery),
            profile_base_url=profile_base_url,
            readiness_timeout=_positive_number(
                recovery.get("timeout_seconds", 720),
                "profile_recovery.timeout_seconds",
            ),
            readiness_poll_interval=_positive_number(
                recovery.get("poll_interval_seconds", 1),
                "profile_recovery.poll_interval_seconds",
            ),
            session=session,
            executor=executor,
            sleep=sleep,
        )

    def _host_command(self, command: list[str]) -> list[str]:
        if self.config.transport == "local":
            return command
        assert self.config.host is not None
        return [
            "ssh",
            *self.config.ssh_options,
            self.config.host,
            shlex.join(command),
        ]

    def _run(self, label: str, command: list[str], *, check: bool = True) -> CommandResult:
        full_command = self._host_command(command)
        try:
            result = self.executor(full_command, self.config.command_timeout_seconds)
        except OSError as exc:
            raise ProfileSupervisorError(f"{label} could not start: {exc}") from exc
        if check and result.returncode != 0:
            suffix = f"\n{result.output.rstrip()}" if result.output.strip() else ""
            raise ProfileSupervisorError(
                f"{label} exited {result.returncode}{suffix}"
            )
        return result

    def _probe_cuda(self) -> tuple[bool, str]:
        outputs: list[str] = []
        smi = self._run(
            "container NVML probe",
            ["docker", "exec", self.config.container, "nvidia-smi", "-L"],
            check=False,
        )
        outputs.append(smi.output)
        if smi.returncode != 0:
            return False, "".join(outputs)
        probe = (
            "import sys, torch; "
            f"sys.exit(0 if torch.cuda.is_available() and "
            f"torch.cuda.device_count() >= {self.config.min_gpus} else 1)"
        )
        torch_result = self._run(
            "container PyTorch CUDA probe",
            [
                "docker",
                "exec",
                self.config.container,
                self.config.torch_python,
                "-c",
                probe,
            ],
            check=False,
        )
        outputs.append(torch_result.output)
        return torch_result.returncode == 0, "".join(outputs)

    def _restart_container(self, output: list[str]) -> None:
        result = self._run(
            "profile container restart",
            ["docker", "restart", self.config.container],
        )
        output.append(result.output)
        deadline = time.monotonic() + self.config.cuda_ready_timeout_seconds
        last_probe = ""
        while time.monotonic() < deadline:
            ok, last_probe = self._probe_cuda()
            if ok:
                output.append("profile container CUDA ready after container restart\n")
                return
            self.sleep(
                min(
                    self.config.poll_interval_seconds,
                    max(deadline - time.monotonic(), 0),
                )
            )
        raise ProfileSupervisorError(
            "profile container CUDA did not recover after docker restart"
            + (f"\n{last_probe.rstrip()}" if last_probe.strip() else "")
        )

    def _cleanup_solution_runners(self, output: list[str]) -> None:
        cleanup = (
            "pattern='flashinfer_bench[.]agents[.]_solution_runner'; "
            "pids=$(pgrep -f \"$pattern\" || true); "
            "[ -z \"$pids\" ] || kill -TERM $pids 2>/dev/null || true; "
            "for _ in $(seq 1 10); do "
            "pids=$(pgrep -f \"$pattern\" || true); "
            "[ -z \"$pids\" ] && exit 0; sleep 1; done; "
            "pids=$(pgrep -f \"$pattern\" || true); "
            "[ -z \"$pids\" ] || kill -KILL $pids 2>/dev/null || true"
        )
        result = self._run(
            "stale profile solution-runner cleanup",
            ["docker", "exec", self.config.container, "bash", "-lc", cleanup],
        )
        output.append(result.output)

    def _restart_service(self, output: list[str]) -> None:
        command = ["docker", "exec"]
        if self.config.fib_devices is not None:
            command.extend(["-e", f"FIB_DEVICES={self.config.fib_devices}"])
        if self.config.profile_max_gpus is not None:
            command.extend(
                ["-e", f"PROFILE_MAX_GPUS={self.config.profile_max_gpus}"]
            )
        if self.config.gpu_cooldown_sleep_s is not None:
            command.extend(
                ["-e", f"FIB_GPU_COOLDOWN_SLEEP_S={self.config.gpu_cooldown_sleep_s}"]
            )
        command.extend(
            [
                self.config.container,
                "bash",
                "-lc",
                shlex.join(["bash", self.config.restart_script]),
            ]
        )
        result = self._run("profiling-service restart", command)
        output.append(result.output)

    def recover(
        self,
        requirements: tuple[ProfileRequirement, ...],
        *,
        definitions: tuple[str, ...] = (),
    ) -> str:
        output: list[str] = []
        container_restarted = False
        cuda_ok, probe_output = self._probe_cuda()
        output.append(probe_output)
        if not cuda_ok:
            self._restart_container(output)
            container_restarted = True
        self._cleanup_solution_runners(output)
        try:
            self._restart_service(output)
        except ProfileSupervisorError as first_error:
            cuda_ok, probe_output = self._probe_cuda()
            output.append(probe_output)
            if cuda_ok or container_restarted:
                raise
            output.append(f"{first_error}\n")
            self._restart_container(output)
            self._cleanup_solution_runners(output)
            self._restart_service(output)
        self.readiness.wait(
            requirements,
            definitions=definitions,
            timeout=self.readiness_timeout,
            poll_interval=self.readiness_poll_interval,
            sleep=self.sleep,
        )
        output.append("profiling service and required workloads are ready\n")
        return "".join(output)[-(1 << 20) :]

    def ensure_ready(
        self,
        requirements: tuple[ProfileRequirement, ...],
        *,
        definitions: tuple[str, ...] = (),
        force_restart: bool = False,
    ) -> str:
        if not force_restart:
            try:
                self.readiness.check(requirements, definitions=definitions)
                return "profiling service already ready\n"
            except ProfileRecoveryError:
                pass
        return self.recover(requirements, definitions=definitions)

    def stop(self) -> None:
        stop = getattr(self.executor, "stop", None)
        if callable(stop):
            stop()


def _requirements_from_experiment(config: dict[str, Any]) -> tuple[ProfileRequirement, ...]:
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("experiment config must contain runs")
    values = {
        ProfileRequirement(
            definition=_non_empty(run.get("definition"), "run.definition"),
            workload_uuid=_non_empty(run.get("workload_uuid"), "run.workload_uuid"),
        )
        for run in runs
        if isinstance(run, dict)
    }
    if len(values) == 0:
        raise ValueError("experiment config contains no valid runs")
    return tuple(sorted(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ptxbench-profile-supervisor")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--profile-base-url")
    parser.add_argument("--transport", choices=("local", "ssh"))
    parser.add_argument("--host")
    parser.add_argument("--container")
    parser.add_argument("--restart-script")
    parser.add_argument("--torch-python")
    parser.add_argument("--min-gpus", type=int)
    parser.add_argument("--fib-devices")
    parser.add_argument("--profile-max-gpus", type=int)
    parser.add_argument("--gpu-cooldown-sleep-s", type=int)
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--definition", action="append", default=[])
    parser.add_argument("--readiness-timeout", type=float, default=720)
    parser.add_argument("--readiness-poll-interval", type=float, default=1)
    parser.add_argument("--command-timeout", type=float, default=600)
    parser.add_argument("--cuda-ready-timeout", type=float, default=360)
    parser.add_argument("--container-poll-interval", type=float, default=5)
    parser.add_argument("--force-restart", action="store_true")
    args = parser.parse_args(argv)

    if args.experiment_config is not None:
        experiment = json.loads(args.experiment_config.read_text())
        if not isinstance(experiment, dict):
            parser.error("experiment config must be an object")
        recovery = experiment.get("profile_recovery")
        if not isinstance(recovery, dict) or not isinstance(
            recovery.get("supervisor"), dict
        ):
            return 0
        profile_base_url = _non_empty(
            experiment.get("profile_base_url"), "profile_base_url"
        )
        requirements = _requirements_from_experiment(experiment)
        supervisor = ProfileSupervisor.from_recovery_config(
            recovery, profile_base_url=profile_base_url
        )
        definitions: tuple[str, ...] = ()
    else:
        if args.transport is None or args.container is None or args.profile_base_url is None:
            parser.error(
                "--transport, --container, and --profile-base-url are required "
                "without --experiment-config"
            )
        value: dict[str, Any] = {
            "transport": args.transport,
            "container": args.container,
            "ssh_options": args.ssh_option,
            "command_timeout_seconds": args.command_timeout,
            "cuda_ready_timeout_seconds": args.cuda_ready_timeout,
            "poll_interval_seconds": args.container_poll_interval,
        }
        for key in (
            "host",
            "restart_script",
            "torch_python",
            "min_gpus",
            "fib_devices",
            "profile_max_gpus",
            "gpu_cooldown_sleep_s",
        ):
            option = getattr(args, key)
            if option is not None:
                value[key] = option
        recovery = {
            "timeout_seconds": args.readiness_timeout,
            "poll_interval_seconds": args.readiness_poll_interval,
            "supervisor": value,
        }
        supervisor = ProfileSupervisor.from_recovery_config(
            recovery, profile_base_url=args.profile_base_url
        )
        requirements = ()
        definitions = tuple(sorted(set(args.definition)))
    try:
        output = supervisor.ensure_ready(
            requirements,
            definitions=definitions,
            force_restart=args.force_restart,
        )
    except (OSError, ValueError, ProfileRecoveryError, ProfileSupervisorError) as exc:
        print(f"PTXBENCH_PROFILE_RECOVERY_ERROR: {exc}", file=sys.stderr)
        return 1
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
