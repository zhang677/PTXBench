from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from .profile_supervisor import ProfileSupervisor
from .recovery import ProfileRequirement
from .util import (
    FileLockUnavailable,
    atomic_write_json,
    file_lock,
    read_json,
    utc_now,
)


class ProfileRecoveryWorker:
    def __init__(
        self,
        *,
        control_dir: Path,
        restart_command: list[str] | None = None,
        supervisor: ProfileSupervisor | None = None,
        restart_timeout: float = 600.0,
    ):
        if bool(restart_command) == (supervisor is not None):
            raise ValueError("configure exactly one restart command or supervisor")
        self.control_dir = control_dir.resolve()
        self.request_dir = self.control_dir / "requests"
        self.result_dir = self.control_dir / "results"
        self.restart_command = list(restart_command or [])
        self.supervisor = supervisor
        self.restart_timeout = restart_timeout
        self._process_lock = threading.RLock()
        self._active_process: subprocess.Popen[str] | None = None

    def process_once(self) -> bool:
        self.request_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        for request_path in sorted(self.request_dir.glob("*.json")):
            result_path = self.result_dir / request_path.name
            if result_path.exists():
                continue
            request = read_json(request_path)
            if not isinstance(request, dict) or not request.get("request_id"):
                atomic_write_json(
                    result_path,
                    {
                        "schema_version": 1,
                        "request_id": request_path.stem,
                        "ok": False,
                        "completed_at": utc_now(),
                        "error": "invalid recovery request",
                    },
                )
                return True
            atomic_write_json(
                self.control_dir / "active.json",
                {
                    "schema_version": 1,
                    "request_id": request["request_id"],
                    "generation": request.get("generation"),
                    "started_at": utc_now(),
                },
            )
            result = self._run_restart(request)
            atomic_write_json(result_path, result)
            atomic_write_json(
                self.control_dir / "active.json",
                {
                    "schema_version": 1,
                    "request_id": None,
                    "completed_at": utc_now(),
                },
            )
            return True
        return False

    def _run_restart(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.supervisor is not None:
            return self._run_supervisor(request)
        try:
            process = subprocess.Popen(
                self.restart_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self._process_lock:
                self._active_process = process
            try:
                output, _ = process.communicate(timeout=self.restart_timeout)
            except subprocess.TimeoutExpired:
                self._terminate_process(process)
                try:
                    output, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    output, _ = process.communicate()
                raise subprocess.TimeoutExpired(
                    self.restart_command, self.restart_timeout, output=output
                )
            finally:
                with self._process_lock:
                    self._active_process = None
            output = (output or "")[-(1 << 20) :]
            ok = process.returncode == 0
            return {
                "schema_version": 1,
                "request_id": request["request_id"],
                "generation": request.get("generation"),
                "ok": ok,
                "returncode": process.returncode,
                "completed_at": utc_now(),
                "output": output,
                "error": None if ok else f"restart command exited {process.returncode}",
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            output = getattr(exc, "stdout", "") or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            return {
                "schema_version": 1,
                "request_id": request["request_id"],
                "generation": request.get("generation"),
                "ok": False,
                "returncode": None,
                "completed_at": utc_now(),
                "output": str(output)[-(1 << 20) :],
                "error": str(exc),
            }

    def _run_supervisor(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.supervisor is not None
        try:
            raw_requirements = request.get("requirements") or []
            if not isinstance(raw_requirements, list):
                raise ValueError("recovery request requirements must be an array")
            requirements = tuple(
                sorted(
                    ProfileRequirement(
                        definition=str(item["definition"]),
                        workload_uuid=str(item["workload_uuid"]),
                    )
                    for item in raw_requirements
                    if isinstance(item, dict)
                )
            )
            if len(requirements) != len(raw_requirements):
                raise ValueError("recovery request contains an invalid requirement")
            if any(
                not item.definition.strip() or not item.workload_uuid.strip()
                for item in requirements
            ):
                raise ValueError("recovery request contains an empty requirement")
            output = self.supervisor.recover(requirements)
            return {
                "schema_version": 1,
                "request_id": request["request_id"],
                "generation": request.get("generation"),
                "ok": True,
                "returncode": 0,
                "completed_at": utc_now(),
                "output": output,
                "error": None,
            }
        except Exception as exc:
            return {
                "schema_version": 1,
                "request_id": request["request_id"],
                "generation": request.get("generation"),
                "ok": False,
                "returncode": None,
                "completed_at": utc_now(),
                "output": "",
                "error": str(exc),
            }

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass

    def stop(self) -> None:
        if self.supervisor is not None:
            self.supervisor.stop()
        with self._process_lock:
            process = self._active_process
        if process is not None:
            self._terminate_process(process)


def _restart_command(value: str) -> list[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid restart command JSON: {exc}") from exc
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise argparse.ArgumentTypeError(
            "restart command JSON must be a non-empty array of non-empty strings"
        )
    return command


def _recovery_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid recovery config JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("recovery config JSON must be an object")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ptxbench-profile-recovery-worker")
    parser.add_argument("--control-dir", required=True, type=Path)
    recovery_source = parser.add_mutually_exclusive_group(required=True)
    recovery_source.add_argument("--restart-command-json", type=_restart_command)
    recovery_source.add_argument("--recovery-config-json", type=_recovery_config)
    parser.add_argument("--profile-base-url")
    parser.add_argument("--restart-timeout", type=float, default=600.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.restart_timeout <= 0 or args.poll_interval <= 0:
        parser.error("timeouts and poll intervals must be positive")

    restart_command = args.restart_command_json
    supervisor = None
    if args.recovery_config_json is not None:
        recovery_config = args.recovery_config_json
        if isinstance(recovery_config.get("supervisor"), dict):
            if not args.profile_base_url:
                parser.error("--profile-base-url is required for supervisor recovery")
            try:
                supervisor = ProfileSupervisor.from_recovery_config(
                    recovery_config,
                    profile_base_url=args.profile_base_url,
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            try:
                restart_command = _restart_command(
                    json.dumps(recovery_config.get("restart_command"))
                )
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
    worker = ProfileRecoveryWorker(
        control_dir=args.control_dir,
        restart_command=restart_command,
        supervisor=supervisor,
        restart_timeout=args.restart_timeout,
    )
    stop = threading.Event()
    wrote_ready = False

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()
        worker.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        with file_lock(args.control_dir / "worker.lock", blocking=False):
            atomic_write_json(
                args.control_dir / "worker.json",
                {"schema_version": 1, "pid": os.getpid(), "started_at": utc_now()},
            )
            wrote_ready = True
            if args.once:
                worker.process_once()
                return 0
            while not stop.is_set():
                if not worker.process_once():
                    stop.wait(args.poll_interval)
            return 0
    except FileLockUnavailable as exc:
        parser.error(str(exc))
    finally:
        if wrote_ready:
            (args.control_dir / "worker.json").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
