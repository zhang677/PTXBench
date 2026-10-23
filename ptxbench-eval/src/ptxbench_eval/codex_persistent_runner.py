#!/usr/bin/env python3
"""Durable Codex sessions and host-side capacity recovery for leased runs.

The container exits 75 only after saving a capacity checkpoint. The host pauses
the batch, drains evaluations, probes once for the batch, then resumes it. GPU
ownership stays with the outer launcher via acquire/release callbacks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import tempfile
import time
from typing import Any

try:
    from . import codex_runner as base
except ImportError:  # Standalone scripts in the agent image.
    import codex_runner as base

CAPACITY_EXIT = 75
CONTINUE_PROMPT = (
    "Continue the existing task from its saved conversation and workspace. "
    "Read /workspace/.ptxbench-resume.json if present: it contains feedback "
    "committed while the batch was stopping. Preserve completed work and the "
    "remaining evaluation/token budgets; do not repeat completed evaluations."
)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class CapacitySettings:
    gpu_policy: str = "retain"
    initial_delay_seconds: float = 30
    max_delay_seconds: float = 300
    max_wait_seconds: float | None = None
    probe_timeout_seconds: float = 120
    drain_timeout_seconds: float = 1000

    @classmethod
    def from_config(cls, config):
        values = config.get("codex_persistent", {})
        if not isinstance(values, dict) or set(values) - cls.__dataclass_fields__.keys():
            raise ValueError("invalid codex_persistent settings")
        if values and config.get("agent") != "codex-persistent":
            raise ValueError("codex_persistent settings require agent=codex-persistent")
        settings = cls(**values)
        if settings.gpu_policy not in {"retain", "release"}:
            raise ValueError("codex_persistent.gpu_policy must be retain or release")
        for name in cls.__dataclass_fields__:
            if name == "gpu_policy":
                continue
            value = getattr(settings, name)
            if name == "max_wait_seconds" and value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"codex_persistent.{name} must be positive and finite")
        if settings.max_delay_seconds < settings.initial_delay_seconds:
            raise ValueError("max_delay_seconds must be >= initial_delay_seconds")
        return settings


def config_identity(config):
    """Ignore only lease-specific endpoints, never experiment/model settings."""
    value = {k: v for k, v in config.items() if k not in {"_config_dir", "profile_base_url"}}
    if "profile_service" in value:
        value["profile_service"] = {
            k: v for k, v in value["profile_service"].items()
            if k not in {"host", "ssh_port", "ssh_options", "ssh_known_hosts"}
        }
    return value


class PersistentClient(base.AppServerClient):
    def __init__(self, timeout, state_path: Path | None):
        saved_state = json.loads(state_path.read_text()) if state_path and state_path.exists() else {}
        super().__init__(timeout)
        self.state_path = state_path
        self.state = saved_state
        self.capacity_error = None
        self.reported_available = False
        self.thread_id = self.state.get("thread_id")
        self.turn_id = None
        self.last_usage.update(self.state.get("usage", {}))

    def checkpoint(self, **changes):
        self.state.update(changes, usage=self.last_usage, updated_at=time.time())
        if self.state_path:
            save_json(self.state_path, self.state)

    def _handle(self, message):
        previous_usage = self.last_usage.copy()
        super()._handle(message)
        if self.last_usage['total_tokens'] < previous_usage['total_tokens']:
            self.last_usage = previous_usage
        params = message.get("params") or {}
        if not self.reported_available and (
            message.get("method") in {"item/agentMessage/delta", "item/reasoning/textDelta",
                                      "item/reasoning/summaryTextDelta"}
            or (message.get("method") == "item/started" and (params.get("item") or {}).get("type")
                in {"agentMessage", "reasoning", "commandExecution", "mcpToolCall"})
        ):
            self.reported_available = True
            base._emit({"event": "codex_model_available"})
        if message.get("method") == "error" and params.get("willRetry") is False:
            error = params.get("error")
            if isinstance(error, dict) and error.get("codexErrorInfo") == "serverOverloaded":
                self.capacity_error = error
        if message.get("method") == "turn/started":
            self.turn_id = (params.get("turn") or {}).get("id")
        if message.get("method") in {"thread/tokenUsage/updated", "turn/completed"}:
            self.checkpoint()

    def run(self, *, prompt, model, effort=None, service_tier=None, disable_web=False,
            goal=False, evaluation_via_mcp=False, probe=False):
        identity = dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        model=model, effort=effort, service_tier=service_tier,
                        disable_web=disable_web, goal=goal, evaluation_via_mcp=evaluation_via_mcp)
        if self.state and self.state.get("identity") != identity:
            raise base.CodexRunnerError("saved Codex session does not match this run")
        self.request("initialize", {"clientInfo": {
            "name": "ptxbench-eval", "title": "PTXBench persistent runner", "version": "0.1.0"}})
        self.notify("initialized")
        params = base._thread_start_params(model=model, disable_web=disable_web,
            service_tier=service_tier, persistent_goal=True, effort=effort,
            evaluation_via_mcp=evaluation_via_mcp)
        params["ephemeral"] = probe
        if probe:
            params["sandbox"] = "read-only"
            params["baseInstructions"] = "Reply with OK only. Do not use tools."
            params["config"].update({"mcp_servers": {}, "web_search": "disabled",
                "features": {"shell_tool": False},
                "sandbox_workspace_write": {"network_access": False}})
        if self.thread_id:
            params.pop("ephemeral", None)
            params.pop("serviceName", None)
            params["threadId"] = self.thread_id
            params["developerInstructions"] = CONTINUE_PROMPT
            thread = self.request("thread/resume", params)["thread"]
        else:
            thread = self.request("thread/start", params)["thread"]
        self.thread_id = thread.get("id")
        if not isinstance(self.thread_id, str):
            raise base.CodexRunnerError("App Server returned no thread id")
        self.checkpoint(identity=identity, thread_id=self.thread_id, status="running")
        resumed = self.state.get("submitted", False)
        self.checkpoint(submitted=True)
        if goal:
            # Updating an existing goal must not replace its objective or reset usage.
            goal_params = {"threadId": self.thread_id, "status": "active"}
            if not resumed:
                goal_params["objective"] = prompt
            try:
                result = self.request("thread/goal/set", goal_params)
            except base.CodexRunnerError as exc:
                base._emit({"event": "codex_goal_warning", "message": "Codex goal setup failed",
                            "upstream_error": str(exc), "objective_characters": len(prompt)})
                raise
            self.goal_statuses[self.thread_id] = result["goal"]["status"]
            while self.goal_statuses[self.thread_id] == "active":
                self._handle(self._read_message())
                if self.capacity_error and any(t.get("status") == "failed" for t in self.completed_turns.values()):
                    break
            status = "completed" if self.goal_statuses[self.thread_id] == "complete" else "failed"
        else:
            turn_params = {"threadId": self.thread_id,
                "input": [{"type": "text", "text": CONTINUE_PROMPT if resumed else prompt}]}
            if effort:
                turn_params["effort"] = effort
            result = self.request("turn/start", turn_params)
            self.turn_id = result["turn"]["id"]
            while self.turn_id not in self.completed_turns:
                self._handle(self._read_message())
            status = self.completed_turns[self.turn_id].get("status", "failed")
        if self.capacity_error:
            self.checkpoint(status="waiting_for_capacity", error=self.capacity_error)
            return CAPACITY_EXIT
        self.checkpoint(status=status)
        base._emit({"event": "result", "source": "codex_app_server", "result": {
            "status": status, "turn_id": self.turn_id, "usage": self.last_usage}})
        return 0 if status == "completed" else 1


def restore_home(state_directory: Path, codex_directory: Path):
    pointer = state_directory / "snapshot.json"
    if not pointer.exists():
        return
    name = json.loads(pointer.read_text())["directory"]
    if not isinstance(name, str) or not name.startswith("snapshot-") or Path(name).name != name:
        raise ValueError("invalid saved Codex snapshot path")
    shutil.copytree(state_directory / name, codex_directory, dirs_exist_ok=True, symlinks=True)


def snapshot_home(state_directory: Path, codex_directory: Path):
    """Called after app-server exits, so its SQLite databases and WALs are stable."""
    pointer = state_directory / "snapshot.json"
    previous = json.loads(pointer.read_text())["directory"] if pointer.exists() else None
    target = Path(tempfile.mkdtemp(prefix="snapshot-", dir=state_directory))
    shutil.copytree(codex_directory, target, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns("auth.json", "tmp", "*.sock"))
    save_json(pointer, {"directory": target.name, "saved_at": time.time()})
    if previous and Path(previous).name == previous and previous.startswith("snapshot-"):
        shutil.rmtree(state_directory / previous)


def probe_model(config):
    """One short inference using the same image, login, model, and service tier."""
    from . import launcher
    settings = CapacitySettings.from_config(config)
    command = ["docker", "run", "--rm", "--name", "ptxbench-capacity-" +
               hashlib.sha256(config["experiment_root"].encode()).hexdigest()[:12],
               "--network", "host", "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=64m",
               *launcher._agent_container_tmpfs_args("codex"), "-i", config["agent_image"],
               "ptxbench-codex-persistent", "--probe", "--model", config["model"],
               "--print-timeout", str(settings.probe_timeout_seconds)]
    for key, flag in (("effort", "--effort"), ("serviceTier", "--service-tier")):
        if config.get(key):
            command += [flag, config[key]]
    try:
        result = subprocess.run(command, input=launcher._codex_plan_auth(config),
                                capture_output=True, text=True,
                                timeout=settings.probe_timeout_seconds + 15)
    finally:
        # Killing the Docker client alone does not stop its container.
        subprocess.run(["docker", "rm", "-f", command[command.index("--name") + 1]],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    with (Path(config["experiment_root"]) / "capacity-probes.jsonl").open("a") as log:
        log.write(json.dumps({"time": time.time(), "returncode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}) + "\n")
    if result.returncode not in {0, CAPACITY_EXIT}:
        raise RuntimeError(f"capacity probe failed with a non-capacity error (exit {result.returncode})")
    return result.returncode == 0


def wait_for_capacity(config, *, probe=probe_model, heartbeat=lambda: None,
                      sleep=time.sleep, monotonic=time.monotonic):
    settings = CapacitySettings.from_config(config)
    state_path = Path(config["experiment_root"]) / "capacity.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    started = monotonic()
    delay = settings.initial_delay_seconds
    attempt = 0
    while True:
        remaining = None if settings.max_wait_seconds is None else settings.max_wait_seconds - (monotonic() - started)
        retry_after = previous.get("retry_after_seconds", 0) if attempt == 0 else 0
        pause = max(min(delay * random.uniform(.9, 1.1), settings.max_delay_seconds), retry_after)
        if remaining is not None and pause >= remaining:
            save_json(state_path, {**previous, "status": "waiting_deadline_exceeded"})
            raise TimeoutError("model capacity waiting deadline exceeded; checkpoints retained")
        save_json(state_path, {**previous, "status": "waiting_for_capacity", "probe_attempt": attempt,
                              "next_probe_at": time.time() + pause, "gpu_policy": settings.gpu_policy})
        print(f'Model at capacity; next probe in {pause:.0f}s (GPU policy: {settings.gpu_policy})', flush=True)
        until = monotonic() + pause
        while monotonic() < until:
            heartbeat()
            sleep(min(1, until - monotonic()))
        heartbeat()
        attempt += 1
        try:
            available = probe(config)
        except Exception as exc:
            save_json(state_path, {**previous, "status": "probe_failed", "message": str(exc)})
            raise
        if settings.max_wait_seconds is not None and monotonic() - started >= settings.max_wait_seconds:
            raise TimeoutError("model capacity waiting deadline exceeded; checkpoints retained")
        previous.update(last_probe_at=time.time(), probe_attempt=attempt, available=available)
        save_json(state_path, {**previous, "status": "available" if available else "waiting_for_capacity"})
        if available:
            return
        delay = min(delay * 2, settings.max_delay_seconds)


def supervise(config, *, acquire, release, run_attempt, heartbeat=lambda: None,
              wait=wait_for_capacity):
    """Own the complete GPU lifecycle; no allocations occur inside agent containers."""
    settings = CapacitySettings.from_config(config)
    try:
        acquire()
        while True:
            status = run_attempt()
            if status != CAPACITY_EXIT or config["agent"] != "codex-persistent":
                return status
            if settings.gpu_policy == "release":
                release()
            wait(config, heartbeat=heartbeat if settings.gpu_policy == "retain" else lambda: None)
            if settings.gpu_policy == "release":
                acquire()
    finally:
        release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt")
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort")
    parser.add_argument("--service-tier")
    parser.add_argument("--disable-web", action="store_true")
    parser.add_argument("--evaluation-via-mcp", action="store_true")
    parser.add_argument("--goal", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--print-timeout", default="24h")
    args = parser.parse_args(argv)
    if not args.probe and args.prompt is None:
        parser.error("--prompt is required unless --probe is used")
    client = None
    state_directory = None
    snapshot_done = False
    lock = None
    signal.signal(signal.SIGTERM, base._exit_on_signal)
    signal.signal(signal.SIGINT, base._exit_on_signal)
    try:
        directory = Path(os.environ.get("CODEX_HOME", "/tmp/ptxbench-codex-home"))
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not args.probe:
            state_directory = Path(os.environ.get("PTXBENCH_CODEX_STATE_DIR",
                                   str(directory.with_name(directory.name + "-persistent-state"))))
            if state_directory == directory or state_directory.is_relative_to(directory):
                raise ValueError("persistent state must be outside the temporary CODEX_HOME")
            state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock = (state_directory / "runner.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            restore_home(state_directory, directory)
        base._install_plan_auth()
        client = PersistentClient(base._duration_seconds(args.print_timeout),
                                  None if args.probe else state_directory / "session.json")
        status = client.run(prompt="Reply with OK only. Do not use tools." if args.probe else args.prompt,
            model=args.model, effort=args.effort, service_tier=args.service_tier,
            disable_web=args.disable_web or args.probe, goal=args.goal,
            evaluation_via_mcp=args.evaluation_via_mcp, probe=args.probe)
        client.close()
        if state_directory:
            snapshot_home(state_directory, directory)
            snapshot_done = True
        if status == CAPACITY_EXIT:
            base._emit({"event": "codex_capacity_checkpoint", "error": client.capacity_error,
                        "thread_id": client.thread_id})
        return status
    except (base.CodexRunnerError, OSError, subprocess.SubprocessError, ValueError) as exc:
        base._emit({"event": "codex_runner_error", "error": str(exc)})
        return 1
    finally:
        if client is not None:
            client.close()
            if state_directory and not snapshot_done:
                snapshot_home(state_directory, directory)
            client.checkpoint()
        if client is not None:
            (directory / "auth.json").unlink(missing_ok=True)
        if lock:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
