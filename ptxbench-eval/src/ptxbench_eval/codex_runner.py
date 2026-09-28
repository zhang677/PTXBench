#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class CodexRunnerError(RuntimeError):
    pass


_PTXBENCH_MCP_SERVER = "ptxbench-eval"
_PTXBENCH_MCP_COMMAND = "/usr/local/bin/ptxbench-eval-mcp"
_PTXBENCH_MCP_TOOL = "evaluate_kernel"
_PTXBENCH_MCP_STARTUP_TIMEOUT_SECONDS = 10
# The adapter gives the evaluator 930 seconds. Leave a small margin so Codex
# receives the adapter's timeout feedback instead of cancelling the MCP call.
_PTXBENCH_MCP_TOOL_TIMEOUT_SECONDS = 960


def _ptxbench_mcp_server_config() -> dict[str, Any]:
    return {
        "command": _PTXBENCH_MCP_COMMAND,
        "args": [],
        "cwd": "/workspace",
        "enabled_tools": [_PTXBENCH_MCP_TOOL],
        "required": True,
        "startup_timeout_sec": _PTXBENCH_MCP_STARTUP_TIMEOUT_SECONDS,
        "tool_timeout_sec": _PTXBENCH_MCP_TOOL_TIMEOUT_SECONDS,
    }


def _thread_start_params(
    *,
    model: str,
    disable_web: bool,
    service_tier: str | None = None,
    persistent_goal: bool = False,
    effort: str | None = None,
    evaluation_via_mcp: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "sandbox_workspace_write": {"network_access": not disable_web},
    }
    if disable_web:
        config["web_search"] = "disabled"
    if evaluation_via_mcp:
        config["mcp_servers"] = {_PTXBENCH_MCP_SERVER: _ptxbench_mcp_server_config()}
    if persistent_goal and effort is not None:
        # Goal mode has no explicit turn/start request on which to override
        # effort, so make the configured effort sticky for every automatic
        # turn started by the goal.
        config["model_reasoning_effort"] = effort
    params: dict[str, Any] = {
        "model": model,
        "cwd": "/workspace",
        "approvalPolicy": "never",
        "sandbox": "workspace-write",
        "serviceName": "ptxbench-eval",
        # Codex goals require a non-ephemeral App Server thread. CODEX_HOME is
        # mounted as a container tmpfs, so this state still disappears with
        # the run container.
        "ephemeral": not persistent_goal,
        "config": config,
    }
    if service_tier is not None:
        params["serviceTier"] = service_tier
    return params


def _emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, separators=(",", ":")), flush=True)


def _duration_seconds(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h|d)?", value.strip())
    if not match:
        raise ValueError(f"invalid timeout: {value}")
    amount = float(match.group(1))
    units = {
        None: 1.0,
        "ms": 0.001,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }
    seconds = amount * units[match.group(2)]
    if seconds <= 0:
        raise ValueError("timeout must be positive")
    return seconds


def _usage_from_notification(params: Any) -> dict[str, int] | None:
    if not isinstance(params, dict):
        return None
    token_usage = params.get("tokenUsage")
    if not isinstance(token_usage, dict):
        return None
    total = token_usage.get("total")
    if not isinstance(total, dict):
        return None

    def count(name: str) -> int:
        try:
            return max(int(total.get(name, 0) or 0), 0)
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": count("inputTokens"),
        "output_tokens": count("outputTokens"),
        "thinking_tokens": count("reasoningOutputTokens"),
        "cache_read_tokens": count("cachedInputTokens"),
        "total_tokens": count("totalTokens"),
    }


def _usage_event(usage: dict[str, int]) -> dict[str, Any]:
    # Keep the launcher/evaluator on the existing agy stream-json usage path.
    # A fixed step index makes each App Server cumulative update replace the
    # previous value instead of being summed as a new Antigravity step.
    return {
        "event": "step_update",
        "source": "codex_app_server",
        "step_update": {
            "step_index": 0,
            "state": "DONE",
            "type": "agent_response",
            "usage": usage,
        },
    }


def _install_plan_auth() -> None:
    payload = sys.stdin.read(1024 * 1024 + 1)
    if not payload:
        raise CodexRunnerError("Codex ChatGPT plan authentication was not provided")
    if len(payload) > 1024 * 1024:
        raise CodexRunnerError("Codex ChatGPT plan authentication is unexpectedly large")
    try:
        auth = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CodexRunnerError("Codex ChatGPT plan authentication is invalid JSON") from exc
    if not isinstance(auth, dict) or auth.get("auth_mode") != "chatgpt":
        raise CodexRunnerError(
            "Codex authentication must use a ChatGPT plan; API-key auth is rejected"
        )
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict) or not isinstance(tokens.get("access_token"), str):
        raise CodexRunnerError("Codex ChatGPT plan authentication has no access token")

    # Defense in depth: never persist an API-key field in the runner's private
    # auth file, even if a caller bypasses the host launcher's sanitization.
    auth.pop("OPENAI_API_KEY", None)
    sanitized_payload = json.dumps(auth, separators=(",", ":"))

    codex_home = Path(os.environ.setdefault("CODEX_HOME", "/tmp/ptxbench-codex-home"))
    codex_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    auth_path = codex_home / "auth.json"
    auth_path.write_text(sanitized_payload)
    auth_path.chmod(0o600)
    # Never let an ambient API key override the explicit ChatGPT plan login.
    os.environ.pop("OPENAI_API_KEY", None)


class AppServerClient:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.deadline = time.monotonic() + timeout
        self.process = subprocess.Popen(
            ["codex", "app-server", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise CodexRunnerError("failed to open Codex App Server pipes")
        self._stdin = self.process.stdin
        self._stdout = self.process.stdout
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._stdout, selectors.EVENT_READ)
        self._buffer = b""
        self._next_id = 1
        self.last_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 0,
        }
        self.completed_turns: dict[str, dict[str, Any]] = {}
        self.goal_statuses: dict[str, str] = {}

    def close(self) -> None:
        self._selector.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def _send(self, message: dict[str, Any]) -> None:
        self._stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        self._stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        while True:
            if b"\n" in self._buffer:
                raw, self._buffer = self._buffer.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    _emit(
                        {
                            "event": "codex_app_server_non_json",
                            "text": raw.decode(errors="replace"),
                        }
                    )
                    continue
                if not isinstance(message, dict):
                    _emit({"event": "codex_app_server_non_object", "payload": message})
                    continue
                return message

            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise CodexRunnerError(f"Codex turn exceeded timeout of {self.timeout:g}s")
            if not self._selector.select(remaining):
                raise CodexRunnerError(f"Codex turn exceeded timeout of {self.timeout:g}s")
            chunk = os.read(self._stdout.fileno(), 65536)
            if not chunk:
                if self._buffer.strip():
                    raw, self._buffer = self._buffer, b""
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise CodexRunnerError(
                            "Codex App Server closed with incomplete JSON output"
                        ) from exc
                    if isinstance(message, dict):
                        return message
                returncode = self.process.poll()
                raise CodexRunnerError(
                    f"Codex App Server exited before turn completion: returncode={returncode}"
                )
            self._buffer += chunk

    def _handle(self, message: dict[str, Any]) -> None:
        _emit({"event": "codex_app_server", "payload": message})
        method = message.get("method")
        if method == "thread/tokenUsage/updated":
            usage = _usage_from_notification(message.get("params"))
            if usage is not None:
                self.last_usage = usage
                _emit(_usage_event(usage))
        elif method == "turn/completed":
            params = message.get("params")
            turn = params.get("turn") if isinstance(params, dict) else None
            if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                self.completed_turns[turn["id"]] = turn
        elif method == "thread/goal/updated":
            params = message.get("params")
            goal = params.get("goal") if isinstance(params, dict) else None
            if isinstance(goal, dict):
                thread_id = goal.get("threadId")
                status = goal.get("status")
                if isinstance(thread_id, str) and isinstance(status, str):
                    self.goal_statuses[thread_id] = status
        elif method == "thread/goal/cleared":
            params = message.get("params")
            thread_id = params.get("threadId") if isinstance(params, dict) else None
            if isinstance(thread_id, str):
                self.goal_statuses[thread_id] = "cleared"

        if method is not None and "id" in message:
            self._send(
                {
                    "id": message["id"],
                    "error": {
                        "code": -32601,
                        "message": "PTXBench headless runner does not support interactive requests",
                    },
                }
            )

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"id": request_id, "method": method, "params": params})
        while True:
            message = self._read_message()
            if message.get("id") != request_id or "method" in message:
                self._handle(message)
                continue
            _emit({"event": "codex_app_server", "payload": message})
            error = message.get("error")
            if error is not None:
                raise CodexRunnerError(f"Codex App Server {method} failed: {error}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise CodexRunnerError(f"Codex App Server {method} returned no result")
            return result

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def run(
        self,
        *,
        prompt: str,
        model: str,
        effort: str | None,
        service_tier: str | None = None,
        disable_web: bool = False,
        goal: bool = False,
        evaluation_via_mcp: bool = False,
    ) -> int:
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "ptxbench-eval",
                    "title": "PTXBench headless runner",
                    "version": "0.1.0",
                }
            },
        )
        self.notify("initialized")
        thread_result = self.request(
            "thread/start",
            _thread_start_params(
                model=model,
                disable_web=disable_web,
                service_tier=service_tier,
                persistent_goal=goal,
                effort=effort,
                evaluation_via_mcp=evaluation_via_mcp,
            ),
        )
        thread = thread_result.get("thread")
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not isinstance(thread_id, str):
            raise CodexRunnerError("Codex App Server thread/start returned no thread id")
        turn_id: str | None = None
        goal_status: str | None = None
        if goal:
            # A goal set on an idle thread starts the initial turn. Use the
            # complete shared prompt as the one objective instead of sending
            # a separate turn and a duplicate completion-only goal.
            try:
                goal_result = self.request(
                    "thread/goal/set",
                    {"threadId": thread_id, "objective": prompt},
                )
            except CodexRunnerError as exc:
                # Let the installed App Server own validation of its goal
                # contract. Surface its version-specific rejection instead of
                # duplicating limits in the PTXBench environment.
                _emit(
                    {
                        "event": "codex_goal_warning",
                        "message": (
                            "Codex App Server could not set the goal objective; "
                            "aborting the run"
                        ),
                        "objective_characters": len(prompt),
                        "upstream_error": str(exc),
                    }
                )
                raise
            goal_payload = goal_result.get("goal")
            goal_status = (
                goal_payload.get("status")
                if isinstance(goal_payload, dict)
                else None
            )
            if not isinstance(goal_status, str):
                raise CodexRunnerError(
                    "Codex App Server thread/goal/set returned no goal status"
                )
            observed_status = self.goal_statuses.get(thread_id)
            if observed_status is None or (
                observed_status == "active" and goal_status != "active"
            ):
                self.goal_statuses[thread_id] = goal_status
            while self.goal_statuses.get(thread_id) == "active":
                self._handle(self._read_message())
            goal_status = self.goal_statuses.get(thread_id, "missing")
            status = "completed" if goal_status == "complete" else "failed"
        else:
            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
            }
            if effort:
                turn_params["effort"] = effort
            turn_result = self.request("turn/start", turn_params)
            turn = turn_result.get("turn")
            turn_id = turn.get("id") if isinstance(turn, dict) else None
            if not isinstance(turn_id, str):
                raise CodexRunnerError(
                    "Codex App Server turn/start returned no turn id"
                )
            while turn_id not in self.completed_turns:
                self._handle(self._read_message())
            completed = self.completed_turns[turn_id]
            status = str(completed.get("status", "failed"))
        _emit(
            {
                "event": "result",
                "source": "codex_app_server",
                "result": {
                    "status": status,
                    "turn_id": turn_id,
                    "goal_status": goal_status,
                    "usage": self.last_usage,
                },
            }
        )
        return 0 if status == "completed" else 1


def _exit_on_signal(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ptxbench-codex")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort")
    parser.add_argument("--service-tier")
    parser.add_argument("--disable-web", action="store_true")
    parser.add_argument("--evaluation-via-mcp", action="store_true")
    parser.add_argument("--goal", action="store_true")
    parser.add_argument("--print-timeout", default="24h")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)

    client: AppServerClient | None = None
    try:
        timeout = _duration_seconds(args.print_timeout)
        _install_plan_auth()
        client = AppServerClient(timeout)
        return client.run(
            prompt=args.prompt,
            model=args.model,
            effort=args.effort,
            service_tier=args.service_tier,
            disable_web=args.disable_web,
            goal=args.goal,
            evaluation_via_mcp=args.evaluation_via_mcp,
        )
    except (CodexRunnerError, OSError, subprocess.SubprocessError, ValueError) as exc:
        _emit({"event": "codex_runner_error", "error": str(exc)})
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
