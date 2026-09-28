#!/usr/bin/env python3
"""Narrow stdio MCP adapter for no-web PTXBench agent evaluations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, TextIO


PROTOCOL_VERSION = "2024-11-05"
EVALUATOR_COMMAND = (
    "/usr/local/bin/ptxbench-eval",
    "/workspace/kernel.cu",
    "/opt/ptxbench/workload.json",
)
EVALUATOR_CWD = "/workspace"
EVALUATOR_TIMEOUT_SECONDS = 930


EVALUATOR_ENVIRONMENT_KEYS = (
    "PTXBENCH_RUN_ID",
    "PTXBENCH_EVAL_EXCHANGE_DIR",
)


def _evaluation_environment(source: dict[str, str]) -> dict[str, str]:
    """Pass only the file-exchange identity required by the evaluator."""

    return {
        key: value
        for key in EVALUATOR_ENVIRONMENT_KEYS
        if (value := source.get(key))
    }


def _tool_result(text: str, *, is_error: bool) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def _evaluate_kernel() -> dict[str, Any]:
    environment = _evaluation_environment(dict(os.environ))
    missing = [key for key in EVALUATOR_ENVIRONMENT_KEYS if key not in environment]
    if missing:
        return _tool_result(
            "PTXBENCH_INFRA_ERROR: evaluator environment is missing "
            + ", ".join(missing),
            is_error=True,
        )

    try:
        completed = subprocess.run(
            list(EVALUATOR_COMMAND),
            cwd=EVALUATOR_CWD,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=EVALUATOR_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _tool_result(
            f"PTXBENCH_INFRA_ERROR: evaluator could not run: {exc}",
            is_error=True,
        )

    output = completed.stdout.rstrip()
    if not output:
        output = f"ptxbench-eval exited with status {completed.returncode}."
    # The evaluator uses 0 for a correct candidate, 1 for ordinary compile or
    # correctness feedback, and 2+ for infrastructure/protocol failures. A
    # candidate failure is useful model feedback, not an MCP transport error.
    return _tool_result(output, is_error=completed.returncode not in (0, 1))


def _success(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if not isinstance(method, str):
        return _error(request_id, -32600, "invalid request")
    if request_id is None:
        return None
    if method == "initialize":
        return _success(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ptxbench-eval", "version": "1"},
            },
        )
    if method == "ping":
        return _success(request_id, {})
    if method == "shutdown":
        return _success(request_id, None)
    if method == "tools/list":
        return _success(
            request_id,
            {
                "tools": [
                    {
                        "name": "evaluate_kernel",
                        "description": (
                            f"Evaluate {EVALUATOR_COMMAND[1]} against the fixed "
                            "PTXBench workload and return compiler, correctness, "
                            "and performance feedback. Takes no arguments and "
                            "waits for the complete result."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    }
                ]
            },
        )
    if method == "tools/call":
        params = request.get("params")
        if not isinstance(params, dict):
            return _error(request_id, -32602, "tools/call params must be an object")
        if params.get("name") != "evaluate_kernel":
            return _error(request_id, -32602, "unknown tool")
        arguments = params.get("arguments", {})
        if arguments not in ({}, None):
            return _error(request_id, -32602, "evaluate_kernel takes no arguments")
        return _success(request_id, _evaluate_kernel())
    return _error(request_id, -32601, f"method not found: {method}")


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    for raw_line in stdin:
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = _handle_request(request)
        except (json.JSONDecodeError, ValueError) as exc:
            response = _error(None, -32700, f"parse error: {exc}")
        if response is not None:
            stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
