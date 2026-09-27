from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests


def _exchange_evaluate(
    directory: Path,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    action_key = hashlib.sha256(str(payload["action_id"]).encode()).hexdigest()
    request_path = directory / f"{action_key}.request.json"
    response_path = directory / f"{action_key}.response.json"
    envelope = {
        "schema_version": 1,
        "timeout_seconds": timeout,
        "request": payload,
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{action_key}.", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(envelope, handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, request_path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass

    deadline = time.monotonic() + timeout + 10
    try:
        while True:
            try:
                result = json.loads(response_path.read_text())
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for evaluator exchange response")
                time.sleep(0.05)
                continue
            if not isinstance(result, dict):
                raise ValueError("evaluator exchange response is not an object")
            error = result.get("transport_error")
            if error:
                raise RuntimeError(str(error))
            status_code = result.get("status_code")
            if isinstance(status_code, bool) or not isinstance(status_code, int):
                raise ValueError("evaluator exchange response has no HTTP status")
            return status_code, result.get("body")
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptxbench-eval",
        description="Synchronously retrieve controlled PTXBench feedback.",
    )
    parser.add_argument("kernel", type=Path)
    parser.add_argument("workload", type=Path)
    parser.add_argument("--action-id", help=argparse.SUPPRESS)
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--gateway-url",
        default=os.environ.get("PTXBENCH_GATEWAY_URL"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("PTXBENCH_RUN_ID"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-token",
        default=os.environ.get("PTXBENCH_RUN_TOKEN"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timeout", type=float, default=900.0, help=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exchange_value = os.environ.get("PTXBENCH_EVAL_EXCHANGE_DIR")
    required = ("run_id",) if exchange_value else ("gateway_url", "run_id", "run_token")
    for name in required:
        if not getattr(args, name):
            raise SystemExit(f"missing --{name.replace('_', '-')} or corresponding PTXBENCH environment variable")
    source = args.kernel.read_text()
    workload_data = json.loads(args.workload.read_text())
    if not isinstance(workload_data, dict):
        raise SystemExit("workload manifest must be a JSON object")
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "action_id": args.action_id or str(uuid.uuid4()),
        "source": source,
        "workload": workload_data,
    }
    try:
        if exchange_value:
            status_code, body = _exchange_evaluate(
                Path(exchange_value), payload, timeout=args.timeout
            )
        else:
            response = requests.post(
                args.gateway_url.rstrip("/") + "/v1/evaluate",
                json=payload,
                headers={"Authorization": f"Bearer {args.run_token}"},
                timeout=(10, args.timeout),
            )
            status_code = response.status_code
            try:
                body = response.json()
            except ValueError:
                print(
                    f"PTXBENCH_INFRA_ERROR: HTTP {status_code}: {response.text}",
                    file=sys.stderr,
                )
                return 2
    except (OSError, ValueError, RuntimeError, TimeoutError, requests.RequestException) as exc:
        print(f"PTXBENCH_INFRA_ERROR: {exc}", file=sys.stderr)
        return 2
    if status_code >= 400:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        print(f"PTXBENCH_INFRA_ERROR: HTTP {status_code}: {detail}", file=sys.stderr)
        return 2
    if not isinstance(body, dict):
        print("PTXBENCH_INFRA_ERROR: evaluator response is not an object", file=sys.stderr)
        return 2
    if args.json_output:
        print(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        print(body["feedback"])
    return int(body.get("returncode", 1))


if __name__ == "__main__":
    raise SystemExit(main())
