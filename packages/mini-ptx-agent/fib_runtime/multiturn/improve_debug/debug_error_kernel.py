#!/usr/bin/env python3
"""Submit one generated CUDA kernel to the FlashInfer-Bench /debug endpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_DEFINITION = "mha_bwd_d128"
DEFAULT_WORKLOAD_UUID = "38c3b07c-f006-5f5e-9860-ba214c805a6b"


def render_debug_metadata(metadata: dict) -> str:
    """Render the /debug metadata response as the legacy readable report text."""
    rendered: list[str] = []
    for title, section in metadata.items():
        if title == "FlashInfer CUDA debug report":
            continue
        if not isinstance(section, dict):
            rendered.append(str(title))
            rendered.append("-" * 80)
            rendered.append(str(section))
            rendered.append("")
            continue
        if not section.get("exist"):
            continue
        msg = section.get("msg", "")
        rendered.append(str(title))
        rendered.append("-" * 80)
        if msg:
            rendered.append(str(msg))
        rendered.append("")
    report = "\n".join(rendered).rstrip()
    return report + "\n" if report else ""


def render_debug_log(entry: dict) -> str:
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("debug task log entry is missing metadata")
    return render_debug_metadata(metadata)


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def infer_entry_symbol(source: str) -> str:
    export = re.search(r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*([A-Za-z_]\w*)\s*,", source)
    if export:
        return export.group(1)
    if re.search(r"\bvoid\s+run\s*\(", source):
        return "run"
    return "run"


def solution_from_kernel(
    kernel_path: Path,
    *,
    definition: str,
    name: str | None,
    author: str,
    target_hardware: str,
) -> dict:
    source = kernel_path.read_text()
    symbol = infer_entry_symbol(source)
    solution_name = name or f"{kernel_path.parent.name}_{kernel_path.stem}"
    return {
        "name": solution_name,
        "definition": definition,
        "author": author,
        "spec": {
            "language": "cuda",
            "target_hardware": [target_hardware],
            "entry_point": f"kernel.cu::{symbol}",
            "dependencies": [],
            "destination_passing_style": True,
            "binding": "tvm-ffi",
        },
        "sources": [{"path": "kernel.cu", "content": source}],
        "description": f"Debug submission from {kernel_path}",
    }


def debug_payload_from_kernel(
    kernel_path: Path,
    *,
    definition: str = DEFAULT_DEFINITION,
    workload_uuid: str = DEFAULT_WORKLOAD_UUID,
    name: str | None = None,
    author: str = "improve_debug",
    target_hardware: str = "cuda",
    timeout: int = 120,
    max_lines: int | None = None,
    print_limit: int | None = 100,
    source_context_lines: int = 4,
    enable_coredump: bool = True,
    coredump_grace_seconds: float = 30,
) -> dict:
    return {
        "solution": solution_from_kernel(
            kernel_path,
            definition=definition,
            name=name,
            author=author,
            target_hardware=target_hardware,
        ),
        "workload_uuids": [workload_uuid],
        "sanitizer_types": ["memcheck"],
        "timeout": timeout,
        "max_lines": max_lines,
        "print_limit": print_limit,
        "source_context_lines": source_context_lines,
        "enable_coredump": enable_coredump,
        "coredump_grace_seconds": coredump_grace_seconds,
    }


def submit_and_wait(
    base_url: str,
    payload: dict,
    *,
    wait_timeout: float,
    http_timeout: float,
    poll_interval: float = 1.0,
) -> dict:
    base_url = base_url.rstrip("/")
    submitted = post_json(f"{base_url}/debug", payload, http_timeout)
    task_id = submitted["task_id"]
    deadline = time.time() + wait_timeout
    result = None
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        poll_timeout = min(5.0, remaining)
        result = get_json(
            f"{base_url}/tasks/{task_id}?timeout={poll_timeout}",
            max(http_timeout, poll_timeout + 5.0),
        )
        if result["status"] in {"completed", "failed"}:
            return result
        time.sleep(poll_interval)
    raise TimeoutError(f"task did not finish before timeout: {task_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path, help="Path to kernel_t*.cu")
    parser.add_argument("--base-url", default="http://localhost:10000")
    parser.add_argument("--definition", default=DEFAULT_DEFINITION)
    parser.add_argument("--workload-uuid", default=DEFAULT_WORKLOAD_UUID)
    parser.add_argument("--name", default=None)
    parser.add_argument("--author", default="improve_debug")
    parser.add_argument("--target-hardware", default="cuda")
    parser.add_argument("--timeout", type=int, default=120, help="/debug tool timeout")
    parser.add_argument("--wait-timeout", type=float, default=180, help="task polling timeout")
    parser.add_argument("--http-timeout", type=float, default=30)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--source-context-lines", type=int, default=4)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--print-limit", type=int, default=100)
    parser.add_argument("--coredump-grace-seconds", type=float, default=30)
    parser.add_argument("--no-coredump", action="store_true")
    parser.add_argument("--dump-payload", type=Path, default=None)
    parser.add_argument(
        "--dump-result",
        type=Path,
        default=None,
        help="Write the raw completed task response JSON.",
    )
    args = parser.parse_args()

    kernel = args.kernel.expanduser().resolve()
    if not kernel.exists():
        print(f"kernel does not exist: {kernel}", file=sys.stderr)
        return 2

    payload = debug_payload_from_kernel(
        kernel,
        definition=args.definition,
        workload_uuid=args.workload_uuid,
        name=args.name,
        author=args.author,
        target_hardware=args.target_hardware,
        timeout=args.timeout,
        max_lines=args.max_lines,
        print_limit=args.print_limit,
        source_context_lines=args.source_context_lines,
        enable_coredump=not args.no_coredump,
        coredump_grace_seconds=args.coredump_grace_seconds,
    )
    if args.dump_payload is not None:
        args.dump_payload.write_text(json.dumps(payload, indent=2) + "\n")

    try:
        result = submit_and_wait(
            args.base_url,
            payload,
            wait_timeout=args.wait_timeout,
            http_timeout=args.http_timeout,
            poll_interval=args.poll_interval,
        )
    except HTTPError as e:
        print(e.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 2
    except (URLError, TimeoutError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 2

    if result["status"] == "failed":
        print(result.get("error") or "debug task failed", file=sys.stderr)
        return 4

    logs = result.get("logs") or []
    if not logs:
        print("debug task completed without logs", file=sys.stderr)
        return 5
    if args.dump_result is not None:
        args.dump_result.write_text(json.dumps(result, indent=2) + "\n")
    try:
        rendered = render_debug_log(logs[0])
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 6
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
