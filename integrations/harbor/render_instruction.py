#!/usr/bin/env python3
"""Render a Harbor instruction from a definition served by FIBServe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_SERVICE_URL = "http://localhost:11000"
TASK_CONTENT_MARKER = "{task_content}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill a Harbor instruction template with a FIBServe definition.",
    )
    parser.add_argument(
        "task_dir",
        type=Path,
        help="Harbor task directory containing environment/task.json and instruction.template.md",
    )
    parser.add_argument(
        "--service-url",
        default=os.environ.get("PTXBENCH_HARBOR_SERVICE_URL", DEFAULT_SERVICE_URL),
        help=(
            "FIBServe base URL (default: PTXBENCH_HARBOR_SERVICE_URL or "
            f"{DEFAULT_SERVICE_URL})"
        ),
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def load_definition(service_url: str, definition_name: str, timeout: float) -> dict:
    url = f"{service_url.rstrip('/')}/definitions/{quote(definition_name, safe='')}"
    try:
        with urlopen(url, timeout=timeout) as response:
            definition = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to fetch definition from {url}: {exc}") from exc

    if not isinstance(definition, dict):
        raise ValueError(f"definition endpoint returned {type(definition).__name__}, expected object")
    if definition.get("name") != definition_name:
        raise ValueError(
            f"definition endpoint returned name {definition.get('name')!r}, expected {definition_name!r}"
        )
    return definition


def render_instruction(template: str, definition: dict) -> str:
    marker_count = template.count(TASK_CONTENT_MARKER)
    if marker_count != 1:
        raise ValueError(
            f"instruction template must contain {TASK_CONTENT_MARKER!r} exactly once; found {marker_count}"
        )

    # Match AccRL fib_runtime/multiturn/common.py: remove tags, then inject
    # json.dumps(definition, indent=2) into the user template.
    task_content = dict(definition)
    task_content.pop("tags", None)
    return template.replace(TASK_CONTENT_MARKER, json.dumps(task_content, indent=2))


def main() -> int:
    args = parse_args()
    task_dir = args.task_dir.resolve()
    manifest_path = task_dir / "environment" / "task.json"
    template_path = task_dir / "instruction.template.md"
    output_path = task_dir / "instruction.md"

    try:
        manifest = json.loads(manifest_path.read_text())
        definition_name = manifest["definition"]
        if not isinstance(definition_name, str) or not definition_name:
            raise ValueError("manifest field 'definition' must be a non-empty string")
        definition = load_definition(args.service_url, definition_name, args.timeout)
        rendered = render_instruction(template_path.read_text(), definition)
        output_path.write_text(rendered)
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Rendered {output_path} from definition {definition_name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
