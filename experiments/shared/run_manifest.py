#!/usr/bin/env python3
"""Check or run a CSV of multiturn workloads through the shared runner."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MULTITURN = ROOT / "multiturn"
REQUIRED = ("model", "arch", "definition", "workload", "exp_dir", "test_path", "config_path")
os.environ.setdefault("PTXBENCH_ROOT", str(ROOT))
os.environ.setdefault("PTXBENCH_DATA_ROOT", str(ROOT / "data"))


def path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve()


def read_manifest(source: Path) -> list[dict[str, str]]:
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: no runs")
    for index, row in enumerate(rows, 2):
        for key in REQUIRED:
            if not row[key].strip():
                raise ValueError(f"{source}:{index}: empty {key}")
        for key in ("test_path", "config_path"):
            if not path(row[key]).is_file():
                raise FileNotFoundError(f"{source}:{index}: missing {key}={path(row[key])}")
        tree = ast.parse(path(row["test_path"]).read_text())
        test_identity = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"DEFINITION_NAME", "WORKLOAD_UUID"}:
                        test_identity[target.id] = node.value.value
        if test_identity != {"DEFINITION_NAME": row["definition"], "WORKLOAD_UUID": row["workload"]}:
            raise ValueError(f"{source}:{index}: definition/workload do not match {path(row['test_path'])}")
        config = json.loads(path(row["config_path"]).read_text())
        if not isinstance(config, list) or not config:
            raise ValueError(f"{source}:{index}: config must be a nonempty list")
        for item in config:
            tag = item.get("prompt_tag")
            if not tag or not (MULTITURN / "prompts/assembled" / f"{tag}.md").is_file():
                raise ValueError(f"{source}:{index}: missing prompt for {tag!r}")
            if item.get("user_template") and not path(item["user_template"]).is_file():
                raise FileNotFoundError(f"{source}:{index}: missing user template")
        if row["arch"] not in {"hopper", "blackwell"}:
            raise ValueError(f"{source}:{index}: unsupported arch {row['arch']!r}")
    return rows


def command(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    output = path(row["exp_dir"])
    cmd = [
        sys.executable, str(MULTITURN / "run_parallel_v2.py"),
        "--model", args.model or row["model"],
        "--output-root", str(output),
        "--service-url", args.service_url,
        "--gpu-arch", row["arch"],
        "--image", args.image,
        "--max-parallel", str(args.max_parallel),
        "--max-profiles", str(args.max_profiles),
        "--timeout", str(args.timeout),
        "--turn-timeout", str(args.turn_timeout),
        "--without-local-gpu",
    ]
    if output.exists():
        if not (output / "plan.json").is_file():
            raise ValueError(f"{output} exists without plan.json; choose a fresh output root")
        cmd.append("--resume")
    else:
        cmd.extend([
            "--config", str(path(row["config_path"])),
            "--definition", row["definition"],
            "--test-path", str(path(row["test_path"])),
        ])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--model", help="Override the model in every manifest row")
    parser.add_argument("--service-url", default=os.environ.get("SERVICE_URL", "http://localhost:10000"))
    parser.add_argument("--image", default="ptxbench-multiturn-eval:latest")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-profiles", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--turn-timeout", type=int, default=980)
    args = parser.parse_args()
    if args.max_parallel <= 0 or args.max_profiles <= 0:
        parser.error("--max-parallel and --max-profiles must be positive")
    rows = read_manifest(args.manifest)
    if args.check:
        print(f"Checked {len(rows)} runnable workload configs in {args.manifest}")
        return 0
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['definition']} -> {path(row['exp_dir'])}", flush=True)
        subprocess.run(command(row, args), cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
