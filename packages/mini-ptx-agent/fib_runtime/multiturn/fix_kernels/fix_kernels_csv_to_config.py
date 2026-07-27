#!/usr/bin/env python3
"""Convert selected failed-kernel CSV rows to run_parallel_fix_v2 config JSON."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = (
    "definition",
    "test_path",
    "error_kernel_path",
    "error_log_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prompt_configs-style JSON for fix_kernels/run_parallel_fix_v2.py "
            "from select_failed_kernels.py output."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV produced by select_failed_kernels.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON config path.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        required=True,
        help="num_trajectories value for every output item.",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        required=True,
        help="num_turns value for every output item.",
    )
    parser.add_argument(
        "--target-speedup",
        type=float,
        required=True,
        help="target_speedup value for every output item.",
    )
    prompt_tag_group = parser.add_mutually_exclusive_group(required=True)
    prompt_tag_group.add_argument(
        "--prompt-tag",
        help="prompt_tag value for every output item.",
    )
    prompt_tag_group.add_argument(
        "--use-stripped-source-prompt-tag",
        action="store_true",
        help=(
            "Use each row's source run prompt_tag from <exp_dir>/summary.json, "
            "after removing the --strip-source-prompt-tag-suffix value."
        ),
    )
    parser.add_argument(
        "--strip-source-prompt-tag-suffix",
        default="-mha-patched",
        help=(
            "Suffix removed when --use-stripped-source-prompt-tag is set. "
            "Defaults to -mha-patched."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation level. Defaults to 4.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be >= 1")
    if args.num_turns < 1:
        raise ValueError("--num-turns must be >= 1")
    if args.target_speedup < 0:
        raise ValueError("--target-speedup must be >= 0")
    if args.indent < 0:
        raise ValueError("--indent must be >= 0")


def require_file(row: dict[str, str], field: str, *, line_number: int) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"missing {field} at CSV line {line_number}")
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"{field} does not exist at CSV line {line_number}: {path}")
    return str(path.resolve())


def require_text(row: dict[str, str], field: str, *, line_number: int) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"missing {field} at CSV line {line_number}")
    return value


def strip_suffix(value: str, suffix: str) -> str:
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def prompt_tag_from_summary(exp_dir: Path, trajectory_id: str) -> str | None:
    summary_path = exp_dir / "summary.json"
    if not summary_path.is_file():
        return None

    summary = json.loads(summary_path.read_text())
    results = summary.get("results", [])
    if not isinstance(results, list):
        return None
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("exp_name") == trajectory_id:
            prompt_tag = result.get("prompt_tag")
            if isinstance(prompt_tag, str) and prompt_tag.strip():
                return prompt_tag.strip()
    return None


def prompt_tag_from_log(exp_dir: Path, trajectory_id: str) -> str | None:
    log_path = exp_dir / "logs" / f"{trajectory_id}.log"
    if not log_path.is_file():
        return None

    for line in log_path.read_text(errors="replace").splitlines():
        if "--prompt-tag" not in line:
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        for index, part in enumerate(parts):
            if part == "--prompt-tag" and index + 1 < len(parts):
                return parts[index + 1]
            if part.startswith("--prompt-tag="):
                return part.split("=", 1)[1]
    return None


def source_prompt_tag(row: dict[str, str], *, line_number: int, strip_source_suffix: str) -> str:
    exp_dir_text = require_text(row, "exp_dir", line_number=line_number)
    trajectory_id = require_text(row, "trajectory_id", line_number=line_number)
    exp_dir = Path(exp_dir_text).expanduser()
    prompt_tag = prompt_tag_from_summary(exp_dir, trajectory_id)
    if prompt_tag is None:
        prompt_tag = prompt_tag_from_log(exp_dir, trajectory_id)
    if prompt_tag is None:
        raise ValueError(
            f"could not find source prompt_tag for {exp_dir}/{trajectory_id} "
            f"at CSV line {line_number}"
        )
    return strip_suffix(prompt_tag, strip_source_suffix)


def build_config(
    rows: list[dict[str, str]],
    *,
    num_trajectories: int,
    num_turns: int,
    target_speedup: float,
    prompt_tag: str | None,
    use_source_prompt_tag: bool,
    strip_source_prompt_tag_suffix: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        definition = require_text(row, "definition", line_number=line_number)
        test_path = require_file(row, "test_path", line_number=line_number)
        error_kernel_path = require_file(row, "error_kernel_path", line_number=line_number)
        error_log_path = require_file(row, "error_log_path", line_number=line_number)
        item_prompt_tag = (
            source_prompt_tag(
                row,
                line_number=line_number,
                strip_source_suffix=strip_source_prompt_tag_suffix,
            )
            if use_source_prompt_tag
            else prompt_tag
        )
        if not item_prompt_tag:
            raise ValueError(f"missing prompt_tag at CSV line {line_number}")

        items.append(
            {
                "num_trajectories": num_trajectories,
                "num_turns": num_turns,
                "target_speedup": target_speedup,
                "prompt_tag": item_prompt_tag,
                "error_kernel_path": error_kernel_path,
                "error_log_path": error_log_path,
                "definition": definition,
                "test_path": test_path,
            }
        )
    return items


def write_json(path: Path, items: list[dict[str, Any]], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=indent) + "\n")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        fieldnames, rows = read_rows(args.input_csv)
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing_columns:
            raise ValueError(
                f"{args.input_csv} is missing required column(s): {', '.join(missing_columns)}"
            )
        items = build_config(
            rows,
            num_trajectories=args.num_trajectories,
            num_turns=args.num_turns,
            target_speedup=args.target_speedup,
            prompt_tag=args.prompt_tag,
            use_source_prompt_tag=args.use_stripped_source_prompt_tag,
            strip_source_prompt_tag_suffix=args.strip_source_prompt_tag_suffix,
        )
        write_json(args.output, items, args.indent)
        print(f"wrote {len(items)} fix config entries to {args.output}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
