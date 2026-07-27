#!/usr/bin/env python3
"""Generate a markdown report comparing successful kernels to originals."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_CSV_COLUMNS = ("kernel_path", "masked_kernel_path")


@dataclass(frozen=True)
class SuccessKernel:
    exp_index: int
    exp_name: str
    kernel_path: Path
    record_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a markdown report for an eval run. Each success kernel is "
            "diffed against the original kernel_path found by joining run "
            "plan.json masked_kernel_path values to --input-csv."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Eval run directory containing plan.json and success/.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Masked-kernels CSV with kernel_path and masked_kernel_path columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output markdown path. Defaults to stdout.",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="Unified diff context lines. Defaults to 3.",
    )
    return parser.parse_args()


def normalize_path(path_text: str) -> str:
    return str(Path(path_text).expanduser().resolve(strict=False))


def read_csv_originals(path: Path) -> dict[str, Path]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = [name for name in REQUIRED_CSV_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

        originals: dict[str, Path] = {}
        for line_number, row in enumerate(reader, start=2):
            masked_kernel_path = row["masked_kernel_path"].strip()
            original_kernel_path = row["kernel_path"].strip()
            if not masked_kernel_path:
                raise ValueError(f"missing masked_kernel_path at CSV line {line_number}")
            if not original_kernel_path:
                raise ValueError(f"missing kernel_path at CSV line {line_number}")

            key = normalize_path(masked_kernel_path)
            original = Path(original_kernel_path).expanduser().resolve(strict=False)
            previous = originals.get(key)
            if previous is not None and previous != original:
                raise ValueError(
                    "conflicting kernel_path values for masked_kernel_path "
                    f"{masked_kernel_path!r}: {previous} and {original}"
                )
            originals[key] = original
        return originals


def read_plan_items(run_dir: Path) -> list[dict[str, Any]]:
    plan_path = run_dir / "plan.json"
    with plan_path.open() as f:
        data = json.load(f)
    items = data.get("plan")
    if not isinstance(items, list):
        raise ValueError(f"{plan_path} does not contain a list-valued 'plan' field")
    return items


def build_exp_original_map(
    plan_items: list[dict[str, Any]],
    csv_originals: dict[str, Path],
) -> dict[int, Path]:
    exp_originals: dict[int, Path] = {}
    for item in plan_items:
        exp_index = item.get("exp_index")
        masked_kernel_path = item.get("masked_kernel_path")
        if not isinstance(exp_index, int):
            raise ValueError(f"plan item has invalid exp_index: {exp_index!r}")
        if not isinstance(masked_kernel_path, str) or not masked_kernel_path.strip():
            raise ValueError(f"plan item exp_{exp_index:03d} is missing masked_kernel_path")

        key = normalize_path(masked_kernel_path)
        original = csv_originals.get(key)
        if original is None:
            raise ValueError(
                f"no --input-csv row matches plan exp_{exp_index:03d} "
                f"masked_kernel_path: {masked_kernel_path}"
            )
        exp_originals[exp_index] = original
    return exp_originals


def parse_exp_index(exp_name: str) -> int | None:
    if not exp_name.startswith("exp_"):
        return None
    try:
        return int(exp_name.removeprefix("exp_"))
    except ValueError:
        return None


def discover_success_kernels(success_dir: Path) -> list[SuccessKernel]:
    kernels: list[SuccessKernel] = []
    if not success_dir.is_dir():
        raise ValueError(f"success directory does not exist: {success_dir}")

    for exp_dir in success_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        exp_index = parse_exp_index(exp_dir.name)
        if exp_index is None:
            continue
        for kernel_path in sorted(exp_dir.glob("kernel_v*.cu")):
            record_path = exp_dir / "record.json"
            kernels.append(
                SuccessKernel(
                    exp_index=exp_index,
                    exp_name=exp_dir.name,
                    kernel_path=kernel_path,
                    record_path=record_path if record_path.is_file() else None,
                )
            )

    return sorted(kernels, key=lambda item: (item.exp_index, item.kernel_path.name))


def latest_record_entry(record_path: Path | None, kernel_path: Path) -> dict[str, Any] | None:
    if record_path is None:
        return None
    version_text = kernel_path.stem.removeprefix("kernel_v")
    try:
        version = int(version_text)
    except ValueError:
        version = None

    try:
        with record_path.open() as f:
            records = json.load(f)
    except Exception:
        return None
    if not isinstance(records, list):
        return None

    matched: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if version is None or record.get("version") == version:
            matched.append(record)
    return matched[-1] if matched else None


def first_trace(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    traces = record.get("traces")
    if isinstance(traces, list) and traces and isinstance(traces[0], dict):
        return traces[0]
    return None


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def render_metadata(
    success: SuccessKernel,
    original_path: Path,
    record: dict[str, Any] | None,
) -> list[str]:
    lines = [
        f"- Success kernel: `{success.kernel_path}`",
        f"- Original kernel: `{original_path}`",
    ]
    trace = first_trace(record)
    if trace is not None:
        definition = trace.get("definition")
        if definition:
            lines.append(f"- Definition: `{definition}`")
        workload = trace.get("workload")
        if isinstance(workload, dict):
            axes = workload.get("axes")
            workload_uuid = workload.get("uuid")
            if axes is not None:
                lines.append(f"- Workload axes: `{compact_json(axes)}`")
            if workload_uuid:
                lines.append(f"- Workload uuid: `{workload_uuid}`")
        evaluation = trace.get("evaluation")
        if isinstance(evaluation, dict):
            performance = evaluation.get("performance")
            if isinstance(performance, dict):
                speedup = performance.get("speedup_factor")
                latency = performance.get("latency_ms")
                reference_latency = performance.get("reference_latency_ms")
                if speedup is not None:
                    lines.append(f"- Speedup: `{speedup}`")
                if latency is not None and reference_latency is not None:
                    lines.append(
                        f"- Latency: `{latency}` ms vs reference `{reference_latency}` ms"
                    )
    return lines


def unified_diff(original_path: Path, success_path: Path, context_lines: int) -> str:
    original_lines = original_path.read_text(errors="replace").splitlines(keepends=True)
    success_lines = success_path.read_text(errors="replace").splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        original_lines,
        success_lines,
        fromfile=str(original_path),
        tofile=str(success_path),
        n=context_lines,
    )
    diff = "".join(diff_lines)
    return diff if diff else "(no changes)\n"


def render_report(
    run_dir: Path,
    input_csv: Path,
    successes: list[SuccessKernel],
    exp_originals: dict[int, Path],
    context_lines: int,
) -> str:
    lines = [
        f"# Success Kernel Diff Report: `{run_dir}`",
        "",
        f"- Input CSV: `{input_csv}`",
        f"- Success kernels: `{len(successes)}`",
        "",
    ]

    for success in successes:
        original_path = exp_originals.get(success.exp_index)
        if original_path is None:
            raise ValueError(f"no plan entry found for {success.exp_name}")
        if not original_path.is_file():
            raise ValueError(f"original kernel does not exist for {success.exp_name}: {original_path}")

        record = latest_record_entry(success.record_path, success.kernel_path)
        lines.append(f"## {success.exp_name} / {success.kernel_path.name}")
        lines.extend(render_metadata(success, original_path, record))
        lines.extend(
            [
                "",
                "```diff",
                unified_diff(original_path, success.kernel_path, context_lines).rstrip("\n"),
                "```",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    try:
        if args.context_lines < 0:
            raise ValueError("--context-lines must be >= 0")

        run_dir = args.run_dir.expanduser().resolve(strict=False)
        input_csv = args.input_csv.expanduser().resolve(strict=False)
        if not run_dir.is_dir():
            raise ValueError(f"--run-dir does not exist: {run_dir}")
        if not input_csv.is_file():
            raise ValueError(f"--input-csv does not exist: {input_csv}")

        csv_originals = read_csv_originals(input_csv)
        plan_items = read_plan_items(run_dir)
        exp_originals = build_exp_original_map(plan_items, csv_originals)
        successes = discover_success_kernels(run_dir / "success")
        report = render_report(run_dir, input_csv, successes, exp_originals, args.context_lines)

        if args.output:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            print(f"wrote {len(successes)} success kernel diffs to {output_path}")
        else:
            print(report, end="")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
