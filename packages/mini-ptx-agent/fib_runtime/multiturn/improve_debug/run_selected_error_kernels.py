#!/usr/bin/env python3
"""Run debug_error_kernel.py over kernels listed in selected_error_kernels.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def safe_name(row: dict, index: int) -> str:
    category = str(row.get("category", "unknown")).lower().replace(" ", "_")
    exp = str(row.get("exp", f"row_{index:03d}"))
    turn = row.get("turn", "unknown")
    return f"{index:03d}_{category}_{exp}_t{turn}"


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    rows = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        if "kernel" not in row:
            raise ValueError(f"row {index} is missing 'kernel'")
        rows.append(row)
    return rows


def build_command(args: argparse.Namespace, kernel: Path, result_path: Path | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(args.debug_script),
        str(kernel),
        "--base-url",
        args.base_url,
        "--definition",
        args.definition,
        "--workload-uuid",
        args.workload_uuid,
        "--timeout",
        str(args.timeout),
        "--wait-timeout",
        str(args.wait_timeout),
        "--http-timeout",
        str(args.http_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--source-context-lines",
        str(args.source_context_lines),
        "--print-limit",
        str(args.print_limit),
        "--coredump-grace-seconds",
        str(args.coredump_grace_seconds),
    ]
    if result_path is not None:
        cmd.extend(["--dump-result", str(result_path)])
    if args.max_lines is not None:
        cmd.extend(["--max-lines", str(args.max_lines)])
    if args.no_coredump:
        cmd.append("--no-coredump")
    return cmd


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "selected_json",
        type=Path,
        nargs="?",
        default=script_dir / "selected_error_kernels.json",
        help="JSON list produced by select_representative_error_kernels.py",
    )
    parser.add_argument("--debug-script", type=Path, default=script_dir / "debug_error_kernel.py")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "debug_selected_outputs")
    parser.add_argument("--base-url", default="http://localhost:10000")
    parser.add_argument("--definition", default="mha_bwd_d128")
    parser.add_argument(
        "--workload-uuid",
        default="38c3b07c-f006-5f5e-9860-ba214c805a6b",
    )
    parser.add_argument("--timeout", type=int, default=45, help="/debug tool timeout per pass")
    parser.add_argument("--wait-timeout", type=float, default=240, help="client wait timeout per row")
    parser.add_argument("--http-timeout", type=float, default=30)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--source-context-lines", type=int, default=4)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--print-limit", type=int, default=8000)
    parser.add_argument("--coredump-grace-seconds", type=float, default=30)
    parser.add_argument("--no-coredump", action="store_true")
    parser.add_argument("--category", action="append", help="Only run rows with this category")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip rows with existing .log files")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    args.selected_json = args.selected_json.expanduser().resolve()
    args.debug_script = args.debug_script.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.selected_json.exists():
        print(f"selected JSON does not exist: {args.selected_json}", file=sys.stderr)
        return 2
    if not args.debug_script.exists():
        print(f"debug script does not exist: {args.debug_script}", file=sys.stderr)
        return 2

    rows = load_rows(args.selected_json)
    selected_categories = set(args.category or [])
    indexed_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if index >= args.start_index
        and (not selected_categories or str(row.get("category")) in selected_categories)
    ]
    if args.limit is not None:
        indexed_rows = indexed_rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.jsonl"
    failures = 0

    with summary_path.open("a") as summary_file:
        for index, row in indexed_rows:
            kernel = Path(str(row["kernel"])).expanduser().resolve()
            stem = safe_name(row, index)
            log_path = args.output_dir / f"{stem}.log"
            meta_path = args.output_dir / f"{stem}.json"
            result_path = args.output_dir / f"{stem}.result.json"

            if args.resume and log_path.exists():
                print(f"[{index}] skip existing {log_path}")
                continue
            if not kernel.exists():
                failures += 1
                record = {
                    "index": index,
                    "status": "missing_kernel",
                    "kernel": str(kernel),
                    "category": row.get("category"),
                    "exp": row.get("exp"),
                    "turn": row.get("turn"),
                    "log_path": str(log_path),
                }
                meta_path.write_text(json.dumps(record, indent=2) + "\n")
                summary_file.write(json.dumps(record) + "\n")
                summary_file.flush()
                print(f"[{index}] missing kernel: {kernel}", file=sys.stderr)
                continue

            cmd = build_command(args, kernel, result_path)
            print(f"[{index}] {row.get('category')} {kernel}")
            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            started = time.time()
            proc = subprocess.run(cmd, text=True, capture_output=True)
            elapsed = time.time() - started
            log_path.write_text(proc.stdout)
            if proc.stderr:
                (args.output_dir / f"{stem}.stderr").write_text(proc.stderr)

            status = "ok" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                failures += 1
            record = {
                "index": index,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "kernel": str(kernel),
                "category": row.get("category"),
                "exp": row.get("exp"),
                "turn": row.get("turn"),
                "original_message": row.get("message"),
                "log_path": str(log_path),
                "result_path": str(result_path) if result_path.exists() else None,
                "stderr_path": str(args.output_dir / f"{stem}.stderr") if proc.stderr else None,
                "command": cmd,
            }
            meta_path.write_text(json.dumps(record, indent=2) + "\n")
            summary_file.write(json.dumps(record) + "\n")
            summary_file.flush()
            print(f"  -> {status} in {elapsed:.1f}s: {log_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
