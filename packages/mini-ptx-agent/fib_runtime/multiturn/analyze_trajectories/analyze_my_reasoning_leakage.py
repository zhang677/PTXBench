#!/usr/bin/env python3
"""Count `my_reasoning` leakage in trajectory reasoning content.

By default this scans the five 2026-0624-0939 eval roots that have
figures/turn_correctness_arch.csv. The leakage signal is an exact substring
match in assistant reasoning_content, with provider_specific_fields used as a
fallback if the top-level field is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUN_GLOB = "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939*"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "my-reasoning-leakage-2026-0624-0939"
)


@dataclass(frozen=True)
class LeakageRow:
    run: str
    exp_dir: str
    trajectory_id: str
    turn: int
    correctness: str
    speedup: str
    arch_tag: str
    reasoning_chars: int
    occurrence_count: int
    first_offset: int
    reasoning_snippet: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-glob",
        default=DEFAULT_RUN_GLOB,
        help="Glob for eval run directories to scan.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        dest="run_dirs",
        help="Specific eval run directory to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--needle",
        default="my_reasoning",
        help="Substring to search for in reasoning_content.",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Use case-insensitive matching for --needle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV, JSON, and markdown summary.",
    )
    parser.add_argument(
        "--snippet-radius",
        type=int,
        default=180,
        help="Characters retained on each side of the first match.",
    )
    parser.add_argument(
        "--include-no-csv",
        action="store_true",
        help="Include run dirs without turn_correctness_arch.csv in skipped_runs manifest only.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def get_reasoning(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning_content")
    if not reasoning:
        provider_fields = message.get("provider_specific_fields") or {}
        if isinstance(provider_fields, dict):
            reasoning = provider_fields.get("reasoning_content")
    if reasoning is None:
        return ""
    if not isinstance(reasoning, str):
        return str(reasoning)
    return reasoning


def shorten_around(text: str, offset: int, needle_len: int, radius: int) -> str:
    if offset < 0:
        return ""
    start = max(0, offset - radius)
    end = min(len(text), offset + needle_len + radius)
    snippet = text[start:end]
    snippet = " ".join(snippet.split())
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def count_occurrences(text: str, needle: str, ignore_case: bool) -> tuple[int, int]:
    if needle == "":
        raise ValueError("--needle must be non-empty")
    haystack = text.lower() if ignore_case else text
    target = needle.lower() if ignore_case else needle
    first = haystack.find(target)
    if first < 0:
        return 0, -1
    count = 0
    pos = first
    while pos >= 0:
        count += 1
        pos = haystack.find(target, pos + len(target))
    return count, first


def discover_run_dirs(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    if args.run_dirs:
        candidates = [path.resolve() for path in args.run_dirs]
    else:
        candidates = sorted(Path(path).resolve() for path in Path("/").glob(args.run_glob.lstrip("/")))

    run_dirs: list[Path] = []
    skipped: list[Path] = []
    for run_dir in candidates:
        csv_path = run_dir / "figures" / "turn_correctness_arch.csv"
        if csv_path.exists():
            run_dirs.append(run_dir)
        else:
            skipped.append(run_dir)
    return run_dirs, skipped


def read_turn_rows(csv_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"trajectory_id", "turn", "correctness"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            rows[(row["trajectory_id"].strip(), int(row["turn"]))] = row
    return rows


def scan_run(
    run_dir: Path,
    needle: str,
    ignore_case: bool,
    snippet_radius: int,
) -> tuple[list[LeakageRow], int, int]:
    turn_rows = read_turn_rows(run_dir / "figures" / "turn_correctness_arch.csv")
    leak_rows: list[LeakageRow] = []
    total_turns = 0
    total_trajectories = 0

    trajectories_dir = run_dir / "trajectories"
    for trajectory_path in sorted(trajectories_dir.glob("*.json")):
        data = read_json(trajectory_path)
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            continue
        total_trajectories += 1
        trajectory_id = trajectory_path.stem
        assistant_turn = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            reasoning = get_reasoning(message)
            count, first = count_occurrences(reasoning, needle, ignore_case)
            total_turns += 1
            if count:
                turn_row = turn_rows.get((trajectory_id, assistant_turn), {})
                leak_rows.append(
                    LeakageRow(
                        run=run_dir.name,
                        exp_dir=str(run_dir),
                        trajectory_id=trajectory_id,
                        turn=assistant_turn,
                        correctness=turn_row.get("correctness", ""),
                        speedup=turn_row.get("speedup", ""),
                        arch_tag=turn_row.get("arch_tag", ""),
                        reasoning_chars=len(reasoning),
                        occurrence_count=count,
                        first_offset=first,
                        reasoning_snippet=shorten_around(
                            reasoning,
                            first,
                            len(needle),
                            snippet_radius,
                        ),
                    )
                )
            assistant_turn += 1
    return leak_rows, total_turns, total_trajectories


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    run_dirs: list[Path],
    skipped_runs: list[Path],
    rows: list[LeakageRow],
    totals_by_run: dict[str, tuple[int, int]],
    needle: str,
    ignore_case: bool,
    include_no_csv: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "my_reasoning_leakage_rows.csv"
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"
    manifest_json = output_dir / "manifest.json"

    row_dicts = [asdict(row) for row in rows]
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(LeakageRow.__dataclass_fields__)
    write_csv(rows_csv, row_dicts, fieldnames)

    summary_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_rows = [row for row in rows if row.run == run_dir.name]
        total_turns, total_trajectories = totals_by_run[run_dir.name]
        leaked_trajectories = len({row.trajectory_id for row in run_rows})
        leaked_turns = len(run_rows)
        summary_rows.append(
            {
                "run": run_dir.name,
                "total_trajectories": total_trajectories,
                "leaked_trajectories": leaked_trajectories,
                "trajectory_leak_rate": (
                    f"{leaked_trajectories / total_trajectories:.3f}"
                    if total_trajectories
                    else "0.000"
                ),
                "total_assistant_turns": total_turns,
                "leaked_turns": leaked_turns,
                "turn_leak_rate": (
                    f"{leaked_turns / total_turns:.3f}" if total_turns else "0.000"
                ),
                "occurrences": sum(row.occurrence_count for row in run_rows),
            }
        )
    totals = {
        "run": "TOTAL",
        "total_trajectories": sum(int(row["total_trajectories"]) for row in summary_rows),
        "leaked_trajectories": sum(int(row["leaked_trajectories"]) for row in summary_rows),
        "trajectory_leak_rate": "0.000",
        "total_assistant_turns": sum(int(row["total_assistant_turns"]) for row in summary_rows),
        "leaked_turns": sum(int(row["leaked_turns"]) for row in summary_rows),
        "turn_leak_rate": "0.000",
        "occurrences": sum(int(row["occurrences"]) for row in summary_rows),
    }
    if totals["total_trajectories"]:
        totals["trajectory_leak_rate"] = (
            f"{totals['leaked_trajectories'] / totals['total_trajectories']:.3f}"
        )
    if totals["total_assistant_turns"]:
        totals["turn_leak_rate"] = (
            f"{totals['leaked_turns'] / totals['total_assistant_turns']:.3f}"
        )
    summary_rows.append(totals)
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()) if summary_rows else [])

    manifest = {
        "needle": needle,
        "ignore_case": ignore_case,
        "run_dirs": [str(path) for path in run_dirs],
        "skipped_no_csv_runs": [str(path) for path in skipped_runs] if include_no_csv else [],
        "outputs": {
            "summary_csv": str(summary_csv),
            "rows_csv": str(rows_csv),
            "summary_md": str(summary_md),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# `my_reasoning` Leakage",
        "",
        f"Signal: substring `{needle}` in assistant `reasoning_content`.",
        "",
        "| run | total trajectories | leaked trajectories | trajectory rate | total turns | leaked turns | turn rate | occurrences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['run']}` | {row['total_trajectories']} | {row['leaked_trajectories']} | "
            f"{row['trajectory_leak_rate']} | {row['total_assistant_turns']} | "
            f"{row['leaked_turns']} | {row['turn_leak_rate']} | {row['occurrences']} |"
        )
    lines.extend(["", "## Leaked Turns", ""])
    if not rows:
        lines.append("No leakage rows found.")
    else:
        for row in rows:
            lines.append(
                f"- `{row.run}` `{row.trajectory_id}` turn `{row.turn}` "
                f"({row.correctness or 'unknown'}), occurrences `{row.occurrence_count}`: "
                f"{row.reasoning_snippet}"
            )
    lines.extend(["", "## Outputs", ""])
    for key, path in manifest["outputs"].items():
        lines.append(f"- {key}: `{path}`")
    summary_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dirs, skipped_runs = discover_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run directories with figures/turn_correctness_arch.csv found.")

    rows: list[LeakageRow] = []
    totals_by_run: dict[str, tuple[int, int]] = {}
    for run_dir in run_dirs:
        run_rows, total_turns, total_trajectories = scan_run(
            run_dir,
            needle=args.needle,
            ignore_case=args.ignore_case,
            snippet_radius=args.snippet_radius,
        )
        rows.extend(run_rows)
        totals_by_run[run_dir.name] = (total_turns, total_trajectories)

    write_outputs(
        args.output_dir,
        run_dirs,
        skipped_runs,
        rows,
        totals_by_run,
        needle=args.needle,
        ignore_case=args.ignore_case,
        include_no_csv=args.include_no_csv,
    )
    leaked_trajectories = sum(
        len({row.trajectory_id for row in rows if row.run == run_dir.name})
        for run_dir in run_dirs
    )
    print(f"Scanned {len(run_dirs)} runs.")
    print(f"Leaked turns: {len(rows)}")
    print(f"Leaked trajectories: {leaked_trajectories}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
