#!/usr/bin/env python3
"""Export assistant expert reasoning from eval trajectories.

This is a companion to analyze_stale_correct_next_reasoning.py. By default it
looks for that script's stale_reasoning_rows.csv and exports the corresponding
next-turn reasoning. If no input CSV exists, pass --run-glob or --run-dir to
export all assistant turns from matching trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUN_GLOB = "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939*"
DEFAULT_STALE_CSV = (
    Path(__file__).resolve().parent
    / "stale-correct-next-reasoning-2026-0624-0939"
    / "stale_reasoning_rows.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "expert-reasoning-2026-0624-0939"
)
CPP_BLOCK_RE = re.compile(
    r"```(?:cpp|cuda|c\+\+)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ExportRow:
    run: str
    exp_dir: str
    trajectory_id: str
    assistant_turn: int
    previous_turn: str
    referenced_turn: str
    referenced_turn_correctness: str
    referenced_turn_distance: str
    referenced_match_score: str
    referenced_match_terms: str
    selection_source: str
    reasoning_chars: int
    content_chars: int
    code_block_count: int
    first_code_chars: int
    previous_feedback_snippet: str
    reasoning_snippet: str
    content_snippet: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_STALE_CSV,
        help="Rows to export. Expected columns include exp_dir, trajectory_id, and next_turn or assistant_turn.",
    )
    parser.add_argument(
        "--ignore-input-csv",
        action="store_true",
        help="Ignore --input-csv and scan all assistant turns from selected run dirs.",
    )
    parser.add_argument(
        "--run-glob",
        default=DEFAULT_RUN_GLOB,
        help="Glob for eval run directories when scanning all assistant turns.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        dest="run_dirs",
        help="Specific eval run directory to scan. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV, JSONL, and markdown report.",
    )
    parser.add_argument(
        "--include-full-reasoning",
        action="store_true",
        help="Write full reasoning/content records to expert_reasoning_full.jsonl.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=600,
        help="Maximum characters retained in CSV/markdown snippets.",
    )
    return parser.parse_args()


def shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 4)].rstrip() + " ..."


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


def get_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)


def get_feedback_before_turn(
    messages: list[dict[str, Any]],
    assistant_entries: list[tuple[int, dict[str, Any]]],
    assistant_turn: int,
) -> str:
    if assistant_turn <= 0:
        return ""
    previous_message_idx = assistant_entries[assistant_turn - 1][0]
    feedback_idx = previous_message_idx + 1
    if feedback_idx >= len(messages):
        return ""
    feedback_message = messages[feedback_idx]
    if feedback_message.get("role") != "user":
        return ""
    content = feedback_message.get("content", "")
    return content if isinstance(content, str) else str(content)


def selections_from_csv(path: Path) -> list[dict[str, str]]:
    selections: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"exp_dir", "trajectory_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            turn_value = row.get("assistant_turn") or row.get("next_turn")
            if turn_value in (None, ""):
                raise ValueError(f"{path} row missing assistant_turn/next_turn: {row!r}")
            selections.append(
                {
                    "run": row.get("run", Path(row["exp_dir"]).name),
                    "exp_dir": row["exp_dir"],
                    "trajectory_id": row["trajectory_id"],
                    "assistant_turn": str(int(turn_value)),
                    "previous_turn": row.get("previous_turn", ""),
                    "referenced_turn": row.get("referenced_turn", ""),
                    "referenced_turn_correctness": row.get("referenced_turn_correctness", ""),
                    "referenced_turn_distance": row.get("referenced_turn_distance", ""),
                    "referenced_match_score": row.get("referenced_match_score", ""),
                    "referenced_match_terms": row.get("referenced_match_terms", ""),
                    "selection_source": str(path),
                }
            )
    return selections


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dirs:
        return [path.resolve() for path in args.run_dirs]
    return sorted(Path(path).resolve() for path in Path("/").glob(args.run_glob.lstrip("/")))


def selections_from_runs(run_dirs: list[Path]) -> list[dict[str, str]]:
    selections: list[dict[str, str]] = []
    for run_dir in run_dirs:
        trajectories_dir = run_dir / "trajectories"
        if not trajectories_dir.exists():
            continue
        for trajectory_path in sorted(trajectories_dir.glob("*.json")):
            data = read_json(trajectory_path)
            messages = data.get("messages", [])
            if not isinstance(messages, list):
                continue
            assistant_count = sum(
                1
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            )
            for assistant_turn in range(assistant_count):
                selections.append(
                    {
                        "run": run_dir.name,
                        "exp_dir": str(run_dir),
                        "trajectory_id": trajectory_path.stem,
                        "assistant_turn": str(assistant_turn),
                        "previous_turn": str(assistant_turn - 1) if assistant_turn else "",
                        "referenced_turn": "",
                        "referenced_turn_correctness": "",
                        "referenced_turn_distance": "",
                        "referenced_match_score": "",
                        "referenced_match_terms": "",
                        "selection_source": "run_scan",
                    }
                )
    return selections


def export_selection(
    selection: dict[str, str],
    snippet_chars: int,
) -> tuple[ExportRow, dict[str, Any]] | None:
    exp_dir = Path(selection["exp_dir"])
    trajectory_id = selection["trajectory_id"]
    assistant_turn = int(selection["assistant_turn"])
    trajectory_path = exp_dir / "trajectories" / f"{trajectory_id}.json"
    if not trajectory_path.exists():
        return None

    data = read_json(trajectory_path)
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return None
    assistant_entries: list[tuple[int, dict[str, Any]]] = [
        (idx, message)
        for idx, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if assistant_turn < 0 or assistant_turn >= len(assistant_entries):
        return None

    message_idx, message = assistant_entries[assistant_turn]
    reasoning = get_reasoning(message)
    content = get_content(message)
    code_blocks = CPP_BLOCK_RE.findall(content)
    feedback = get_feedback_before_turn(messages, assistant_entries, assistant_turn)
    row = ExportRow(
        run=selection.get("run") or exp_dir.name,
        exp_dir=str(exp_dir),
        trajectory_id=trajectory_id,
        assistant_turn=assistant_turn,
        previous_turn=selection.get("previous_turn", ""),
        referenced_turn=selection.get("referenced_turn", ""),
        referenced_turn_correctness=selection.get("referenced_turn_correctness", ""),
        referenced_turn_distance=selection.get("referenced_turn_distance", ""),
        referenced_match_score=selection.get("referenced_match_score", ""),
        referenced_match_terms=selection.get("referenced_match_terms", ""),
        selection_source=selection.get("selection_source", ""),
        reasoning_chars=len(reasoning),
        content_chars=len(content),
        code_block_count=len(code_blocks),
        first_code_chars=len(code_blocks[0]) if code_blocks else 0,
        previous_feedback_snippet=shorten(feedback, snippet_chars),
        reasoning_snippet=shorten(reasoning, snippet_chars),
        content_snippet=shorten(content, snippet_chars),
    )
    full_record = {
        **asdict(row),
        "trajectory_path": str(trajectory_path),
        "message_idx": message_idx,
        "previous_feedback": feedback,
        "reasoning_content": reasoning,
        "assistant_content": content,
    }
    return row, full_record


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    rows: list[ExportRow],
    full_records: list[dict[str, Any]],
    include_full_reasoning: bool,
    selection_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "expert_reasoning_rows.csv"
    full_jsonl = output_dir / "expert_reasoning_full.jsonl"
    report_md = output_dir / "expert_reasoning_report.md"
    manifest_json = output_dir / "manifest.json"

    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ExportRow.__dataclass_fields__)
    write_csv(rows_csv, [asdict(row) for row in rows], fieldnames)

    if include_full_reasoning:
        with full_jsonl.open("w") as f:
            for record in full_records:
                f.write(json.dumps(record) + "\n")
    elif full_jsonl.exists():
        full_jsonl.unlink()

    lines = [
        "# Expert Reasoning Export",
        "",
        f"Selected rows: {selection_count}",
        f"Exported rows: {len(rows)}",
        "",
        "| run | trajectory | assistant turn | referenced turn | referenced correctness | reasoning chars | code blocks | reasoning snippet |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.run}` | `{row.trajectory_id}` | {row.assistant_turn} | "
            f"{row.referenced_turn or ''} | {row.referenced_turn_correctness or ''} | "
            f"{row.reasoning_chars} | {row.code_block_count} | {row.reasoning_snippet} |"
        )
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- rows_csv: `{rows_csv}`")
    if include_full_reasoning:
        lines.append(f"- full_jsonl: `{full_jsonl}`")
    lines.append(f"- report_md: `{report_md}`")
    lines.append(f"- manifest_json: `{manifest_json}`")
    report_md.write_text("\n".join(lines) + "\n")

    manifest = {
        "selected_rows": selection_count,
        "exported_rows": len(rows),
        "include_full_reasoning": include_full_reasoning,
        "outputs": {
            "rows_csv": str(rows_csv),
            "full_jsonl": str(full_jsonl) if include_full_reasoning else "",
            "report_md": str(report_md),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if not args.ignore_input_csv and args.input_csv.exists():
        selections = selections_from_csv(args.input_csv)
    else:
        selections = selections_from_runs(discover_run_dirs(args))

    exported_rows: list[ExportRow] = []
    full_records: list[dict[str, Any]] = []
    for selection in selections:
        exported = export_selection(selection, args.snippet_chars)
        if exported is None:
            continue
        row, full_record = exported
        exported_rows.append(row)
        full_records.append(full_record)

    write_outputs(
        args.output_dir,
        exported_rows,
        full_records,
        include_full_reasoning=args.include_full_reasoning,
        selection_count=len(selections),
    )
    print(f"Selected {len(selections)} rows, exported {len(exported_rows)} rows.")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
