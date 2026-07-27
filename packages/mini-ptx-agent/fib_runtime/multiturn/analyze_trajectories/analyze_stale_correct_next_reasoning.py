#!/usr/bin/env python3
"""Find next-turn reasoning that ignores a correct previous turn.

The correctness source of truth is each eval run's
figures/turn_correctness_arch.csv. The reasoning source is the next assistant
message in trajectories/<trajectory_id>.json.

By default this scans the five 2026-0624-0939 eval roots that have
turn_correctness_arch.csv and writes summary/detail artifacts under:

  stale-correct-next-reasoning-2026-0624-0939/
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
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "stale-correct-next-reasoning-2026-0624-0939"
)

ACK_RE = re.compile(
    r"\b("
    r"PASSED|passed|correct results?|produces correct|kernel is correct|"
    r"correctness|correct but|speedup|performance|slow(?:er)?|optimi[sz]|target"
    r")\b",
    re.IGNORECASE,
)
FAILURE_RE = re.compile(
    r"\b("
    r"INCORRECT_NUMERICAL|RUNTIME_ERROR|COMPILATION|compilation error|"
    r"compile error|fails? to compile|failed|failure|timeout|illegal memory|"
    r"invalid argument|wrong results?|produces wrong|incorrect numerical|"
    r"error message says|what went wrong|how to fix it|bug"
    r")\b",
    re.IGNORECASE,
)
CORRECT_PERF_CONTEXT_RE = re.compile(
    r"correct[^.]{0,120}(but|slow|speedup)|"
    r"passed[^.]{0,120}(but|speedup)|"
    r"speedup[^.]{0,120}(target|only|slow)",
    re.IGNORECASE,
)
REFERENCE_FEATURE_RES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bINCORRECT_NUMERICAL\b",
        r"\bRUNTIME_ERROR\b",
        r"\bTIMEOUT\b",
        r"CUDA error [^`\n<.]+(?: at kernel\.cu:\d+)?",
        r"kernel\.cu\(\d+\)",
        r"kernel\.cu:\d+",
        r"max_abs_error\s*=\s*(?:NaN/Inf|[-+0-9.eE]+)",
        r"max_rel_error\s*=\s*(?:NaN/Inf|[-+0-9.eE]+)",
        r'identifier\s+"[^"]+"\s+is undefined',
        r'argument of type\s+"[^"]+"[^`\n]*?incompatible with parameter of type\s+"[^"]+"',
        r'a value of type\s+"[^"]+"[^`\n]*?cannot be used[^`\n]*',
        r'has no member\s+"[^"]+"',
        r"Failed to create TMA descriptor for [A-Za-z0-9_]+",
        r"Could not extract a ```cpp code block",
        r"\bUsed \d+ registers\b",
        r"\bused \d+ barriers\b",
        r"\b\d+ bytes smem\b",
        r"\b\d+ bytes stack frame\b",
        r"\b\d+ bytes spill stores\b",
        r"\b\d+ bytes spill loads\b",
        r"\bspeedup:\s*[0-9.]+x\b",
        r"illegal memory access",
        r"invalid argument",
        r"invalid configuration",
        r"illegal instruction",
        r"fails? to compile",
        r"compilation errors?",
    )
]
QUOTED_DIAGNOSTIC_RE = re.compile(r'["`]([A-Za-z_][A-Za-z0-9_:.<>-]{0,80})["`]')


@dataclass(frozen=True)
class ReasoningRow:
    run: str
    exp_dir: str
    trajectory_id: str
    previous_turn: int
    next_turn: int
    referenced_turn: str
    referenced_turn_correctness: str
    referenced_turn_speedup: str
    referenced_turn_distance: str
    referenced_match_score: int
    referenced_match_terms: str
    previous_correctness: str
    previous_speedup: str
    previous_arch_tag: str
    classification: str
    has_failure_cue: bool
    has_correct_or_perf_cue: bool
    first_failure_cue: str
    first_correct_or_perf_cue: str
    next_reasoning_chars: int
    previous_feedback_snippet: str
    referenced_feedback_snippet: str
    next_reasoning_snippet: str


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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV, JSON, and markdown summary.",
    )
    parser.add_argument(
        "--window-chars",
        type=int,
        default=2500,
        help="Leading reasoning window used for classification.",
    )
    parser.add_argument(
        "--early-failure-chars",
        type=int,
        default=700,
        help="Treat a failure cue before this character offset as stale when it precedes any correct/perf cue.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=360,
        help="Maximum characters retained in CSV/markdown snippets.",
    )
    parser.add_argument(
        "--include-no-csv",
        action="store_true",
        help="Include run directories without figures/turn_correctness_arch.csv in manifest skipped_runs only.",
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


def get_feedback_after(messages: list[dict[str, Any]], assistant_msg_idx: int) -> str:
    next_idx = assistant_msg_idx + 1
    if next_idx >= len(messages):
        return ""
    message = messages[next_idx]
    if message.get("role") != "user":
        return ""
    content = message.get("content", "")
    return content if isinstance(content, str) else str(content)


def read_turn_rows(csv_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    required = {"trajectory_id", "turn", "correctness"}
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                turn = int(row["turn"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path} has invalid turn value: {row!r}") from exc
            rows[(row["trajectory_id"].strip(), turn)] = row
    return rows


def read_correct_turns(csv_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    return {
        key: row
        for key, row in read_turn_rows(csv_path).items()
        if row.get("correctness") == "Correct"
    }


def classify_reasoning(
    reasoning: str,
    window_chars: int,
    early_failure_chars: int,
) -> tuple[str, bool, bool, str, str]:
    window = reasoning[:window_chars]
    failure_match = FAILURE_RE.search(window)
    ack_match = ACK_RE.search(window)
    has_failure = failure_match is not None
    has_ack = ack_match is not None

    stale = has_failure and not has_ack
    if failure_match and (not ack_match or failure_match.start() < ack_match.start()):
        early_failure = failure_match.start() < early_failure_chars
        if early_failure and not CORRECT_PERF_CONTEXT_RE.search(reasoning[:1200]):
            stale = True

    classification = "stale_failure_after_correct" if stale else "aligned_correct_or_perf"
    return (
        classification,
        has_failure,
        has_ack,
        failure_match.group(0) if failure_match else "",
        ack_match.group(0) if ack_match else "",
    )


def normalize_feature(value: str) -> str:
    return " ".join(value.lower().strip("`'\".,;:()[]{} ").split())


def extract_reference_features(text: str) -> set[str]:
    features: set[str] = set()
    for pattern in REFERENCE_FEATURE_RES:
        for match in pattern.finditer(text):
            feature = normalize_feature(match.group(0))
            if feature:
                features.add(feature)
    for match in QUOTED_DIAGNOSTIC_RE.finditer(text):
        feature = normalize_feature(match.group(1))
        if (
            "__" in feature
            or "kernel.cu" in feature
            or feature.startswith(("smem", "mbar", "tma", "wgmma", "cuda"))
            or feature in {"out", "out1", "tid", "d"}
        ):
            features.add(feature)
    return features


def score_referenced_feedback(reasoning: str, feedback: str) -> tuple[int, list[str]]:
    reasoning_features = extract_reference_features(reasoning[:12000])
    feedback_features = extract_reference_features(feedback)
    matched = sorted(reasoning_features.intersection(feedback_features))
    score = 0
    for feature in matched:
        if feature.startswith(("incorrect_numerical", "runtime_error", "timeout")):
            score += 4
        elif "kernel.cu" in feature or "max_" in feature or "cuda error" in feature:
            score += 4
        elif feature.startswith(("identifier ", "argument of type", "a value of type")):
            score += 5
        elif feature in {"invalid argument", "invalid configuration", "illegal memory access"}:
            score += 3
        elif re.search(r"\b(registers|barriers|smem|stack frame|spill stores|spill loads|speedup)\b", feature):
            score += 1
        else:
            score += 2
    return score, matched


def infer_referenced_turn(
    reasoning: str,
    feedback_by_turn: dict[int, str],
    turn_rows: dict[tuple[str, int], dict[str, str]],
    trajectory_id: str,
    next_turn: int,
) -> tuple[str, str, str, str, int, str, str]:
    best_turn = ""
    best_score = 0
    best_terms: list[str] = []
    best_feedback = ""

    for turn, feedback in feedback_by_turn.items():
        if turn >= next_turn:
            continue
        score, terms = score_referenced_feedback(reasoning, feedback)
        if score > best_score or (score == best_score and score > 0 and str(turn) > best_turn):
            best_turn = str(turn)
            best_score = score
            best_terms = terms
            best_feedback = feedback

    if best_score <= 0 or best_turn == "":
        return "", "", "", "", 0, "", ""

    referenced_row = turn_rows.get((trajectory_id, int(best_turn)), {})
    distance = str(next_turn - int(best_turn))
    return (
        best_turn,
        referenced_row.get("correctness", ""),
        referenced_row.get("speedup", ""),
        distance,
        best_score,
        "; ".join(best_terms[:12]),
        best_feedback,
    )


def scan_run(
    run_dir: Path,
    window_chars: int,
    early_failure_chars: int,
    snippet_chars: int,
) -> tuple[list[ReasoningRow], int]:
    csv_path = run_dir / "figures" / "turn_correctness_arch.csv"
    turn_rows = read_turn_rows(csv_path)
    correct_turns = {
        key: row for key, row in turn_rows.items() if row.get("correctness") == "Correct"
    }
    rows: list[ReasoningRow] = []
    terminal_correct = 0

    trajectory_ids = sorted({trajectory_id for trajectory_id, _ in correct_turns})
    for trajectory_id in trajectory_ids:
        trajectory_path = run_dir / "trajectories" / f"{trajectory_id}.json"
        if not trajectory_path.exists():
            continue
        data = read_json(trajectory_path)
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            continue

        assistant_entries: list[tuple[int, dict[str, Any]]] = [
            (idx, message)
            for idx, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        feedback_by_turn = {
            assistant_turn: get_feedback_after(messages, message_idx)
            for assistant_turn, (message_idx, _) in enumerate(assistant_entries)
        }
        for assistant_turn, (message_idx, _) in enumerate(assistant_entries):
            correct_row = correct_turns.get((trajectory_id, assistant_turn))
            if correct_row is None:
                continue
            if assistant_turn == len(assistant_entries) - 1:
                terminal_correct += 1
                continue

            next_message = assistant_entries[assistant_turn + 1][1]
            reasoning = get_reasoning(next_message)
            (
                classification,
                has_failure,
                has_ack,
                first_failure,
                first_ack,
            ) = classify_reasoning(reasoning, window_chars, early_failure_chars)
            feedback = get_feedback_after(messages, message_idx)
            (
                referenced_turn,
                referenced_correctness,
                referenced_speedup,
                referenced_distance,
                referenced_score,
                referenced_terms,
                referenced_feedback,
            ) = infer_referenced_turn(
                reasoning,
                feedback_by_turn,
                turn_rows,
                trajectory_id,
                assistant_turn + 1,
            )
            rows.append(
                ReasoningRow(
                    run=run_dir.name,
                    exp_dir=str(run_dir),
                    trajectory_id=trajectory_id,
                    previous_turn=assistant_turn,
                    next_turn=assistant_turn + 1,
                    referenced_turn=referenced_turn,
                    referenced_turn_correctness=referenced_correctness,
                    referenced_turn_speedup=referenced_speedup,
                    referenced_turn_distance=referenced_distance,
                    referenced_match_score=referenced_score,
                    referenced_match_terms=referenced_terms,
                    previous_correctness=correct_row.get("correctness", ""),
                    previous_speedup=correct_row.get("speedup", ""),
                    previous_arch_tag=correct_row.get("arch_tag", ""),
                    classification=classification,
                    has_failure_cue=has_failure,
                    has_correct_or_perf_cue=has_ack,
                    first_failure_cue=first_failure,
                    first_correct_or_perf_cue=first_ack,
                    next_reasoning_chars=len(reasoning),
                    previous_feedback_snippet=shorten(feedback, snippet_chars),
                    referenced_feedback_snippet=shorten(referenced_feedback, snippet_chars),
                    next_reasoning_snippet=shorten(reasoning, snippet_chars),
                )
            )
    return rows, terminal_correct


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    run_dirs: list[Path],
    skipped_runs: list[Path],
    rows: list[ReasoningRow],
    terminal_by_run: dict[str, int],
    include_no_csv: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ReasoningRow.__dataclass_fields__)

    all_csv = output_dir / "correct_next_reasoning_rows.csv"
    stale_csv = output_dir / "stale_reasoning_rows.csv"
    summary_csv = output_dir / "summary.csv"
    referenced_counts_csv = output_dir / "referenced_turn_counts.csv"
    manifest_json = output_dir / "manifest.json"
    summary_md = output_dir / "summary.md"

    write_csv(all_csv, row_dicts, fieldnames)
    write_csv(
        stale_csv,
        [row for row in row_dicts if row["classification"] == "stale_failure_after_correct"],
        fieldnames,
    )

    summary_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_rows = [row for row in rows if row.run == run_dir.name]
        stale_count = sum(
            1 for row in run_rows if row.classification == "stale_failure_after_correct"
        )
        aligned_count = len(run_rows) - stale_count
        terminal_count = terminal_by_run.get(run_dir.name, 0)
        correct_total = len(run_rows) + terminal_count
        summary_rows.append(
            {
                "run": run_dir.name,
                "correct_total": correct_total,
                "correct_with_next": len(run_rows),
                "terminal_correct": terminal_count,
                "stale_failure_after_correct": stale_count,
                "aligned_correct_or_perf": aligned_count,
                "stale_rate_of_correct_with_next": (
                    f"{stale_count / len(run_rows):.3f}" if run_rows else "0.000"
                ),
            }
        )
    totals = {
        "run": "TOTAL",
        "correct_total": sum(int(row["correct_total"]) for row in summary_rows),
        "correct_with_next": sum(int(row["correct_with_next"]) for row in summary_rows),
        "terminal_correct": sum(int(row["terminal_correct"]) for row in summary_rows),
        "stale_failure_after_correct": sum(
            int(row["stale_failure_after_correct"]) for row in summary_rows
        ),
        "aligned_correct_or_perf": sum(
            int(row["aligned_correct_or_perf"]) for row in summary_rows
        ),
        "stale_rate_of_correct_with_next": "0.000",
    }
    if totals["correct_with_next"]:
        totals["stale_rate_of_correct_with_next"] = (
            f"{totals['stale_failure_after_correct'] / totals['correct_with_next']:.3f}"
        )
    summary_rows.append(totals)
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()) if summary_rows else [])

    referenced_count_map: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        if row.classification != "stale_failure_after_correct":
            continue
        key = (
            row.run,
            row.referenced_turn,
            row.referenced_turn_correctness,
            row.referenced_turn_distance,
        )
        referenced_count_map[key] = referenced_count_map.get(key, 0) + 1
    referenced_count_rows = [
        {
            "run": run,
            "referenced_turn": referenced_turn,
            "referenced_turn_correctness": referenced_correctness,
            "referenced_turn_distance": referenced_distance,
            "stale_rows": count,
        }
        for (run, referenced_turn, referenced_correctness, referenced_distance), count in sorted(
            referenced_count_map.items()
        )
    ]
    write_csv(
        referenced_counts_csv,
        referenced_count_rows,
        [
            "run",
            "referenced_turn",
            "referenced_turn_correctness",
            "referenced_turn_distance",
            "stale_rows",
        ],
    )

    manifest = {
        "run_dirs": [str(path) for path in run_dirs],
        "skipped_no_csv_runs": [str(path) for path in skipped_runs] if include_no_csv else [],
        "outputs": {
            "summary_csv": str(summary_csv),
            "referenced_counts_csv": str(referenced_counts_csv),
            "all_rows_csv": str(all_csv),
            "stale_rows_csv": str(stale_csv),
            "summary_md": str(summary_md),
            "manifest_json": str(manifest_json),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Stale Correct Next-Turn Reasoning",
        "",
        "Predicate: previous turn is `Correct`, a next assistant turn exists, and the next turn's leading reasoning is failure-oriented without acknowledging the correct/performance feedback.",
        "",
        "| run | correct turns | correct with next | terminal correct | stale after correct | aligned correct/perf | stale rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['run']}` | {row['correct_total']} | {row['correct_with_next']} | "
            f"{row['terminal_correct']} | {row['stale_failure_after_correct']} | "
            f"{row['aligned_correct_or_perf']} | {row['stale_rate_of_correct_with_next']} |"
        )
    lines.extend(["", "## Stale Rows", ""])
    stale_rows = [
        row for row in rows if row.classification == "stale_failure_after_correct"
    ]
    if not stale_rows:
        lines.append("No stale rows found.")
    else:
        for row in stale_rows:
            lines.extend(
                [
                    f"- `{row.run}` `{row.trajectory_id}` prev turn `{row.previous_turn}` -> next turn `{row.next_turn}`, speedup `{row.previous_speedup}`",
                    f"  - inferred referenced turn: `{row.referenced_turn or 'unknown'}` ({row.referenced_turn_correctness or 'unknown'}), distance `{row.referenced_turn_distance or 'unknown'}`, score `{row.referenced_match_score}`",
                    f"  - matched terms: {row.referenced_match_terms or 'none'}",
                    f"  - feedback: {row.previous_feedback_snippet}",
                    f"  - referenced feedback: {row.referenced_feedback_snippet or 'none'}",
                    f"  - reasoning: {row.next_reasoning_snippet}",
                ]
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

    rows: list[ReasoningRow] = []
    terminal_by_run: dict[str, int] = {}
    for run_dir in run_dirs:
        run_rows, terminal_correct = scan_run(
            run_dir,
            window_chars=args.window_chars,
            early_failure_chars=args.early_failure_chars,
            snippet_chars=args.snippet_chars,
        )
        rows.extend(run_rows)
        terminal_by_run[run_dir.name] = terminal_correct

    write_outputs(
        args.output_dir,
        run_dirs,
        skipped_runs,
        rows,
        terminal_by_run,
        args.include_no_csv,
    )
    total_stale = sum(1 for row in rows if row.classification == "stale_failure_after_correct")
    print(f"Scanned {len(run_dirs)} runs, {len(rows)} correct turns with next reasoning.")
    print(f"Stale after correct: {total_stale}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
