#!/usr/bin/env python3
"""Count collected fix-it error kernels by definition and error type."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_PROJECT = Path("/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm")
DEFAULT_SOURCE_RUNS_CSV = Path(__file__).resolve().with_name("fixit-v5-source-runs.csv")
DEFAULT_INPUT_CSV = DEFAULT_PROJECT / "fixit-v5-error-kernels.enriched.csv"
DEFAULT_MANIFEST = DEFAULT_PROJECT / "fixit-v5-manifest.json"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().with_name("fixit-v5-error-type-distribution.csv")
DEFAULT_FAILURE_LABELS = {
    "Runtime error",
    "Kernel Execution Timeout",
    "Numerical error",
    "Compilation error",
}
ERROR_TYPE_ORDER = [
    "Compilation error",
    "Runtime error",
    "Numerical error",
    "Kernel Execution Timeout",
    "Other error",
    "Extraction error",
]
FWD_PROMPT_TAGS = {"hopper-07", "hopper-08"}
BWD_PROMPT_TAGS = {"hopper-012", "hopper-013"}
MAX_SOURCE_TURN_EXCLUSIVE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a compact matrix for rows collected by 00_collect_error_kernels.sh. "
            "The collected CSV does not store the error label, so this script "
            "joins each row back to its source turn_correctness_arch.csv."
        )
    )
    parser.add_argument(
        "--source-runs-csv",
        type=Path,
        default=DEFAULT_SOURCE_RUNS_CSV,
        help=f"Source runs CSV used by 00_collect_error_kernels.sh. Default: {DEFAULT_SOURCE_RUNS_CSV}",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Collected enriched CSV. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Optional manifest to cross-check counts. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--skip-manifest-check",
        action="store_true",
        help="Do not compare the input row count against the balanced collection manifest.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Compact distribution table to write. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--print-turn-table",
        action="store_true",
        help="Also print a compact error_type x turn table to stdout.",
    )
    parser.add_argument(
        "--top-examples",
        type=int,
        default=0,
        help="Print this many source rows per definition/error_type bucket.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def int_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def get_error_type(
    row: dict[str, str],
    turn_row_cache: dict[Path, dict[tuple[str, str], str]],
) -> str:
    turn_csv = Path(row["turn_csv"])
    if turn_csv not in turn_row_cache:
        _, turn_rows = read_csv_rows(turn_csv)
        turn_row_cache[turn_csv] = {
            (turn_row.get("trajectory_id", ""), turn_row.get("turn", "")): (
                turn_row.get("correctness", "").strip() or "Unknown"
            )
            for turn_row in turn_rows
        }

    key = (row.get("trajectory_id", ""), row.get("turn", ""))
    try:
        return turn_row_cache[turn_csv][key]
    except KeyError as exc:
        raise SystemExit(
            "missing source turn row for "
            f"{row.get('turn_csv')} trajectory_id={key[0]} turn={key[1]}"
        ) from exc


def allowed_prompt_tags(definition: str) -> set[str]:
    if definition.startswith("mha_bwd"):
        return BWD_PROMPT_TAGS
    return FWD_PROMPT_TAGS


def plan_items_by_trajectory(exp_dir: Path) -> dict[str, dict[str, object]]:
    data = json.loads((exp_dir / "plan.json").read_text())
    return {
        f"exp_{int(item['exp_index']):03d}": item
        for item in data.get("plan", [])
    }


def is_first_source_turns(turn: str) -> bool:
    try:
        return int(turn) < MAX_SOURCE_TURN_EXCLUSIVE
    except ValueError:
        return False


def enrich_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    turn_row_cache: dict[Path, dict[tuple[str, str], str]] = {}
    enriched = []
    for row in rows:
        out = dict(row)
        out["error_type"] = get_error_type(row, turn_row_cache)
        enriched.append(out)
    return enriched


def assistant_turn_ends_with_code_fence(exp_dir: Path, trajectory_id: str, turn: str) -> bool:
    try:
        target_turn = int(turn)
    except ValueError:
        return False

    trajectory_path = exp_dir / "trajectories" / f"{trajectory_id}.json"
    data = json.loads(trajectory_path.read_text())
    assistant_turn = 0
    for message in data.get("messages", []):
        if message.get("role") != "assistant":
            continue
        if assistant_turn == target_turn:
            return (message.get("content") or "").rstrip().endswith("```")
        assistant_turn += 1
    return False


def load_source_turn_rows(source_runs_csv: Path) -> list[dict[str, str]]:
    _, source_runs = read_csv_rows(source_runs_csv)
    source_rows = []
    for source_run in source_runs:
        exp_dir = Path(source_run["exp_dir"])
        plan_items = plan_items_by_trajectory(exp_dir)
        turn_csv = exp_dir / "figures" / "turn_correctness_arch.csv"
        _, turn_rows = read_csv_rows(turn_csv)
        for turn_row in turn_rows:
            trajectory_id = turn_row.get("trajectory_id", "")
            plan_item = plan_items.get(trajectory_id, {})
            source_rows.append(
                {
                    "exp_dir": str(exp_dir),
                    "definition": source_run.get("definition", "") or "Unknown",
                    "trajectory_id": trajectory_id,
                    "turn": turn_row.get("turn", ""),
                    "error_type": turn_row.get("correctness", "").strip() or "Unknown",
                    "prompt_tag": str(plan_item.get("prompt_tag") or ""),
                }
            )
    return source_rows


def classify_source_rows(
    source_rows: Iterable[dict[str, str]],
    collected_rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    collected_keys = {
        (row.get("exp_dir", ""), row.get("trajectory_id", ""), row.get("turn", ""))
        for row in collected_rows
    }
    classified = []
    for row in source_rows:
        out = dict(row)
        key = (row.get("exp_dir", ""), row.get("trajectory_id", ""), row.get("turn", ""))
        if key in collected_keys:
            out["collection_status"] = "collected"
            out["filter_reason"] = ""
        elif row.get("prompt_tag") not in allowed_prompt_tags(row.get("definition", "")):
            out["collection_status"] = "filtered"
            out["filter_reason"] = "prompt_tag"
        elif not is_first_source_turns(row.get("turn", "")):
            out["collection_status"] = "filtered"
            out["filter_reason"] = "source_turn"
        elif row.get("error_type") not in DEFAULT_FAILURE_LABELS:
            out["collection_status"] = "filtered"
            out["filter_reason"] = "excluded_label"
        else:
            exp_dir = Path(row["exp_dir"])
            try:
                has_fence = assistant_turn_ends_with_code_fence(
                    exp_dir,
                    row.get("trajectory_id", ""),
                    row.get("turn", ""),
                )
            except (FileNotFoundError, json.JSONDecodeError):
                has_fence = False
            out["collection_status"] = "filtered"
            out["filter_reason"] = "non_fenced_assistant_turn" if not has_fence else "missing_collected_row"
        classified.append(out)
    return classified


def count_by(rows: Iterable[dict[str, str]], *fields: str) -> list[dict[str, str | int]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        counter[tuple(row.get(field, "") or "Unknown" for field in fields)] += 1

    def sort_key(item: tuple[tuple[str, ...], int]) -> tuple:
        values, count = item
        normalized = []
        for field, value in zip(fields, values, strict=True):
            normalized.append(int_sort_key(value) if field == "turn" else value)
        return (*normalized, -count)

    output = []
    for values, count in sorted(counter.items(), key=sort_key):
        output.append({field: value for field, value in zip(fields, values, strict=True)} | {"count": count})
    return output


def ordered_error_types(rows: Iterable[dict[str, str]]) -> list[str]:
    seen = {row.get("error_type", "") or "Unknown" for row in rows}
    ordered = [error_type for error_type in ERROR_TYPE_ORDER if error_type in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def build_definition_error_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, str | int]], list[str]]:
    error_types = ordered_error_types(rows)
    definitions = sorted({row.get("definition", "") or "Unknown" for row in rows})
    counter: Counter[tuple[str, str]] = Counter(
        ((row.get("definition", "") or "Unknown"), row.get("error_type", "") or "Unknown")
        for row in rows
    )

    output = []
    total_row: dict[str, str | int] = {"definition": "Total"}
    for definition in definitions:
        row: dict[str, str | int] = {"definition": definition}
        total = 0
        for error_type in error_types:
            count = counter[(definition, error_type)]
            row[error_type] = count
            total += count
        row["total"] = total
        output.append(row)

    total_count = 0
    for error_type in error_types:
        count = sum(int(row[error_type]) for row in output)
        total_row[error_type] = count
        total_count += count
    total_row["total"] = total_count
    output.append(total_row)
    return output, ["definition", "total", *error_types]


def add_source_audit_columns(
    table: list[dict[str, str | int]],
    columns: list[str],
    classified_source_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str | int]], list[str]]:
    definitions = [str(row["definition"]) for row in table if row["definition"] != "Total"]
    turns = sorted({row.get("turn", "") or "Unknown" for row in classified_source_rows}, key=int_sort_key)
    filter_reasons = [
        "prompt_tag",
        "source_turn",
        "non_fenced_assistant_turn",
        "excluded_label",
        "missing_collected_row",
    ]

    source_total = Counter(row["definition"] for row in classified_source_rows)
    collected_total = Counter(
        row["definition"] for row in classified_source_rows if row["collection_status"] == "collected"
    )
    filtered_total = Counter(
        row["definition"] for row in classified_source_rows if row["collection_status"] == "filtered"
    )
    filter_counter = Counter(
        (row["definition"], row["filter_reason"])
        for row in classified_source_rows
        if row["collection_status"] == "filtered"
    )
    collected_turn_counter = Counter(
        (row["definition"], row["turn"])
        for row in classified_source_rows
        if row["collection_status"] == "collected"
    )
    filtered_turn_counter = Counter(
        (row["definition"], row["turn"])
        for row in classified_source_rows
        if row["collection_status"] == "filtered"
    )

    audit_columns = [
        "source_total",
        "collected_total",
        "filtered_total",
        *(f"filter_{reason}" for reason in filter_reasons),
        *(f"collected_turn_{turn}" for turn in turns),
        *(f"filtered_turn_{turn}" for turn in turns),
    ]

    total_row = next(row for row in table if row["definition"] == "Total")
    for row in table:
        definition = str(row["definition"])
        keys = definitions if definition == "Total" else [definition]
        row["source_total"] = sum(source_total[key] for key in keys)
        row["collected_total"] = sum(collected_total[key] for key in keys)
        row["filtered_total"] = sum(filtered_total[key] for key in keys)
        for reason in filter_reasons:
            row[f"filter_{reason}"] = sum(filter_counter[(key, reason)] for key in keys)
        for turn in turns:
            row[f"collected_turn_{turn}"] = sum(collected_turn_counter[(key, turn)] for key in keys)
            row[f"filtered_turn_{turn}"] = sum(filtered_turn_counter[(key, turn)] for key in keys)

    reordered_columns = ["definition", *audit_columns, *(column for column in columns if column != "definition")]
    # Keep the original error-total column as a compatibility alias for collected_total.
    total_row["total"] = total_row["collected_total"]
    return table, reordered_columns


def build_error_turn_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, str | int]], list[str]]:
    error_types = ordered_error_types(rows)
    turns = sorted({row.get("turn", "") or "Unknown" for row in rows}, key=int_sort_key)
    counter: Counter[tuple[str, str]] = Counter(
        ((row.get("error_type", "") or "Unknown"), row.get("turn", "") or "Unknown")
        for row in rows
    )

    output = []
    for error_type in error_types:
        row: dict[str, str | int] = {"error_type": error_type}
        total = 0
        for turn in turns:
            count = counter[(error_type, turn)]
            row[f"turn_{turn}"] = count
            total += count
        row["total"] = total
        output.append(row)
    return output, ["error_type", "total", *(f"turn_{turn}" for turn in turns)]


def print_table(title: str, rows: list[dict[str, str | int]], columns: list[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no rows)")
        return
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def write_csv(path: Path, rows: list[dict[str, str | int]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def maybe_print_examples(rows: list[dict[str, str]], limit: int) -> None:
    if limit <= 0:
        return
    examples: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("definition", "Unknown"), row.get("error_type", "Unknown"))
        examples.setdefault(key, [])
        if len(examples[key]) < limit:
            examples[key].append(row)

    print("\nExamples by definition/error_type")
    for key in sorted(examples):
        definition, error_type = key
        print(f"{definition} / {error_type}")
        for row in examples[key]:
            print(
                "  "
                f"{Path(row.get('exp_dir', '')).name}/"
                f"{row.get('trajectory_id')} turn {row.get('turn')}: "
                f"{row.get('error_kernel_path')}"
            )


def main() -> None:
    args = parse_args()
    _, rows = read_csv_rows(args.input_csv)
    enriched_rows = enrich_rows(rows)
    source_rows = load_source_turn_rows(args.source_runs_csv)
    classified_source_rows = classify_source_rows(source_rows, enriched_rows)

    definition_error_table, definition_error_columns = build_definition_error_table(enriched_rows)
    definition_error_table, definition_error_columns = add_source_audit_columns(
        definition_error_table,
        definition_error_columns,
        classified_source_rows,
    )
    error_turn_table, error_turn_columns = build_error_turn_table(enriched_rows)

    filtered_count = sum(1 for row in classified_source_rows if row["collection_status"] == "filtered")
    print(f"input_csv: {args.input_csv}")
    print(f"source_runs_csv: {args.source_runs_csv}")
    print(f"total_source_turns: {len(classified_source_rows)}")
    print(f"total_collected_turns: {len(enriched_rows)}")
    print(f"total_filtered_turns: {filtered_count}")

    if not args.skip_manifest_check and args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text())
        manifest_count = manifest.get("num_error_kernels")
        print(f"manifest_num_error_kernels: {manifest_count}")
        if manifest_count != len(enriched_rows):
            print(f"WARNING: manifest count differs from input rows: {manifest_count} != {len(enriched_rows)}")

    print_table("Error-type distribution by definition", definition_error_table, definition_error_columns)
    if args.print_turn_table:
        print_table("Error-type distribution by turn", error_turn_table, error_turn_columns)
    maybe_print_examples(enriched_rows, args.top_examples)

    write_csv(args.output_csv, definition_error_table, definition_error_columns)
    print(f"\nwrote distribution table to {args.output_csv}")


if __name__ == "__main__":
    main()
