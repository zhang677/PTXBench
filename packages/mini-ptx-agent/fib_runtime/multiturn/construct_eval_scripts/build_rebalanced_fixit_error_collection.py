#!/usr/bin/env python3
"""Build a balanced fix-it error-kernel collection and prompt config.

This consumes the raw CSV from fix_kernels/select_failed_kernels.py and writes:

- an enriched CSV with prompt_tag and error_type
- a balanced raw CSV with the same schema as the raw selector output
- a fix-it prompt config JSON
- a manifest JSON

When --per-definition-cap is set, each definition is capped to exactly that
many rows. Downsampling is deterministic and stratified by error_type and turn
to preserve the source mix as much as possible.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FAILURE_LABELS = {
    "Runtime error",
    "Kernel Execution Timeout",
    "Numerical error",
    "Compilation error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--balanced-csv", type=Path, required=True)
    parser.add_argument("--enriched-csv", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--source-markdown", required=True)
    parser.add_argument("--source-runs-csv", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-definition-cap", type=int, default=0)
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--target-speedup", type=float, default=0.15)
    parser.add_argument("--min-speedup-for-later-collection", type=float, default=0.0)
    parser.add_argument(
        "--allow-underfilled-definitions",
        action="store_true",
        help="Keep definitions with fewer rows than --per-definition-cap instead of failing.",
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


def stable_row_key(row: dict[str, str]) -> tuple:
    return (
        row.get("definition", ""),
        int_sort_key(row.get("turn", "")),
        row.get("trajectory_id", ""),
        row.get("exp_dir", ""),
        row.get("error_kernel_path", ""),
    )


def plan_item(
    plan_cache: dict[Path, dict[str, dict[str, object]]],
    exp_dir_text: str,
    trajectory_id: str,
) -> dict[str, object]:
    exp_dir = Path(exp_dir_text)
    if exp_dir not in plan_cache:
        data = json.loads((exp_dir / "plan.json").read_text())
        plan_cache[exp_dir] = {
            f"exp_{int(item['exp_index']):03d}": item
            for item in data.get("plan", [])
        }
    item = plan_cache[exp_dir].get(trajectory_id)
    if item is None:
        raise SystemExit(f"missing plan item for {exp_dir}/{trajectory_id}")
    return item


def error_type_for_row(
    turn_row_cache: dict[Path, dict[tuple[str, str], str]],
    row: dict[str, str],
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
            f"{turn_csv} trajectory_id={key[0]} turn={key[1]}"
        ) from exc


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    plan_cache: dict[Path, dict[str, dict[str, object]]] = {}
    turn_row_cache: dict[Path, dict[tuple[str, str], str]] = {}
    enriched: list[dict[str, str]] = []
    for row in rows:
        item = plan_item(plan_cache, row["exp_dir"], row["trajectory_id"])
        prompt_tag = str(item.get("prompt_tag") or "")
        if not prompt_tag:
            raise SystemExit(f"missing prompt_tag for {row['exp_dir']}/{row['trajectory_id']}")
        error_type = error_type_for_row(turn_row_cache, row)
        if error_type not in DEFAULT_FAILURE_LABELS:
            raise SystemExit(
                f"unexpected selected non-failure label {error_type!r} for "
                f"{row['exp_dir']}/{row['trajectory_id']} turn {row['turn']}"
            )
        out = dict(row)
        out["prompt_tag"] = prompt_tag
        out["error_type"] = error_type
        enriched.append(out)
    return enriched


def allocate_stratified_counts(
    bucket_sizes: dict[tuple[str, str], int],
    *,
    cap: int,
    total: int,
) -> dict[tuple[str, str], int]:
    allocations: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, int, tuple[str, str]]] = []
    assigned = 0
    for key, size in sorted(bucket_sizes.items()):
        exact = size * cap / total
        count = int(exact)
        allocations[key] = count
        assigned += count
        remainders.append((exact - count, size, key))

    remaining = cap - assigned
    for _, _, key in sorted(remainders, key=lambda item: (-item[0], -item[1], item[2])):
        if remaining <= 0:
            break
        if allocations[key] < bucket_sizes[key]:
            allocations[key] += 1
            remaining -= 1

    if remaining != 0:
        raise SystemExit(f"internal allocation error: remaining={remaining}")
    return allocations


def rebalance_rows(
    rows: list[dict[str, str]],
    *,
    per_definition_cap: int,
    allow_underfilled_definitions: bool,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    if per_definition_cap <= 0:
        counts = Counter(row["definition"] for row in rows)
        return sorted(rows, key=stable_row_key), {
            definition: {"available": count, "kept": count}
            for definition, count in sorted(counts.items())
        }

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["definition"]].append(row)

    kept: list[dict[str, str]] = []
    balance: dict[str, dict[str, int]] = {}
    for definition in sorted(grouped):
        definition_rows = sorted(grouped[definition], key=stable_row_key)
        available = len(definition_rows)
        if available < per_definition_cap and not allow_underfilled_definitions:
            raise SystemExit(
                f"{definition} has only {available} rows, need {per_definition_cap}; "
                "lower --per-definition-cap or pass --allow-underfilled-definitions"
            )
        cap = min(per_definition_cap, available)
        by_stratum: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in definition_rows:
            by_stratum[(row.get("error_type", ""), row.get("turn", ""))].append(row)

        allocations = allocate_stratified_counts(
            {key: len(value) for key, value in by_stratum.items()},
            cap=cap,
            total=available,
        )
        selected: list[dict[str, str]] = []
        for key in sorted(by_stratum):
            selected.extend(by_stratum[key][: allocations[key]])
        kept.extend(sorted(selected, key=stable_row_key))
        balance[definition] = {"available": available, "kept": len(selected), "dropped": available - len(selected)}

    return kept, balance


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_config(rows: list[dict[str, str]], *, num_turns: int, target_speedup: float) -> list[dict[str, object]]:
    config: list[dict[str, object]] = []
    for row in rows:
        config.append(
            {
                "num_trajectories": 1,
                "num_turns": num_turns,
                "target_speedup": target_speedup,
                "prompt_tag": row["prompt_tag"],
                "error_kernel_path": str(Path(row["error_kernel_path"]).resolve()),
                "error_log_path": str(Path(row["error_log_path"]).resolve()),
                "definition": row["definition"],
                "test_path": str(Path(row["test_path"]).resolve()),
            }
        )
    return config


def main() -> None:
    args = parse_args()
    input_fields, raw_rows = read_csv_rows(args.input_csv)
    enriched_all = enrich_rows(raw_rows)
    enriched, balance = rebalance_rows(
        enriched_all,
        per_definition_cap=args.per_definition_cap,
        allow_underfilled_definitions=args.allow_underfilled_definitions,
    )
    config = build_config(
        enriched,
        num_turns=args.num_turns,
        target_speedup=args.target_speedup,
    )

    output_fields = list(input_fields)
    for field in ("prompt_tag", "error_type"):
        if field not in output_fields:
            output_fields.append(field)
    write_csv(args.balanced_csv, enriched, input_fields)
    write_csv(args.enriched_csv, enriched, output_fields)
    args.config_json.parent.mkdir(parents=True, exist_ok=True)
    args.config_json.write_text(json.dumps(config, indent=4) + "\n")

    manifest = {
        "source_markdown": args.source_markdown,
        "source_runs_csv": str(args.source_runs_csv),
        "prebalanced_error_kernels_csv": str(args.input_csv),
        "balanced_error_kernels_csv": str(args.balanced_csv),
        "enriched_error_kernels_csv": str(args.enriched_csv),
        "gemini_config": str(args.config_json),
        "output_root": args.output_root,
        "num_raw_error_kernels": len(raw_rows),
        "num_error_kernels": len(enriched),
        "num_config_entries": len(config),
        "num_trajectories_per_entry": 1,
        "num_turns_per_entry": args.num_turns,
        "target_speedup": args.target_speedup,
        "min_speedup_for_later_collection": args.min_speedup_for_later_collection,
        "per_definition_cap": args.per_definition_cap or None,
        "balance_by_definition": balance,
        "balance_strategy": "proportional_by_definition_error_type_turn",
        "note": (
            "Rows are capped per definition before Gemini fix-it collection to keep "
            "problem counts balanced across definitions."
        ),
        "by_definition": dict(sorted(Counter(r["definition"] for r in enriched).items())),
        "by_turn": dict(sorted(Counter(r["turn"] for r in enriched).items(), key=lambda kv: int_sort_key(kv[0]))),
        "by_error_type": dict(sorted(Counter(r["error_type"] for r in enriched).items())),
        "by_prompt_tag": dict(sorted(Counter(r["prompt_tag"] for r in enriched).items())),
        "by_source_root": dict(sorted(Counter(Path(r["exp_dir"]).name for r in enriched).items())),
    }
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"raw_rows={len(raw_rows)}")
    print(f"balanced_rows={len(enriched)}")
    print(f"wrote_balanced_csv={args.balanced_csv}")
    print(f"wrote_enriched_csv={args.enriched_csv}")
    print(f"wrote_config_json={args.config_json}")
    print(f"wrote_manifest_json={args.manifest_json}")


if __name__ == "__main__":
    main()
