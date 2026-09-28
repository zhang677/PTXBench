#!/usr/bin/env python3
"""Filter selected fix-it error kernels before balancing."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FWD_PROMPT_TAGS = {"hopper-07", "hopper-08"}
BWD_PROMPT_TAGS = {"hopper-012", "hopper-013"}
MAX_SOURCE_TURN_EXCLUSIVE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def allowed_prompt_tags(definition: str) -> set[str]:
    if definition.startswith("mha_bwd"):
        return BWD_PROMPT_TAGS
    return FWD_PROMPT_TAGS


def is_first_source_turns(turn: str) -> bool:
    try:
        return int(turn) < MAX_SOURCE_TURN_EXCLUSIVE
    except ValueError:
        return False


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_csv_rows(args.input_csv)
    plan_cache: dict[Path, dict[str, dict[str, object]]] = {}
    kept: list[dict[str, str]] = []
    dropped_by_definition: Counter[str] = Counter()
    kept_by_definition: Counter[str] = Counter()
    kept_by_prompt_tag: Counter[str] = Counter()
    dropped_by_prompt_tag: Counter[str] = Counter()
    dropped_by_turn: Counter[str] = Counter()

    for row in rows:
        item = plan_item(plan_cache, row["exp_dir"], row["trajectory_id"])
        prompt_tag = str(item.get("prompt_tag") or "")
        definition = row.get("definition", "")
        if prompt_tag not in allowed_prompt_tags(definition):
            dropped_by_definition[definition] += 1
            dropped_by_prompt_tag[prompt_tag or "missing"] += 1
        elif not is_first_source_turns(row.get("turn", "")):
            dropped_by_definition[definition] += 1
            dropped_by_turn[row.get("turn", "") or "missing"] += 1
        else:
            kept.append(row)
            kept_by_definition[definition] += 1
            kept_by_prompt_tag[prompt_tag] += 1

    write_csv(args.output_csv, kept, fieldnames)
    print(f"input_rows={len(rows)}")
    print(f"kept_rows={len(kept)}")
    print(f"dropped_rows={len(rows) - len(kept)}")
    print(f"max_source_turn_exclusive={MAX_SOURCE_TURN_EXCLUSIVE}")
    print(f"kept_by_definition={dict(sorted(kept_by_definition.items()))}")
    print(f"dropped_by_definition={dict(sorted(dropped_by_definition.items()))}")
    print(f"kept_by_prompt_tag={dict(sorted(kept_by_prompt_tag.items()))}")
    print(f"dropped_by_prompt_tag={dict(sorted(dropped_by_prompt_tag.items()))}")
    print(f"dropped_by_turn={dict(sorted(dropped_by_turn.items()))}")
    print(f"wrote_filtered_csv={args.output_csv}")


if __name__ == "__main__":
    main()
