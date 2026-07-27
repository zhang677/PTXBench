#!/usr/bin/env bash
set -euo pipefail

python - "$@" <<'PY'
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path("/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm")
CONFIG_JSON = PROJECT / "fixit-v5-gemini-source-prompt-config.json"
UNFILTERED_CSV = PROJECT / "fixit-v5-error-kernels.unfiltered.csv"
PAIRS_CSV = PROJECT / "fixit-v5-gemini-kernel-pairs.csv"
EXTENSION_CSV = PROJECT / "fixit-v5-mha-bwd-d128-causal-hopper-012-013-error-kernels.csv"
EXTENSION_CONFIG = PROJECT / "fixit-v5-mha-bwd-d128-causal-hopper-012-013-prompt-config.json"
REASONING_MD = PROJECT / "fixit-v5-mha-bwd-d128-causal-hopper-012-013-reasoning.md"

TARGET_DEFINITION = "mha_bwd_d128_causal"
TARGET_PROMPT_TAGS = ["hopper-012", "hopper-013"]
FIX_NUM_TURNS = 5
MIN_SOURCE_TURN = 4
MAX_SOURCE_TURN = 8
TARGET_SPEEDUP = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append fixit-v5 mha_bwd_d128_causal / hopper-012 and hopper-013 rows "
            "without increasing Gemini fix turns."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Write audit artifacts but do not modify the source config.")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_source_plan_item(cache: dict[Path, dict[str, dict]], exp_dir_text: str, trajectory_id: str) -> dict:
    exp_dir = Path(exp_dir_text)
    if exp_dir not in cache:
        data = json.loads((exp_dir / "plan.json").read_text())
        cache[exp_dir] = {
            f"exp_{int(item['exp_index']):03d}": item
            for item in data.get("plan", [])
        }
    return cache[exp_dir][trajectory_id]


def speedup_from_pair_rows() -> list[dict[str, object]]:
    if not PAIRS_CSV.is_file():
        return []
    _, pair_rows = read_csv(PAIRS_CSV)
    record_cache: dict[tuple[str, str], dict[int, tuple[int, float | None]]] = {}
    out: list[dict[str, object]] = []
    for row in pair_rows:
        key = (row["exp_dir"], row["trajectory_id"])
        if key not in record_cache:
            record_path = Path(row["exp_dir"]) / "success" / row["trajectory_id"] / "record.json"
            records = json.loads(record_path.read_text())
            version_map: dict[int, tuple[int, float | None]] = {}
            for entry in records if isinstance(records, list) else []:
                speeds = []
                for trace in entry.get("traces", []):
                    speedup = (
                        trace.get("evaluation", {})
                        .get("performance", {})
                        .get("speedup_factor")
                    )
                    if speedup is not None:
                        speeds.append(float(speedup))
                version_map[int(entry["version"])] = (
                    int(entry["turn"]),
                    min(speeds) if speeds else None,
                )
            record_cache[key] = version_map
        turn, speedup = record_cache[key][int(row["correct_kernel_version"])]
        if speedup is None:
            continue
        out.append(
            {
                "definition": row["definition"],
                "prompt_tag": row["prompt_tag"],
                "correct_turn": turn,
                "speedup": speedup,
            }
        )
    return out


def stats_line(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={statistics.mean(values):.4f} "
        f"median={statistics.median(values):.4f} min={min(values):.4f} max={max(values):.4f}"
    )


def item_key(item: dict) -> tuple[str, str, str, str, str]:
    return (
        str(item.get("prompt_tag", "")),
        str(item.get("definition", "")),
        str(Path(str(item.get("error_kernel_path", ""))).expanduser()),
        str(Path(str(item.get("error_log_path", ""))).expanduser()),
        str(Path(str(item.get("test_path", ""))).expanduser()),
    )


def row_sort_key(row: dict[str, str]) -> tuple[int, int]:
    prompt_order = TARGET_PROMPT_TAGS.index(row["prompt_tag"])
    try:
        trajectory_num = int(row["trajectory_id"].split("_", 1)[1])
    except (IndexError, ValueError):
        trajectory_num = 10**9
    return prompt_order, trajectory_num, int(row["turn"])


def config_item_from_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "num_trajectories": 1,
        "num_turns": FIX_NUM_TURNS,
        "target_speedup": TARGET_SPEEDUP,
        "prompt_tag": row["prompt_tag"],
        "error_kernel_path": str(Path(row["error_kernel_path"]).resolve()),
        "error_log_path": str(Path(row["error_log_path"]).resolve()),
        "definition": row["definition"],
        "test_path": str(Path(row["test_path"]).resolve()),
    }


def is_managed_item(item: dict) -> bool:
    return (
        str(item.get("definition", "")) == TARGET_DEFINITION
        and str(item.get("prompt_tag", "")) in TARGET_PROMPT_TAGS
    )


def main() -> None:
    args = parse_args()
    if not UNFILTERED_CSV.is_file():
        raise SystemExit(f"missing input CSV: {UNFILTERED_CSV}")
    if not CONFIG_JSON.is_file():
        raise SystemExit(f"missing config JSON: {CONFIG_JSON}")

    fields, rows = read_csv(UNFILTERED_CSV)
    current_config = json.loads(CONFIG_JSON.read_text())
    if not isinstance(current_config, list):
        raise SystemExit(f"{CONFIG_JSON} has no list-valued config")

    source_plan_cache: dict[Path, dict[str, dict]] = {}
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("definition", "") != TARGET_DEFINITION:
            continue
        try:
            source_turn = int(row.get("turn", ""))
        except ValueError:
            continue
        if not (MIN_SOURCE_TURN < source_turn < MAX_SOURCE_TURN):
            continue

        source_item = load_source_plan_item(source_plan_cache, row["exp_dir"], row["trajectory_id"])
        prompt_tag = str(source_item.get("prompt_tag") or "")
        if prompt_tag not in TARGET_PROMPT_TAGS:
            continue

        out = dict(row)
        out["prompt_tag"] = prompt_tag
        selected.append(out)

    selected.sort(key=row_sort_key)
    output_fields = list(fields)
    if "prompt_tag" not in output_fields:
        output_fields.append("prompt_tag")
    write_csv(EXTENSION_CSV, selected, output_fields)

    selected_items = []
    selected_keys = set()
    for row in selected:
        item = config_item_from_row(row)
        key = item_key(item)
        if key in selected_keys:
            continue
        selected_items.append(item)
        selected_keys.add(key)
    EXTENSION_CONFIG.write_text(json.dumps(selected_items, indent=4) + "\n")

    base_config = list(current_config)
    base_keys = {item_key(item) for item in base_config if isinstance(item, dict)}
    output_config = list(base_config)
    for item in selected_items:
        if item_key(item) not in base_keys:
            output_config.append(item)
            base_keys.add(item_key(item))

    speedups = speedup_from_pair_rows()
    target_speedups = [
        float(item["speedup"])
        for item in speedups
        if item["definition"] == TARGET_DEFINITION and item["prompt_tag"] in TARGET_PROMPT_TAGS
    ]
    target_turns = Counter(
        int(item["correct_turn"])
        for item in speedups
        if item["definition"] == TARGET_DEFINITION and item["prompt_tag"] in TARGET_PROMPT_TAGS
    )
    target_pair_counts = Counter(
        str(item["prompt_tag"])
        for item in speedups
        if item["definition"] == TARGET_DEFINITION and item["prompt_tag"] in TARGET_PROMPT_TAGS
    )
    selected_counts = Counter(row["prompt_tag"] for row in selected)
    existing_managed = [
        item
        for item in current_config
        if isinstance(item, dict) and is_managed_item(item)
    ]
    existing_turn_counts = Counter(
        (str(item.get("prompt_tag", "")), int(item.get("num_turns", 0)))
        for item in existing_managed
    )

    reasoning_lines = [
        "# fixit-v5 mha_bwd_d128_causal hopper-012/013 extension",
        "",
        f"Target: `{TARGET_DEFINITION}` / `{', '.join(TARGET_PROMPT_TAGS)}`.",
        (
            f"This script selects source error kernels with "
            f"`{MIN_SOURCE_TURN} < turn < {MAX_SOURCE_TURN}` and appends missing rows "
            f"with `num_turns={FIX_NUM_TURNS}`."
        ),
        "Gemini fix turns are not increased; existing rows are not reordered or deleted.",
        "",
        "## Current target success pairs",
        f"- pair_counts_by_prompt: {dict(sorted(target_pair_counts.items()))}",
        f"- speedups: {stats_line(target_speedups)}",
        f"- >0.05={sum(speedup > 0.05 for speedup in target_speedups)}",
        f"- >0.10={sum(speedup > 0.10 for speedup in target_speedups)}",
        f"- >0.15={sum(speedup > 0.15 for speedup in target_speedups)}",
        f"- correct_turn_counts: {dict(sorted(target_turns.items()))}",
        "",
        "## Config rewrite",
        f"- existing_managed_items={len(existing_managed)}",
        f"- existing_num_turn_counts={dict(sorted(existing_turn_counts.items()))}",
        f"- selected_rows={len(selected)}",
        f"- selected_rows_by_prompt={dict(sorted(selected_counts.items()))}",
        f"- selected_unique_items={len(selected_items)}",
        f"- appended_unique_items={len(output_config) - len(current_config)}",
        f"- source_turn_filter={MIN_SOURCE_TURN} < turn < {MAX_SOURCE_TURN}",
        f"- fix_num_turns={FIX_NUM_TURNS}",
        "",
        f"- Source CSV: `{UNFILTERED_CSV}`",
        f"- Extension CSV: `{EXTENSION_CSV}`",
        f"- Extension config: `{EXTENSION_CONFIG}`",
        f"- Config target: `{CONFIG_JSON}`",
        f"- Dry run: `{args.dry_run}`",
    ]
    REASONING_MD.write_text("\n".join(reasoning_lines) + "\n")

    backup = None
    if not args.dry_run and output_config != current_config:
        backup = CONFIG_JSON.with_name(f"{CONFIG_JSON.name}.{int(time.time())}.bak")
        shutil.copy2(CONFIG_JSON, backup)
        CONFIG_JSON.write_text(json.dumps(output_config, indent=4) + "\n")

    print(f"target_definition={TARGET_DEFINITION}")
    print(f"target_prompt_tags={','.join(TARGET_PROMPT_TAGS)}")
    print(f"source_turn_filter={MIN_SOURCE_TURN}<turn<{MAX_SOURCE_TURN}")
    print(f"fix_num_turns={FIX_NUM_TURNS}")
    print(f"selected_rows={len(selected)}")
    for prompt_tag in TARGET_PROMPT_TAGS:
        print(f"selected_{prompt_tag}={selected_counts[prompt_tag]}")
    print(f"old_config_items={len(current_config)}")
    print(f"managed_items_seen={len(existing_managed)}")
    print(f"extension_config_items={len(selected_items)}")
    print(f"appended_config_items={len(output_config) - len(current_config)}")
    print(f"output_config_items={len(output_config)}")
    print(f"config_changed={output_config != current_config}")
    print(f"extension_csv={EXTENSION_CSV}")
    print(f"extension_config={EXTENSION_CONFIG}")
    print(f"output_config={CONFIG_JSON}")
    print(f"reasoning_md={REASONING_MD}")
    if backup is not None:
        print(f"backup_config={backup}")


if __name__ == "__main__":
    main()
PY
