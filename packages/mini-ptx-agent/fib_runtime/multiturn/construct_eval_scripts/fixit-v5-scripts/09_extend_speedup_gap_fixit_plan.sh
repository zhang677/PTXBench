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
EXTENSION_CSV = PROJECT / "fixit-v5-speedup-gap-error-kernels.csv"
EXTENSION_CONFIG = PROJECT / "fixit-v5-speedup-gap-prompt-config.json"
REASONING_MD = PROJECT / "fixit-v5-speedup-gap-reasoning.md"

DEFAULT_NUM_TURNS = 5
TARGET_SPEEDUP = 0.15
TARGETS = {
    "mha_bwd_d128": ["hopper-010"],
    "mha_bwd_d128_causal": ["hopper-no-hint"],
}
TARGET_NUM_TURNS = {
    ("mha_bwd_d128", "hopper-010"): 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append targeted fixit-v5 rows for bwd definitions with weak speedup-bin coverage."
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
    record_cache: dict[tuple[str, str], dict[int, float | None]] = {}
    out: list[dict[str, object]] = []
    for row in pair_rows:
        key = (row["exp_dir"], row["trajectory_id"])
        if key not in record_cache:
            record_path = Path(row["exp_dir"]) / "success" / row["trajectory_id"] / "record.json"
            records = json.loads(record_path.read_text())
            version_map: dict[int, float | None] = {}
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
                version_map[int(entry["version"])] = min(speeds) if speeds else None
            record_cache[key] = version_map
        speedup = record_cache[key][int(row["correct_kernel_version"])]
        if speedup is None:
            continue
        out.append({"definition": row["definition"], "speedup": speedup})
    return out


def speedup_bin(speedup: float) -> str:
    if speedup < 0.05:
        return "<0.05"
    if speedup < 0.10:
        return "0.05-<0.10"
    if speedup < TARGET_SPEEDUP:
        return "0.10-<0.15"
    return ">=0.15"


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


def row_sort_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    definition_order = list(TARGETS).index(row["definition"])
    prompt_order = TARGETS[row["definition"]].index(row["prompt_tag"])
    try:
        trajectory_num = int(row["trajectory_id"].split("_", 1)[1])
    except (IndexError, ValueError):
        trajectory_num = 10**9
    return definition_order, prompt_order, trajectory_num, int(row["turn"])


def num_turns_for(row: dict[str, str]) -> int:
    return TARGET_NUM_TURNS.get(
        (row["definition"], row["prompt_tag"]),
        DEFAULT_NUM_TURNS,
    )


def config_item_from_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "num_trajectories": 1,
        "num_turns": num_turns_for(row),
        "target_speedup": TARGET_SPEEDUP,
        "prompt_tag": row["prompt_tag"],
        "error_kernel_path": str(Path(row["error_kernel_path"]).resolve()),
        "error_log_path": str(Path(row["error_log_path"]).resolve()),
        "definition": row["definition"],
        "test_path": str(Path(row["test_path"]).resolve()),
    }


def is_managed_item(item: dict) -> bool:
    definition = str(item.get("definition", ""))
    prompt_tag = str(item.get("prompt_tag", ""))
    return definition in TARGETS and prompt_tag in TARGETS[definition]


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
        definition = row.get("definition", "")
        if definition not in TARGETS:
            continue
        try:
            source_turn = int(row.get("turn", ""))
        except ValueError:
            continue
        source_item = load_source_plan_item(source_plan_cache, row["exp_dir"], row["trajectory_id"])
        prompt_tag = str(source_item.get("prompt_tag") or "")
        if prompt_tag not in TARGETS[definition]:
            continue
        out = dict(row)
        out["prompt_tag"] = prompt_tag
        if source_turn >= num_turns_for(out):
            continue
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

    base_config = [
        item
        for item in current_config
        if not (isinstance(item, dict) and is_managed_item(item))
    ]
    base_keys = {item_key(item) for item in base_config if isinstance(item, dict)}
    output_config = list(base_config)
    for item in selected_items:
        if item_key(item) not in base_keys:
            output_config.append(item)
            base_keys.add(item_key(item))

    speedups = speedup_from_pair_rows()
    by_definition: dict[str, list[float]] = defaultdict(list)
    bins_by_definition: dict[str, Counter[str]] = defaultdict(Counter)
    for item in speedups:
        definition = str(item["definition"])
        speedup = float(item["speedup"])
        by_definition[definition].append(speedup)
        bins_by_definition[definition][speedup_bin(speedup)] += 1

    selected_counts = Counter((row["definition"], row["prompt_tag"]) for row in selected)
    reasoning_lines = [
        "# fixit-v5 speedup-gap extension reasoning",
        "",
        "This extension follows the current success-pair speedup-bin table.",
        "It adds one source-prompt slice for each bwd definition with weak target-speedup coverage.",
        f"Default num_turns remains {DEFAULT_NUM_TURNS}; only explicit target overrides use a different value.",
        "",
        "## Target slices",
    ]
    for definition, prompt_tags in TARGETS.items():
        for prompt_tag in prompt_tags:
            reasoning_lines.append(
                f"- `{definition}` / `{prompt_tag}`: "
                f"num_turns={TARGET_NUM_TURNS.get((definition, prompt_tag), DEFAULT_NUM_TURNS)} "
                f"selected={selected_counts[(definition, prompt_tag)]}"
            )
    reasoning_lines.extend(["", "## Current speedup-bin counts"])
    bin_labels = ["<0.05", "0.05-<0.10", "0.10-<0.15", ">=0.15"]
    for definition in TARGETS:
        counts = bins_by_definition[definition]
        values = by_definition[definition]
        reasoning_lines.append(
            f"- `{definition}`: "
            + " ".join(f"{label}={counts[label]}" for label in bin_labels)
            + f" {stats_line(values)}"
        )
    reasoning_lines.extend(
        [
            "",
            f"- Source CSV: `{UNFILTERED_CSV}`",
            f"- Extension CSV: `{EXTENSION_CSV}`",
            f"- Extension config: `{EXTENSION_CONFIG}`",
            f"- Config target: `{CONFIG_JSON}`",
            f"- Dry run: `{args.dry_run}`",
        ]
    )
    REASONING_MD.write_text("\n".join(reasoning_lines) + "\n")

    backup = None
    if not args.dry_run and output_config != current_config:
        backup = CONFIG_JSON.with_name(f"{CONFIG_JSON.name}.{int(time.time())}.bak")
        shutil.copy2(CONFIG_JSON, backup)
        CONFIG_JSON.write_text(json.dumps(output_config, indent=4) + "\n")

    print(f"selected_rows={len(selected)}")
    for definition, prompt_tags in TARGETS.items():
        for prompt_tag in prompt_tags:
            print(f"selected_{definition}_{prompt_tag}={selected_counts[(definition, prompt_tag)]}")
    print(f"old_config_items={len(current_config)}")
    print(f"base_config_items={len(base_config)}")
    print(f"extension_config_items={len(selected_items)}")
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
