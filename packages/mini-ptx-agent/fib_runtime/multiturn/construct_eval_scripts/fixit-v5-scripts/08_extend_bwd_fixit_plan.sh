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
EXTENSION_CSV = PROJECT / "fixit-v5-bwd-extra-error-kernels.csv"
EXTENSION_CONFIG = PROJECT / "fixit-v5-bwd-extra-prompt-config.json"
REASONING_MD = PROJECT / "fixit-v5-bwd-extra-reasoning.md"

NUM_TURNS = 5
TARGET_SPEEDUP = 0.15
MANAGED_PROMPT_TAGS = {"hopper-010", "hopper-011", "hopper-no-hint"}
TARGETS = {
    "mha_bwd_d128_causal": ["hopper-010", "hopper-011"],
    "mha_bwd_d128": ["hopper-011"],
    "mha_bwd_d64": ["hopper-011"],
    "mha_bwd_d64_causal": ["hopper-011"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write fixit-v5 source config with targeted extra bwd rows.")
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
                "arch_tag": row["arch_tag"],
                "correct_turn": turn,
                "speedup": speedup,
            }
        )
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def stats_line(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={statistics.mean(values):.4f} "
        f"median={statistics.median(values):.4f} p25={percentile(values, 25):.4f} "
        f"p75={percentile(values, 75):.4f} min={min(values):.4f} max={max(values):.4f}"
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
    trajectory = row["trajectory_id"]
    try:
        trajectory_num = int(trajectory.split("_", 1)[1])
    except (IndexError, ValueError):
        trajectory_num = 10**9
    return definition_order, prompt_order, trajectory_num, int(row["turn"])


def apply_target_caps(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_definition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_definition[row["definition"]].append(row)

    selected: list[dict[str, str]] = []
    for definition in TARGETS:
        definition_rows = sorted(by_definition.get(definition, []), key=row_sort_key)
        selected.extend(definition_rows)
    return selected


def config_item_from_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "num_trajectories": 1,
        "num_turns": NUM_TURNS,
        "target_speedup": TARGET_SPEEDUP,
        "prompt_tag": row["prompt_tag"],
        "error_kernel_path": str(Path(row["error_kernel_path"]).resolve()),
        "error_log_path": str(Path(row["error_log_path"]).resolve()),
        "definition": row["definition"],
        "test_path": str(Path(row["test_path"]).resolve()),
    }


def is_managed_extension_item(item: dict) -> bool:
    return (
        str(item.get("definition", "")) in TARGETS
        and str(item.get("prompt_tag", "")) in MANAGED_PROMPT_TAGS
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
    raw_selected: list[dict[str, str]] = []
    for row in rows:
        definition = row.get("definition", "")
        if definition not in TARGETS:
            continue
        try:
            source_turn = int(row.get("turn", ""))
        except ValueError:
            continue
        if source_turn >= NUM_TURNS:
            continue
        source_item = load_source_plan_item(source_plan_cache, row["exp_dir"], row["trajectory_id"])
        prompt_tag = str(source_item.get("prompt_tag") or "")
        if prompt_tag not in TARGETS[definition]:
            continue
        out = dict(row)
        out["prompt_tag"] = prompt_tag
        raw_selected.append(out)

    selected = apply_target_caps(raw_selected)
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
        if not (isinstance(item, dict) and is_managed_extension_item(item))
    ]
    base_keys = {item_key(item) for item in base_config if isinstance(item, dict)}
    output_config = list(base_config)
    for item in selected_items:
        if item_key(item) not in base_keys:
            output_config.append(item)
            base_keys.add(item_key(item))

    speedups = speedup_from_pair_rows()
    all_speeds = [float(item["speedup"]) for item in speedups]
    by_definition: dict[str, list[float]] = defaultdict(list)
    for item in speedups:
        by_definition[str(item["definition"])].append(float(item["speedup"]))

    selected_counts = Counter((row["definition"], row["prompt_tag"]) for row in selected)
    threshold_lines = [
        f"- `>{threshold}`: {sum(1 for speedup in all_speeds if speedup > threshold)} / {len(all_speeds)}"
        for threshold in (0.05, 0.10, 0.15)
    ]
    definition_lines = [
        f"- `{definition}`: {stats_line(values)}"
        for definition, values in sorted(by_definition.items())
        if definition in TARGETS
    ]
    append_lines = [
        f"- `{definition}` / `{prompt_tag}`: selected={selected_counts[(definition, prompt_tag)]}"
        for definition in TARGETS
        for prompt_tag in TARGETS[definition]
    ]
    reasoning = "\n".join(
        [
            "# fixit-v5 extra bwd plan reasoning",
            "",
            "The existing successful fix collection is weaker for the bwd definitions than for most forward/LSE definitions.",
            "The goal of this extension is to add targeted source prompt tags for the bwd groups without changing the fix-it turn budget or target speedup.",
            "",
            "## Current successful-kernel speedup thresholds",
            *threshold_lines,
            "",
            "## Current bwd speedup distributions",
            *definition_lines,
            "",
            "## New source rows",
            *append_lines,
            "",
            f"- Source CSV: `{UNFILTERED_CSV}`",
            f"- Extension CSV: `{EXTENSION_CSV}`",
            f"- Extension config: `{EXTENSION_CONFIG}`",
            f"- Config target: `{CONFIG_JSON}`",
            f"- Dry run: `{args.dry_run}`",
        ]
    )
    REASONING_MD.write_text(reasoning + "\n")

    if not args.dry_run and output_config != current_config:
        backup = CONFIG_JSON.with_name(f"{CONFIG_JSON.name}.{int(time.time())}.bak")
        shutil.copy2(CONFIG_JSON, backup)
        CONFIG_JSON.write_text(json.dumps(output_config, indent=4) + "\n")
    else:
        backup = None

    print(f"selected_rows={len(selected)}")
    print(f"candidate_rows={len(raw_selected)}")
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
