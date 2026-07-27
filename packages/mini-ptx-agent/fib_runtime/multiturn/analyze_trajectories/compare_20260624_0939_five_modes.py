#!/usr/bin/env python3
"""Compare shared 2026-0624-0939 d128 MHA prompt tags across six modes."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "compare_20260624_0939_five_modes"
STAGES = [
    "regular",
    "regular-patched",
    "linfo",
    "linfo-patched",
    "linfo-singleuser",
    "latest-pair-patched",
]
STAGE_LABELS = {
    "regular": "Regular",
    "regular-patched": "Regular patched",
    "linfo": "Linfo",
    "linfo-patched": "Linfo patched",
    "linfo-singleuser": "Linfo single-user",
    "latest-pair-patched": "Latest-pair patched",
}
MODE_LABEL = f"{len(STAGES)}-mode"
MODE_TITLE = {5: "Five-Mode", 6: "Six-Mode"}.get(len(STAGES), f"{len(STAGES)}-Mode")
ARCH_TAG_BY_ARCH = {
    "hopper": "H",
    "blackwell": "B",
}
REQUIRED_ARCH_TAG = ARCH_TAG_BY_ARCH["hopper"]
GROUPS = {
    "regular": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-bwd-d128-causal",
    ],
    "regular-patched": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128-causal",
    ],
    "linfo": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-mha-bwd-d128-causal",
    ],
    "linfo-patched": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-patched-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-patched-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-patched-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-patched-mha-bwd-d128-causal",
    ],
    "linfo-singleuser": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-singleuser-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-singleuser-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-singleuser-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-linfo-singleuser-mha-bwd-d128-causal",
    ],
    "latest-pair-patched": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-latest-pair-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-latest-pair-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-latest-pair-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-latest-pair-mha-bwd-d128-causal",
    ],
}
STATE_ORDER = [
    "Correct",
    "Compilation error",
    "Runtime error",
    "Kernel Execution Timeout",
    "Numerical error",
    "Extraction error",
    "Other error",
    "Sanitize Timeout",
    "Profiling Service Timeout",
]
STATE_COLORS = {
    "Correct": "#2a9d55",
    "Compilation error": "#7b6fd6",
    "Runtime error": "#d55e00",
    "Kernel Execution Timeout": "#d33f49",
    "Numerical error": "#e4a72b",
    "Other error": "#6b7280",
    "Extraction error": "#0f766e",
    "Sanitize Timeout": "#b45309",
    "Profiling Service Timeout": "#9333ea",
}


def normalize_prompt_tag(tag: str) -> str:
    return tag.removesuffix("-mha-patched")


def load_plan(run_dir: Path) -> list[dict]:
    payload = json.loads((run_dir / "plan.json").read_text())
    return payload["plan"] if isinstance(payload, dict) and "plan" in payload else payload


def load_turn_rows(run_dir: Path) -> dict[tuple[str, int], dict]:
    path = run_dir / "figures" / "turn_correctness_arch.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required source CSV: {path}")
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["trajectory_id"], int(row["turn"]))] = row
    return rows


def as_float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def geomean(values: list[float]) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return 0.0
    return math.exp(statistics.fmean(math.log(value) for value in positives))


def format_metric(value: float) -> str:
    return "0.0" if value == 0 else f"{value:.6f}"


def uses_required_arch_instruction(arch_tag: str | None) -> bool:
    tags = {tag.strip() for tag in (arch_tag or "").split(",") if tag.strip()}
    return REQUIRED_ARCH_TAG in tags


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def state_sort_key(state: str) -> tuple[int, str]:
    try:
        return (STATE_ORDER.index(state), state)
    except ValueError:
        return (len(STATE_ORDER), state)


def collect_rows() -> list[dict]:
    rows = []
    for group, roots in GROUPS.items():
        for root in roots:
            run_dir = Path(root)
            plan = sorted(load_plan(run_dir), key=lambda item: int(item["exp_index"]))
            turn_rows = load_turn_rows(run_dir)
            replica_by_pair: dict[tuple[str, str], int] = defaultdict(int)
            for item in plan:
                trajectory_id = f"exp_{int(item['exp_index']):03d}"
                definition = item.get("definition", "")
                prompt_tag_raw = item.get("prompt_tag", "")
                prompt_tag = normalize_prompt_tag(prompt_tag_raw)
                pair = (definition, prompt_tag)
                replica = replica_by_pair[pair]
                replica_by_pair[pair] += 1
                for turn in range(int(item.get("num_turns", 8))):
                    turn_row = turn_rows.get((trajectory_id, turn))
                    if not turn_row:
                        continue
                    correctness = turn_row["correctness"]
                    rows.append(
                        {
                            "group": group,
                            "run": run_dir.name,
                            "run_dir": str(run_dir),
                            "definition": definition,
                            "prompt_tag": prompt_tag,
                            "prompt_tag_raw": prompt_tag_raw,
                            "replica": replica,
                            "trajectory_id": trajectory_id,
                            "turn": turn,
                            "correctness": correctness,
                            "is_correct": int(correctness == "Correct"),
                            "speedup": as_float(turn_row.get("speedup")),
                            "arch_tag": turn_row.get("arch_tag", ""),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shared_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    pairs_by_group: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        pairs_by_group[row["group"]].add((row["definition"], row["prompt_tag"]))
    return set.intersection(*(pairs_by_group[group] for group in STAGES))


def matched_turn_rows(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str, int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["definition"], row["prompt_tag"], int(row["replica"]), int(row["turn"]))
        by_key[key][row["group"]] = row

    matched = []
    for key, group_rows in sorted(by_key.items()):
        if not all(group in group_rows for group in STAGES):
            continue
        definition, prompt_tag, replica, turn = key
        out = {
            "definition": definition,
            "prompt_tag": prompt_tag,
            "replica": replica,
            "turn": turn,
        }
        for group in STAGES:
            prefix = group.replace("-", "_")
            row = group_rows[group]
            out[f"{prefix}_correctness"] = row["correctness"]
            out[f"{prefix}_speedup"] = row["speedup"]
            out[f"{prefix}_arch_tag"] = row["arch_tag"]
            out[f"{prefix}_run"] = row["run"]
            out[f"{prefix}_trajectory_id"] = row["trajectory_id"]
        matched.append(out)
    return matched


def stage_state_counts(matched: list[dict]) -> Counter:
    counts: Counter = Counter()
    for item in matched:
        for stage in STAGES:
            prefix = stage.replace("-", "_")
            counts[(stage, item[f"{prefix}_correctness"])] += 1
    return counts


def rows_from_matched(matched: list[dict]) -> list[dict]:
    rows = []
    for item in matched:
        for group in STAGES:
            prefix = group.replace("-", "_")
            rows.append(
                {
                    "group": group,
                    "definition": item["definition"],
                    "prompt_tag": item["prompt_tag"],
                    "replica": item["replica"],
                    "turn": item["turn"],
                    "correctness": item[f"{prefix}_correctness"],
                    "is_correct": int(item[f"{prefix}_correctness"] == "Correct"),
                    "speedup": item[f"{prefix}_speedup"],
                    "arch_tag": item[f"{prefix}_arch_tag"],
                    "run": item[f"{prefix}_run"],
                    "trajectory_id": item[f"{prefix}_trajectory_id"],
                }
            )
    return rows


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)

    out = []
    def sort_key(item: tuple[tuple, list[dict]]) -> tuple:
        key_values, _items = item
        if keys and keys[0] == "group":
            return (STAGES.index(key_values[0]),) + key_values[1:]
        return key_values

    for key_values, items in sorted(buckets.items(), key=sort_key):
        trajectory_keys = {(item["run"], item["trajectory_id"]) for item in items}
        n_traj = len(trajectory_keys)
        horizon_metrics = []
        for horizon in (1, 4, 8):
            horizon_items = [item for item in items if int(item["turn"]) < horizon]
            horizon_turns = len(horizon_items)
            horizon_correct = sum(int(item["is_correct"]) for item in horizon_items)
            horizon_correct_use_instruction = sum(
                1
                for item in horizon_items
                if item["correctness"] == "Correct"
                and uses_required_arch_instruction(item.get("arch_tag"))
            )
            horizon_correct_trajectories = {
                (item["run"], item["trajectory_id"])
                for item in horizon_items
                if item["correctness"] == "Correct"
            }
            horizon_correct_instruction_trajectories = {
                (item["run"], item["trajectory_id"])
                for item in horizon_items
                if item["correctness"] == "Correct"
                and uses_required_arch_instruction(item.get("arch_tag"))
            }
            horizon_correct_speedups = [
                float(item["speedup"])
                for item in horizon_items
                if item["correctness"] == "Correct" and float(item["speedup"]) > 0
            ]
            horizon_metrics.append(
                (
                    horizon_correct,
                    horizon_correct / horizon_turns if horizon_turns else 0.0,
                    horizon_correct_use_instruction / horizon_turns
                    if horizon_turns
                    else 0.0,
                    len(horizon_correct_trajectories) / n_traj if n_traj else 0.0,
                    len(horizon_correct_instruction_trajectories) / n_traj
                    if n_traj
                    else 0.0,
                    max(
                        (float(item["speedup"]) for item in horizon_items),
                        default=0.0,
                    ),
                    geomean(horizon_correct_speedups),
                )
            )
        n_turns = len(items)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "n_trajectories": n_traj,
                "n_turns": n_turns,
                "correct_turns": " / ".join(
                    str(metric[0]) for metric in horizon_metrics
                ),
                "correctness_rate": " / ".join(
                    format_metric(metric[1]) for metric in horizon_metrics
                ),
                "correct_and_use_instruction_rate": " / ".join(
                    format_metric(metric[2]) for metric in horizon_metrics
                ),
                "trajectory_correctness_rate": " / ".join(
                    format_metric(metric[3]) for metric in horizon_metrics
                ),
                "trajectory_correct_and_use_instruction_rate": " / ".join(
                    format_metric(metric[4]) for metric in horizon_metrics
                ),
                "best_speedup": " / ".join(
                    format_metric(metric[5]) for metric in horizon_metrics
                ),
                "correct_turn_speedup_geomean": " / ".join(
                    format_metric(metric[6]) for metric in horizon_metrics
                ),
            }
        )
        out.append(row)
    return out


def state_counts(rows: list[dict], keys: list[str]) -> list[dict]:
    buckets: dict[tuple, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for row in rows:
        key = tuple(row[key_name] for key_name in keys)
        buckets[key][row["correctness"]] += 1
        totals[key] += 1

    out = []
    for key, counts in sorted(buckets.items()):
        for state, count in sorted(counts.items()):
            row = {key_name: value for key_name, value in zip(keys, key)}
            row.update({"correctness": state, "count": count, "fraction": count / totals[key]})
            out.append(row)
    return out


def final_state_counts(rows: list[dict]) -> list[dict]:
    by_traj: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_traj[(row["group"], row["run"], row["trajectory_id"])].append(row)
    counts: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for (group, _run, _traj), items in by_traj.items():
        final = max(items, key=lambda row: int(row["turn"]))
        counts[group][final["correctness"]] += 1
        totals[group] += 1
    out = []
    for group in STAGES:
        for state, count in sorted(counts[group].items()):
            out.append(
                {
                    "group": group,
                    "final_correctness": state,
                    "count": count,
                    "fraction": count / totals[group] if totals[group] else 0.0,
                }
            )
    return out


def holistic_turn_rows(
    rows: list[dict],
    shared: set[tuple[str, str]],
    matched: list[dict],
) -> list[dict]:
    matched_keys = {
        (
            item["definition"],
            item["prompt_tag"],
            int(item["replica"]),
            int(item["turn"]),
        )
        for item in matched
    }

    out = []
    for row in rows:
        key = (
            row["definition"],
            row["prompt_tag"],
            int(row["replica"]),
            int(row["turn"]),
        )
        item = dict(row)
        item["stage_index"] = STAGES.index(row["group"])
        item["is_shared_pair"] = int((row["definition"], row["prompt_tag"]) in shared)
        item["is_matched_turn"] = int(key in matched_keys)
        out.append(item)
    return sorted(
        out,
        key=lambda item: (
            item["stage_index"],
            item["definition"],
            item["prompt_tag"],
            int(item["replica"]),
            int(item["turn"]),
        ),
    )


def svg_sankey(matched_subset: list[dict], title: str) -> str:
    width = max(1120, 230 * len(STAGES))
    margin_x = 88
    top = 72
    bottom = 46
    col_w = 24
    gap = 10
    unit = min(3.0, max(1.4, 700.0 / max(len(matched_subset), 1)))
    height = int(top + bottom + max(180, max(len(matched_subset), 1) * unit + 90))

    counts = stage_state_counts(matched_subset)
    states_by_stage = {
        stage: sorted(
            [state for (stage_name, state), count in counts.items() if stage_name == stage and count],
            key=state_sort_key,
        )
        for stage in STAGES
    }
    x_pos = {
        stage: margin_x + i * ((width - margin_x * 2) / max(1, len(STAGES) - 1))
        for i, stage in enumerate(STAGES)
    }

    node_pos = {}
    for stage in STAGES:
        y = top + 36
        for state in states_by_stage[stage]:
            h = max(4.0, counts[(stage, state)] * unit)
            node_pos[(stage, state)] = (x_pos[stage], y, h)
            y += h + gap

    pair_counts = Counter()
    for item in matched_subset:
        for left, right in zip(STAGES, STAGES[1:]):
            left_prefix = left.replace("-", "_")
            right_prefix = right.replace("-", "_")
            pair_counts[
                (
                    left,
                    item[f"{left_prefix}_correctness"],
                    right,
                    item[f"{right_prefix}_correctness"],
                )
            ] += 1

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.small{font-size:12px}.title{font-size:18px;font-weight:700}.stage{font-size:13px;font-weight:700}.node-label{font-size:11px}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
        f'<text class="small" x="24" y="52">Matched {MODE_LABEL} turn rows: {len(matched_subset)}</text>',
    ]

    outgoing_used = defaultdict(float)
    incoming_used = defaultdict(float)
    for (left, left_state, right, right_state), count in sorted(pair_counts.items()):
        if (left, left_state) not in node_pos or (right, right_state) not in node_pos:
            continue
        x1, y1, _h1 = node_pos[(left, left_state)]
        x2, y2, _h2 = node_pos[(right, right_state)]
        thickness = max(1.0, count * unit)
        sy = y1 + outgoing_used[(left, left_state)] + thickness / 2
        ty = y2 + incoming_used[(right, right_state)] + thickness / 2
        outgoing_used[(left, left_state)] += thickness
        incoming_used[(right, right_state)] += thickness
        color = STATE_COLORS.get(right_state, "#6b7280")
        c1 = x1 + (x2 - x1) * 0.45
        c2 = x1 + (x2 - x1) * 0.55
        d = f"M {x1 + col_w} {sy:.1f} C {c1:.1f} {sy:.1f}, {c2:.1f} {ty:.1f}, {x2:.1f} {ty:.1f}"
        parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-opacity="0.28" '
            f'stroke-width="{thickness:.1f}"><title>{html.escape(STAGE_LABELS[left])}: '
            f'{html.escape(left_state)} -> {html.escape(STAGE_LABELS[right])}: '
            f'{html.escape(right_state)}: {count}</title></path>'
        )

    for stage in STAGES:
        x = x_pos[stage]
        parts.append(
            f'<text class="stage" x="{x - 24:.1f}" y="{top}">{html.escape(STAGE_LABELS[stage])}</text>'
        )
        for state in states_by_stage[stage]:
            _x, y, h = node_pos[(stage, state)]
            color = STATE_COLORS.get(state, "#6b7280")
            count = counts[(stage, state)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{col_w}" height="{h:.1f}" '
                f'rx="3" fill="{color}"><title>{html.escape(state)}: {count}</title></rect>'
            )
            label_x = x + col_w + 6 if stage != STAGES[-1] else x - 8
            anchor = "start" if stage != STAGES[-1] else "end"
            parts.append(
                f'<text class="node-label" x="{label_x:.1f}" y="{y + min(h, 14):.1f}" '
                f'text-anchor="{anchor}">{html.escape(state)} ({count})</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def svg_turn_transition_sankey(rows: list[dict], title: str) -> str:
    sequences: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for row in rows:
        sequences[(row["run"], row["trajectory_id"])][int(row["turn"])] = row["correctness"]

    max_turn = max((max(turns) for turns in sequences.values() if turns), default=0)
    width = max(1120, 180 * (max_turn + 1))
    margin_x = 72
    top = 76
    bottom = 42
    col_w = 18
    gap = 8
    total_sequences = max(len(sequences), 1)
    unit = min(6.0, max(1.2, 620.0 / total_sequences))
    height = int(top + bottom + max(180, total_sequences * unit + 90))

    counts = Counter()
    for turns in sequences.values():
        for turn, state in turns.items():
            counts[(turn, state)] += 1

    states_by_turn = {
        turn: sorted(
            [state for (turn_idx, state), count in counts.items() if turn_idx == turn and count],
            key=state_sort_key,
        )
        for turn in range(max_turn + 1)
    }
    x_pos = {
        turn: margin_x + turn * ((width - margin_x * 2) / max(1, max_turn))
        for turn in range(max_turn + 1)
    }

    node_pos = {}
    for turn in range(max_turn + 1):
        y = top + 32
        for state in states_by_turn[turn]:
            h = max(4.0, counts[(turn, state)] * unit)
            node_pos[(turn, state)] = (x_pos[turn], y, h)
            y += h + gap

    pair_counts = Counter()
    for turns in sequences.values():
        for turn in range(max_turn):
            left = turns.get(turn)
            right = turns.get(turn + 1)
            if left is not None and right is not None:
                pair_counts[(turn, left, turn + 1, right)] += 1

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.small{font-size:12px}.title{font-size:18px;font-weight:700}.stage{font-size:13px;font-weight:700}.node-label{font-size:11px}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="30">{html.escape(title)}</text>',
        f'<text class="small" x="24" y="52">Trajectories: {len(sequences)}; shared definition/prompt-tag subset</text>',
    ]

    outgoing_used = defaultdict(float)
    incoming_used = defaultdict(float)
    for (left_turn, left_state, right_turn, right_state), count in sorted(pair_counts.items()):
        if (left_turn, left_state) not in node_pos or (right_turn, right_state) not in node_pos:
            continue
        x1, y1, _h1 = node_pos[(left_turn, left_state)]
        x2, y2, _h2 = node_pos[(right_turn, right_state)]
        thickness = max(1.0, count * unit)
        sy = y1 + outgoing_used[(left_turn, left_state)] + thickness / 2
        ty = y2 + incoming_used[(right_turn, right_state)] + thickness / 2
        outgoing_used[(left_turn, left_state)] += thickness
        incoming_used[(right_turn, right_state)] += thickness
        color = STATE_COLORS.get(right_state, "#6b7280")
        c1 = x1 + (x2 - x1) * 0.45
        c2 = x1 + (x2 - x1) * 0.55
        d = f"M {x1 + col_w} {sy:.1f} C {c1:.1f} {sy:.1f}, {c2:.1f} {ty:.1f}, {x2:.1f} {ty:.1f}"
        parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-opacity="0.30" '
            f'stroke-width="{thickness:.1f}"><title>turn {left_turn} {html.escape(left_state)} -> '
            f'turn {right_turn} {html.escape(right_state)}: {count}</title></path>'
        )

    for turn in range(max_turn + 1):
        x = x_pos[turn]
        parts.append(f'<text class="stage" x="{x - 12:.1f}" y="{top}">turn {turn}</text>')
        for state in states_by_turn[turn]:
            _x, y, h = node_pos[(turn, state)]
            color = STATE_COLORS.get(state, "#6b7280")
            count = counts[(turn, state)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{col_w}" height="{h:.1f}" '
                f'rx="3" fill="{color}"><title>{html.escape(state)}: {count}</title></rect>'
            )
            label_x = x + col_w + 5
            anchor = "start"
            if turn == max_turn:
                label_x = x - 7
                anchor = "end"
            parts.append(
                f'<text class="node-label" x="{label_x:.1f}" y="{y + min(h, 14):.1f}" '
                f'text-anchor="{anchor}">{html.escape(state)} ({count})</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def write_alluvial_outputs(_matched: list[dict], matched_long: list[dict]) -> None:
    for stale in [
        OUT_DIR / "alluvial_overall.svg",
        OUT_DIR / "alluvial_index.md",
    ]:
        stale.unlink(missing_ok=True)
    for stale_svg in OUT_DIR.glob("turn_transition_alluvial_*.svg"):
        stale_svg.unlink()
    for stale_dir in [
        OUT_DIR / "alluvial_by_definition",
        OUT_DIR / "alluvial_by_prompt_tag",
        OUT_DIR / "turn_transition_alluvial_by_definition",
    ]:
        shutil.rmtree(stale_dir, ignore_errors=True)

    out_dir = OUT_DIR / "turn_transition_alluvial_by_definition"
    out_dir.mkdir(exist_ok=True)
    turn_index_lines = [
        f"# 2026-0624-0939 {MODE_TITLE} Turn-Transition Alluvial Diagrams",
        "",
        "Each SVG is for one model and one definition. Columns are turns 0..7; each turn aggregates all matched prompt tags and trajectories for that definition.",
        "",
    ]
    for group in STAGES:
        turn_index_lines.extend([f"## {STAGE_LABELS[group]}", ""])
        group_rows = [row for row in matched_long if row["group"] == group]
        for definition in sorted({row["definition"] for row in group_rows}):
            subset = [row for row in group_rows if row["definition"] == definition]
            filename = f"{safe_stem(group)}__{safe_stem(definition)}.svg"
            title = f"{STAGE_LABELS[group]} / {definition}: turn-to-turn correctness flow"
            (out_dir / filename).write_text(svg_turn_transition_sankey(subset, title))
            turn_index_lines.extend(
                [
                    f"### `{definition}`",
                    "",
                    f"![{STAGE_LABELS[group]} / {definition}](turn_transition_alluvial_by_definition/{filename})",
                    "",
                ]
            )

    (OUT_DIR / "turn_transition_alluvial_index.md").write_text("\n".join(turn_index_lines))


def md_table(rows: list[dict], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(
    all_rows: list[dict],
    shared_rows: list[dict],
    matched: list[dict],
    overall: list[dict],
    by_definition: list[dict],
    pairs: list[dict],
) -> None:
    shared = sorted(shared_pairs(all_rows))
    lines = [
        f"# 2026-0624-0939 {MODE_TITLE} Shared MHA Comparison",
        "",
        "Source of truth: each run's `figures/turn_correctness_arch.csv`, joined to `plan.json` for definition and prompt-tag metadata.",
        "",
        "Modes:",
        "",
    ]
    for group in STAGES:
        lines.append(f"- `{group}`")
    lines.extend(
        [
            "",
            "## Run Roots",
            "",
        ]
    )
    for group in STAGES:
        lines.append(f"### {group}")
        lines.extend(f"- `{root}`" for root in GROUPS[group])
        lines.append("")
    lines.extend(
        [
            f"All source rows loaded: {len(all_rows)}",
            f"Shared `(definition, prompt_tag)` pairs: {len(shared)}",
            f"Shared rows after filtering: {len(shared_rows)}",
            f"Matched {MODE_LABEL} turn rows: {len(matched)}",
            "",
            "Turn-transition alluvial outputs:",
            "",
            "- `turn_transition_alluvial_index.md`",
            "- `turn_transition_alluvial_by_definition/`",
            "",
            "## Matched-Prompt Overall Metrics",
            "",
            "All metric triplets are ordered `≤1 / ≤4 / ≤8` turns.",
            "",
            md_table(
                overall,
                [
                    "group",
                    "n_trajectories",
                    "n_turns",
                    "correct_turns",
                    "correctness_rate",
                    "correct_and_use_instruction_rate",
                    "trajectory_correctness_rate",
                    "trajectory_correct_and_use_instruction_rate",
                    "best_speedup",
                    "correct_turn_speedup_geomean",
                ],
            ),
            "",
            "## Matched-Definition Collective Metrics",
            "",
            "Each table pools all matched prompt tags for one problem and group. Metric triplets remain ordered `≤1 / ≤4 / ≤8` turns.",
            "",
        ]
    )
    collective_fields = [
        "group",
        "n_trajectories",
        "n_turns",
        "correct_turns",
        "correctness_rate",
        "correct_and_use_instruction_rate",
        "trajectory_correctness_rate",
        "trajectory_correct_and_use_instruction_rate",
        "best_speedup",
        "correct_turn_speedup_geomean",
    ]
    for definition in sorted({row["definition"] for row in by_definition}):
        definition_rows = sorted(
            (row for row in by_definition if row["definition"] == definition),
            key=lambda row: STAGES.index(row["group"]),
        )
        lines.extend(
            [
                f"### `{definition}`",
                "",
                md_table(definition_rows, collective_fields),
                "",
            ]
        )
    lines.extend(
        [
            "## Matched Definition / Prompt-Tag Rows",
            "",
            md_table(
                pairs,
                [
                    "definition",
                    "prompt_tag",
                    "group",
                    "n_trajectories",
                    "n_turns",
                    "correct_turns",
                    "correctness_rate",
                    "correct_and_use_instruction_rate",
                    "trajectory_correctness_rate",
                    "trajectory_correct_and_use_instruction_rate",
                    "best_speedup",
                    "correct_turn_speedup_geomean",
                ],
            ),
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "holistic_turns.csv").unlink(missing_ok=True)
    all_rows = collect_rows()
    shared = shared_pairs(all_rows)
    shared_rows = [
        row for row in all_rows if (row["definition"], row["prompt_tag"]) in shared
    ]
    matched = matched_turn_rows(shared_rows)
    matched_long = rows_from_matched(matched)
    overall = summarize(matched_long, ["group"])
    by_definition = summarize(matched_long, ["definition", "group"])
    by_pair = summarize(matched_long, ["definition", "prompt_tag", "group"])
    holistic = holistic_turn_rows(all_rows, shared, matched)

    write_csv(
        OUT_DIR / "holistic_turns.csv",
        holistic,
        ["stage_index", "group", "run", "run_dir", "definition", "prompt_tag", "prompt_tag_raw", "replica", "trajectory_id", "turn", "correctness", "is_correct", "speedup", "arch_tag", "is_shared_pair", "is_matched_turn"],
    )
    write_alluvial_outputs(matched, matched_long)
    write_summary(all_rows, shared_rows, matched, overall, by_definition, by_pair)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
