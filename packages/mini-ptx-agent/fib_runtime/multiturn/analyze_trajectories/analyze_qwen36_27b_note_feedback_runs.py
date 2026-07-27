#!/usr/bin/env python3
"""Analyze Qwen3.6-27B retrieved-note and note-feedback2 eval runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "qwen36_27b_note_feedback_analysis"
WATCHERS = {
    "retrieved_notes_d128": "/home/ubuntu/AccRL-exps/tasks/collect_notes/watch_qwen36_27b_retrieved_notes_d128_4defs.sh",
    "note_feedback2_d128_d96": "/home/ubuntu/AccRL-exps/tasks/collect_notes/watch_qwen36_27b_note_feedback2_mha_d128_d96_8defs.sh",
}
GROUPS = {
    "fixit_v2_glm_nopatched_d128": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-nopatched-mha-bwd-d128-causal",
    ],
    "fixit_v2_glm_d128": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128-causal",
    ],
    "fixit_v2_glm_d96": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d96",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d96-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d96",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d96-causal",
    ],
    "retrieved_notes_d128": [
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-retrieved-notes-d128-4defs-mha-bwd-d128-causal",
    ],
    "note_feedback2_d128": [
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal",
    ],
    "note_feedback2_d96": [
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96",
        "/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal",
    ],
}
GROUP_LABELS = {
    "fixit_v2_glm_nopatched_d128": "Fixit v2 GLM non-patched d128",
    "fixit_v2_glm_d128": "Fixit v2 GLM d128",
    "fixit_v2_glm_d96": "Fixit v2 GLM d96",
    "retrieved_notes_d128": "Retrieved notes d128",
    "note_feedback2_d128": "Note feedback2 d128",
    "note_feedback2_d96": "Note feedback2 d96",
}
GROUP_ORDER = list(GROUPS)
GLM_PROMPT_CONFIGS = {
    ("fixit_v2_glm_nopatched_d128", "forward"): Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-2-r8-p4.json"
    ),
    ("fixit_v2_glm_nopatched_d128", "backward"): Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-bwd-2-r8-p4.json"
    ),
    ("fixit_v2_glm_d128", "forward"): Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-2-r8-p4-patched.json"
    ),
    ("fixit_v2_glm_d128", "backward"): Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-bwd-2-r8-p4-patched.json"
    ),
    ("fixit_v2_glm_d96", "forward"): Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-2-r8-p4-patched.json"
    ),
    ("fixit_v2_glm_d96", "backward"): Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-bwd-2-r8-p4-patched.json"
    ),
}
ARCH_TAG_BY_ARCH = {
    "hopper": "H",
    "blackwell": "B",
}
REQUIRED_ARCH_TAG = ARCH_TAG_BY_ARCH["hopper"]
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
    "Missing turn",
]


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


def uses_instruction(arch_tag: str | None) -> bool:
    tags = {tag.strip() for tag in (arch_tag or "").split(",") if tag.strip()}
    return REQUIRED_ARCH_TAG in tags


def resolution_for_definition(definition: str) -> str:
    if "_d96" in definition:
        return "d96"
    if "_d128" in definition:
        return "d128"
    return ""


def definition_family(definition: str) -> str:
    return "backward" if definition.startswith("mha_bwd_") else "forward"


def load_prompt_tags(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in prompt config: {path}")
    tags = {item.get("prompt_tag", "") for item in payload}
    if "" in tags:
        raise ValueError(f"Missing prompt_tag in prompt config: {path}")
    return tags


def definition_sort_key(definition: str) -> tuple[int, str]:
    resolution_order = {"d128": 0, "d96": 1, "": 2}
    return (resolution_order[resolution_for_definition(definition)], definition)


def collect_rows() -> list[dict]:
    rows = []
    glm_prompt_tags = {
        key: load_prompt_tags(path) for key, path in GLM_PROMPT_CONFIGS.items()
    }
    for stage_index, (group, roots) in enumerate(GROUPS.items()):
        for root in roots:
            run_dir = Path(root)
            plan = sorted(load_plan(run_dir), key=lambda item: int(item["exp_index"]))
            turn_rows = load_turn_rows(run_dir)
            replica_by_pair: dict[tuple[str, str], int] = defaultdict(int)
            for item in plan:
                trajectory_id = f"exp_{int(item['exp_index']):03d}"
                definition = item.get("definition", "")
                prompt_tag_raw = item.get("prompt_tag", "")
                allowed_tags = glm_prompt_tags.get(
                    (group, definition_family(definition))
                )
                if allowed_tags is not None and prompt_tag_raw not in allowed_tags:
                    continue
                prompt_tag = prompt_tag_raw
                pair = (definition, prompt_tag)
                replica = replica_by_pair[pair]
                replica_by_pair[pair] += 1
                for turn in range(int(item.get("num_turns", 8))):
                    turn_row = turn_rows.get((trajectory_id, turn))
                    has_source_turn = int(turn_row is not None)
                    correctness = turn_row["correctness"] if turn_row else "Missing turn"
                    arch_tag = turn_row.get("arch_tag", "") if turn_row else ""
                    is_correct = int(correctness == "Correct")
                    use_instruction = int(is_correct and uses_instruction(arch_tag))
                    rows.append(
                        {
                            "stage_index": stage_index,
                            "group": group,
                            "resolution": resolution_for_definition(definition),
                            "run": run_dir.name,
                            "run_dir": str(run_dir),
                            "definition": definition,
                            "prompt_tag": prompt_tag,
                            "prompt_tag_raw": prompt_tag_raw,
                            "replica": replica,
                            "trajectory_id": trajectory_id,
                            "turn": turn,
                            "has_source_turn": has_source_turn,
                            "correctness": correctness,
                            "is_correct": is_correct,
                            "speedup": as_float(turn_row.get("speedup")) if turn_row else 0.0,
                            "arch_tag": arch_tag,
                            "uses_instruction": use_instruction,
                            "correct_and_use_instruction": use_instruction,
                        }
                    )
    return rows


def validate_glm_prompt_tags(rows: list[dict]) -> None:
    configured = {
        key: load_prompt_tags(path) for key, path in GLM_PROMPT_CONFIGS.items()
    }
    observed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (row["group"], definition_family(row["definition"]))
        if key in configured:
            observed[key].add(row["prompt_tag"])
    for key, expected_tags in configured.items():
        if observed[key] != expected_tags:
            raise ValueError(
                f"Prompt tags for {key} do not match config: "
                f"expected={sorted(expected_tags)}, observed={sorted(observed[key])}"
            )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)

    out = []
    def sort_key(item: tuple[tuple, list[dict]]) -> tuple:
        key_values, _items = item
        if "group" not in keys:
            return key_values
        group_index = keys.index("group")
        return (
            *key_values[:group_index],
            GROUP_ORDER.index(key_values[group_index]),
            *key_values[group_index + 1 :],
        )

    for key_values, items in sorted(buckets.items(), key=sort_key):
        trajectory_keys = {(item["run"], item["trajectory_id"]) for item in items}
        n_traj = len(trajectory_keys)
        horizon_metrics = []
        for horizon in (1, 4, 8):
            horizon_items = [item for item in items if int(item["turn"]) < horizon]
            horizon_turns = len(horizon_items)
            horizon_correct = sum(int(item["is_correct"]) for item in horizon_items)
            horizon_correct_use_instruction = sum(
                int(item["correct_and_use_instruction"])
                for item in horizon_items
            )
            horizon_correct_trajectories = {
                (item["run"], item["trajectory_id"])
                for item in horizon_items
                if item["correctness"] == "Correct"
            }
            horizon_correct_instruction_trajectories = {
                (item["run"], item["trajectory_id"])
                for item in horizon_items
                if item["correct_and_use_instruction"]
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
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "n_trajectories": n_traj,
                "n_turns": len(items),
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

    def state_key(state: str) -> tuple[int, str]:
        try:
            return (STATE_ORDER.index(state), state)
        except ValueError:
            return (len(STATE_ORDER), state)

    out = []
    for key, counts in sorted(buckets.items()):
        for state, count in sorted(counts.items(), key=lambda item: state_key(item[0])):
            row = {key_name: value for key_name, value in zip(keys, key)}
            row.update({"correctness": state, "count": count, "fraction": count / totals[key]})
            out.append(row)
    return out


def source_coverage(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["group"]].append(row)
    out = []
    for group in GROUP_ORDER:
        items = buckets[group]
        trajectories = {(item["run"], item["trajectory_id"]) for item in items}
        observed_turns = sum(int(item["has_source_turn"]) for item in items)
        out.append(
            {
                "group": group,
                "n_runs": len({item["run"] for item in items}),
                "n_definitions": len({item["definition"] for item in items}),
                "n_prompt_pairs": len(
                    {(item["definition"], item["prompt_tag"]) for item in items}
                ),
                "n_trajectories": len(trajectories),
                "planned_turns": len(items),
                "observed_turns": observed_turns,
                "missing_turns": len(items) - observed_turns,
                "definitions": ", ".join(
                    sorted({item["definition"] for item in items})
                ),
            }
        )
    return out


def md_table(rows: list[dict], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row[field]
            vals.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(
    rows: list[dict],
    overall: list[dict],
    by_definition: list[dict],
    pairs: list[dict],
    coverage: list[dict],
    states: list[dict],
) -> None:
    lines = [
        "# Qwen3.6-27B Retrieved Notes / Note Feedback2 Analysis",
        "",
        "Source of truth: each run's `figures/turn_correctness_arch.csv`, joined to `plan.json` for definition and prompt metadata.",
        "Prompt tags are kept as recorded in `plan.json`; patched tags keep their `-mha-patched` suffix.",
        "No cross-group prompt-tag alignment or intersection is applied. Retrieved-notes and feedback2 groups include every available prompt tag; the fixit-v2-glm groups use the exact applicable prompt-tag allowlists below.",
        "Missing planned turns are retained in `holistic_turns.csv` with `has_source_turn=0` and count against planned-turn rates.",
        "",
        "Fixit-v2-GLM prompt configurations:",
        "",
    ]
    for path in dict.fromkeys(GLM_PROMPT_CONFIGS.values()):
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
        "Watchers:",
        "",
        ]
    )
    for label, path in WATCHERS.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(
        [
            "",
            "Groups:",
            "",
            "- `fixit_v2_glm_nopatched_d128`: fixit-v2-glm non-patched baseline over the four d128 definitions and two config-selected prompt tags per definition.",
            "- `fixit_v2_glm_d128`: fixit-v2-glm patched baseline over the four d128 definitions and two config-selected `-mha-patched` prompt tags per definition.",
            "- `fixit_v2_glm_d96`: fixit-v2-glm patched baseline over the four d96 definitions and two config-selected `-mha-patched` prompt tags per definition.",
            "- `retrieved_notes_d128`: retrieved-note mode over the four d128 definitions.",
            "- `note_feedback2_d128`: feedback2 mode over the four d128 definitions.",
            "- `note_feedback2_d96`: feedback2 mode over the four d96 definitions.",
            "",
            f"Holistic CSV: `holistic_turns.csv` ({len(rows)} planned turn rows)",
            "",
            "## Source Coverage",
            "",
            md_table(
                coverage,
                [
                    "group",
                    "n_runs",
                    "n_definitions",
                    "n_prompt_pairs",
                    "n_trajectories",
                    "planned_turns",
                    "observed_turns",
                    "missing_turns",
                    "definitions",
                ],
            ),
            "",
            "## Included-Prompt Overall Metrics",
            "",
            "All metric triplets are ordered `≤1 / ≤4 / ≤8` turns. Missing planned turns remain in the denominator.",
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
            "## Included-Prompt-Definition Collective Metrics",
            "",
            "Each table pools every included prompt tag for one problem and group without cross-group prompt-tag matching. Metric triplets remain ordered `≤1 / ≤4 / ≤8` turns.",
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
    for definition in sorted(
        {row["definition"] for row in by_definition}, key=definition_sort_key
    ):
        definition_rows = [
            row for row in by_definition if row["definition"] == definition
        ]
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
            "## Included Definition / Prompt-Tag Rows",
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
            "## State Counts",
            "",
            md_table(states, ["group", "correctness", "count", "fraction"]),
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "holistic_turns.csv").unlink(missing_ok=True)
    rows = collect_rows()
    validate_glm_prompt_tags(rows)
    overall = summarize(rows, ["group"])
    by_definition = summarize(rows, ["definition", "group"])
    by_pair = summarize(rows, ["definition", "prompt_tag", "group"])
    coverage = source_coverage(rows)
    states = state_counts(rows, ["group"])
    write_csv(
        OUT_DIR / "holistic_turns.csv",
        rows,
        [
            "stage_index",
            "group",
            "resolution",
            "run",
            "run_dir",
            "definition",
            "prompt_tag",
            "prompt_tag_raw",
            "replica",
            "trajectory_id",
            "turn",
            "has_source_turn",
            "correctness",
            "is_correct",
            "speedup",
            "arch_tag",
            "uses_instruction",
            "correct_and_use_instruction",
        ],
    )
    write_summary(rows, overall, by_definition, by_pair, coverage, states)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
