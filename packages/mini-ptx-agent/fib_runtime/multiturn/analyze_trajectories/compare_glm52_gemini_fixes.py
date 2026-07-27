#!/usr/bin/env python3
"""Compare direct GLM 5.2 and Gemini fix-it runs.

This intentionally emits tables only. The paired input plans make an alluvial
view unnecessary for the main question: which model fixed more Qwen kernels,
and which fixed kernels were faster?
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "compare_glm52_gemini_fixes"
EXPORTER_PATH = Path("/home/ubuntu/AccRL/benchmark/export_turn_correctness_arch.py")
TARGET_SPEEDUP = 0.15
REQUIRED_ARCH_TAG = "H"

RUNS = {
    "glm52": {
        "label": "GLM 5.2",
        "root": Path("/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-glm52"),
    },
    "gemini": {
        "label": "Gemini",
        "root": Path("/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-gemini"),
    },
}


def load_exporter() -> Any:
    spec = importlib.util.spec_from_file_location("export_turn_correctness_arch", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load exporter from {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plan(root: Path) -> list[dict[str, Any]]:
    payload = json.loads((root / "plan.json").read_text())
    plan = payload["plan"] if isinstance(payload, dict) and "plan" in payload else payload
    return sorted(plan, key=lambda item: int(item.get("exp_index", len(plan))))


def success_kernel_counts(root: Path) -> tuple[set[str], int]:
    success_dir = root / "success"
    fixed_trajectories: set[str] = set()
    kernel_files = 0
    if not success_dir.exists():
        return fixed_trajectories, kernel_files
    for exp_dir in sorted(success_dir.glob("exp_*")):
        if not exp_dir.is_dir() or ".bak" in exp_dir.name:
            continue
        kernels = list(exp_dir.glob("kernel_v*.cu"))
        if kernels:
            fixed_trajectories.add(exp_dir.name)
            kernel_files += len(kernels)
    return fixed_trajectories, kernel_files


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def uses_required_arch_instruction(arch_tag: str | None) -> bool:
    tags = {tag.strip() for tag in (arch_tag or "").split(",") if tag.strip()}
    return REQUIRED_ARCH_TAG in tags


def trajectory_turn_rows(root: Path, exporter: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "trajectories").glob("*.json")):
        traj = json.loads(path.read_text())
        sequence = exporter.extract_turn_sequence(traj)
        speedups = exporter.extract_turn_speedups(traj)
        arch_tags = {}
        for turn, assistant_message, _eval_message in exporter.assistant_eval_turns(traj):
            if turn >= len(sequence) or sequence[turn] != "Correct":
                arch_tags[turn] = ""
                continue
            content = assistant_message.get("content", "")
            kernel_source = exporter.extract_code_block(
                content if isinstance(content, str) else "",
                languages=["cpp"],
                keep_separators=False,
            )
            arch_tags[turn] = exporter.check_arch_from_text(kernel_source) if kernel_source else ""
        for turn, correctness in enumerate(sequence):
            arch_tag = arch_tags.get(turn, "")
            instruction_correct = correctness == "Correct" and uses_required_arch_instruction(arch_tag)
            rows.append(
                {
                    "trajectory_id": path.stem,
                    "turn": turn,
                    "correctness": correctness,
                    "is_correct": int(correctness == "Correct"),
                    "arch_tag": arch_tag,
                    "correct_and_use_instruction": int(instruction_correct),
                    "speedup": float_or_none(speedups.get(turn)),
                }
            )
    return rows


def summarize_run(name: str, exporter: Any) -> dict[str, Any]:
    root = RUNS[name]["root"]
    plan = load_plan(root)
    turn_rows = trajectory_turn_rows(root, exporter)
    materialized_trajectories, materialized_kernel_files = success_kernel_counts(root)

    turns_by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in turn_rows:
        turns_by_traj[row["trajectory_id"]].append(row)

    prompt_rows: list[dict[str, Any]] = []
    for index, item in enumerate(plan):
        trajectory_id = f"exp_{int(item.get('exp_index', index)):03d}"
        turns = sorted(turns_by_traj.get(trajectory_id, []), key=lambda row: int(row["turn"]))
        correct = [row for row in turns if row["correctness"] == "Correct"]
        instruction_correct = [row for row in turns if int(row["correct_and_use_instruction"])]
        correct_with_speedup = [row for row in correct if row["speedup"] is not None]
        instruction_correct_with_speedup = [
            row for row in instruction_correct if row["speedup"] is not None
        ]
        best_row = max(correct_with_speedup, key=lambda row: float(row["speedup"]), default=None)
        best_instruction_row = max(
            instruction_correct_with_speedup,
            key=lambda row: float(row["speedup"]),
            default=None,
        )
        best_speedup = float(best_row["speedup"]) if best_row else None
        best_instruction_speedup = (
            float(best_instruction_row["speedup"]) if best_instruction_row else None
        )
        prompt_rows.append(
            {
                "run": name,
                "label": RUNS[name]["label"],
                "trajectory_id": trajectory_id,
                "exp_index": int(item.get("exp_index", index)),
                "definition": item.get("definition", ""),
                "prompt_tag": item.get("prompt_tag", ""),
                "source_error_kernel_path": item.get("error_kernel_path", ""),
                "source_error_log_path": item.get("error_log_path", ""),
                "n_turns": len(turns),
                "any_correct": int(bool(correct)),
                "any_instruction_correct": int(bool(instruction_correct)),
                "first_correct_turn": min((int(row["turn"]) for row in correct), default=""),
                "first_instruction_correct_turn": min(
                    (int(row["turn"]) for row in instruction_correct),
                    default="",
                ),
                "best_correct_turn": int(best_row["turn"]) if best_row else "",
                "best_instruction_correct_turn": (
                    int(best_instruction_row["turn"]) if best_instruction_row else ""
                ),
                "best_correct_speedup": best_speedup if best_speedup is not None else "",
                "best_instruction_correct_speedup": (
                    best_instruction_speedup if best_instruction_speedup is not None else ""
                ),
                "target_hit": int(best_speedup is not None and best_speedup >= TARGET_SPEEDUP),
                "instruction_target_hit": int(
                    best_instruction_speedup is not None
                    and best_instruction_speedup >= TARGET_SPEEDUP
                ),
                "materialized_success_kernel": int(trajectory_id in materialized_trajectories),
            }
        )

    return {
        "name": name,
        "label": RUNS[name]["label"],
        "root": root,
        "plan": plan,
        "turn_rows": [{**row, "run": name, "label": RUNS[name]["label"]} for row in turn_rows],
        "prompt_rows": prompt_rows,
        "materialized_trajectories": materialized_trajectories,
        "materialized_kernel_files": materialized_kernel_files,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{count / total:.1%}"


def best_values(prompt_rows: list[dict[str, Any]], key: str = "best_correct_speedup") -> list[float]:
    return [float(row[key]) for row in prompt_rows if row[key] != ""]


def overall_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, result in results.items():
        prompt_rows = result["prompt_rows"]
        values = best_values(prompt_rows)
        instruction_values = best_values(prompt_rows, "best_instruction_correct_speedup")
        target_hits = sum(1 for row in prompt_rows if int(row["target_hit"]))
        instruction_target_hits = sum(1 for row in prompt_rows if int(row["instruction_target_hit"]))
        rows.append(
            {
                "run": name,
                "label": result["label"],
                "root": str(result["root"]),
                "planned_prompts": len(result["plan"]),
                "trajectory_files": len(list((result["root"] / "trajectories").glob("*.json"))),
                "turn_rows": len(result["turn_rows"]),
                "correct_turns": sum(1 for row in result["turn_rows"] if row["correctness"] == "Correct"),
                "correct_and_use_instruction_turns": sum(
                    1 for row in result["turn_rows"] if int(row["correct_and_use_instruction"])
                ),
                "any_correct": len(values),
                "any_correct_rate": len(values) / len(prompt_rows) if prompt_rows else 0.0,
                "any_instruction_correct": len(instruction_values),
                "any_instruction_correct_rate": (
                    len(instruction_values) / len(prompt_rows) if prompt_rows else 0.0
                ),
                "target_hits": target_hits,
                "target_hit_rate": target_hits / len(prompt_rows) if prompt_rows else 0.0,
                "target_hit_rate_among_fixed": target_hits / len(values) if values else 0.0,
                "instruction_target_hits": instruction_target_hits,
                "instruction_target_hit_rate": (
                    instruction_target_hits / len(prompt_rows) if prompt_rows else 0.0
                ),
                "instruction_target_hit_rate_among_instruction_fixed": (
                    instruction_target_hits / len(instruction_values) if instruction_values else 0.0
                ),
                "best_correct_mean_speedup": statistics.mean(values) if values else 0.0,
                "best_correct_median_speedup": statistics.median(values) if values else 0.0,
                "best_correct_max_speedup": max(values) if values else 0.0,
                "best_instruction_correct_mean_speedup": (
                    statistics.mean(instruction_values) if instruction_values else 0.0
                ),
                "best_instruction_correct_median_speedup": (
                    statistics.median(instruction_values) if instruction_values else 0.0
                ),
                "best_instruction_correct_max_speedup": (
                    max(instruction_values) if instruction_values else 0.0
                ),
                "materialized_success_trajectories": len(result["materialized_trajectories"]),
                "materialized_kernel_files": result["materialized_kernel_files"],
            }
        )
    return rows


def definition_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, result in results.items():
        by_definition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in result["prompt_rows"]:
            by_definition[row["definition"]].append(row)
        for definition, prompt_rows in sorted(by_definition.items()):
            values = best_values(prompt_rows)
            instruction_values = best_values(prompt_rows, "best_instruction_correct_speedup")
            target_hits = sum(1 for row in prompt_rows if int(row["target_hit"]))
            instruction_target_hits = sum(1 for row in prompt_rows if int(row["instruction_target_hit"]))
            rows.append(
                {
                    "run": name,
                    "label": result["label"],
                    "definition": definition,
                    "planned_prompts": len(prompt_rows),
                    "any_correct": len(values),
                    "any_correct_rate": len(values) / len(prompt_rows) if prompt_rows else 0.0,
                    "any_instruction_correct": len(instruction_values),
                    "any_instruction_correct_rate": (
                        len(instruction_values) / len(prompt_rows) if prompt_rows else 0.0
                    ),
                    "target_hits": target_hits,
                    "target_hit_rate": target_hits / len(prompt_rows) if prompt_rows else 0.0,
                    "instruction_target_hits": instruction_target_hits,
                    "instruction_target_hit_rate": (
                        instruction_target_hits / len(prompt_rows) if prompt_rows else 0.0
                    ),
                    "best_correct_mean_speedup": statistics.mean(values) if values else 0.0,
                    "best_correct_median_speedup": statistics.median(values) if values else 0.0,
                    "best_correct_max_speedup": max(values) if values else 0.0,
                    "best_instruction_correct_mean_speedup": (
                        statistics.mean(instruction_values) if instruction_values else 0.0
                    ),
                    "best_instruction_correct_median_speedup": (
                        statistics.median(instruction_values) if instruction_values else 0.0
                    ),
                    "best_instruction_correct_max_speedup": (
                        max(instruction_values) if instruction_values else 0.0
                    ),
                }
            )
    return rows


def turn_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, result in results.items():
        by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in result["turn_rows"]:
            by_turn[int(row["turn"])].append(row)
        for turn, rows_for_turn in sorted(by_turn.items()):
            correct = [row for row in rows_for_turn if row["correctness"] == "Correct"]
            instruction_correct = [
                row for row in rows_for_turn if int(row["correct_and_use_instruction"])
            ]
            speedups = [float(row["speedup"]) for row in correct if row["speedup"] is not None]
            instruction_speedups = [
                float(row["speedup"])
                for row in instruction_correct
                if row["speedup"] is not None
            ]
            target_hits = sum(1 for value in speedups if value >= TARGET_SPEEDUP)
            instruction_target_hits = sum(
                1 for value in instruction_speedups if value >= TARGET_SPEEDUP
            )
            rows.append(
                {
                    "run": name,
                    "label": result["label"],
                    "turn": turn,
                    "attempts": len(rows_for_turn),
                    "correct_turns": len(correct),
                    "correct_rate": len(correct) / len(rows_for_turn) if rows_for_turn else 0.0,
                    "correct_and_use_instruction_turns": len(instruction_correct),
                    "correct_and_use_instruction_rate": (
                        len(instruction_correct) / len(rows_for_turn) if rows_for_turn else 0.0
                    ),
                    "target_hits": target_hits,
                    "target_hit_rate": target_hits / len(rows_for_turn) if rows_for_turn else 0.0,
                    "instruction_target_hits": instruction_target_hits,
                    "instruction_target_hit_rate": (
                        instruction_target_hits / len(rows_for_turn) if rows_for_turn else 0.0
                    ),
                    "correct_median_speedup": statistics.median(speedups) if speedups else 0.0,
                    "correct_mean_speedup": statistics.mean(speedups) if speedups else 0.0,
                    "instruction_correct_median_speedup": (
                        statistics.median(instruction_speedups) if instruction_speedups else 0.0
                    ),
                    "instruction_correct_mean_speedup": (
                        statistics.mean(instruction_speedups) if instruction_speedups else 0.0
                    ),
                }
            )
    return rows


def paired_rows(results: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = {row["trajectory_id"]: row for row in results["glm52"]["prompt_rows"]}
    right = {row["trajectory_id"]: row for row in results["gemini"]["prompt_rows"]}
    paired = []
    outcome_rows = []
    counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    deltas: list[float] = []
    instruction_deltas: list[float] = []

    for trajectory_id in sorted(set(left) | set(right)):
        glm = left.get(trajectory_id)
        gemini = right.get(trajectory_id)
        if not glm or not gemini:
            continue
        glm_speed = float_or_none(glm["best_correct_speedup"])
        gemini_speed = float_or_none(gemini["best_correct_speedup"])
        glm_instruction_speed = float_or_none(glm["best_instruction_correct_speedup"])
        gemini_instruction_speed = float_or_none(gemini["best_instruction_correct_speedup"])
        if glm_speed is not None and gemini_speed is not None:
            outcome = "both_fixed"
            delta = glm_speed - gemini_speed
            deltas.append(delta)
        elif glm_speed is not None:
            outcome = "glm_only"
            delta = ""
        elif gemini_speed is not None:
            outcome = "gemini_only"
            delta = ""
        else:
            outcome = "neither"
            delta = ""
        counts[outcome] += 1

        if glm_instruction_speed is not None and gemini_instruction_speed is not None:
            instruction_outcome = "both_instruction_fixed"
            instruction_delta = glm_instruction_speed - gemini_instruction_speed
            instruction_deltas.append(instruction_delta)
        elif glm_instruction_speed is not None:
            instruction_outcome = "glm_instruction_only"
            instruction_delta = ""
        elif gemini_instruction_speed is not None:
            instruction_outcome = "gemini_instruction_only"
            instruction_delta = ""
        else:
            instruction_outcome = "neither_instruction_fixed"
            instruction_delta = ""
        instruction_counts[instruction_outcome] += 1

        paired.append(
            {
                "trajectory_id": trajectory_id,
                "definition": glm["definition"],
                "prompt_tag": glm["prompt_tag"],
                "glm_best_correct_speedup": glm_speed if glm_speed is not None else "",
                "gemini_best_correct_speedup": gemini_speed if gemini_speed is not None else "",
                "glm_best_instruction_correct_speedup": (
                    glm_instruction_speed if glm_instruction_speed is not None else ""
                ),
                "gemini_best_instruction_correct_speedup": (
                    gemini_instruction_speed if gemini_instruction_speed is not None else ""
                ),
                "glm_first_correct_turn": glm["first_correct_turn"],
                "gemini_first_correct_turn": gemini["first_correct_turn"],
                "glm_first_instruction_correct_turn": glm["first_instruction_correct_turn"],
                "gemini_first_instruction_correct_turn": gemini["first_instruction_correct_turn"],
                "glm_minus_gemini_best_speedup": delta,
                "glm_minus_gemini_best_instruction_speedup": instruction_delta,
                "outcome": outcome,
                "instruction_outcome": instruction_outcome,
            }
        )

    for outcome in ["both_fixed", "glm_only", "gemini_only", "neither"]:
        outcome_rows.append({"outcome": outcome, "count": counts[outcome]})
    outcome_rows.extend(
        [
            {
                "outcome": "glm_better_when_both_fixed",
                "count": sum(1 for value in deltas if value > 0),
            },
            {
                "outcome": "gemini_better_when_both_fixed",
                "count": sum(1 for value in deltas if value < 0),
            },
            {
                "outcome": "mean_glm_minus_gemini_best_speedup_when_both_fixed",
                "count": statistics.mean(deltas) if deltas else "",
            },
            {
                "outcome": "median_glm_minus_gemini_best_speedup_when_both_fixed",
                "count": statistics.median(deltas) if deltas else "",
            },
        ]
    )
    for outcome in [
        "both_instruction_fixed",
        "glm_instruction_only",
        "gemini_instruction_only",
        "neither_instruction_fixed",
    ]:
        outcome_rows.append({"outcome": outcome, "count": instruction_counts[outcome]})
    outcome_rows.extend(
        [
            {
                "outcome": "glm_better_when_both_instruction_fixed",
                "count": sum(1 for value in instruction_deltas if value > 0),
            },
            {
                "outcome": "gemini_better_when_both_instruction_fixed",
                "count": sum(1 for value in instruction_deltas if value < 0),
            },
            {
                "outcome": "mean_glm_minus_gemini_best_instruction_speedup_when_both_instruction_fixed",
                "count": statistics.mean(instruction_deltas) if instruction_deltas else "",
            },
            {
                "outcome": "median_glm_minus_gemini_best_instruction_speedup_when_both_instruction_fixed",
                "count": statistics.median(instruction_deltas) if instruction_deltas else "",
            },
        ]
    )
    return paired, outcome_rows


def top_rows(rows: list[dict[str, Any]], key: str, limit: int = 20) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row[key] != ""),
        key=lambda row: float(row[key]),
        reverse=True,
    )[:limit]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(
    results: dict[str, dict[str, Any]],
    overall: list[dict[str, Any]],
    by_definition: list[dict[str, Any]],
    by_turn: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> None:
    overall_by_run = {row["run"]: row for row in overall}
    outcome_counts = {row["outcome"]: row["count"] for row in outcomes}

    glm_wins = [
        row
        for row in paired
        if row["outcome"] == "glm_only"
        or (
            row["outcome"] == "both_fixed"
            and row["glm_minus_gemini_best_speedup"] != ""
            and float(row["glm_minus_gemini_best_speedup"]) > 0
        )
    ]
    glm_only_top = top_rows(glm_wins, "glm_best_correct_speedup", limit=10)
    gemini_only_top = top_rows(
        [row for row in paired if row["outcome"] == "gemini_only"],
        "gemini_best_correct_speedup",
        limit=10,
    )

    lines = [
        "# GLM 5.2 vs Gemini Fix-It Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Scope",
        "",
        "This compares the direct fix-it runs that use the same planned d128 Qwen failed kernels:",
        "",
    ]
    for name, result in results.items():
        lines.append(f"- {result['label']}: `{result['root']}`")
    lines.extend(
        [
            "",
            "The analysis reads current `trajectories/*.json` and extracts per-turn correctness and speedup with "
            "`/home/ubuntu/AccRL/benchmark/export_turn_correctness_arch.py`. Success directories are used only "
            "as a materialized-kernel cross-check.",
            "",
            "## Overall",
            "",
            markdown_table(
                ["metric", "GLM 5.2", "Gemini"],
                [
                    [
                        "planned prompts",
                        overall_by_run["glm52"]["planned_prompts"],
                        overall_by_run["gemini"]["planned_prompts"],
                    ],
                    [
                        "trajectory files",
                        overall_by_run["glm52"]["trajectory_files"],
                        overall_by_run["gemini"]["trajectory_files"],
                    ],
                    [
                        "turn rows",
                        overall_by_run["glm52"]["turn_rows"],
                        overall_by_run["gemini"]["turn_rows"],
                    ],
                    [
                        "correct turns",
                        overall_by_run["glm52"]["correct_turns"],
                        overall_by_run["gemini"]["correct_turns"],
                    ],
                    [
                        f"correct turns using `{REQUIRED_ARCH_TAG}` instruction",
                        overall_by_run["glm52"]["correct_and_use_instruction_turns"],
                        overall_by_run["gemini"]["correct_and_use_instruction_turns"],
                    ],
                    [
                        "any correct",
                        f"{overall_by_run['glm52']['any_correct']} ({pct(int(overall_by_run['glm52']['any_correct']), int(overall_by_run['glm52']['planned_prompts']))})",
                        f"{overall_by_run['gemini']['any_correct']} ({pct(int(overall_by_run['gemini']['any_correct']), int(overall_by_run['gemini']['planned_prompts']))})",
                    ],
                    [
                        f"any correct using `{REQUIRED_ARCH_TAG}` instruction",
                        f"{overall_by_run['glm52']['any_instruction_correct']} ({pct(int(overall_by_run['glm52']['any_instruction_correct']), int(overall_by_run['glm52']['planned_prompts']))})",
                        f"{overall_by_run['gemini']['any_instruction_correct']} ({pct(int(overall_by_run['gemini']['any_instruction_correct']), int(overall_by_run['gemini']['planned_prompts']))})",
                    ],
                    [
                        f"target hits >= {TARGET_SPEEDUP}",
                        f"{overall_by_run['glm52']['target_hits']} ({pct(int(overall_by_run['glm52']['target_hits']), int(overall_by_run['glm52']['planned_prompts']))})",
                        f"{overall_by_run['gemini']['target_hits']} ({pct(int(overall_by_run['gemini']['target_hits']), int(overall_by_run['gemini']['planned_prompts']))})",
                    ],
                    [
                        f"`{REQUIRED_ARCH_TAG}` instruction target hits >= {TARGET_SPEEDUP}",
                        f"{overall_by_run['glm52']['instruction_target_hits']} ({pct(int(overall_by_run['glm52']['instruction_target_hits']), int(overall_by_run['glm52']['planned_prompts']))})",
                        f"{overall_by_run['gemini']['instruction_target_hits']} ({pct(int(overall_by_run['gemini']['instruction_target_hits']), int(overall_by_run['gemini']['planned_prompts']))})",
                    ],
                    [
                        "best-correct median speedup",
                        fmt_float(float(overall_by_run["glm52"]["best_correct_median_speedup"])),
                        fmt_float(float(overall_by_run["gemini"]["best_correct_median_speedup"])),
                    ],
                    [
                        "best-correct mean speedup",
                        fmt_float(float(overall_by_run["glm52"]["best_correct_mean_speedup"])),
                        fmt_float(float(overall_by_run["gemini"]["best_correct_mean_speedup"])),
                    ],
                    [
                        f"best `{REQUIRED_ARCH_TAG}`-instruction-correct median speedup",
                        fmt_float(float(overall_by_run["glm52"]["best_instruction_correct_median_speedup"])),
                        fmt_float(float(overall_by_run["gemini"]["best_instruction_correct_median_speedup"])),
                    ],
                    [
                        "materialized success trajectories",
                        overall_by_run["glm52"]["materialized_success_trajectories"],
                        overall_by_run["gemini"]["materialized_success_trajectories"],
                    ],
                ],
            ),
            "",
            "## Paired Outcomes",
            "",
            markdown_table(
                ["outcome", "count"],
                [[row["outcome"], row["count"]] for row in outcomes],
            ),
            "",
            "## By Definition",
            "",
        ]
    )

    definitions = sorted({row["definition"] for row in by_definition})
    definition_table = []
    for definition in definitions:
        glm = next(row for row in by_definition if row["run"] == "glm52" and row["definition"] == definition)
        gemini = next(row for row in by_definition if row["run"] == "gemini" and row["definition"] == definition)
        definition_table.append(
            [
                definition,
                f"{glm['any_correct']}/{glm['planned_prompts']}",
                f"{gemini['any_correct']}/{gemini['planned_prompts']}",
                f"{glm['any_instruction_correct']}/{glm['planned_prompts']}",
                f"{gemini['any_instruction_correct']}/{gemini['planned_prompts']}",
                glm["target_hits"],
                gemini["target_hits"],
                glm["instruction_target_hits"],
                gemini["instruction_target_hits"],
                fmt_float(float(glm["best_correct_median_speedup"])),
                fmt_float(float(gemini["best_correct_median_speedup"])),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "definition",
                    "GLM fixed",
                    "Gemini fixed",
                    "GLM instr fixed",
                    "Gemini instr fixed",
                    "GLM target hits",
                    "Gemini target hits",
                    "GLM instr target hits",
                    "Gemini instr target hits",
                    "GLM median speedup",
                    "Gemini median speedup",
                ],
                definition_table,
            ),
            "",
            "## By Turn",
            "",
        ]
    )

    turns = sorted({int(row["turn"]) for row in by_turn})
    turn_table = []
    for turn in turns:
        glm = next(row for row in by_turn if row["run"] == "glm52" and int(row["turn"]) == turn)
        gemini = next(row for row in by_turn if row["run"] == "gemini" and int(row["turn"]) == turn)
        turn_table.append(
            [
                turn,
                glm["attempts"],
                glm["correct_turns"],
                glm["correct_and_use_instruction_turns"],
                glm["target_hits"],
                glm["instruction_target_hits"],
                fmt_float(float(glm["correct_median_speedup"])),
                gemini["attempts"],
                gemini["correct_turns"],
                gemini["correct_and_use_instruction_turns"],
                gemini["target_hits"],
                gemini["instruction_target_hits"],
                fmt_float(float(gemini["correct_median_speedup"])),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "turn",
                    "GLM attempts",
                    "GLM correct",
                    "GLM instr correct",
                    "GLM target hits",
                    "GLM instr target hits",
                    "GLM median correct speedup",
                    "Gemini attempts",
                    "Gemini correct",
                    "Gemini instr correct",
                    "Gemini target hits",
                    "Gemini instr target hits",
                    "Gemini median correct speedup",
                ],
                turn_table,
            ),
            "",
            "## Notable GLM Wins",
            "",
            markdown_table(
                ["trajectory", "definition", "prompt tag", "GLM best", "Gemini best", "outcome"],
                [
                    [
                        row["trajectory_id"],
                        row["definition"],
                        row["prompt_tag"],
                        fmt_float(float(row["glm_best_correct_speedup"])),
                        fmt_float(float(row["gemini_best_correct_speedup"])) if row["gemini_best_correct_speedup"] != "" else "",
                        row["outcome"],
                    ]
                    for row in glm_only_top
                ],
            ),
            "",
            "## Notable Gemini-Only Wins",
            "",
            markdown_table(
                ["trajectory", "definition", "prompt tag", "Gemini best"],
                [
                    [
                        row["trajectory_id"],
                        row["definition"],
                        row["prompt_tag"],
                        fmt_float(float(row["gemini_best_correct_speedup"])),
                    ]
                    for row in gemini_only_top
                ],
            ),
            "",
            "## Outputs",
            "",
            "- `summary.md`",
            "- `prompt_summary.csv`: one paired row per prompt, including ordinary correctness and instruction-correctness fields",
            "",
            "## Interpretation",
            "",
        ]
    )

    glm_fixed = int(overall_by_run["glm52"]["any_correct"])
    gemini_fixed = int(overall_by_run["gemini"]["any_correct"])
    glm_instruction_fixed = int(overall_by_run["glm52"]["any_instruction_correct"])
    gemini_instruction_fixed = int(overall_by_run["gemini"]["any_instruction_correct"])
    glm_targets = int(overall_by_run["glm52"]["target_hits"])
    gemini_targets = int(overall_by_run["gemini"]["target_hits"])
    glm_instruction_targets = int(overall_by_run["glm52"]["instruction_target_hits"])
    gemini_instruction_targets = int(overall_by_run["gemini"]["instruction_target_hits"])
    lines.extend(
        [
            f"Gemini fixes substantially more prompts than GLM 5.2 on this paired set: `{gemini_fixed}` vs `{glm_fixed}`.",
            f"It also has many more target-speedup hits at `{TARGET_SPEEDUP}`: `{gemini_targets}` vs `{glm_targets}`.",
            f"Using the Hopper instruction tag `{REQUIRED_ARCH_TAG}`, Gemini also leads on instruction-correct prompts: `{gemini_instruction_fixed}` vs `{glm_instruction_fixed}`, and instruction-correct target hits: `{gemini_instruction_targets}` vs `{glm_instruction_targets}`.",
            "GLM 5.2 still has isolated useful wins, so it is worth mining GLM-only successes, but Gemini is the stronger primary fixed-kernel source for this run pair.",
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python /home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/compare_glm52_gemini_fixes.py",
            "```",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exporter = load_exporter()
    results = {name: summarize_run(name, exporter) for name in RUNS}

    overall = overall_rows(results)
    by_definition = definition_rows(results)
    by_turn = turn_rows(results)
    paired, outcomes = paired_rows(results)

    for stale_name in [
        "overall.csv",
        "definition_summary.csv",
        "turn_summary.csv",
        "paired_prompt_summary.csv",
        "paired_outcomes.csv",
        "turn_rows.csv",
    ]:
        (OUT_DIR / stale_name).unlink(missing_ok=True)

    write_csv(OUT_DIR / "prompt_summary.csv", paired, list(paired[0].keys()))
    write_summary(results, overall, by_definition, by_turn, paired, outcomes)

    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
