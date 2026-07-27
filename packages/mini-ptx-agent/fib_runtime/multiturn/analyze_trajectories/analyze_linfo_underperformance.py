#!/usr/bin/env python3
"""Explain linfo underperformance from matched outcomes and trajectory behavior."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
FIVE_MODE_CSV = HERE / "compare_20260624_0939_five_modes" / "holistic_turns.csv"
FIXIT_CSV = (
    HERE
    / "compare_fixit_until_v6"
    / "holistic_turns.csv"
)
OUTPUT_DIR = HERE / "linfo_underperformance_analysis"

SFT_ROOT = Path(
    "/home/ubuntu/AccRL-exps/sft_experiments/"
    "test-fixit-qwen36-27b-gemini-glm"
)
BASE_PARQUET = SFT_ROOT / "data" / "glm-5.2-mha-d128-4def-full.parquet"
V5_PARQUET = SFT_ROOT / "data" / "glm-5.2-fixit-v5.parquet"
BASE_TRAIN_CONFIG = (
    SFT_ROOT
    / "runs"
    / "qwen36-27b-glm52-fixit-e5-lr4.65e-4-lora32-"
    "Qwen-Qwen3.6-27B-2026-06-24-09-39"
    / "config.json"
)
V5_TRAIN_CONFIG = (
    SFT_ROOT
    / "runs"
    / "qwen36-27b-glm52-fixit-v5-e5-lr4.65e-4-lora32-"
    "Qwen-Qwen3.6-27B-2026-07-12-00-58"
    / "config.json"
)

DIRECT_GROUPS = ("regular", "linfo", "regular-patched", "linfo-patched")
TRAINING_GROUPS = ("fixit-v2-glm", "fixit-v5")

FEATURE_PATTERNS = {
    "tma": ("CUtensorMap", "cp.async.bulk.tensor"),
    "wgmma": ("wgmma.",),
    "mbarrier": ("mbarrier",),
    "setmaxnreg": ("setmaxnreg",),
    "extended_launch": ("cudaLaunchKernelEx", "cudaLaunchConfig_t"),
    "cluster": ("cluster",),
}

FEEDBACK_MARKERS = {
    "precise_fault": "Most precise CUDA fault",
    "primary_diagnostics": "Primary diagnostics",
    "source_locations": "Source locations",
    "core_dump": "CUDA core dump analysis",
}

SHORT_OUTCOME = {
    "Correct": "OK",
    "Compilation error": "compile",
    "Runtime error": "runtime",
    "Numerical error": "numeric",
    "Kernel Execution Timeout": "timeout",
    "Extraction error": "extract",
    "Other error": "other",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def matched_rows(path: Path, groups: Iterable[str]) -> list[dict[str, str]]:
    group_set = set(groups)
    return [
        row
        for row in read_csv(path)
        if row["group"] in group_set and row["is_matched_turn"] == "1"
    ]


def trajectory_records(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["group"], row["run_dir"], row["trajectory_id"])
        record = records.setdefault(
            key,
            {
                "group": row["group"],
                "run_dir": row["run_dir"],
                "trajectory_id": row["trajectory_id"],
                "definition": row["definition"],
                "prompt_tag": row["prompt_tag"],
                "replica": int(row["replica"]),
                "outcomes": {},
            },
        )
        record["outcomes"][int(row["turn"])] = row["correctness"]
    return list(records.values())


def line_jaccard(left: str, right: str) -> float:
    left_lines = {line.strip() for line in left.splitlines() if line.strip()}
    right_lines = {line.strip() for line in right.splitlines() if line.strip()}
    union = left_lines | right_lines
    if not union:
        return 1.0
    return len(left_lines & right_lines) / len(union)


def assistant_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in trajectory["messages"] if m.get("role") == "assistant"]


def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    path = (
        Path(record["run_dir"])
        / "trajectories"
        / f"{record['trajectory_id']}.json"
    )
    with path.open() as f:
        trajectory = json.load(f)
    all_messages = assistant_messages(trajectory)
    matched_turns = sorted(record["outcomes"])
    messages = [all_messages[turn] for turn in matched_turns]
    all_feedback = [
        message
        for message in trajectory["messages"][2:]
        if message.get("role") == "user"
    ]
    feedback = [all_feedback[turn]["content"] for turn in matched_turns]
    codes = [str(message.get("content") or "") for message in messages]
    reasons = [str(message.get("reasoning_content") or "") for message in messages]
    outcomes = [record["outcomes"][turn] for turn in matched_turns]
    similarities = [
        line_jaccard(codes[index - 1], codes[index])
        for index in range(1, len(codes))
    ]

    feature_turns = {}
    for feature, patterns in FEATURE_PATTERNS.items():
        feature_turns[feature] = sum(
            any(pattern in code for pattern in patterns) for code in codes
        )

    correct_turns = [
        turn for turn, outcome in enumerate(outcomes) if outcome == "Correct"
    ]
    model_config = trajectory["info"]["config"]["model"]
    model_kwargs = dict(model_config["model_kwargs"])
    api_base = model_kwargs.pop("api_base", "")
    model_kwargs.pop("api_key", None)
    system = trajectory["messages"][0]["content"]
    initial_user = trajectory["messages"][1]["content"]
    record.update(
        {
            "trajectory_path": str(path),
            "system_chars": len(system),
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "initial_user_sha256": hashlib.sha256(initial_user.encode()).hexdigest(),
            "model_name": model_config["model_name"],
            "sampling_signature": json.dumps(model_kwargs, sort_keys=True),
            "api_base": api_base,
            "n_turns": len(outcomes),
            "code_chars_mean": statistics.mean(map(len, codes)),
            "code_chars_initial": len(codes[0]),
            "reasoning_chars_mean": statistics.mean(map(len, reasons)),
            "reasoning_chars_initial": len(reasons[0]),
            "edit_similarity_mean": (
                statistics.mean(similarities) if similarities else 1.0
            ),
            "major_rewrites": sum(value < 0.55 for value in similarities),
            "correct_turns": len(correct_turns),
            "ever_correct": bool(correct_turns),
            "first_correct_turn": min(correct_turns) if correct_turns else None,
            "correct_then_regressed": bool(correct_turns)
            and any(
                outcome != "Correct"
                for outcome in outcomes[min(correct_turns) + 1 :]
            ),
            "outcome_sequence": " -> ".join(
                SHORT_OUTCOME.get(outcome, outcome) for outcome in outcomes
            ),
            "feedback_chars_mean": statistics.mean(map(len, feedback)),
            **{f"{key}_turns": value for key, value in feature_turns.items()},
            **{
                f"feedback_{key}": sum(marker in text for text in feedback)
                for key, marker in FEEDBACK_MARKERS.items()
            },
        }
    )
    for outcome in SHORT_OUTCOME:
        record[f"outcome_{SHORT_OUTCOME[outcome]}"] = outcomes.count(outcome)
    return record


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else math.nan


def group_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group"]].append(record)

    summaries = {}
    for group, group_records in by_group.items():
        turns = sum(record["n_turns"] for record in group_records)
        summary = {
            "n_trajectories": len(group_records),
            "n_turns": turns,
            "correct_turns": sum(record["correct_turns"] for record in group_records),
            "ever_correct": sum(record["ever_correct"] for record in group_records),
            "first_turn_correct": sum(
                record["first_correct_turn"] == 0 for record in group_records
            ),
            "correct_then_regressed": sum(
                record["correct_then_regressed"] for record in group_records
            ),
            "mean_initial_code_chars": safe_mean(
                record["code_chars_initial"] for record in group_records
            ),
            "mean_code_chars": safe_mean(
                record["code_chars_mean"] for record in group_records
            ),
            "mean_initial_reasoning_chars": safe_mean(
                record["reasoning_chars_initial"] for record in group_records
            ),
            "mean_reasoning_chars": safe_mean(
                record["reasoning_chars_mean"] for record in group_records
            ),
            "mean_feedback_chars": safe_mean(
                record["feedback_chars_mean"] for record in group_records
            ),
            "mean_edit_similarity": safe_mean(
                record["edit_similarity_mean"] for record in group_records
            ),
            "major_rewrites": sum(record["major_rewrites"] for record in group_records),
        }
        for feature in FEATURE_PATTERNS:
            feature_turns = sum(
                record[f"{feature}_turns"] for record in group_records
            )
            summary[f"{feature}_turn_rate"] = feature_turns / turns
        for marker in FEEDBACK_MARKERS:
            marker_turns = sum(
                record[f"feedback_{marker}"] for record in group_records
            )
            summary[f"feedback_{marker}_rate"] = marker_turns / turns
        for outcome, short in SHORT_OUTCOME.items():
            summary[short] = sum(
                record[f"outcome_{short}"] for record in group_records
            )
        summaries[group] = summary
    return summaries


def recovery_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, tuple[int, int]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group"]].append(record)
    result = {}
    for group, group_records in by_group.items():
        transitions: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for record in group_records:
            outcomes = [
                record["outcomes"][turn] for turn in sorted(record["outcomes"])
            ]
            for before, after in zip(outcomes, outcomes[1:]):
                if before == "Correct":
                    continue
                transitions[before][1] += 1
                transitions[before][0] += after == "Correct"
        result[group] = {
            outcome: (counts[0], counts[1])
            for outcome, counts in transitions.items()
        }
    return result


def parquet_summary() -> dict[str, Any]:
    import pandas as pd

    base = pd.read_parquet(BASE_PARQUET)
    v5 = pd.read_parquet(V5_PARQUET)
    base_ids = set(base["id"])
    v5_ids = set(v5["id"])

    def last_message_lengths(frame: Any, role: str) -> list[int]:
        result = []
        for messages in frame["messages"]:
            matches = [message for message in messages if message["role"] == role]
            result.append(len(matches[-1]["content"]))
        return result

    def dimension_rows(frame: Any, dimension: str) -> int:
        return sum(
            str(metadata["definition"]).endswith(f"_{dimension}")
            or f"_{dimension}_" in str(metadata["definition"])
            for metadata in frame["metadata"]
        )

    return {
        "base_rows": len(base),
        "v5_rows": len(v5),
        "overlap_rows": len(base_ids & v5_ids),
        "base_d128_rows": dimension_rows(base, "d128"),
        "v5_d128_rows": dimension_rows(v5, "d128"),
        "v5_d64_rows": dimension_rows(v5, "d64"),
        "base_wrong_turns": Counter(
            int(metadata["wrong_turn"]) for metadata in base["metadata"]
        ),
        "v5_wrong_turns": Counter(
            int(metadata["wrong_turn"]) for metadata in v5["metadata"]
        ),
        "v5_linfo_source_rows": sum(
            "linfo" in str(metadata.get("wrong_trajectory_path", "")).lower()
            for metadata in v5["metadata"]
        ),
        "base_feedback_mean": statistics.mean(last_message_lengths(base, "user")),
        "v5_feedback_mean": statistics.mean(last_message_lengths(v5, "user")),
        "base_target_mean": statistics.mean(
            last_message_lengths(base, "assistant")
        ),
        "v5_target_mean": statistics.mean(
            last_message_lengths(v5, "assistant")
        ),
    }


def train_config_summary() -> dict[str, Any]:
    with BASE_TRAIN_CONFIG.open() as f:
        base = json.load(f)
    with V5_TRAIN_CONFIG.open() as f:
        v5 = json.load(f)
    keys = ("learning_rate", "num_epochs", "lora_rank", "model_name")
    return {
        "base": {key: base[key] for key in keys},
        "v5": {key: v5[key] for key in keys},
        "base_train_on": base["dataset_builder"]["common_config"]["train_on_what"],
        "v5_train_on": v5["dataset_builder"]["common_config"]["train_on_what"],
    }


def fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def record_index(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            record["group"],
            record["definition"],
            record["prompt_tag"],
            record["replica"],
        ): record
        for record in records
    }


def paired_condition_parity(
    records: list[dict[str, Any]], left: str, right: str
) -> dict[str, int]:
    index = record_index(records)
    pairs = []
    for key, record in index.items():
        if key[0] != left:
            continue
        pairs.append((record, index[(right, *key[1:])]))
    return {
        "pairs": len(pairs),
        "same_system": sum(
            left_record["system_sha256"] == right_record["system_sha256"]
            for left_record, right_record in pairs
        ),
        "same_initial_user": sum(
            left_record["initial_user_sha256"]
            == right_record["initial_user_sha256"]
            for left_record, right_record in pairs
        ),
        "same_model": sum(
            left_record["model_name"] == right_record["model_name"]
            for left_record, right_record in pairs
        ),
        "same_sampling": sum(
            left_record["sampling_signature"]
            == right_record["sampling_signature"]
            for left_record, right_record in pairs
        ),
    }


def case_rows(
    index: dict[tuple[Any, ...], dict[str, Any]],
    case_specs: list[tuple[str, str, str, int]],
) -> list[list[Any]]:
    rows = []
    for group, definition, prompt_tag, replica in case_specs:
        record = index[(group, definition, prompt_tag, replica)]
        rows.append(
            [
                group,
                definition,
                prompt_tag,
                replica,
                record["trajectory_id"],
                record["outcome_sequence"],
                fmt_int(record["code_chars_initial"]),
                fmt_int(record["reasoning_chars_initial"]),
            ]
        )
    return rows


def write_metrics(records: list[dict[str, Any]]) -> None:
    fields = [
        "group",
        "run_dir",
        "trajectory_id",
        "trajectory_path",
        "definition",
        "prompt_tag",
        "replica",
        "n_turns",
        "correct_turns",
        "ever_correct",
        "first_correct_turn",
        "correct_then_regressed",
        "outcome_sequence",
        "system_chars",
        "system_sha256",
        "initial_user_sha256",
        "model_name",
        "sampling_signature",
        "api_base",
        "feedback_chars_mean",
        "code_chars_initial",
        "code_chars_mean",
        "reasoning_chars_initial",
        "reasoning_chars_mean",
        "edit_similarity_mean",
        "major_rewrites",
        *[f"{feature}_turns" for feature in FEATURE_PATTERNS],
        *[f"feedback_{marker}" for marker in FEEDBACK_MARKERS],
    ]
    with (OUTPUT_DIR / "trajectory_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def write_report(
    direct_records: list[dict[str, Any]],
    training_records: list[dict[str, Any]],
) -> None:
    direct = group_summary(direct_records)
    training = group_summary(training_records)
    recoveries = recovery_summary(direct_records + training_records)
    parquet = parquet_summary()
    configs = train_config_summary()
    unpatched_parity = paired_condition_parity(
        direct_records, "regular", "linfo"
    )
    patched_parity = paired_condition_parity(
        direct_records, "regular-patched", "linfo-patched"
    )
    checkpoint_parity = paired_condition_parity(
        training_records, "fixit-v2-glm", "fixit-v5"
    )
    direct_index = record_index(direct_records)
    training_index = record_index(training_records)

    metric_rows = []
    for group in (*DIRECT_GROUPS, *TRAINING_GROUPS):
        summary = direct.get(group, training.get(group))
        metric_rows.append(
            [
                group,
                f"{summary['correct_turns']}/{summary['n_turns']}",
                f"{summary['ever_correct']}/{summary['n_trajectories']}",
                f"{summary['first_turn_correct']}/{summary['n_trajectories']}",
                summary["compile"],
                summary["runtime"],
                summary["numeric"],
                summary["timeout"],
                fmt_int(summary["mean_initial_code_chars"]),
                fmt_int(summary["mean_initial_reasoning_chars"]),
                fmt_int(summary["mean_feedback_chars"]),
                fmt_pct(summary["feedback_primary_diagnostics_rate"]),
            ]
        )

    feature_rows = []
    for group in (*DIRECT_GROUPS, *TRAINING_GROUPS):
        summary = direct.get(group, training.get(group))
        feature_rows.append(
            [
                group,
                fmt_pct(summary["tma_turn_rate"]),
                fmt_pct(summary["wgmma_turn_rate"]),
                fmt_pct(summary["mbarrier_turn_rate"]),
                fmt_pct(summary["extended_launch_turn_rate"]),
                f"{summary['major_rewrites']}",
                f"{summary['mean_edit_similarity']:.3f}",
            ]
        )

    recovery_rows = []
    for group in (*DIRECT_GROUPS, *TRAINING_GROUPS):
        for outcome in ("Compilation error", "Runtime error", "Numerical error"):
            recovered, attempts = recoveries[group].get(outcome, (0, 0))
            recovery_rows.append(
                [
                    group,
                    outcome,
                    f"{recovered}/{attempts}",
                    fmt_pct(recovered / attempts) if attempts else "n/a",
                ]
            )

    direct_cases = case_rows(
        direct_index,
        [
            ("regular-patched", "mha_with_lse_d128", "hopper-07", 1),
            ("linfo-patched", "mha_with_lse_d128", "hopper-07", 1),
            ("regular", "mha_bwd_d128", "hopper-012", 0),
            ("linfo", "mha_bwd_d128", "hopper-012", 0),
        ],
    )
    training_cases = case_rows(
        training_index,
        [
            ("fixit-v2-glm", "mha_with_lse_d128", "hopper-07", 0),
            (
                "fixit-v5",
                "mha_with_lse_d128",
                "hopper-07",
                0,
            ),
        ],
    )

    report = f"""# Why linfo underperforms

## Conclusion

These folders do **not** provide a clean “short feedback versus long feedback” ablation. The unpatched `regular` and `linfo` arms are matched stochastic repeats: they use identical prompts, checkpoint, sampling settings, and structured diagnostics. The patched pair is closer to a feedback comparison, but its system-reference text also changed. The checkpoint comparison is `fixit-v2-glm` versus `fixit-v5`; it changes the checkpoint, system prompt, inference-time diagnostics, and SFT data distribution together.

Across all three views, the most consistent mechanism is **repair thrashing around a brittle initial architecture**. Detailed diagnostics are read and often localized correctly, but the next full-code rewrite preserves inconsistent TMA descriptor, WGMMA layout, mbarrier, shared-memory, or launch assumptions—or introduces a new defect. Because part of the performance gap already exists on turn 0, feedback alone cannot explain it.

## Outcome and behavior shift

{markdown_table(
    [
        "group",
        "correct turns",
        "ever-correct trajectories",
        "turn-0 correct",
        "compile",
        "runtime",
        "numeric",
        "timeout",
        "mean initial code chars",
        "mean initial reasoning chars",
        "mean feedback chars",
        "feedback with primary diagnostics",
    ],
    metric_rows,
)}

The v2-to-v5 drop is large on the matched 797 turns: correct turns fall from {training['fixit-v2-glm']['correct_turns']} to {training['fixit-v5']['correct_turns']}, and ever-correct trajectories from {training['fixit-v2-glm']['ever_correct']}/100 to {training['fixit-v5']['ever_correct']}/100. V5 has {training['fixit-v5']['compile'] - training['fixit-v2-glm']['compile']} more compile errors and {training['fixit-v5']['timeout'] - training['fixit-v2-glm']['timeout']} more timeouts. It also starts from longer samples: mean initial code grows by {fmt_int(training['fixit-v5']['mean_initial_code_chars'] - training['fixit-v2-glm']['mean_initial_code_chars'])} characters and initial reasoning by {fmt_int(training['fixit-v5']['mean_initial_reasoning_chars'] - training['fixit-v2-glm']['mean_initial_reasoning_chars'])}.

Some of both gaps exists before evaluator feedback is shown. `regular-patched` has 2 turn-0 successes versus 0 for `linfo-patched`; `fixit-v2-glm` has {training['fixit-v2-glm']['first_turn_correct']} versus {training['fixit-v5']['first_turn_correct']} for v5. Feedback affects recovery, not those initial-sample differences.

## Condition audit: what is actually being compared

- Unpatched `regular` versus `linfo`: {unpatched_parity['same_system']}/{unpatched_parity['pairs']} matched trajectories have the same system prompt, {unpatched_parity['same_initial_user']}/{unpatched_parity['pairs']} the same task prompt, and every pair uses the same checkpoint and sampling settings. Both arms contain structured CUDA diagnostics. This is principally a stochastic repeat, not a feedback-detail ablation.
- `regular-patched` versus `linfo-patched`: all {patched_parity['pairs']} pairs use the same checkpoint, task prompt, and sampling settings, but 0/{patched_parity['pairs']} system prompts are byte-identical because the Hopper reference text changed. The old regular run has no `Primary diagnostics` blocks, while linfo-patched has them on {direct['linfo-patched']['feedback_primary_diagnostics_rate'] * direct['linfo-patched']['n_turns']:.0f}/{direct['linfo-patched']['n_turns']} turns. This is the closest feedback comparison, though it is still not perfectly isolated.
- `fixit-v2-glm` versus `fixit-v5`: all {checkpoint_parity['same_initial_user']}/{checkpoint_parity['pairs']} pairs have the same task prompt and all {checkpoint_parity['same_sampling']}/{checkpoint_parity['pairs']} use the same sampling settings, but 0/{checkpoint_parity['pairs']} use the same checkpoint or byte-identical system prompt. Structured `Primary diagnostics` rise from 0 to {training['fixit-v5']['feedback_primary_diagnostics_rate'] * training['fixit-v5']['n_turns']:.0f}/{training['fixit-v5']['n_turns']} turns. This measures the combined v5 checkpoint and evaluation stack, not feedback length alone.

## What the model does with the information

{markdown_table(
    [
        "group",
        "turns with TMA",
        "turns with WGMMA",
        "turns with mbarrier",
        "turns with extended launch",
        "major rewrites",
        "mean adjacent-code Jaccard",
    ],
    feature_rows,
)}

All arms already use TMA, WGMMA, and mbarrier on nearly every turn. V5 does not merely use these mechanisms more often; it begins with larger code and performs {training['fixit-v5']['major_rewrites']} major rewrites versus {training['fixit-v2-glm']['major_rewrites']} for v2. The reasoning can be specific to the latest reported fault while the overall descriptor/layout/barrier/launch contract remains inconsistent.

## Representative trajectories

### Same checkpoint, patched forward attention

{markdown_table(
    [
        "group", "definition", "tag", "replica", "trajectory",
        "turn outcomes", "initial code chars", "initial reasoning chars",
    ],
    direct_cases[:2],
)}

`regular-patched/exp_005` fixes a compile-time shared-memory pointer problem, then notices an unused `kv_offset` after a numerical failure and reaches correctness at turn 2. `linfo-patched/exp_001` times out before receiving feedback. Its coredump correctly localizes a V-buffer mbarrier hang, but later rewrites move through compile failure and persistent NaN/Inf. The model proposes several locally plausible fixes without making the whole double-buffered TMA/WGMMA kernel coherent.

### Same checkpoint, unpatched backward attention

{markdown_table(
    [
        "group", "definition", "tag", "replica", "trajectory",
        "turn outcomes", "initial code chars", "initial reasoning chars",
    ],
    direct_cases[2:],
)}

`regular/exp_012` progresses from a nonexistent BF16 intrinsic through layout failures and passes at turn 5. `linfo/exp_000` samples a larger two-kernel design with extended cluster launch, TMA, WGMMA, and explicit register allocation. It bounces among host API, launch-argument, runtime, and numerical failures without passing. Since the condition is otherwise identical, this pair demonstrates sensitivity to the initial sampled architecture.

### Fixit v2 versus v5

{markdown_table(
    [
        "group", "definition", "tag", "replica", "trajectory",
        "turn outcomes", "initial code chars", "initial reasoning chars",
    ],
    training_cases,
)}

For the same forward-attention prompt, tag, and replica, `fixit-v2-glm/exp_004` is correct on turn 0 and recovers once again at turn 5 after an optimization regression. `fixit-v5/exp_004` starts with a declaration-order compile error. It fixes that issue, then moves through an illegal shared-memory read, a numerical failure, an ambiguous BF16 conversion, NaN/Inf, a shared-memory alias, and finally a newly introduced undefined identifier. Several diagnoses are correct locally, but the eight rewrites never produce a correct kernel. This is the clearest v5 example of symptom-by-symptom repair without stable global invariants.

## What changed in v5 training

The two training configs use the same base model, learning rate ({configs['base']['learning_rate']}), epochs ({configs['base']['num_epochs']}), LoRA rank ({configs['base']['lora_rank']}), and `train_on_what={configs['base_train_on']}`. The parquets are different lineages with {parquet['overlap_rows']} shared IDs:

- V2 uses {parquet['base_rows']} rows, all d128, with wrong-turn distribution {dict(sorted(parquet['base_wrong_turns'].items()))}.
- V5 uses {parquet['v5_rows']} rows: {parquet['v5_d128_rows']} d128 and {parquet['v5_d64_rows']} d64. All {parquet['v5_linfo_source_rows']} wrong kernels come from linfo trajectories. Its wrong-turn distribution is {dict(sorted(parquet['v5_wrong_turns'].items()))}; {sum(count for turn, count in parquet['v5_wrong_turns'].items() if turn >= 4)} rows are from turns 4 or later.
- Mean evaluator-feedback length rises from {fmt_int(parquet['base_feedback_mean'])} to {fmt_int(parquet['v5_feedback_mean'])} characters, and mean target length from {fmt_int(parquet['base_target_mean'])} to {fmt_int(parquet['v5_target_mean'])}.

For this d128 evaluation, v5 therefore has fewer than half as many in-domain d128 SFT rows as v2 ({parquet['v5_d128_rows']} versus {parquet['base_d128_rows']}), while adding d64 and linfo-failure coverage. That distribution shift is a plausible contributor to the lower d128 result. It also means the v5 checkpoint comparison cannot identify a simple “more feedback is worse” effect: the checkpoint was trained on a different task mix and failure-source distribution.

## Recovery evidence

{markdown_table(
    ["group", "previous failure", "next-turn recoveries", "rate"],
    recovery_rows,
)}

The low absolute recovery rates support the trajectory reading. Detailed logs expose a real defect, but these coupled kernels commonly contain several defects at once. Fixing one does not make the next kernel correct, and full-file reconstruction can reintroduce already-solved errors.

## Bottom line

1. The corrected checkpoint comparison is `fixit-v2-glm` versus `fixit-v5`.
2. V5 is materially worse on these matched d128 trajectories: 27 versus 70 correct turns and 19 versus 38 ever-correct trajectories.
3. The gap is not attributable to feedback alone. V5 changes checkpoint, system prompt, diagnostic stack, and SFT distribution; it also has only 76 d128 training rows versus v2's 158.
4. Across representative failures, structured diagnostics improve local fault localization but do not enforce rollback or whole-kernel invariants. The model keeps repairing a brittle design and often creates the next error.

The practical implication is to pair detailed diagnostics with control rules: establish a compiling minimal kernel first, add one optimization pattern at a time, and after repeated compile/runtime failures force a revert to the last correct kernel or a restart from a simpler architecture.

## Reproduction

```bash
python analyze_linfo_underperformance.py
```

The script reads the two existing `holistic_turns.csv` files, joins matched trajectory IDs to source JSONs, reads the v2 and v5 SFT parquets and training configs, and writes this report plus `trajectory_metrics.csv`.
"""
    (OUTPUT_DIR / "summary.md").write_text(report)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    direct_records = [
        enrich_record(record)
        for record in trajectory_records(matched_rows(FIVE_MODE_CSV, DIRECT_GROUPS))
    ]
    training_records = [
        enrich_record(record)
        for record in trajectory_records(matched_rows(FIXIT_CSV, TRAINING_GROUPS))
    ]
    all_records = direct_records + training_records
    write_metrics(all_records)
    write_report(direct_records, training_records)
    print(f"Wrote {OUTPUT_DIR / 'summary.md'}")
    print(f"Wrote {OUTPUT_DIR / 'trajectory_metrics.csv'}")


if __name__ == "__main__":
    main()
