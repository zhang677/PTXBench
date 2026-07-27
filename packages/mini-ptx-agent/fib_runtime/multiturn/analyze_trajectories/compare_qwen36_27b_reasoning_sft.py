#!/usr/bin/env python3
"""Compare reasoning behavior for matched Qwen3.6-27B eval rows.

Defaults compare the five registry-matched Hopper evals:

  Qwen3.6-27B vs Qwen3.6-27B-fixit-v2-glm

The right side is constrained to the 2026-0624-0939 run family. Registry
matching uses (arch, definition, workload). Turn outcomes come from each run's
figures/turn_correctness_arch.csv, while reasoning and response-shape metrics
come from trajectories/*.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_EXPERIMENTS_CSV = Path("/home/ubuntu/AccRL/benchmark/experiments.csv")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "qwen36-27b-reasoning-2026-0624-0939"
)
DEFAULT_BASELINE_MODEL = "Qwen3.6-27B"
DEFAULT_SFT_MODEL = "Qwen3.6-27B-fixit-v2-glm"
DEFAULT_SFT_RUN_FRAGMENT = "2026-0624-0939"
DEFAULT_DEFINITIONS = {
    "gemm_n7168_k5120",
    "mha_with_lse_d128",
    "mha_with_lse_d128_causal",
    "mha_bwd_d128",
    "mha_bwd_d128_causal",
}
MHA_DEFINITIONS = {
    "mha_with_lse_d128",
    "mha_with_lse_d128_causal",
    "mha_bwd_d128",
    "mha_bwd_d128_causal",
}

CPP_BLOCK_RE = re.compile(r"```(?:cpp|cuda|c\+\+)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Experiment:
    group: str
    model: str
    arch: str
    definition: str
    workload: str
    exp_dir: Path

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.arch, self.definition, self.workload)

    @property
    def turn_csv_path(self) -> Path:
        return self.exp_dir / "figures" / "turn_correctness_arch.csv"

    @property
    def trajectories_dir(self) -> Path:
        return self.exp_dir / "trajectories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-csv", type=Path, default=DEFAULT_EXPERIMENTS_CSV)
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--sft-model", default=DEFAULT_SFT_MODEL)
    parser.add_argument(
        "--sft-run-fragment",
        default=DEFAULT_SFT_RUN_FRAGMENT,
        help="Substring required in SFT exp_dir values.",
    )
    parser.add_argument(
        "--definition",
        action="append",
        dest="definitions",
        help="Definition to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=None,
        help="Only include turns with turn < this value.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def nested_get(value: object, path: tuple[str, ...]) -> object | None:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_registry(
    path: Path,
    baseline_model: str,
    sft_model: str,
    definitions: set[str],
    sft_run_fragment: str,
) -> list[Experiment]:
    required = {"model", "arch", "definition", "workload", "exp_dir"}
    experiments: list[Experiment] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            model = row["model"].strip()
            definition = row["definition"].strip()
            exp_dir = Path(row["exp_dir"].strip())
            if definition not in definitions:
                continue
            if model == baseline_model:
                group = "baseline"
            elif model == sft_model:
                if sft_run_fragment and sft_run_fragment not in str(exp_dir):
                    continue
                group = "sft"
            else:
                continue
            experiments.append(
                Experiment(
                    group=group,
                    model=model,
                    arch=row["arch"].strip(),
                    definition=definition,
                    workload=row["workload"].strip(),
                    exp_dir=exp_dir,
                )
            )
    return experiments


def pair_experiments(experiments: list[Experiment]) -> list[tuple[Experiment, Experiment]]:
    by_key: dict[tuple[str, str, str], dict[str, list[Experiment]]] = defaultdict(
        lambda: {"baseline": [], "sft": []}
    )
    for exp in experiments:
        by_key[exp.key][exp.group].append(exp)

    pairs: list[tuple[Experiment, Experiment]] = []
    problems: list[str] = []
    for key in sorted(by_key):
        grouped = by_key[key]
        baseline = grouped["baseline"]
        sft = grouped["sft"]
        if len(baseline) != 1 or len(sft) != 1:
            problems.append(
                f"{key}: expected one baseline and one sft, got "
                f"baseline={len(baseline)} sft={len(sft)}"
            )
            continue
        pairs.append((baseline[0], sft[0]))

    if problems:
        raise ValueError("Unpaired or ambiguous registry rows:\n" + "\n".join(problems))
    if len(pairs) != 5:
        raise ValueError(f"Expected five matched pairs, found {len(pairs)}")
    return pairs


def read_turn_metrics(exp: Experiment, turn_limit: int | None) -> dict[tuple[str, int], dict[str, Any]]:
    if not exp.turn_csv_path.exists():
        raise FileNotFoundError(f"Missing turn correctness CSV: {exp.turn_csv_path}")

    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    with exp.turn_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"trajectory_id", "turn", "correctness"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{exp.turn_csv_path} missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            turn = as_int(row.get("turn"))
            if turn is None:
                continue
            if turn_limit is not None and turn >= turn_limit:
                continue
            metrics[(row["trajectory_id"].strip(), turn)] = {
                "correctness": (row.get("correctness") or "").strip() or "Unknown",
                "speedup": as_float(row.get("speedup")),
                "arch_tag": (row.get("arch_tag") or "").strip(),
            }
    return metrics


def assistant_eval_turns(traj: dict[str, Any], turn_limit: int | None) -> list[tuple[int, dict, dict]]:
    """Pair each assistant kernel response with its following feedback message."""
    turns: list[tuple[int, dict, dict]] = []
    saw_initial_user = False
    pending_assistant: dict | None = None

    for msg in traj.get("messages", []):
        role = msg.get("role")
        if role == "user":
            if not saw_initial_user:
                saw_initial_user = True
                continue
            if pending_assistant is not None:
                turn = len(turns)
                if turn_limit is None or turn < turn_limit:
                    turns.append((turn, pending_assistant, msg))
                pending_assistant = None
                if turn_limit is not None and len(turns) >= turn_limit:
                    break
        elif role == "assistant" and saw_initial_user:
            pending_assistant = msg

    return turns


def extract_usage(message: dict[str, Any]) -> dict[str, int | None]:
    usage = nested_get(message, ("extra", "response", "usage"))
    if not isinstance(usage, dict):
        usage = {}
    return {
        "prompt_tokens": as_int(usage.get("prompt_tokens", usage.get("input_tokens"))),
        "completion_tokens": as_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        ),
        "total_tokens": as_int(usage.get("total_tokens")),
        "reasoning_tokens": (
            as_int(nested_get(usage, ("completion_tokens_details", "reasoning_tokens")))
            or as_int(nested_get(usage, ("completion_tokens_details", "thinking_tokens")))
            or as_int(nested_get(usage, ("output_tokens_details", "thinking_tokens")))
            or as_int(nested_get(usage, ("output_tokens_details", "reasoning_tokens")))
            or as_int(usage.get("reasoning_tokens"))
            or as_int(usage.get("thinking_tokens"))
        ),
        "text_tokens": (
            as_int(nested_get(usage, ("completion_tokens_details", "text_tokens")))
            or as_int(nested_get(usage, ("output_tokens_details", "text_tokens")))
        ),
    }


def first_choice_message(message: dict[str, Any]) -> dict[str, Any]:
    choices = nested_get(message, ("extra", "response", "choices"))
    if isinstance(choices, list) and choices:
        choice_message = choices[0].get("message")
        if isinstance(choice_message, dict):
            return choice_message
    return {}


def response_shape(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)

    choice = first_choice_message(message)
    provider_content = choice.get("content") or ""
    provider_reasoning = choice.get("reasoning_content") or ""
    if not isinstance(provider_content, str):
        provider_content = str(provider_content)
    if not isinstance(provider_reasoning, str):
        provider_reasoning = str(provider_reasoning)

    code_blocks = CPP_BLOCK_RE.findall(content)
    first_code_pos = content.find("```")
    pre_code = content if first_code_pos < 0 else content[:first_code_pos]
    think_blocks = THINK_RE.findall(content)
    open_think = content.lower().count("<think>")
    close_think = content.lower().count("</think>")

    return {
        "content_chars": len(content),
        "provider_visible_chars": len(provider_content),
        "provider_reasoning_chars": len(provider_reasoning),
        "pre_code_chars": len(pre_code),
        "code_chars": sum(len(block) for block in code_blocks),
        "code_block_count": len(code_blocks),
        "think_block_chars": sum(len(block) for block in think_blocks),
        "open_think_tags": open_think,
        "close_think_tags": close_think,
        "unclosed_think_tags": max(0, open_think - close_think),
    }


def collect_turn_rows(
    exp: Experiment,
    pair_key: str,
    turn_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not exp.trajectories_dir.is_dir():
        raise FileNotFoundError(f"Missing trajectories dir: {exp.trajectories_dir}")

    turn_metrics = read_turn_metrics(exp, turn_limit)
    rows: list[dict[str, Any]] = []
    counts = Counter()

    for traj_path in sorted(exp.trajectories_dir.glob("*.json")):
        trajectory_id = traj_path.stem
        traj = json.loads(traj_path.read_text())
        for turn, assistant_message, eval_message in assistant_eval_turns(traj, turn_limit):
            metrics = turn_metrics.get((trajectory_id, turn))
            if metrics is None:
                counts["missing_turn_metrics"] += 1
                continue

            usage = extract_usage(assistant_message)
            shape = response_shape(assistant_message)
            reasoning_tokens = usage["reasoning_tokens"]
            completion_tokens = usage["completion_tokens"]
            reasoning_fraction = ""
            if reasoning_tokens is not None and completion_tokens:
                reasoning_fraction = reasoning_tokens / completion_tokens

            eval_content = eval_message.get("content") or ""
            if not isinstance(eval_content, str):
                eval_content = str(eval_content)

            rows.append(
                {
                    "group": exp.group,
                    "model": exp.model,
                    "pair_key": pair_key,
                    "arch": exp.arch,
                    "definition": exp.definition,
                    "workload": exp.workload,
                    "exp_dir": str(exp.exp_dir),
                    "trajectory_id": trajectory_id,
                    "turn": turn,
                    "correctness": metrics["correctness"],
                    "speedup": metrics["speedup"] if metrics["speedup"] is not None else "",
                    "arch_tag": metrics["arch_tag"],
                    "prompt_tokens": usage["prompt_tokens"] if usage["prompt_tokens"] is not None else "",
                    "completion_tokens": completion_tokens if completion_tokens is not None else "",
                    "total_tokens": usage["total_tokens"] if usage["total_tokens"] is not None else "",
                    "reasoning_tokens": reasoning_tokens if reasoning_tokens is not None else "",
                    "text_tokens": usage["text_tokens"] if usage["text_tokens"] is not None else "",
                    "reasoning_fraction": reasoning_fraction,
                    "feedback_chars": len(eval_content),
                    **shape,
                }
            )
            counts["turn_rows"] += 1

    return rows, dict(counts)


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value == "":
        return None
    return as_float(value)


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def pct(value: float) -> str:
    return f"{value:.2%}"


def fmt_number(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.1f}"


def fmt_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correctness = Counter(row["correctness"] for row in rows)

    def values(key: str) -> list[float]:
        return [value for row in rows if (value := numeric(row, key)) is not None]

    result: dict[str, Any] = {
        "turn_rows": total,
        "correct_rows": correctness["Correct"],
        "correct_rate": rate(correctness["Correct"], total),
        "compile_error_rate": rate(correctness["Compilation error"], total),
        "runtime_error_rate": rate(correctness["Runtime error"], total),
        "timeout_rate": rate(correctness["Kernel Execution Timeout"], total),
        "reasoning_token_rows": sum(1 for row in rows if numeric(row, "reasoning_tokens") is not None),
    }
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "reasoning_fraction",
        "content_chars",
        "provider_visible_chars",
        "provider_reasoning_chars",
        "pre_code_chars",
        "code_chars",
        "feedback_chars",
    ]:
        vals = values(key)
        result[f"mean_{key}"] = mean(vals) if vals else ""
        result[f"median_{key}"] = median(vals) if vals else ""
    result["mean_code_block_count"] = mean(values("code_block_count")) if total else ""
    result["unclosed_think_turns"] = sum(1 for row in rows if numeric(row, "unclosed_think_tags"))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate_rows(turn_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in turn_rows:
        grouped[
            (
                "overall",
                "",
                "",
                row["group"],
                row["model"],
            )
        ].append(row)
        grouped[
            (
                "definition",
                row["definition"],
                row["workload"],
                row["group"],
                row["model"],
            )
        ].append(row)
        grouped[
            (
                "definition_by_turn",
                row["definition"],
                row["workload"],
                row["group"],
                row["model"] + f"::turn={row['turn']}",
            )
        ].append(row)

    rows: list[dict[str, Any]] = []
    for (scope, definition, workload, group, model_key), grouped_rows in sorted(grouped.items()):
        if "::turn=" in model_key:
            model, turn_text = model_key.split("::turn=", 1)
            turn = turn_text
        else:
            model = model_key
            turn = ""
        summary = summarize_group(grouped_rows)
        rows.append(
            {
                "scope": scope,
                "definition": definition,
                "workload": workload,
                "turn": turn,
                "group": group,
                "model": model,
                **summary,
            }
        )
    return rows


def build_aligned_delta_rows(turn_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in turn_rows:
        by_key[
            (
                row["definition"],
                row["workload"],
                row["trajectory_id"],
                int(row["turn"]),
            )
        ][row["group"]] = row

    delta_rows: list[dict[str, Any]] = []
    metric_keys = [
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "reasoning_fraction",
        "content_chars",
        "pre_code_chars",
        "code_chars",
        "provider_visible_chars",
        "feedback_chars",
    ]
    for (definition, workload, trajectory_id, turn), grouped in sorted(by_key.items()):
        baseline = grouped.get("baseline")
        sft = grouped.get("sft")
        if baseline is None or sft is None:
            continue
        row: dict[str, Any] = {
            "definition": definition,
            "workload": workload,
            "trajectory_id": trajectory_id,
            "turn": turn,
            "baseline_correctness": baseline["correctness"],
            "sft_correctness": sft["correctness"],
            "baseline_correct": int(baseline["correctness"] == "Correct"),
            "sft_correct": int(sft["correctness"] == "Correct"),
            "correct_delta": int(sft["correctness"] == "Correct")
            - int(baseline["correctness"] == "Correct"),
            "baseline_exp_dir": baseline["exp_dir"],
            "sft_exp_dir": sft["exp_dir"],
        }
        for key in metric_keys:
            base_value = numeric(baseline, key)
            sft_value = numeric(sft, key)
            row[f"baseline_{key}"] = "" if base_value is None else base_value
            row[f"sft_{key}"] = "" if sft_value is None else sft_value
            row[f"delta_{key}"] = (
                "" if base_value is None or sft_value is None else sft_value - base_value
            )
        delta_rows.append(row)
    return delta_rows


def build_pair_rows(pairs: list[tuple[Experiment, Experiment]]) -> list[dict[str, Any]]:
    return [
        {
            "arch": baseline.arch,
            "definition": baseline.definition,
            "workload": baseline.workload,
            "baseline_model": baseline.model,
            "baseline_exp_dir": str(baseline.exp_dir),
            "sft_model": sft.model,
            "sft_exp_dir": str(sft.exp_dir),
        }
        for baseline, sft in pairs
    ]


def find_definition_pair(
    pairs: list[tuple[Experiment, Experiment]],
    definition: str,
) -> tuple[Experiment, Experiment]:
    matches = [pair for pair in pairs if pair[0].definition == definition]
    if len(matches) != 1:
        raise ValueError(f"Expected one pair for {definition}, found {len(matches)}")
    return matches[0]


def overall_for_group(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    grouped = [row for row in rows if row["group"] == group]
    if not grouped:
        raise ValueError(f"No rows for group {group}")
    return summarize_group(grouped)


def correctness_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row["correctness"] for row in rows)


def metric_table_rows(baseline: dict[str, Any], sft: dict[str, Any]) -> list[tuple[str, Any, Any, Any]]:
    return [
        ("turn rows", baseline["turn_rows"], sft["turn_rows"], sft["turn_rows"] - baseline["turn_rows"]),
        (
            "correct rate",
            pct(baseline["correct_rate"]),
            pct(sft["correct_rate"]),
            pct(sft["correct_rate"] - baseline["correct_rate"]),
        ),
        (
            "rows with reasoning-token counters",
            baseline["reasoning_token_rows"],
            sft["reasoning_token_rows"],
            sft["reasoning_token_rows"] - baseline["reasoning_token_rows"],
        ),
        (
            "mean reasoning tokens",
            fmt_number(baseline["mean_reasoning_tokens"]),
            fmt_number(sft["mean_reasoning_tokens"]),
            fmt_number(float(sft["mean_reasoning_tokens"]) - float(baseline["mean_reasoning_tokens"])),
        ),
        (
            "mean completion tokens",
            fmt_number(baseline["mean_completion_tokens"]),
            fmt_number(sft["mean_completion_tokens"]),
            fmt_number(float(sft["mean_completion_tokens"]) - float(baseline["mean_completion_tokens"])),
        ),
        (
            "mean provider reasoning chars",
            fmt_number(baseline["mean_provider_reasoning_chars"]),
            fmt_number(sft["mean_provider_reasoning_chars"]),
            fmt_number(float(sft["mean_provider_reasoning_chars"]) - float(baseline["mean_provider_reasoning_chars"])),
        ),
        (
            "mean visible content chars",
            fmt_number(baseline["mean_provider_visible_chars"]),
            fmt_number(sft["mean_provider_visible_chars"]),
            fmt_number(float(sft["mean_provider_visible_chars"]) - float(baseline["mean_provider_visible_chars"])),
        ),
        (
            "mean pre-code chars",
            fmt_number(baseline["mean_pre_code_chars"]),
            fmt_number(sft["mean_pre_code_chars"]),
            fmt_number(float(sft["mean_pre_code_chars"]) - float(baseline["mean_pre_code_chars"])),
        ),
        (
            "mean code chars",
            fmt_number(baseline["mean_code_chars"]),
            fmt_number(sft["mean_code_chars"]),
            fmt_number(float(sft["mean_code_chars"]) - float(baseline["mean_code_chars"])),
        ),
    ]


def write_mha_eval_reports(
    output_dir: Path,
    pairs: list[tuple[Experiment, Experiment]],
    turn_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    turn_limit: int | None,
    sft_run_fragment: str,
) -> dict[str, Path]:
    reports_dir = output_dir / "mha_eval_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Path] = {}

    for definition in sorted(MHA_DEFINITIONS):
        baseline_exp, sft_exp = find_definition_pair(pairs, definition)
        definition_rows = [row for row in turn_rows if row["definition"] == definition]
        definition_delta_rows = [
            row for row in delta_rows if row["definition"] == definition
        ]
        baseline_rows = [row for row in definition_rows if row["group"] == "baseline"]
        sft_rows = [row for row in definition_rows if row["group"] == "sft"]
        baseline_summary = overall_for_group(definition_rows, "baseline")
        sft_summary = overall_for_group(definition_rows, "sft")
        baseline_counts = correctness_counts(baseline_rows)
        sft_counts = correctness_counts(sft_rows)
        categories = sorted(set(baseline_counts) | set(sft_counts), key=lambda c: (c != "Correct", c))
        aligned_improved = sum(1 for row in definition_delta_rows if row["correct_delta"] > 0)
        aligned_regressed = sum(1 for row in definition_delta_rows if row["correct_delta"] < 0)

        lines = [
            f"# {definition} Reasoning Comparison",
            "",
            f"- baseline: `{baseline_exp.model}`",
            f"- sft: `{sft_exp.model}`",
            f"- sft run family: `{sft_run_fragment}`",
            f"- turn_limit: `{turn_limit if turn_limit is not None else 'all'}`",
            f"- workload: `{baseline_exp.workload}`",
            f"- baseline exp_dir: `{baseline_exp.exp_dir}`",
            f"- sft exp_dir: `{sft_exp.exp_dir}`",
            "",
            "## Overall",
            "",
            "| metric | baseline | sft | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
        for metric, base_value, sft_value, delta in metric_table_rows(baseline_summary, sft_summary):
            lines.append(f"| {metric} | {base_value} | {sft_value} | {delta} |")

        lines.extend(
            [
                "",
                "## Correctness Buckets",
                "",
                "| correctness | baseline count | baseline pct | sft count | sft pct | delta pct |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for category in categories:
            base_count = baseline_counts[category]
            sft_count = sft_counts[category]
            base_rate = rate(base_count, len(baseline_rows))
            sft_rate = rate(sft_count, len(sft_rows))
            lines.append(
                f"| {category} | {base_count} | {pct(base_rate)} | "
                f"{sft_count} | {pct(sft_rate)} | {pct(sft_rate - base_rate)} |"
            )

        lines.extend(
            [
                "",
                "## By Turn",
                "",
                "| turn | baseline correct | sft correct | delta | baseline mean reasoning tokens | sft mean reasoning tokens | baseline mean provider reasoning chars | sft mean provider reasoning chars |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        turns = sorted({int(row["turn"]) for row in definition_rows})
        for turn in turns:
            baseline_turn = summarize_group(
                [row for row in baseline_rows if int(row["turn"]) == turn]
            )
            sft_turn = summarize_group([row for row in sft_rows if int(row["turn"]) == turn])
            lines.append(
                f"| {turn} | {pct(baseline_turn['correct_rate'])} | "
                f"{pct(sft_turn['correct_rate'])} | "
                f"{pct(sft_turn['correct_rate'] - baseline_turn['correct_rate'])} | "
                f"{fmt_number(baseline_turn['mean_reasoning_tokens'])} | "
                f"{fmt_number(sft_turn['mean_reasoning_tokens'])} | "
                f"{fmt_number(baseline_turn['mean_provider_reasoning_chars'])} | "
                f"{fmt_number(sft_turn['mean_provider_reasoning_chars'])} |"
            )

        lines.extend(
            [
                "",
                "## Aligned Turns",
                "",
                f"- aligned baseline/SFT turns: `{len(definition_delta_rows)}`",
                f"- SFT improved correctness on aligned turns: `{aligned_improved}`",
                f"- SFT regressed correctness on aligned turns: `{aligned_regressed}`",
                "",
                "## Most Changed Aligned Turns",
                "",
                "| trajectory | turn | baseline correctness | sft correctness | delta reasoning tokens | delta completion tokens | delta code chars |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        changed_rows = sorted(
            definition_delta_rows,
            key=lambda row: (
                abs(float(row["delta_reasoning_tokens"]))
                if row.get("delta_reasoning_tokens") != ""
                else -1.0
            ),
            reverse=True,
        )[:10]
        for row in changed_rows:
            lines.append(
                f"| `{row['trajectory_id']}` | {row['turn']} | "
                f"{row['baseline_correctness']} | {row['sft_correctness']} | "
                f"{fmt_cell(row['delta_reasoning_tokens'])} | "
                f"{fmt_cell(row['delta_completion_tokens'])} | "
                f"{fmt_cell(row['delta_code_chars'])} |"
            )
        lines.append("")

        report_path = reports_dir / f"{definition}.md"
        report_path.write_text("\n".join(lines))
        reports[definition] = report_path

    return reports


def write_summary(
    path: Path,
    pairs: list[tuple[Experiment, Experiment]],
    aggregate_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    outputs: dict[str, Path],
    turn_limit: int | None,
    sft_run_fragment: str,
) -> None:
    overall = {
        row["group"]: row
        for row in aggregate_rows
        if row["scope"] == "overall"
    }
    baseline = overall["baseline"]
    sft = overall["sft"]
    aligned_total = len(delta_rows)
    aligned_improved = sum(1 for row in delta_rows if row["correct_delta"] > 0)
    aligned_regressed = sum(1 for row in delta_rows if row["correct_delta"] < 0)

    lines = [
        "# Qwen3.6-27B Reasoning Comparison",
        "",
        f"- baseline: `{pairs[0][0].model}`",
        f"- sft: `{pairs[0][1].model}`",
        f"- sft run family: `{sft_run_fragment}`",
        f"- turn_limit: `{turn_limit if turn_limit is not None else 'all'}`",
        f"- registry-matched rows: `{len(pairs)}`",
        "",
        "## Paired Registry Rows",
        "",
        "| definition | workload | baseline exp_dir | sft exp_dir |",
        "| --- | --- | --- | --- |",
    ]
    for base, sft_exp in pairs:
        lines.append(
            f"| `{base.definition}` | `{base.workload}` | "
            f"`{base.exp_dir}` | `{sft_exp.exp_dir}` |"
        )

    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| metric | baseline | sft | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric, base_value, sft_value, delta in metric_table_rows(baseline, sft):
        lines.append(f"| {metric} | {base_value} | {sft_value} | {delta} |")

    lines.extend(
        [
            "",
            "## Aligned Turns",
            "",
            f"- aligned baseline/SFT turns: `{aligned_total}`",
            f"- SFT improved correctness on aligned turns: `{aligned_improved}`",
            f"- SFT regressed correctness on aligned turns: `{aligned_regressed}`",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, output_path in outputs.items():
        lines.append(f"- {name}: `{output_path}`")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    definitions = set(args.definitions or DEFAULT_DEFINITIONS)
    experiments = read_registry(
        args.experiments_csv,
        baseline_model=args.baseline_model,
        sft_model=args.sft_model,
        definitions=definitions,
        sft_run_fragment=args.sft_run_fragment,
    )
    pairs = pair_experiments(experiments)

    turn_rows: list[dict[str, Any]] = []
    collection_counts: dict[str, dict[str, int]] = {}
    for baseline, sft in pairs:
        pair_key = "|".join(baseline.key)
        for exp in (baseline, sft):
            rows, counts = collect_turn_rows(exp, pair_key, args.turn_limit)
            turn_rows.extend(rows)
            collection_counts[str(exp.exp_dir)] = counts

    aggregate_rows = build_aggregate_rows(turn_rows)
    delta_rows = build_aligned_delta_rows(turn_rows)
    pair_rows = build_pair_rows(pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_turn_csv = args.output_dir / "reasoning_turns.csv"
    aggregate_csv = args.output_dir / "reasoning_aggregates.csv"
    aligned_delta_csv = args.output_dir / "aligned_turn_deltas.csv"
    paired_runs_csv = args.output_dir / "paired_registry_rows.csv"
    summary_md = args.output_dir / "reasoning_summary.md"
    manifest_json = args.output_dir / "manifest.json"
    mha_reports = write_mha_eval_reports(
        args.output_dir,
        pairs,
        turn_rows,
        delta_rows,
        args.turn_limit,
        args.sft_run_fragment,
    )

    turn_fieldnames = [
        "group",
        "model",
        "pair_key",
        "arch",
        "definition",
        "workload",
        "exp_dir",
        "trajectory_id",
        "turn",
        "correctness",
        "speedup",
        "arch_tag",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "text_tokens",
        "reasoning_fraction",
        "content_chars",
        "provider_visible_chars",
        "provider_reasoning_chars",
        "pre_code_chars",
        "code_chars",
        "code_block_count",
        "think_block_chars",
        "open_think_tags",
        "close_think_tags",
        "unclosed_think_tags",
        "feedback_chars",
    ]
    aggregate_fieldnames = list(aggregate_rows[0].keys()) if aggregate_rows else []
    delta_fieldnames = list(delta_rows[0].keys()) if delta_rows else []
    pair_fieldnames = list(pair_rows[0].keys()) if pair_rows else []

    write_csv(per_turn_csv, turn_rows, turn_fieldnames)
    write_csv(aggregate_csv, aggregate_rows, aggregate_fieldnames)
    write_csv(aligned_delta_csv, delta_rows, delta_fieldnames)
    write_csv(paired_runs_csv, pair_rows, pair_fieldnames)

    outputs = {
        "per_turn_csv": per_turn_csv,
        "aggregate_csv": aggregate_csv,
        "aligned_delta_csv": aligned_delta_csv,
        "paired_runs_csv": paired_runs_csv,
        "mha_reports_dir": args.output_dir / "mha_eval_reports",
        "summary_md": summary_md,
        "manifest_json": manifest_json,
    }
    write_summary(
        summary_md,
        pairs,
        aggregate_rows,
        delta_rows,
        outputs,
        args.turn_limit,
        args.sft_run_fragment,
    )
    manifest_json.write_text(
        json.dumps(
            {
                "baseline_model": args.baseline_model,
                "sft_model": args.sft_model,
                "sft_run_fragment": args.sft_run_fragment,
                "turn_limit": args.turn_limit,
                "definitions": sorted(definitions),
                "collection_counts": collection_counts,
                "outputs": {name: str(path) for name, path in outputs.items()},
                "mha_reports": {
                    definition: str(path) for definition, path in sorted(mha_reports.items())
                },
                "paired_runs": pair_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"Wrote {per_turn_csv}")
    print(f"Wrote {aggregate_csv}")
    print(f"Wrote {aligned_delta_csv}")
    print(f"Wrote {paired_runs_csv}")
    for report_path in mha_reports.values():
        print(f"Wrote {report_path}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {manifest_json}")


if __name__ == "__main__":
    main()
