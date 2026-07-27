#!/usr/bin/env python3
"""Compare three Gemini 3.1 Pro MHA-backward d128 evaluation conditions."""

from __future__ import annotations

import json
import statistics
from itertools import combinations
from pathlib import Path

import compare_fixit_until_v6 as comparison
from render_comparison_table_figure import render


BASE = Path(__file__).resolve().parent

comparison.OUT_DIR = BASE / "compare_gemini31_linfo_june_may_mha_bwd_d128"
comparison.REPORT_TITLE = (
    "Gemini 3.1 Pro MHA Backward d128: Linfo vs June Baseline vs May NCU"
)
comparison.REQUIRED_ARCH_TAG = "H"
comparison.STAGES = [
    "gemini-31-pro-linfo",
    "gemini-31-pro-june",
    "gemini-31-pro-may-ncu",
]
comparison.STAGE_LABELS = {
    "gemini-31-pro-linfo": "Gemini 3.1 Pro linfo",
    "gemini-31-pro-june": "Gemini 3.1 Pro June baseline",
    "gemini-31-pro-may-ncu": "Gemini 3.1 Pro May NCU",
}
comparison.GROUPS = {
    "gemini-31-pro-linfo": [
        "/home/ubuntu/AccRL-exps/eval_runs/gemini-31-pro-linfo-mha-bwd-d128",
    ],
    "gemini-31-pro-june": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240",
    ],
    "gemini-31-pro-may-ncu": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0506-1130",
    ],
}
comparison.RUN_DEFINITION_OVERRIDES = {
    "gemini-31-pro-linfo-mha-bwd-d128": "mha_bwd_d128",
    "2026-0609-2240": "mha_bwd_d128",
    "2026-0506-1130": "mha_bwd_d128",
}
comparison.PLAN_PATH_OVERRIDES = {}
comparison.PARQUET_STAGES = set()
comparison.SKIP_MISSING_TURN_CSV = False
comparison.DATA_NOTES = [
    "All three runs use `gemini/gemini-3.1-pro-preview`, four replicas per selected prompt tag, and eight turns per trajectory.",
    "The paired slice is the four tags shared by all runs: `hopper-010`, `hopper-011`, `hopper-012`, and `hopper-013` (16 trajectories and 128 turns per condition).",
    "The June run's additional `hopper-no-hint` trajectories remain in `holistic_turns.csv` with `is_comparison_pair=0` and are excluded from comparison metrics and diagrams.",
    "The May task name is the legacy alias `mha_bwd_h48_d128`; its tensor shapes and operation match the canonical `mha_bwd_d128` task used by the other two runs.",
]
comparison.PROMPT_CONFIG_BY_DEFINITION = {
    "mha_bwd_d128": Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/2026-0504-fa3-bwd.json"
    ),
}
comparison.UNPATCHED_PROMPT_TAG_STAGES = set(comparison.STAGES)
comparison.REQUIRE_FULL_CONFIG_PROMPT_TAG_STAGES = set(comparison.STAGES)
comparison.PROMPT_RULE_HEADING = "Shared Prompt-Tag Rules"
comparison.PROMPT_RULE_POLICY = (
    "All three conditions must cover all four tags in the selected config. "
    "Matching is exact by definition, prompt tag, replica, and turn."
)


def condition_audit() -> list[dict[str, object]]:
    rows = []
    selected_tags = comparison.configured_prompt_tags()["mha_bwd_d128"]
    for group in comparison.STAGES:
        run_dir = Path(comparison.GROUPS[group][0])
        plan = comparison.load_plan(run_dir)
        paths = [
            run_dir / "trajectories" / f"exp_{int(item['exp_index']):03d}.json"
            for item in plan
            if item.get("prompt_tag") in selected_tags
        ]
        feedback_lengths = []
        system_lengths = []
        system_prompts = set()
        primary_diagnostics = 0
        ncu_feedback = 0
        first = None
        for path in paths:
            trajectory = json.loads(path.read_text())
            first = first or trajectory
            system = trajectory["messages"][0]["content"]
            system_lengths.append(len(system))
            system_prompts.add(system)
            for message in trajectory.get("messages", [])[2:]:
                if message.get("role") != "user":
                    continue
                content = str(message.get("content") or "")
                feedback_lengths.append(len(content))
                primary_diagnostics += "Primary diagnostics" in content
                ncu_feedback += "NCU" in content or "ncu" in content
        assert first is not None
        config = first["info"]["config"]
        system = first["messages"][0]["content"]
        rows.append(
            {
                "group": group,
                "model": config["model"]["model_name"],
                "mean_system_chars": round(statistics.fmean(system_lengths)),
                "distinct_system_prompts": len(system_prompts),
                "mean_feedback_chars": round(statistics.fmean(feedback_lengths)),
                "primary_diagnostics": primary_diagnostics,
                "ncu_feedback": ncu_feedback,
                "n_feedback": len(feedback_lengths),
            }
        )
    return rows


def append_interpretation() -> None:
    all_rows = comparison.collect_rows()
    selected_pairs = comparison.shared_pairs(all_rows)
    selected = [
        row
        for row in all_rows
        if (row["definition"], row["prompt_tag_match"]) in selected_pairs
    ]
    overall = {
        row["group"]: row for row in comparison.summarize(selected, ["group"])
    }
    matched = comparison.matched_turn_rows(selected)

    def final_metric(group: str, field: str) -> str:
        return str(overall[group][field]).split(" / ")[-1]

    correctness_order = sorted(
        comparison.STAGES,
        key=lambda group: float(final_metric(group, "correctness_rate")),
        reverse=True,
    )
    trajectory_order = sorted(
        comparison.STAGES,
        key=lambda group: float(final_metric(group, "trajectory_correctness_rate")),
        reverse=True,
    )

    def ranking(order: list[str], field: str) -> str:
        return ", ".join(
            f"`{group}` {final_metric(group, field)}" for group in order
        )

    lines = [
        "",
        "## Interpretation",
        "",
        f"- The paired comparison contains {len(matched)} exact tag/replica/turn positions per condition.",
        f"- At ≤8 turns, turn correctness ranks: {ranking(correctness_order, 'correctness_rate')}.",
        f"- At ≤8 turns, trajectory correctness ranks: {ranking(trajectory_order, 'trajectory_correctness_rate')}.",
        "",
        "### Pairwise correctness disagreements",
        "",
        "| left | right | both correct | left only | right only | neither |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for left, right in combinations(comparison.STAGES, 2):
        left_key = f"{left.replace('-', '_')}_correctness"
        right_key = f"{right.replace('-', '_')}_correctness"
        both = sum(
            row[left_key] == "Correct" and row[right_key] == "Correct"
            for row in matched
        )
        left_only = sum(
            row[left_key] == "Correct" and row[right_key] != "Correct"
            for row in matched
        )
        right_only = sum(
            row[left_key] != "Correct" and row[right_key] == "Correct"
            for row in matched
        )
        neither = len(matched) - both - left_only - right_only
        lines.append(
            f"| {left} | {right} | {both} | {left_only} | {right_only} | {neither} |"
        )

    lines.extend(
        [
            "",
            "### Condition audit",
            "",
            "| group | model | mean system chars | distinct system prompts | mean feedback chars | primary-diagnostic turns | NCU-feedback turns |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    audits = condition_audit()
    for row in audits:
        lines.append(
            f"| {row['group']} | `{row['model']}` | {row['mean_system_chars']} | "
            f"{row['distinct_system_prompts']} | {row['mean_feedback_chars']} | "
            f"{row['primary_diagnostics']}/{row['n_feedback']} | "
            f"{row['ncu_feedback']}/{row['n_feedback']} |"
        )
    lines.extend(
        [
            "",
            "All three conditions use the same model ID, but their system prompts and feedback stacks differ. "
            "The result is therefore a condition-level comparison, not a clean single-variable ablation.",
        ]
    )

    summary_path = comparison.OUT_DIR / "summary.md"
    summary_path.write_text(
        summary_path.read_text().rstrip() + "\n" + "\n".join(lines) + "\n"
    )


def main() -> None:
    comparison.main()
    append_interpretation()
    render(
        comparison.OUT_DIR / "summary.md",
        comparison.OUT_DIR / "presentation_tables.svg",
        comparison.REPORT_TITLE,
        set(),
    )
    print(f"Wrote {comparison.OUT_DIR / 'presentation_tables.svg'}")


if __name__ == "__main__":
    main()
