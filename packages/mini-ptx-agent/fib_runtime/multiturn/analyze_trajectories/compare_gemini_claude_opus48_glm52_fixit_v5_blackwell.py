#!/usr/bin/env python3
"""Compare Gemini, Claude Opus 4.8, GLM-5.2, and Fixit v5 on Blackwell."""

from __future__ import annotations

from pathlib import Path

import compare_fixit_until_v6 as comparison


BASE = Path(__file__).resolve().parent

comparison.OUT_DIR = BASE / "compare_gemini_claude_opus48_glm52_fixit_v5_blackwell"
comparison.REPORT_TITLE = "Blackwell: Gemini vs Claude Opus 4.8 vs GLM-5.2 vs Fixit v5"
comparison.REQUIRED_ARCH_TAG = "B"
comparison.STAGES = [
    "gemini-3.1-pro-preview",
    "claude-opus-4.8-xhigh",
    "glm-5.2",
    "fixit-v5",
]
comparison.STAGE_LABELS = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "claude-opus-4.8-xhigh": "Claude Opus 4.8 xhigh",
    "glm-5.2": "GLM-5.2",
    "fixit-v5": "Fixit v5",
}
comparison.GROUPS = {
    "gemini-3.1-pro-preview": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0529-1900-complete",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0529-1040",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0040",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0240",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0513-1125",
    ],
    "claude-opus-4.8-xhigh": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0530-1540-complete",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0530-1740",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0530-1940",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0530-2140",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0601-2320",
    ],
    "glm-5.2": [
        "/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-bwd-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-gemm",
    ],
    "fixit-v5": [
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-bwd-d128",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-bwd-d128-causal",
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-gemm",
    ],
}
comparison.RUN_DEFINITION_OVERRIDES = {
    "2026-0513-1125": "gemm_n7168_k5120",
    "2026-0529-1900-complete": "mha_with_lse_d128",
    "2026-0529-1040": "mha_with_lse_d128_causal",
    "2026-0530-0040": "mha_bwd_d128",
    "2026-0530-0240": "mha_bwd_d128_causal",
    "2026-0601-2320": "gemm_n7168_k5120",
    "2026-0530-1540-complete": "mha_with_lse_d128",
    "2026-0530-1740": "mha_with_lse_d128_causal",
    "2026-0530-1940": "mha_bwd_d128",
    "2026-0530-2140": "mha_bwd_d128_causal",
}
comparison.PLAN_PATH_OVERRIDES = {
    "2026-0529-1900-complete": Path(
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0529-1900/plan.json"
    ),
    "2026-0530-1540-complete": Path(
        "/home/ubuntu/AccRL-exps/eval_runs/2026-0530-1540/plan.json"
    ),
}
comparison.PARQUET_STAGES = {"fixit-v5"}
comparison.SKIP_MISSING_TURN_CSV = False
comparison.DATA_NOTES = [
    "No Claude Opus 4.7 Blackwell artifacts exist in the repository; the available registered family is Claude Opus 4.8 xhigh (`anthropic/claude-opus-4-8`).",
    "The 15 comparison pairs come from the B200 GEMM, forward-MHA, and backward-MHA prompt configs. Gemini and Claude may omit configured tags without reducing GLM-5.2 or Fixit v5 coverage.",
    "Rows retain their original turn numbers and trajectory budgets are not equalized across groups.",
]
comparison.PROMPT_CONFIG_BY_DEFINITION = {
    "gemm_n7168_k5120": Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/b200-gemm-3-r8-p4.json"
    ),
    "mha_with_lse_d128": Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/b200-mha-3-r8-p4.json"
    ),
    "mha_with_lse_d128_causal": Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/b200-mha-3-r8-p4.json"
    ),
    "mha_bwd_d128": Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/b200-mha-bwd-3-r8-p4.json"
    ),
    "mha_bwd_d128_causal": Path(
        "/home/ubuntu/AccRL-exps/prompt_configs/b200-mha-bwd-3-r8-p4.json"
    ),
}
comparison.UNPATCHED_PROMPT_TAG_STAGES = set(comparison.STAGES)
comparison.REQUIRE_FULL_CONFIG_PROMPT_TAG_STAGES = {"glm-5.2", "fixit-v5"}
comparison.PROMPT_RULE_HEADING = "Blackwell Prompt-Tag Rules"
comparison.PROMPT_RULE_POLICY = (
    "GLM-5.2 and Fixit v5 must cover all configured B200 tags. Gemini and Claude "
    "use the same raw, unpatched tag names and may omit configured tags; their "
    "omissions do not reduce GLM-5.2 or Fixit v5 coverage."
)


def append_interpretation() -> None:
    all_rows = comparison.collect_rows()
    shared = comparison.shared_pairs(all_rows)
    rows = [
        row
        for row in all_rows
        if (row["definition"], row["prompt_tag_match"]) in shared
    ]
    overall = comparison.summarize(rows, ["group"])
    by_group = {row["group"]: row for row in overall}

    def last_rate(group: str, field: str) -> float:
        return float(str(by_group[group][field]).split(" / ")[-1])

    correctness_order = sorted(
        comparison.STAGES,
        key=lambda group: last_rate(group, "correctness_rate"),
        reverse=True,
    )
    instruction_order = sorted(
        comparison.STAGES,
        key=lambda group: last_rate(group, "correct_and_use_instruction_rate"),
        reverse=True,
    )
    trajectory_correctness_order = sorted(
        comparison.STAGES,
        key=lambda group: last_rate(group, "trajectory_correctness_rate"),
        reverse=True,
    )
    trajectory_instruction_order = sorted(
        comparison.STAGES,
        key=lambda group: last_rate(
            group, "trajectory_correct_and_use_instruction_rate"
        ),
        reverse=True,
    )

    def ranking(order: list[str], field: str) -> str:
        parts = []
        for group in order:
            value = str(by_group[group][field]).split(" / ")[-1]
            parts.append(f"`{group}` {value}")
        return ", ".join(parts)

    lines = [
        "",
        "## Interpretation",
        "",
        f"- Configured comparison coverage is {len(shared)} definition/prompt-tag pairs across {len({row['definition'] for row in rows})} definitions.",
        f"- At ≤8 turns, correctness ranks: {ranking(correctness_order, 'correctness_rate')}.",
        f"- At ≤8 turns, correct-and-Blackwell-instruction use ranks: {ranking(instruction_order, 'correct_and_use_instruction_rate')}.",
        f"- At ≤8 turns, trajectory correctness ranks: {ranking(trajectory_correctness_order, 'trajectory_correctness_rate')}.",
        f"- At ≤8 turns, trajectory correct-and-Blackwell-instruction use ranks: {ranking(trajectory_instruction_order, 'trajectory_correct_and_use_instruction_rate')}.",
        "- Treat these rankings as descriptive rather than fully paired because Gemini and Claude use smaller trajectory budgets.",
    ]
    summary_path = comparison.OUT_DIR / "summary.md"
    summary_path.write_text(summary_path.read_text().rstrip() + "\n" + "\n".join(lines) + "\n")


def main() -> None:
    comparison.main()
    append_interpretation()


if __name__ == "__main__":
    main()
