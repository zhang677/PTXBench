#!/usr/bin/env python3
"""Plot matched reasoning-token means for regular-patched vs linfo-singleuser."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BASE = Path(__file__).resolve().parent
COMPARISON_DIR = BASE / "compare_20260624_0939_five_modes"
INPUT_CSV = COMPARISON_DIR / "holistic_turns.csv"
OUTPUT_STEM = "mean_reasoning_tokens_by_turn_regular_patched_vs_linfo_singleuser"

GROUPS = ("regular-patched", "linfo-singleuser")
GROUP_LABELS = {
    "regular-patched": "Regular patched",
    "linfo-singleuser": "Linfo single-user",
}
GROUP_LINESTYLES = {
    "regular-patched": (0, (5, 3)),
    "linfo-singleuser": "solid",
}
DEFINITION_LABELS = {
    "mha_with_lse_d128": "MHA fwd d128",
    "mha_with_lse_d128_causal": "MHA fwd d128 causal",
    "mha_bwd_d128": "MHA bwd d128",
    "mha_bwd_d128_causal": "MHA bwd d128 causal",
}
DEFINITION_COLORS = {
    "mha_with_lse_d128": "#2f6f9f",
    "mha_with_lse_d128_causal": "#c5523c",
    "mha_bwd_d128": "#5f8f3a",
    "mha_bwd_d128_causal": "#7a5aa6",
}


def nested_get(value: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def assistant_eval_turns(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return assistant kernel responses that are followed by evaluator feedback."""
    turns: list[dict[str, Any]] = []
    saw_initial_user = False
    pending_assistant: dict[str, Any] | None = None

    for message in trajectory.get("messages", []):
        role = message.get("role")
        if role == "user":
            if not saw_initial_user:
                saw_initial_user = True
                continue
            if pending_assistant is not None:
                turns.append(pending_assistant)
                pending_assistant = None
        elif role == "assistant" and saw_initial_user:
            pending_assistant = message

    return turns


def reasoning_tokens(message: dict[str, Any]) -> int | None:
    usage = nested_get(message, ("extra", "response", "usage"))
    if not isinstance(usage, dict):
        return None

    candidates = (
        nested_get(usage, ("completion_tokens_details", "reasoning_tokens")),
        nested_get(usage, ("completion_tokens_details", "thinking_tokens")),
        nested_get(usage, ("output_tokens_details", "reasoning_tokens")),
        nested_get(usage, ("output_tokens_details", "thinking_tokens")),
        usage.get("reasoning_tokens"),
        usage.get("thinking_tokens"),
    )
    for candidate in candidates:
        parsed = as_int(candidate)
        if parsed is not None:
            return parsed
    return None


def read_matched_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "group",
            "run_dir",
            "definition",
            "prompt_tag",
            "replica",
            "trajectory_id",
            "turn",
            "is_matched_turn",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{INPUT_CSV} missing columns: {', '.join(sorted(missing))}")
        return [
            row
            for row in reader
            if row["group"] in GROUPS and row["is_matched_turn"] == "1"
        ]


def collect_reasoning_rows(matched_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    trajectory_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    output: list[dict[str, Any]] = []

    for row in matched_rows:
        cache_key = (row["run_dir"], row["trajectory_id"])
        if cache_key not in trajectory_cache:
            trajectory_path = (
                Path(row["run_dir"]) / "trajectories" / f"{row['trajectory_id']}.json"
            )
            trajectory = json.loads(trajectory_path.read_text())
            trajectory_cache[cache_key] = assistant_eval_turns(trajectory)

        turn = int(row["turn"])
        turns = trajectory_cache[cache_key]
        if turn >= len(turns):
            raise ValueError(f"Missing assistant turn {turn} for {cache_key}")
        tokens = reasoning_tokens(turns[turn])
        if tokens is None:
            raise ValueError(f"Missing reasoning-token counter for turn {turn} in {cache_key}")

        output.append(
            {
                "definition": row["definition"],
                "prompt_tag": row["prompt_tag"],
                "replica": int(row["replica"]),
                "turn": turn,
                "group": row["group"],
                "run_dir": row["run_dir"],
                "trajectory_id": row["trajectory_id"],
                "reasoning_tokens": tokens,
            }
        )
    return output


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["definition"], row["turn"], row["group"])].append(
            row["reasoning_tokens"]
        )

    output: list[dict[str, Any]] = []
    for (definition, turn, group), values in sorted(grouped.items()):
        output.append(
            {
                "definition": definition,
                "definition_label": DEFINITION_LABELS.get(definition, definition),
                "turn": turn,
                "group": group,
                "group_label": GROUP_LABELS[group],
                "rows": len(values),
                "mean_reasoning_tokens": f"{mean(values):.6f}",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    turns = sorted({int(row["turn"]) for row in rows})
    values = {
        (row["definition"], row["group"], int(row["turn"])): float(
            row["mean_reasoning_tokens"]
        )
        for row in rows
    }

    figure, axis = plt.subplots(figsize=(10.5, 6.0), dpi=160)
    for definition in DEFINITION_LABELS:
        for group in GROUPS:
            axis.plot(
                turns,
                [values.get((definition, group, turn)) for turn in turns],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=DEFINITION_COLORS[definition],
                linestyle=GROUP_LINESTYLES[group],
                label=f"{DEFINITION_LABELS[definition]} - {GROUP_LABELS[group]}",
            )

    axis.set_title(
        "Mean Reasoning Tokens by Turn: Regular Patched vs Linfo Single-user"
    )
    axis.set_xlabel("Turn")
    axis.set_ylabel("Mean reasoning tokens")
    axis.set_xticks(turns)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    matched_rows = read_matched_rows()
    expected_rows = 2 * 32 * 8
    if len(matched_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} matched rows, found {len(matched_rows)}")

    reasoning_rows = collect_reasoning_rows(matched_rows)
    aggregates = aggregate_rows(reasoning_rows)
    if len(aggregates) != 4 * 8 * 2:
        raise ValueError(f"Expected 64 aggregate rows, found {len(aggregates)}")

    aggregate_path = COMPARISON_DIR / f"{OUTPUT_STEM}.csv"
    plot_path = COMPARISON_DIR / f"{OUTPUT_STEM}.png"
    write_csv(aggregate_path, aggregates)
    make_plot(aggregates, plot_path)

    print(f"Matched reasoning rows: {len(reasoning_rows)}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
