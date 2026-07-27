#!/usr/bin/env python3
"""Scan a multiturn_v2 run for experiments with fewer turns than plan.json.

The output is a JSON list suitable for rerun_failed_experiments.py. Each entry
uses the number of completed assistant/evaluation pairs as `starting_turn`.
Partially completed runs resume at the first missing turn while preserving prior
context; runs that never started are emitted with `starting_turn: 0` so the
launcher can start them fresh.

Output schema:
    {
        "base_path":      "<absolute path to the run dir>",
        "exp_id":         "exp_NNN",
        "starting_turn":  <number of completed turns>,
        "num_turns":      <planned max turns from plan.json>,
        "target_speedup": <planned target_speedup>,
        "prompt_tag":     <planned prompt_tag>,
        "actual_turns":   <number of completed turns>,
        "planned_turns":  <planned max turns from plan.json>,
        "source_status":  "partial" | "fresh_trajectory" |
                          "missing_trajectory" | "empty_trajectory"
    }

Usage:
    python scan_incomplete_turns.py \\
        --base-path /home/ubuntu/AccRL-exps/eval_runs/2026-0525-0845 \\
        --output /home/ubuntu/AccRL-exps/eval_runs/2026-0525-0845/incomplete_turns.json
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--base-path",
        required=True,
        help="Path to a multiturn_v2 run directory containing plan.json and trajectories/.",
    )
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    return parser.parse_args()


def load_plan(base_path: Path) -> dict[int, dict]:
    """Return exp_index -> {num_turns, target_speedup, prompt_tag}."""
    plan_path = base_path / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"{plan_path} not found")
    with open(plan_path) as f:
        data = json.load(f)
    plan = data.get("plan") if isinstance(data, dict) else data
    if not isinstance(plan, list):
        raise ValueError(f"{plan_path} must contain a list or a dict with a 'plan' list")

    by_index: dict[int, dict] = {}
    for item in plan:
        exp_index = int(item["exp_index"])
        by_index[exp_index] = {
            "num_turns": int(item["num_turns"]),
            "target_speedup": float(item["target_speedup"]),
            "prompt_tag": item["prompt_tag"],
        }
    return by_index


def count_completed_turns(trajectory_path: Path) -> int:
    """Count complete assistant/evaluation turn pairs in a trajectory.

    Message layout is [system, user, (assistant, observation) * N, optional exit].
    Observations are stored as user-role messages by the mini-swe-agent model
    formatter. A trailing assistant message without its observation is not a
    completed turn and is not included.
    """
    with open(trajectory_path) as f:
        data = json.load(f)
    messages = data.get("messages", [])

    turns = 0
    while True:
        assistant_idx = 2 + 2 * turns
        observation_idx = assistant_idx + 1
        if observation_idx >= len(messages):
            return turns
        if messages[assistant_idx].get("role") != "assistant":
            return turns
        if messages[observation_idx].get("role") != "user":
            return turns
        turns += 1


def add_entry(
    entries: list[dict],
    *,
    base_path: Path,
    exp_id: str,
    plan_entry: dict,
    actual_turns: int,
    source_status: str,
) -> None:
    planned_turns = plan_entry["num_turns"]
    entries.append({
        "base_path": str(base_path),
        "exp_id": exp_id,
        "starting_turn": actual_turns,
        "num_turns": planned_turns,
        "target_speedup": plan_entry["target_speedup"],
        "prompt_tag": plan_entry["prompt_tag"],
        "actual_turns": actual_turns,
        "planned_turns": planned_turns,
        "source_status": source_status,
    })


def is_empty_file(path: Path) -> bool:
    try:
        return path.read_text().strip() == ""
    except OSError:
        return False


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    if not base_path.is_dir():
        raise SystemExit(f"Base path {base_path} is not a directory.")

    traj_dir = base_path / "trajectories"
    if not traj_dir.is_dir():
        raise SystemExit(f"No trajectories/ under {base_path}.")

    plan_by_index = load_plan(base_path)

    entries: list[dict] = []
    fresh_starts: list[str] = []
    malformed: list[str] = []
    turn_histogram: dict[int, int] = {}

    for exp_index in sorted(plan_by_index):
        plan_entry = plan_by_index[exp_index]
        exp_id = f"exp_{exp_index:03d}"
        traj_path = traj_dir / f"{exp_id}.json"

        if not traj_path.exists():
            add_entry(
                entries,
                base_path=base_path,
                exp_id=exp_id,
                plan_entry=plan_entry,
                actual_turns=0,
                source_status="missing_trajectory",
            )
            fresh_starts.append(exp_id)
            turn_histogram[0] = turn_histogram.get(0, 0) + 1
            continue

        try:
            actual_turns = count_completed_turns(traj_path)
        except (json.JSONDecodeError, OSError) as e:
            if is_empty_file(traj_path):
                add_entry(
                    entries,
                    base_path=base_path,
                    exp_id=exp_id,
                    plan_entry=plan_entry,
                    actual_turns=0,
                    source_status="empty_trajectory",
                )
                fresh_starts.append(exp_id)
                turn_histogram[0] = turn_histogram.get(0, 0) + 1
                continue
            malformed.append(f"{exp_id}: {e}")
            continue

        planned_turns = plan_entry["num_turns"]
        if actual_turns >= planned_turns:
            continue

        source_status = "fresh_trajectory" if actual_turns == 0 else "partial"
        add_entry(
            entries,
            base_path=base_path,
            exp_id=exp_id,
            plan_entry=plan_entry,
            actual_turns=actual_turns,
            source_status=source_status,
        )
        if actual_turns == 0:
            fresh_starts.append(exp_id)
        turn_histogram[actual_turns] = turn_histogram.get(actual_turns, 0) + 1

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Scanned {len(plan_by_index)} planned experiments under {base_path}")
    print(f"Incomplete experiments: {len(entries)}")
    if turn_histogram:
        print("Starting-turn distribution:")
        for turn in sorted(turn_histogram):
            print(f"  turn {turn}: {turn_histogram[turn]} experiment(s)")
    if fresh_starts:
        print(f"Fresh-start experiments: {len(fresh_starts)}")
        print("  " + ", ".join(fresh_starts))
    if malformed:
        print(f"Malformed trajectories skipped: {len(malformed)}", file=sys.stderr)
        for item in malformed:
            print(f"  {item}", file=sys.stderr)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
