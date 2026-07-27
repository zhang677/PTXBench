#!/usr/bin/env python3
"""Scan a completed multiturn_v2 run for experiments that hit 'No such container'
mid-run and emit a JSON list of entries suitable for rerun_failed_experiments.py.

When the Docker container for an experiment dies (e.g. --container-timeout
expired while the agent was still iterating), subsequent turns fail with:

    test.py failed (returncode 1):
    Error response from daemon: No such container: <id>

The agent keeps polling the LLM until its step limit is hit, so the run exits
cleanly and summary.json marks it "success" — but the last few turns produced
no real evaluation. This script detects the first such dead turn per experiment
by scanning trajectories/exp_*.json, and joins with plan.json to recover the
original config parameters needed for a resume rerun.

Output schema (JSON array):
    {
        "base_path":      "<absolute path to the run dir>",
        "exp_id":         "exp_NNN",
        "starting_turn":  <first dead turn, 0-indexed>,
        "num_turns":      <original max_turns from plan.json>,
        "target_speedup": <original target_speedup>,
        "prompt_tag":     <original prompt_tag>
    }

Usage:
    python scan_container_kills.py \\
        --base-path /home/ubuntu/AccRL-exps/eval_runs/2026-0422-2352 \\
        --output    container_kills_0422.json
"""

import argparse
import json
import re
from pathlib import Path

CONTAINER_KILLED_SIGNATURE = "No such container"
EXP_NAME_RE = re.compile(r"^exp_(\d{3,})$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-path", required=True,
                        help="Path to a completed multiturn_v2 run directory "
                             "(contains plan.json, trajectories/, etc.)")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path.")
    return parser.parse_args()


def first_dead_turn(trajectory_path: Path) -> int | None:
    """Return the smallest turn index whose observation contains the container-killed
    signature, or None if the trajectory never hit it.

    Message layout: [system, user, (assistant, observation) * N, optional exit].
    Observation for turn k is messages[3 + 2*k].
    """
    with open(trajectory_path) as f:
        data = json.load(f)
    messages = data.get("messages", [])
    k = 0
    while True:
        idx = 3 + 2 * k
        if idx >= len(messages):
            return None
        msg = messages[idx]
        if msg.get("role") != "user":
            # Reached the exit marker or a malformed trajectory — stop looking.
            return None
        content = msg.get("content") or ""
        if CONTAINER_KILLED_SIGNATURE in content:
            return k
        k += 1


def load_plan(base_path: Path) -> dict[int, dict]:
    """Return exp_index -> {num_turns, target_speedup, prompt_tag}."""
    plan_path = base_path / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"{plan_path} not found")
    with open(plan_path) as f:
        data = json.load(f)
    plan = data.get("plan") if isinstance(data, dict) else data
    by_index = {}
    for item in plan:
        by_index[int(item["exp_index"])] = {
            "num_turns": int(item["num_turns"]),
            "target_speedup": float(item["target_speedup"]),
            "prompt_tag": item["prompt_tag"],
        }
    return by_index


def main():
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    if not base_path.is_dir():
        raise SystemExit(f"Base path {base_path} is not a directory.")

    traj_dir = base_path / "trajectories"
    if not traj_dir.is_dir():
        raise SystemExit(f"No trajectories/ under {base_path}.")

    plan_by_index = load_plan(base_path)

    entries = []
    scanned = 0
    turn_histogram: dict[int, int] = {}

    for traj_path in sorted(traj_dir.glob("exp_*.json")):
        m = EXP_NAME_RE.match(traj_path.stem)
        if not m:
            continue
        scanned += 1
        exp_index = int(m.group(1))

        try:
            turn = first_dead_turn(traj_path)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! Failed to parse {traj_path.name}: {e}")
            continue
        if turn is None:
            continue

        plan_entry = plan_by_index.get(exp_index)
        if plan_entry is None:
            print(f"  ! {traj_path.name} has no matching plan.json entry; skipping")
            continue

        entries.append({
            "base_path": str(base_path),
            "exp_id": traj_path.stem,
            "starting_turn": turn,
            "num_turns": plan_entry["num_turns"],
            "target_speedup": plan_entry["target_speedup"],
            "prompt_tag": plan_entry["prompt_tag"],
        })
        turn_histogram[turn] = turn_histogram.get(turn, 0) + 1

    entries.sort(key=lambda e: int(e["exp_id"].split("_")[1]))

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Scanned {scanned} trajectories under {base_path}")
    print(f"Container-killed experiments: {len(entries)}")
    if turn_histogram:
        print("Starting-turn distribution:")
        for t in sorted(turn_histogram):
            print(f"  turn {t}: {turn_histogram[t]} experiment(s)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
