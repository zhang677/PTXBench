#!/usr/bin/env python3
"""Extract the generated kernel at each turn from multiturn experiment trajectories.

For each experiment in the run directory, reads the trajectory JSON and extracts
the kernel code from each assistant message's ```cpp code block (mirroring
KernelAgent.step() in fib_runtime/multiturn/common.py). Writes each kernel to:

    <run-dir>/kernels/<exp_name>/kernel_t<turn>.cu
    <run-dir>/kernels/<exp_name>/log_t<turn>.txt    (profile/evaluation output)

Usage:
    python analyze_kernel_per_turn.py --run-dir /path/to/eval_runs/run_name

    # Limit extraction to specific experiments from the run directory.
    python analyze_kernel_per_turn.py --run-dir /path/to/eval_runs/run_name \
        --run-exps 19,exp_023,030-032,exp_040-exp_041
"""

import argparse
import json
import sys
from pathlib import Path

# Add the mini-ptx-agent package root so we can import accrl.utils.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from accrl.utils.code_utils import extract_code_block


def normalize_exp_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty experiment id")
    if value.startswith("exp_"):
        suffix = value[4:]
    else:
        suffix = value
    if not suffix.isdigit():
        raise ValueError(f"invalid experiment id {value!r}; expected N, NNN, or exp_NNN")
    return f"exp_{int(suffix):03d}"


def parse_run_exps(value: str | None) -> set[str] | None:
    """Parse comma-separated experiment ids/ranges into normalized exp_NNN names."""
    if value is None:
        return None

    selected: set[str] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" not in part:
            selected.add(normalize_exp_name(part))
            continue

        start_raw, end_raw = part.split("-", 1)
        start = int(normalize_exp_name(start_raw)[4:])
        end = int(normalize_exp_name(end_raw)[4:])
        if end < start:
            raise ValueError(f"invalid experiment range {part!r}; end is before start")
        selected.update(f"exp_{idx:03d}" for idx in range(start, end + 1))

    if not selected:
        raise ValueError("--run-exps did not select any experiments")
    return selected


def extract_turns_from_trajectory(trajectory: dict) -> list[tuple[int, str, str]]:
    """Extract kernel code and observation from each turn in a trajectory.

    Returns a list of (turn_index, kernel_source, observation) tuples.
    Turns where no ```cpp block is found are skipped.
    The observation is the user message content immediately following the
    assistant message (the profile/evaluation output the agent received).
    """
    msgs = trajectory.get("messages", [])
    turns = []
    turn = 0
    for i, msg in enumerate(msgs):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        kernel_code = extract_code_block(content, languages=["cpp"], keep_separators=False)
        if kernel_code:
            # The next message is the user observation for this turn
            observation = ""
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
                observation = msgs[i + 1].get("content", "") or ""
            turns.append((turn, kernel_code, observation))
        turn += 1
    return turns


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-turn kernels from multiturn experiment trajectories",
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the eval run directory (contains trajectories/ subdirectory)",
    )
    parser.add_argument(
        "--run-exps",
        default=None,
        help=(
            "Optional comma-separated experiment filter, e.g. "
            "'19,exp_023,exp_030-exp_035'. Accepts N, NNN, or exp_NNN."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    try:
        selected_exps = parse_run_exps(args.run_exps)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    traj_dir = run_dir / "trajectories"
    if not traj_dir.is_dir():
        print(f"Error: {traj_dir} does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)

    kernel_root = run_dir / "kernels"
    kernel_root.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(traj_dir.glob("exp_*.json"))
    if selected_exps is not None:
        traj_files = [path for path in traj_files if path.stem in selected_exps]
        missing = sorted(selected_exps - {path.stem for path in traj_files})
        if missing:
            print(
                f"Warning: selected experiment(s) not found in {traj_dir}: {', '.join(missing)}",
                file=sys.stderr,
            )
    if not traj_files:
        print(f"No trajectory files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)

    total_experiments = 0
    total_kernels = 0

    for traj_path in traj_files:
        exp_name = traj_path.stem  # e.g. "exp_000"
        try:
            trajectory = json.loads(traj_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: failed to read {traj_path}: {e}", file=sys.stderr)
            continue

        turns = extract_turns_from_trajectory(trajectory)
        if not turns:
            continue

        exp_kernel_dir = kernel_root / exp_name
        exp_kernel_dir.mkdir(parents=True, exist_ok=True)

        for turn, kernel_source, observation in turns:
            (exp_kernel_dir / f"kernel_t{turn}.cu").write_text(kernel_source)
            (exp_kernel_dir / f"log_t{turn}.txt").write_text(observation)

        total_experiments += 1
        total_kernels += len(turns)
        print(f"  {exp_name}: {len(turns)} kernels extracted")

    print(f"\nDone: {total_experiments} experiments, {total_kernels} kernels total")
    print(f"Output: {kernel_root}")


if __name__ == "__main__":
    main()
