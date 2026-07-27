#!/usr/bin/env python3
"""Extract per-turn records from eval trajectories.

Usage:
    python -m accrl.distill.extract_turns \
        ~/work/AccRL-exps/eval_runs/2026-0422-1002 \
        ~/work/AccRL-exps/distill/gemini_turns_0422.jsonl
"""

import argparse
import json
from pathlib import Path

from accrl.distill.utils import extract_eval_dir


def main():
    parser = argparse.ArgumentParser(description="Extract per-turn records from eval trajectories")
    parser.add_argument("eval_dir", type=Path, help="Path to eval run directory (must contain trajectories/)")
    parser.add_argument("output", type=Path, nargs="?", default=None,
                        help="Output JSONL path (default: <eval_dir>/turns.jsonl)")
    args = parser.parse_args()

    output = args.output or args.eval_dir / "turns.jsonl"

    turns = extract_eval_dir(args.eval_dir)
    if not turns:
        print(f"No turns extracted from {args.eval_dir}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")

    # Stats
    by_exp = {}
    for t in turns:
        by_exp.setdefault(t["exp_id"], []).append(t)

    n_code = sum(1 for t in turns if t.get("kernel_code"))
    n_pass = sum(1 for t in turns if t.get("passed"))
    defs = set(t["definition_name"] for t in turns)

    print(f"{len(turns)} turns from {len(by_exp)} experiments → {output}")
    print(f"  with kernel_code: {n_code}")
    print(f"  passed: {n_pass}")
    print(f"  definitions: {', '.join(sorted(defs))}")


if __name__ == "__main__":
    main()
