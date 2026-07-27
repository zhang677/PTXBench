#!/usr/bin/env python3
"""Plot stacked bar of distillation output length (reasoning + thinking) per pair.

Mirrors fib_runtime/mini_swe_agent_docker/plots/plot_token_breakdown.py but
operates on distill `reasoning_pairs.jsonl` records:
  - reasoning  ← chars in the extracted <my_reasoning> block (visible output)
  - thinking   ← chars in the raw reasoning_content/reasoning channel (hidden CoT)

When multiple --exp-dirs are given, all figures share the same y range so they
can be compared side-by-side.

Usage:
  python -m accrl.distill.plots.plot_length_breakdown \
    --exp-dirs ../AccRL-exps/distill/experiments/trajectory_reasoning_glm_v3 \
               ../AccRL-exps/distill/experiments/trajectory_reasoning_kimi_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_pairs(exp_dir: Path) -> list[dict]:
    pairs_path = exp_dir / "reasoning_pairs.jsonl"
    return [json.loads(l) for l in pairs_path.open() if l.strip()]


def load_label(exp_dir: Path) -> str:
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        return exp_dir.name
    cfg = yaml.safe_load(cfg_path.read_text())
    model = cfg.get("reasoning_model", "?")
    thinking = cfg.get("enable_thinking", "?")
    return f"{model} (thinking={thinking})"


def per_pair_arrays(pairs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort pairs by (exp_id, turn) and return (reasoning_lens, thinking_lens, x_labels)."""
    sorted_pairs = sorted(
        pairs,
        key=lambda r: (r.get("metadata", {}).get("exp_id", ""),
                       r.get("metadata", {}).get("turn", 0)),
    )
    reas = np.array([len(p.get("reasoning", "")) for p in sorted_pairs])
    think = np.array([len(p.get("thinking", "")) for p in sorted_pairs])
    labels = [
        f"{p['metadata'].get('exp_id','?')}.t{p['metadata'].get('turn','?')}"
        for p in sorted_pairs
    ]
    return reas, think, labels


def per_turn_means(pairs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group pairs by turn, return (turns, mean_reasoning, mean_thinking)."""
    by_turn: dict[int, list[tuple[int, int]]] = {}
    for p in pairs:
        t = p.get("metadata", {}).get("turn", 0)
        by_turn.setdefault(t, []).append(
            (len(p.get("reasoning", "")), len(p.get("thinking", "")))
        )
    turns = np.array(sorted(by_turn))
    mean_reas = np.array([np.mean([r for r, _ in by_turn[t]]) for t in turns])
    mean_think = np.array([np.mean([th for _, th in by_turn[t]]) for t in turns])
    return turns, mean_reas, mean_think


def plot_per_pair(exp_dir: Path, out_dir: Path, label: str,
                  pairs: list[dict], y_max: float):
    reas, think, labels = per_pair_arrays(pairs)
    if reas.size == 0:
        print(f"  (no pairs in {exp_dir})")
        return

    has_think = think.sum() > 0
    fig, ax = plt.subplots(figsize=(max(10, len(reas) * 0.18), 5))
    x = np.arange(len(reas))

    ax.bar(x, reas, width=0.85, color="#66BB6A", alpha=0.85, label="reasoning (visible <my_reasoning>)")
    if has_think:
        ax.bar(x, think, width=0.85, bottom=reas, color="#FF7043", alpha=0.85,
               label="thinking (hidden CoT)")

    ax.set_xlabel("pair (sorted by exp_id, turn)")
    ax.set_ylabel("characters")
    ax.set_title(f"Distill output length per pair — {label}\n{exp_dir.name} (n={len(reas)})")
    ax.set_xticks(x[::max(1, len(reas) // 30)])
    ax.set_xticklabels([labels[i] for i in range(0, len(reas), max(1, len(reas) // 30))],
                       rotation=70, fontsize=7)
    ax.legend(loc="upper right")
    ax.set_ylim(0, y_max * 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = out_dir / "length_breakdown_per_pair.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_per_turn(exp_dir: Path, out_dir: Path, label: str,
                  pairs: list[dict], turn_max: int, y_max: float):
    turns, mean_reas, mean_think = per_turn_means(pairs)
    if turns.size == 0:
        return

    has_think = mean_think.sum() > 0
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(turns, mean_reas, width=0.7, color="#66BB6A", alpha=0.85, label="reasoning")
    if has_think:
        ax.bar(turns, mean_think, width=0.7, bottom=mean_reas, color="#FF7043", alpha=0.85,
               label="thinking")

    counts = {t: sum(1 for p in pairs if p["metadata"].get("turn") == t) for t in turns}
    for t, r, th in zip(turns, mean_reas, mean_think):
        total = r + th
        ax.text(t, total + y_max * 0.01, f"n={counts[t]}", ha="center", fontsize=8, color="dimgray")

    ax.set_xlabel("turn")
    ax.set_ylabel("characters (mean across pairs at this turn)")
    ax.set_title(f"Distill output length by turn — {label}\n{exp_dir.name}")
    ax.set_xticks(np.arange(0, turn_max + 1))
    ax.set_xlim(-0.6, turn_max + 0.6)
    ax.set_ylim(0, y_max * 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = out_dir / "length_breakdown_per_turn.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_comparison(exp_dirs: list[Path], pairs_by_dir: dict, labels_by_dir: dict,
                    out_dir: Path, y_max: float):
    """Single side-by-side per-turn comparison across all runs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_w = 0.8 / len(exp_dirs)
    colors_reas = ["#66BB6A", "#42A5F5", "#AB47BC", "#FFA726"]
    colors_think = ["#FF7043", "#FF9800", "#7E57C2", "#FFEB3B"]

    all_turns = sorted({p["metadata"].get("turn", 0) for d in exp_dirs for p in pairs_by_dir[d]})

    for i, d in enumerate(exp_dirs):
        turns, mean_reas, mean_think = per_turn_means(pairs_by_dir[d])
        # Pad to all_turns
        reas_full = np.zeros(len(all_turns)); think_full = np.zeros(len(all_turns))
        for j, t in enumerate(all_turns):
            for k, tk in enumerate(turns):
                if tk == t:
                    reas_full[j] = mean_reas[k]; think_full[j] = mean_think[k]
        x_offset = (i - (len(exp_dirs) - 1) / 2) * bar_w
        ax.bar(np.array(all_turns) + x_offset, reas_full, width=bar_w * 0.9,
               color=colors_reas[i % 4], alpha=0.85,
               label=f"{labels_by_dir[d]} — reasoning")
        if think_full.sum() > 0:
            ax.bar(np.array(all_turns) + x_offset, think_full, width=bar_w * 0.9,
                   bottom=reas_full, color=colors_think[i % 4], alpha=0.85,
                   label=f"{labels_by_dir[d]} — thinking")

    ax.set_xlabel("turn")
    ax.set_ylabel("characters (mean)")
    ax.set_title("Distill output length by turn — comparison")
    ax.set_xticks(all_turns)
    ax.set_ylim(0, y_max * 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = out_dir / "length_breakdown_comparison_per_turn.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dirs", type=str, nargs="+", required=True,
                        help="Path(s) to distill experiment dirs (must contain reasoning_pairs.jsonl)")
    parser.add_argument("--comparison-out-dir", type=str, default=None,
                        help="Where to save the side-by-side comparison plot. "
                             "Default: parent of first exp_dir / 'figures'.")
    args = parser.parse_args()

    exp_dirs = [Path(d) for d in args.exp_dirs]

    # Load all pairs and labels
    pairs_by_dir = {d: load_pairs(d) for d in exp_dirs}
    labels_by_dir = {d: load_label(d) for d in exp_dirs}

    # Compute global y_max: max stacked total across all runs
    y_max_per_pair = 0
    y_max_per_turn = 0
    turn_max = 0
    for d in exp_dirs:
        reas, think, _ = per_pair_arrays(pairs_by_dir[d])
        y_max_per_pair = max(y_max_per_pair, (reas + think).max() if reas.size else 0)
        turns, mr, mt = per_turn_means(pairs_by_dir[d])
        y_max_per_turn = max(y_max_per_turn, (mr + mt).max() if turns.size else 0)
        turn_max = max(turn_max, int(turns.max()) if turns.size else 0)

    print(f"global y_max: per_pair={y_max_per_pair:.0f} per_turn={y_max_per_turn:.0f}")

    # Per-experiment plots in each exp's figures/
    for d in exp_dirs:
        out_dir = d / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"plotting {d.name} ({len(pairs_by_dir[d])} pairs):")
        plot_per_pair(d, out_dir, labels_by_dir[d], pairs_by_dir[d], y_max_per_pair)
        plot_per_turn(d, out_dir, labels_by_dir[d], pairs_by_dir[d], turn_max, y_max_per_turn)

    # Comparison plot
    if len(exp_dirs) > 1:
        comp_out = Path(args.comparison_out_dir) if args.comparison_out_dir \
                   else exp_dirs[0].parent / "figures"
        comp_out.mkdir(parents=True, exist_ok=True)
        print(f"plotting comparison:")
        plot_comparison(exp_dirs, pairs_by_dir, labels_by_dir, comp_out, y_max_per_turn)


if __name__ == "__main__":
    main()
