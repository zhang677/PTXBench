#!/usr/bin/env python3
"""Plot per-turn reasoning-token count against kernel performance score.

Each datapoint is one trajectory turn, capped at the requested turn limit. The
default score is:

    speedup + score_weight * I(correct and uses the required architecture instruction)

Plots are grouped by workload and architecture, matching the grouping used by
plot_correctness_rate_by_release_date.py.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

from export_turn_correctness_arch import load_experiment_rows
from plot_correctness_rate_by_release_date import (
    ARCH_TAG_BY_ARCH,
    DEFAULT_MIGRATION_PATH,
    DEFINITIONS_DIR,
    WORKLOADS_DIR,
    build_model_color_map,
    enrich_experiment_rows,
    load_definition_var_axes,
    load_workload_index,
    load_workload_migrations,
    slugify,
    sorted_model_names,
    trajectory_paths,
    workload_arch_title,
)


DEFAULT_MAX_TURNS = 8
DEFAULT_SCORE_WEIGHT = 0.0
DEFAULT_QWEN_REASONING_TOKENIZER = (
    "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
)
DEFAULT_KIMI_REASONING_TOKENIZER = (
    "/home/ubuntu/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6"
)


def nested_get(value: object, path: tuple[str, ...]) -> object | None:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def as_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_qwen36_family(model: str) -> bool:
    return "qwen3.6" in model.casefold() or "qwen36" in model.casefold()


def is_kimi_family(model: str) -> bool:
    model_key = model.casefold()
    return "kimi" in model_key or "moonshot" in model_key


def resolve_tokenizer_path(model_path: str) -> str:
    path = Path(model_path)
    if not path.exists():
        return model_path
    if (path / "tokenizer_config.json").exists():
        return str(path)

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.is_dir():
        return model_path

    refs_main = path / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = snapshots_dir / revision
        if (snapshot / "tokenizer_config.json").exists():
            return str(snapshot)

    snapshots = [
        snapshot
        for snapshot in snapshots_dir.iterdir()
        if snapshot.is_dir() and (snapshot / "tokenizer_config.json").exists()
    ]
    if not snapshots:
        return model_path
    return str(max(snapshots, key=lambda snapshot: snapshot.stat().st_mtime))


def load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    try:
        os.environ.setdefault("HF_MODULES_CACHE", "/tmp/huggingface_modules")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            resolve_tokenizer_path(model_path),
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as exc:
        print(f"warning: failed to load tokenizer from {model_path}: {exc}")
        return None


def count_text_tokens(tokenizer, text: str) -> int | None:
    if tokenizer is None:
        return None
    try:
        if tokenizer.__class__.__name__ == "TikTokenTokenizer":
            return len(tokenizer.encode(text))
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
        encoded = tokenizer(text, add_special_tokens=False)
        if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
            return len(encoded["input_ids"])
    except Exception:
        return None
    return None


def extract_reasoning_tokens(
    message: dict,
    missing_mode: str,
    model: str = "",
    qwen_reasoning_tokenizer=None,
    kimi_reasoning_tokenizer=None,
) -> tuple[int | None, str]:
    """Return reasoning/thinking tokens for an assistant message.

    Providers use different usage schemas. Explicit reasoning/thinking counters
    are preferred. The default fallback matches
    fib_runtime/mini_swe_agent_docker/plots/plot_token_breakdown.py: estimate
    from reasoning_content when present, otherwise use zero reasoning tokens.
    """
    usage = nested_get(message, ("extra", "response", "usage"))
    if not isinstance(usage, dict):
        usage = {}
    choices = nested_get(message, ("extra", "response", "choices"))

    candidates = [
        (("completion_tokens_details", "reasoning_tokens"), "completion_tokens_details.reasoning_tokens"),
        (("completion_tokens_details", "thinking_tokens"), "completion_tokens_details.thinking_tokens"),
        (("output_tokens_details", "thinking_tokens"), "output_tokens_details.thinking_tokens"),
        (("output_tokens_details", "reasoning_tokens"), "output_tokens_details.reasoning_tokens"),
        (("reasoning_tokens",), "reasoning_tokens"),
        (("thinking_tokens",), "thinking_tokens"),
    ]
    for path, source in candidates:
        value = as_number(nested_get(usage, path))
        if value is not None:
            return int(value), source

    if missing_mode == "approximate":
        if isinstance(choices, list) and choices:
            msg_obj = choices[0].get("message", {})
            if isinstance(msg_obj, dict):
                reasoning_content = msg_obj.get("reasoning_content") or ""
                if reasoning_content and is_qwen36_family(model):
                    reasoning_token_count = count_text_tokens(
                        qwen_reasoning_tokenizer,
                        reasoning_content,
                    )
                    if reasoning_token_count is not None:
                        return (
                            reasoning_token_count,
                            "qwen_hf_reasoning_content_tokens",
                        )
                if reasoning_content and is_kimi_family(model):
                    reasoning_token_count = count_text_tokens(
                        kimi_reasoning_tokenizer,
                        reasoning_content,
                    )
                    if reasoning_token_count is not None:
                        return (
                            reasoning_token_count,
                            "kimi_k2.6_reasoning_content_tokens",
                        )

        completion = as_number(usage.get("completion_tokens", usage.get("output_tokens")))
        if completion is not None and isinstance(choices, list) and choices:
            msg_obj = choices[0].get("message", {})
            if isinstance(msg_obj, dict):
                reasoning_content = msg_obj.get("reasoning_content") or ""
                content = msg_obj.get("content") or ""
                if reasoning_content and (len(reasoning_content) + len(content)) > 0:
                    reasoning_ratio = len(reasoning_content) / (
                        len(reasoning_content) + len(content)
                    )
                    return (
                        round(completion * reasoning_ratio),
                        "estimated_from_reasoning_content",
                    )
        if completion is not None:
            return 0, "missing->zero_no_reasoning_content"

    if missing_mode == "zero":
        return 0, "missing->zero"
    if missing_mode == "completion":
        value = as_number(usage.get("completion_tokens", usage.get("output_tokens")))
        if value is not None:
            return int(value), "missing->completion_tokens"

    return None, "missing"


def assistant_eval_turns(traj: dict, max_turns: int) -> list[tuple[int, dict, dict]]:
    """Pair each assistant kernel response with its following evaluation message."""
    turns = []
    saw_initial_user = False
    pending_assistant = None

    for msg in traj.get("messages", []):
        role = msg.get("role")
        if role == "user":
            if not saw_initial_user:
                saw_initial_user = True
                continue
            if pending_assistant is not None:
                turns.append((len(turns), pending_assistant, msg))
                pending_assistant = None
                if len(turns) >= max_turns:
                    break
        elif role == "assistant" and saw_initial_user:
            pending_assistant = msg

    return turns


def load_turn_metrics(run_dir: Path) -> dict[tuple[str, int], dict[str, object]]:
    csv_path = run_dir / "figures" / "turn_correctness_arch.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing exported turn CSV: {csv_path}")

    metrics = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            turn = as_number(row.get("turn"))
            if turn is None:
                continue
            metrics[(row["trajectory_id"], int(turn))] = {
                "correctness": row.get("correctness", ""),
                "speedup": as_number(row.get("speedup")) or 0.0,
                "arch_tag": row.get("arch_tag", ""),
            }
    return metrics


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    den = x_den * y_den
    if den == 0:
        return None
    return num / den


def reasoning_token_density(xs: list[float]) -> tuple[list[float], list[float]] | None:
    if len(xs) < 2:
        return None

    x_min = min(xs)
    x_max = max(xs)
    if x_min == x_max:
        return None

    n = len(xs)
    x_mean = mean(xs)
    std = math.sqrt(sum((x - x_mean) ** 2 for x in xs) / (n - 1))
    bandwidth = 1.06 * std * (n ** (-1 / 5)) if std > 0 else 0
    if bandwidth <= 0:
        return None

    pad = 0.04 * (x_max - x_min)
    grid = [x_min - pad + (x_max - x_min + 2 * pad) * i / 199 for i in range(200)]
    scale = 1 / (n * bandwidth * math.sqrt(2 * math.pi))
    density = [
        scale * sum(math.exp(-0.5 * ((x - xi) / bandwidth) ** 2) for xi in xs)
        for x in grid
    ]
    return grid, density


def density_label(density: list[float]) -> str:
    max_density = max(density, default=0.0)
    if max_density == 0:
        return "Reasoning-token density"
    return f"Reasoning-token density (max={max_density:.2e})"


def collect_points(
    experiment_rows: list[dict[str, str]],
    max_turns: int,
    score_weight: float,
    missing_reasoning: str,
    qwen_reasoning_tokenizer=None,
    kimi_reasoning_tokenizer=None,
) -> tuple[list[dict], dict[str, int]]:
    rows = []
    counts = {
        "points": 0,
        "missing_reasoning": 0,
        "missing_arch_requirement": 0,
        "missing_turn_metrics": 0,
    }

    for exp_row in experiment_rows:
        run_dir = Path(exp_row["exp_dir"])
        model = exp_row.get("model", "")
        arch = exp_row["arch"].strip().lower()
        required_arch_tag = ARCH_TAG_BY_ARCH.get(arch)
        if required_arch_tag is None:
            counts["missing_arch_requirement"] += 1
            continue

        turn_metrics = load_turn_metrics(run_dir)
        for traj_path in trajectory_paths(run_dir):
            trajectory_id = traj_path.stem
            traj = json.loads(traj_path.read_text())
            for turn, assistant_message, _eval_message in assistant_eval_turns(traj, max_turns):
                metrics = turn_metrics.get((trajectory_id, turn))
                if metrics is None:
                    counts["missing_turn_metrics"] += 1
                    continue

                reasoning_tokens, reasoning_source = extract_reasoning_tokens(
                    assistant_message,
                    missing_reasoning,
                    model=model,
                    qwen_reasoning_tokenizer=qwen_reasoning_tokenizer,
                    kimi_reasoning_tokenizer=kimi_reasoning_tokenizer,
                )
                if reasoning_tokens is None:
                    counts["missing_reasoning"] += 1
                    continue

                correctness = str(metrics["correctness"])
                speedup = float(metrics["speedup"])
                arch_tag = str(metrics["arch_tag"])
                correct_with_arch = (
                    correctness == "Correct"
                    and required_arch_tag in [tag.strip() for tag in arch_tag.split(",")]
                )
                score = speedup + score_weight * int(correct_with_arch)

                rows.append(
                    {
                        "model": model,
                        "arch": exp_row["arch"],
                        "required_arch_tag": required_arch_tag,
                        "workload": exp_row.get("workload", ""),
                        "workload_name": exp_row.get("workload_name", ""),
                        "definition": exp_row.get("definition", ""),
                        "exp_dir": exp_row["exp_dir"],
                        "trajectory_id": trajectory_id,
                        "turn": turn,
                        "reasoning_tokens": reasoning_tokens,
                        "reasoning_token_source": reasoning_source,
                        "correctness": correctness,
                        "speedup": speedup,
                        "arch_tag": arch_tag,
                        "correct_with_arch": int(correct_with_arch),
                        "performance_score": score,
                    }
                )
                counts["points"] += 1

    return rows, counts


def write_points_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "model",
        "arch",
        "required_arch_tag",
        "workload",
        "workload_name",
        "definition",
        "exp_dir",
        "trajectory_id",
        "turn",
        "reasoning_tokens",
        "reasoning_token_source",
        "correctness",
        "speedup",
        "arch_tag",
        "correct_with_arch",
        "performance_score",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_by_workload_arch(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped = {}
    for row in rows:
        key = (row.get("workload_name", ""), row["arch"].strip().lower())
        grouped.setdefault(key, []).append(row)
    return grouped


def plot_group(
    rows: list[dict],
    out_path: Path,
    score_weight: float,
    model_colors: dict[str, str] | None = None,
    x_limit: tuple[float, float] | None = None,
) -> None:
    rows = sorted(rows, key=lambda row: (row["model"], row["trajectory_id"], int(row["turn"])))
    xs = [float(row["reasoning_tokens"]) for row in rows]
    ys = [float(row["performance_score"]) for row in rows]
    r = pearson_r(xs, ys)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    present_models = set(row["model"] for row in rows)
    if model_colors is None:
        model_colors = build_model_color_map(present_models)
    models = [model for model in sorted_model_names(model_colors.keys()) if model in present_models]

    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        ax.scatter(
            [row["reasoning_tokens"] for row in model_rows],
            [row["performance_score"] for row in model_rows],
            s=34,
            alpha=0.68,
            color=model_colors[model],
            edgecolor="white",
            linewidth=0.35,
            label=model,
        )

    ax2 = None
    density_plotted = False
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        model_xs = [float(row["reasoning_tokens"]) for row in model_rows]
        density = reasoning_token_density(model_xs)
        if density is None:
            continue
        if ax2 is None:
            ax2 = ax.twinx()
        density_x, density_y = density
        model_density_max = max(density_y, default=0.0)
        if model_density_max == 0:
            continue
        density_y = [value / model_density_max for value in density_y]
        color = model_colors[model]
        ax2.plot(
            density_x,
            density_y,
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=f"{model} token density",
        )
        ax2.fill_between(density_x, density_y, color=color, alpha=0.14)
        density_plotted = True

    if ax2 is not None and density_plotted:
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Relative reasoning-token density")
        ax2.tick_params(axis="y", labelcolor="#222222")

    title_context = workload_arch_title(rows[0])
    r_text = "undefined" if r is None else f"{r:.3f}"
    ax.set_title(
        f"Reasoning Tokens vs Performance Score\n"
        f"{title_context}  "
        f"(n={len(rows)}, Pearson r={r_text})"
    )
    ax.set_xlabel("Reasoning tokens")
    ax.set_ylabel(f"Performance score = speedup + {score_weight:g} * I(correct and arch)")
    ax.set_ylim(0, 1.3)
    if x_limit is not None:
        ax.set_xlim(x_limit)
        if ax2 is not None:
            ax2.set_xlim(x_limit)
    ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        density_handles, density_labels = ax2.get_legend_handles_labels()
        handles += density_handles
        labels += density_labels
    ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments.csv",
        help="CSV with model, arch, definition, workload, and exp_dir columns",
    )
    parser.add_argument(
        "--workloads-dir",
        type=Path,
        default=WORKLOADS_DIR,
        help="Directory containing workload JSONL files used to resolve workload UUIDs",
    )
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=DEFINITIONS_DIR,
        help="Directory containing definition JSON files used to identify variable axes",
    )
    parser.add_argument(
        "--migration-csv",
        type=Path,
        default=DEFAULT_MIGRATION_PATH,
        help="CSV-style migration.csv mapping old definition/workload to new definition/workload",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Maximum number of turns to include per trajectory",
    )
    parser.add_argument(
        "--score-weight",
        type=float,
        default=DEFAULT_SCORE_WEIGHT,
        help="Bonus added when a turn is correct and uses the required architecture instruction",
    )
    parser.add_argument(
        "--missing-reasoning",
        choices=["approximate", "skip", "zero", "completion"],
        default="approximate",
        help=(
            "How to handle turns without explicit reasoning/thinking token counts; "
            "approximate matches plot_token_breakdown.py"
        ),
    )
    parser.add_argument(
        "--qwen-reasoning-tokenizer",
        default=DEFAULT_QWEN_REASONING_TOKENIZER,
        help=(
            "Tokenizer path used to count Qwen3.6-family reasoning_content tokens "
            "instead of using provider token details or character-ratio approximation"
        ),
    )
    parser.add_argument(
        "--kimi-reasoning-tokenizer",
        default=DEFAULT_KIMI_REASONING_TOKENIZER,
        help=(
            "Tokenizer path used to count Kimi-family reasoning_content tokens "
            "when provider reasoning/thinking token details are missing"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <experiments-csv parent>/figures)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Grouped plot directory (default: <out-dir>/by_workload_arch)",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write reasoning-token CSV and skip grouped PNG generation",
    )
    args = parser.parse_args()

    if args.max_turns < 1:
        raise ValueError("--max-turns must be at least 1")

    definition_var_axes = load_definition_var_axes(args.definitions_dir)
    workload_index = load_workload_index(args.workloads_dir, definition_var_axes)
    workload_migrations = load_workload_migrations(args.migration_csv)
    experiment_rows = enrich_experiment_rows(
        load_experiment_rows(args.experiments_csv),
        workload_index,
        workload_migrations,
    )
    model_colors = build_model_color_map(row["model"] for row in experiment_rows)
    qwen_reasoning_tokenizer = None
    if (
        args.missing_reasoning == "approximate"
        and any(is_qwen36_family(row.get("model", "")) for row in experiment_rows)
    ):
        qwen_reasoning_tokenizer = load_tokenizer(args.qwen_reasoning_tokenizer)
    kimi_reasoning_tokenizer = None
    if (
        args.missing_reasoning == "approximate"
        and any(is_kimi_family(row.get("model", "")) for row in experiment_rows)
    ):
        kimi_reasoning_tokenizer = load_tokenizer(args.kimi_reasoning_tokenizer)

    out_dir = args.out_dir or (args.experiments_csv.resolve().parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, counts = collect_points(
        experiment_rows=experiment_rows,
        max_turns=args.max_turns,
        score_weight=args.score_weight,
        missing_reasoning=args.missing_reasoning,
        qwen_reasoning_tokenizer=qwen_reasoning_tokenizer,
        kimi_reasoning_tokenizer=kimi_reasoning_tokenizer,
    )
    csv_path = out_dir / "reasoning_tokens_vs_performance.csv"
    write_points_csv(rows, csv_path)
    print(f"Wrote {counts['points']} points to {csv_path}")

    if args.csv_only:
        print("Skipped grouped PNG generation (--csv-only)")
        if counts["missing_reasoning"]:
            print(
                f"Skipped {counts['missing_reasoning']} turns without explicit reasoning tokens "
                f"(use --missing-reasoning completion to use completion/output tokens as a fallback)"
            )
        if counts["missing_arch_requirement"]:
            print(f"Skipped {counts['missing_arch_requirement']} experiments with unsupported arch")
        if counts["missing_turn_metrics"]:
            print(
                f"Skipped {counts['missing_turn_metrics']} turns missing from "
                "figures/turn_correctness_arch.csv"
            )
        return

    plot_dir = args.plot_dir or (out_dir / "by_workload_arch")
    plot_dir.mkdir(parents=True, exist_ok=True)

    max_reasoning_tokens = max((float(row["reasoning_tokens"]) for row in rows), default=0.0)
    x_max = math.ceil(max_reasoning_tokens * 1.05 / 10000) * 10000 if max_reasoning_tokens else 1.0
    x_limit = (0.0, x_max)

    plot_paths = []
    for (workload_name, arch), group_rows in sorted(grouped_by_workload_arch(rows).items()):
        out_path = plot_dir / (
            "reasoning_tokens_vs_performance__"
            f"{slugify(workload_name)}__{slugify(arch)}.png"
        )
        plot_group(
            group_rows,
            out_path,
            args.score_weight,
            model_colors=model_colors,
            x_limit=x_limit,
        )
        plot_paths.append(out_path)

    print(f"Wrote {len(plot_paths)} grouped plots to {plot_dir}")
    if counts["missing_reasoning"]:
        print(
            f"Skipped {counts['missing_reasoning']} turns without explicit reasoning tokens "
            f"(use --missing-reasoning completion to use completion/output tokens as a fallback)"
        )
    if counts["missing_arch_requirement"]:
        print(f"Skipped {counts['missing_arch_requirement']} experiments with unsupported arch")
    if counts["missing_turn_metrics"]:
        print(
            f"Skipped {counts['missing_turn_metrics']} turns missing from "
            "figures/turn_correctness_arch.csv"
        )


if __name__ == "__main__":
    main()
