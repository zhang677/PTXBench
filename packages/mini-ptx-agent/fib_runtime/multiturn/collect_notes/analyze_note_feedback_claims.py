#!/usr/bin/env python3
"""Build plots and tables for note-feedback retrieval claims."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


EVAL_ROOT = Path("/home/ubuntu/AccRL-exps/eval_runs")
OUT_DIR = Path("/home/ubuntu/AccRL/fib_runtime/multiturn/collect_notes/analysis_note_feedback_claims")
NOTES_JSONL = Path(
    "/home/ubuntu/AccRL-exps/tasks/collect_notes/outputs/"
    "mha-d128-4def-kernel-fix-notes-full/notes.jsonl"
)

NOTE_ONLY_GLOB = "qwen36-27b-retrieved-notes-d128-4defs-*"
KERNEL_GLOB = "qwen36-27b-note-feedback2-mha-d128-d96-8defs-*"

HEADER_RE = re.compile(
    r"^### Similar error kernel (?P<rank>\d+): bm25=(?P<bm25>[0-9.]+), "
    r"definition=(?P<definition>[^,]+), fixed_variants=(?P<fixed_variants>\d+)",
    re.MULTILINE,
)
BEST_RE = re.compile(r"^best_fixed_kernel_path: (.+)$", re.MULTILINE)
NOTE_HEADER_RE = re.compile(
    r"^### Similar wrong kernel (?P<rank>\d+): bm25=(?P<bm25>[0-9.]+), "
    r"definition=(?P<definition>[^,]+), fixed_variants=(?P<fixed_variants>\d+)",
    re.MULTILINE,
)
SHAPE_RESULT_RE = re.compile(
    r"^\[(?P<shape>.*?)\]\s+(?P<status>PASSED|[A-Z_]+)"
    r"(?:\s+[—-]\s+speedup:\s+(?P<speedup>[0-9.]+)x)?",
    re.MULTILINE,
)
SPEED_RE = re.compile(r"speedup:\s+([0-9.]+)x")


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def definition_stem(run_name: str) -> str:
    if "note-feedback2-mha-d128-d96-8defs-" in run_name:
        return run_name.split("note-feedback2-mha-d128-d96-8defs-", 1)[1]
    if "retrieved-notes-d128-4defs-" in run_name:
        return run_name.split("retrieved-notes-d128-4defs-", 1)[1]
    return run_name


def task_family(definition: str) -> str:
    return "bwd" if "_bwd_" in definition or definition.startswith("mha_bwd") else "fwd"


def task_dim(definition: str) -> str:
    match = re.search(r"d(\d+)", definition)
    return f"d{match.group(1)}" if match else "unknown"


def is_causal(definition: str) -> bool:
    return definition.endswith("_causal") or "-causal" in definition


def planned_definition(run: Path) -> str:
    plan_path = run / "plan.json"
    if plan_path.exists():
        try:
            plan = load_json(plan_path).get("plan") or []
            if plan:
                return str(plan[0].get("definition") or definition_stem(run.name))
        except (OSError, json.JSONDecodeError):
            pass
    stem = definition_stem(run.name)
    return stem.replace("mha-d128", "mha_with_lse_d128")


def parse_feedback(text: str) -> dict[str, Any]:
    shape_results = []
    for match in SHAPE_RESULT_RE.finditer(text):
        speed = match.group("speedup")
        shape_results.append(
            {
                "shape": match.group("shape"),
                "status": match.group("status"),
                "speedup": float(speed) if speed else None,
            }
        )
    if shape_results:
        if all(item["status"] == "PASSED" for item in shape_results):
            speeds = [item["speedup"] for item in shape_results if item["speedup"] is not None]
            return {
                "status": "PASSED",
                "speedup": min(speeds) if speeds else None,
                "detail": "PASSED",
            }
        counts = Counter(item["status"] for item in shape_results)
        return {
            "status": "FAILED",
            "speedup": None,
            "detail": ",".join(f"{key}x{value}" for key, value in counts.most_common()),
        }
    if "Failed to compile kernel" in text or "error detected in the compilation" in text:
        return {"status": "COMPILE_ERROR", "speedup": None, "detail": "COMPILE_ERROR"}
    if "TIMEOUT" in text or "timed out" in text.lower():
        return {"status": "TIMEOUT", "speedup": None, "detail": "TIMEOUT"}
    if "Evaluation FAILED" in text:
        return {"status": "FAILED", "speedup": None, "detail": "FAILED"}
    if "<returncode>0</returncode>" in text:
        match = SPEED_RE.search(text)
        return {
            "status": "PASSED_OR_OK",
            "speedup": float(match.group(1)) if match else None,
            "detail": "PASSED_OR_OK",
        }
    return {"status": "UNKNOWN", "speedup": None, "detail": "UNKNOWN"}


def completed_turns(messages: list[dict[str, Any]]) -> int:
    count = 0
    while True:
        assistant_idx = 2 + 2 * count
        observation_idx = assistant_idx + 1
        if observation_idx >= len(messages):
            return count
        if messages[assistant_idx].get("role") != "assistant":
            return count
        if messages[observation_idx].get("role") != "user":
            return count
        count += 1


def success_speedups(success_dir: Path) -> list[float]:
    record = success_dir / "record.json"
    if not record.exists():
        return []
    try:
        records = load_json(record)
    except (OSError, json.JSONDecodeError):
        return []
    values: list[float] = []
    for item in records if isinstance(records, list) else []:
        for trace in item.get("traces") or []:
            perf = ((trace.get("evaluation") or {}).get("performance") or {})
            speed = perf.get("speedup_factor")
            if isinstance(speed, (int, float)):
                values.append(float(speed))
    return values


def trajectory_summary(method: str, run: Path) -> list[dict[str, Any]]:
    definition = planned_definition(run)
    rows = []
    for trajectory in sorted((run / "trajectories").glob("exp_*.json")):
        data = load_json(trajectory)
        messages = data.get("messages") or []
        feedback = [
            (idx, parse_feedback(str(message.get("content") or "")))
            for idx, message in enumerate(messages)
            if message.get("role") == "user" and idx >= 3
        ]
        pass_speeds = [item["speedup"] for _, item in feedback if item["status"].startswith("PASSED") and item["speedup"] is not None]
        exp_name = trajectory.stem
        saved_speeds = success_speedups(run / "success" / exp_name)
        rows.append(
            {
                "method": method,
                "run": run.name,
                "exp": exp_name,
                "definition": definition,
                "family": task_family(definition),
                "dim": task_dim(definition),
                "causal": is_causal(definition),
                "completed_turns": completed_turns(messages),
                "exit_status": data.get("info", {}).get("exit_status"),
                "feedback_turns": len(feedback),
                "pass_turns": len(pass_speeds),
                "has_pass": bool(pass_speeds or saved_speeds),
                "best_turn_speedup": max(pass_speeds) if pass_speeds else None,
                "saved_success_kernels": len(list((run / "success" / exp_name).glob("kernel_v*.cu"))),
                "best_saved_speedup": max(saved_speeds) if saved_speeds else None,
            }
        )
    return rows


def notes_path_to_speedup() -> dict[str, float | None]:
    mapping: dict[str, float | None] = {}
    if not NOTES_JSONL.exists():
        return mapping
    for line in NOTES_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        meta = payload.get("metadata") or {}
        correct_path = meta.get("correct_kernel_path")
        if not correct_path:
            continue
        exp_dir = meta.get("exp_dir")
        trajectory_id = meta.get("trajectory_id")
        version = safe_int(meta.get("correct_kernel_version"))
        if not exp_dir or not trajectory_id or version is None:
            mapping[correct_path] = None
            continue
        record = Path(exp_dir) / "success" / trajectory_id / "record.json"
        speed = None
        try:
            records = load_json(record)
            for item in records if isinstance(records, list) else []:
                if safe_int(item.get("version")) != version:
                    continue
                vals = []
                for trace in item.get("traces") or []:
                    perf = ((trace.get("evaluation") or {}).get("performance") or {})
                    value = perf.get("speedup_factor")
                    if isinstance(value, (int, float)):
                        vals.append(float(value))
                speed = min(vals) if vals else None
                break
        except (OSError, json.JSONDecodeError):
            speed = None
        mapping[correct_path] = speed
    return mapping


def retrieval_rows(runs: list[Path]) -> list[dict[str, Any]]:
    speedups = notes_path_to_speedup()
    rows = []
    for run in runs:
        definition = planned_definition(run)
        for trajectory in sorted((run / "trajectories").glob("exp_*.json")):
            data = load_json(trajectory)
            messages = data.get("messages") or []
            for idx, message in enumerate(messages):
                if message.get("role") != "user" or idx < 3:
                    continue
                text = str(message.get("content") or "")
                for match in HEADER_RE.finditer(text):
                    next_msg = messages[idx + 2] if idx + 2 < len(messages) else None
                    next_feedback = parse_feedback(str(next_msg.get("content") or "")) if isinstance(next_msg, dict) and next_msg.get("role") == "user" else None
                    block = text[match.start():]
                    best = BEST_RE.search(block)
                    best_path = best.group(1).strip() if best else ""
                    rows.append(
                        {
                            "run": run.name,
                            "exp": trajectory.stem,
                            "task_definition": definition,
                            "task_dim": task_dim(definition),
                            "task_family": task_family(definition),
                            "feedback_after_turn": (idx - 1) // 2,
                            "retrieved_definition": match.group("definition"),
                            "retrieved_dim": task_dim(match.group("definition")),
                            "retrieved_family": task_family(match.group("definition")),
                            "bm25": float(match.group("bm25")),
                            "best_fixed_kernel_path": best_path,
                            "retrieved_speedup": speedups.get(best_path),
                            "next_status": (next_feedback or {}).get("status") or "NO_NEXT_EVAL",
                            "next_speedup": (next_feedback or {}).get("speedup"),
                        }
                    )
                    break
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_pass_rates(traj: pd.DataFrame) -> None:
    grouped = (
        traj.groupby(["method", "definition"], as_index=False)
        .agg(trajectories=("exp", "count"), any_pass=("has_pass", "sum"), kernels=("saved_success_kernels", "sum"))
    )
    method_order = {"notes_only": 0, "fixed_kernel": 1}
    grouped["method_order"] = grouped["method"].map(method_order)
    grouped = grouped.sort_values(["method_order", "definition"]).reset_index(drop=True)
    grouped["pass_rate"] = grouped["any_pass"] / grouped["trajectories"]
    labels = [f"{row.method}\n{row.definition}" for row in grouped.itertuples()]
    colors = ["#8884d8" if row.method == "notes_only" else "#2f9e44" for row in grouped.itertuples()]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    bars = ax.bar(range(len(grouped)), grouped["pass_rate"], color=colors)
    ax.set_ylabel("Trajectories with >=1 passing kernel")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Trajectory pass rate by retrieval mode")
    for bar, row in zip(bars, grouped.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{int(row.any_pass)}/{int(row.trajectories)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "pass_rate_by_run.png", dpi=180)
    plt.close(fig)


def plot_dim_generalization(traj: pd.DataFrame) -> None:
    data = traj[traj["method"] == "fixed_kernel"].copy()
    grouped = (
        data.groupby(["dim"], as_index=False)
        .agg(trajectories=("exp", "count"), any_pass=("has_pass", "sum"), kernels=("saved_success_kernels", "sum"))
    )
    grouped["pass_rate"] = grouped["any_pass"] / grouped["trajectories"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(grouped["dim"], grouped["pass_rate"], color=["#2f9e44" if x == "d128" else "#f08c00" for x in grouped["dim"]])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Trajectory pass rate")
    ax.set_title("Fixed-kernel retrieval: d128 source examples transfer poorly to d96")
    for bar, row in zip(bars, grouped.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{int(row.any_pass)}/{int(row.trajectories)}\n{int(row.kernels)} kernels",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "fixed_kernel_d128_vs_d96.png", dpi=180)
    plt.close(fig)


def plot_retrieval_heatmap(retrieval: pd.DataFrame) -> None:
    table = pd.crosstab(retrieval["task_definition"], retrieval["retrieved_definition"])
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    image = ax.imshow(table.values, cmap="Blues")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_title("BM25 retrieved-definition counts")
    ax.set_xlabel("Retrieved fixed-kernel definition")
    ax.set_ylabel("Target task definition")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = int(table.values[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "retrieval_definition_heatmap.png", dpi=180)
    plt.close(fig)


def plot_next_turn_by_retrieved_speed(retrieval: pd.DataFrame) -> None:
    data = retrieval.dropna(subset=["retrieved_speedup"]).copy()
    data["next_pass"] = data["next_status"].str.startswith("PASSED")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = data["next_pass"].map({True: "#2f9e44", False: "#c92a2a"})
    ax.scatter(data["retrieved_speedup"], data["bm25"], c=colors, alpha=0.75, s=35, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Retrieved fixed-kernel measured speedup (log scale)")
    ax.set_ylabel("BM25 score")
    ax.set_title("Retrieved-kernel speedup vs next-turn result")
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.text(0.02, 0.97, "green: next turn passed\nred: next turn failed/no eval", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "retrieved_speedup_vs_next_turn.png", dpi=180)
    plt.close(fig)


def write_memo(traj: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    note = traj[traj["method"] == "notes_only"]
    fixed = traj[traj["method"] == "fixed_kernel"]
    note_agg = note.agg(
        trajectories=("exp", "count"),
        any_pass=("has_pass", "sum"),
        kernels=("saved_success_kernels", "sum"),
    )
    fixed_by_dim = (
        fixed.groupby("dim", as_index=False)
        .agg(trajectories=("exp", "count"), any_pass=("has_pass", "sum"), kernels=("saved_success_kernels", "sum"))
    )
    bad_request_stops = int((fixed["exit_status"] == "BadRequestError").sum())
    missing_turns = int((8 - fixed["completed_turns"]).clip(lower=0).sum())
    next_counts = retrieval["next_status"].value_counts()
    retrieval_defs = retrieval["retrieved_definition"].value_counts()
    pass_by_speed = retrieval.assign(next_pass=retrieval["next_status"].str.startswith("PASSED"))
    top_pairs = (
        pass_by_speed.groupby(["retrieved_definition", "retrieved_speedup"], dropna=False)
        .agg(uses=("run", "count"), next_pass=("next_pass", "sum"))
        .reset_index()
        .sort_values(["uses", "next_pass"], ascending=False)
        .head(10)
    )

    lines = [
        "# Note Feedback Retrieval Claims",
        "",
        "## Scope",
        "",
        f"- Note-only roots: `{NOTE_ONLY_GLOB}`.",
        f"- Fixed-kernel roots: `{KERNEL_GLOB}`.",
        "- Unit of analysis is the trajectory and the per-turn feedback messages in `trajectories/exp_*.json`; `summary.json` is not used as the primary outcome because several roots were resumed subsets.",
        "",
        "## Claim 1: Notes alone did not help",
        "",
        f"The four note-only d128 roots contain `{len(note)}` trajectories. They produced `{int(note['has_pass'].sum())}` trajectories with any passing turn and `{int(note['saved_success_kernels'].sum())}` saved passing kernels.",
        "",
        "This is the strongest available artifact-backed conclusion: the notes were present in feedback, but no trajectory reached a saved correct kernel. The failure mode is not that retrieval was absent; it is that prose repair notes were insufficient for Qwen3.6-27B to synthesize a legal, correct Hopper implementation.",
        "",
        "## Claim 2: BM25 is adequate for selecting concrete success kernels, but the model mostly imitates",
        "",
        f"The fixed-kernel run has `{len(fixed)}` trajectories and `{int(fixed['saved_success_kernels'].sum())}` saved passing kernels. By dimension:",
        "",
    ]
    for row in fixed_by_dim.itertuples():
        rate = row.any_pass / row.trajectories if row.trajectories else 0
        lines.append(f"- `{row.dim}`: `{int(row.any_pass)}/{int(row.trajectories)}` trajectories passed, `{int(row.kernels)}` saved kernels, pass rate `{rate:.1%}`.")
    lines.extend(
        [
            "",
            "The retrieval corpus is d128-only, so the d96 rows are a transfer/generalization test. The much stronger d128 result and weak d96 result are consistent with in-context imitation or local adaptation of concrete examples, not robust extrapolation of the algorithm across head dimensions.",
            "",
            "BM25 itself looks adequate as a sparse selector: retrieved definitions are concentrated on structurally related MHA forward/backward and causal/noncausal variants rather than arbitrary kernels.",
            "",
            "Retrieved definition counts:",
        ]
    )
    for key, value in retrieval_defs.items():
        lines.append(f"- `{key}`: `{int(value)}`")
    lines.extend(["", "Next-turn outcomes after a retrieved fixed kernel was injected:"])
    for key, value in next_counts.items():
        lines.append(f"- `{key}`: `{int(value)}`")
    lines.extend(["", "Most-used retrieved definition/speedup pairs:"])
    for row in top_pairs.itertuples():
        speed = "unknown" if pd.isna(row.retrieved_speedup) else f"{row.retrieved_speedup:.6g}x"
        lines.append(f"- `{row.retrieved_definition}` @ `{speed}`: `{int(row.uses)}` uses, `{int(row.next_pass)}` next-turn passes.")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `pass_rate_by_run.png`: note-only versus fixed-kernel pass rates by task.",
            "- `fixed_kernel_d128_vs_d96.png`: d128 versus d96 pass rate under fixed-kernel retrieval.",
            "- `retrieval_definition_heatmap.png`: BM25 target-definition to retrieved-definition counts.",
            "- `retrieved_speedup_vs_next_turn.png`: retrieved fixed-kernel speedup versus next-turn behavior.",
            "",
            "## Interpretation",
            "",
            "A good next experiment is to make the success-kernel injection less copy-heavy and more transform-heavy: show the correct kernel plus a compact structured diff against the retrieved wrong kernel, and explicitly ask for which constants/layout choices must change for the target definition. That would test whether the model can adapt the example instead of copying its shape-specific implementation.",
            "",
            "## Caveats",
            "",
            f"This is observational evidence from the current artifacts, not a randomized causal estimate. The fixed-kernel run also has `{bad_request_stops}` trajectories ending in context-length `BadRequestError`, leaving `{missing_turns}` planned turns unattempted. The pass-rate comparison is therefore trajectory-level and uses any observed passing kernel, not final-turn-only success.",
            "",
        ]
    )
    (OUT_DIR / "note_feedback_claims_writeup.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    note_runs = sorted(EVAL_ROOT.glob(NOTE_ONLY_GLOB))
    fixed_runs = sorted(EVAL_ROOT.glob(KERNEL_GLOB))

    traj_rows = []
    for run in note_runs:
        traj_rows.extend(trajectory_summary("notes_only", run))
    for run in fixed_runs:
        traj_rows.extend(trajectory_summary("fixed_kernel", run))
    retrieval = retrieval_rows(fixed_runs)

    save_csv(OUT_DIR / "trajectory_summary.csv", traj_rows)
    save_csv(OUT_DIR / "fixed_kernel_retrieval_rows.csv", retrieval)

    traj_df = pd.DataFrame(traj_rows)
    retrieval_df = pd.DataFrame(retrieval)
    plot_pass_rates(traj_df)
    plot_dim_generalization(traj_df)
    plot_retrieval_heatmap(retrieval_df)
    plot_next_turn_by_retrieved_speed(retrieval_df)
    write_memo(traj_df, retrieval_df)

    print(f"Wrote {OUT_DIR}")
    print(f"trajectory_rows={len(traj_rows)} retrieval_rows={len(retrieval)}")


if __name__ == "__main__":
    main()
