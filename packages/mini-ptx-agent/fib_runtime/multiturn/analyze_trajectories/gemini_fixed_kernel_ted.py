#!/usr/bin/env python3
"""Measure CFG TED from original failed kernels to Gemini fix-it turns.

For each row in a fix-it run's `figures/turn_correctness_arch.csv`, this script
finds the source failed kernel from `plan.json`, extracts the generated kernel
for that trajectory turn from `trajectories/exp_*.json`, and computes the same
normalized Zhang-Shasha tree edit distance used by
`../analyze_patterns_cross_run.py`:

    simple_distance(cfg_a, cfg_b) / max(node_count_a, node_count_b, 1)

Rows with no extractable generated C++ kernel are preserved with blank TED
fields, which makes the output row count line up with the correctness CSV.

Usage:
    python gemini_fixed_kernel_ted.py

    python gemini_fixed_kernel_ted.py \
        --run-dir /home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-gemini \
        --output-dir gemini-fixed-kernel-ted
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from zss import simple_distance

SCRIPT_DIR = Path(__file__).resolve().parent
ACCRL_ROOT = SCRIPT_DIR.parents[2]
MULTITURN_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(ACCRL_ROOT))
sys.path.insert(0, str(MULTITURN_DIR))

from accrl.utils.code_utils import extract_code_block  # noqa: E402
from analyze_patterns_batch import (  # noqa: E402
    DEFAULT_REFERENCE_FNS,
    ZssCache,
    analyze_one,
    load_reference_fn_names,
)


DEFAULT_RUN_DIR = Path(
    "/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-gemini"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gemini-fixed-kernel-ted"

KERNEL_T_RE = re.compile(r"kernel_t(\d+)\.cu$")
EXP_RE = re.compile(r"^exp_(\d+)$")


def load_plan(run_dir: Path) -> dict[str, dict[str, Any]]:
    plan_path = run_dir / "plan.json"
    data = json.loads(plan_path.read_text())
    entries = data.get("plan", data)
    if not isinstance(entries, list):
        raise SystemExit(f"{plan_path}: expected a list or a dict with a 'plan' list")

    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        exp_index = entry.get("exp_index")
        if exp_index is None:
            continue
        out[f"exp_{int(exp_index):03d}"] = entry
    return out


def load_turn_metrics(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"trajectory_id", "turn", "correctness", "speedup", "arch_tag"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"{csv_path}: missing required columns: {sorted(missing)}")
    return rows


def extract_turn_kernels(trajectory: dict[str, Any]) -> dict[int, str]:
    """Return assistant-turn index -> extracted C++ kernel source.

    This mirrors the turn counting used by `analyze_kernel_per_turn.py` and the
    benchmark correctness exporter: every assistant message advances the turn
    counter, even when the message has no extractable C++ block.
    """
    kernels: dict[int, str] = {}
    turn = 0
    for msg in trajectory.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        kernel = extract_code_block(
            content if isinstance(content, str) else "",
            languages=["cpp"],
            keep_separators=False,
        )
        if kernel:
            kernels[turn] = kernel
        turn += 1
    return kernels


def source_run_name(path: Path) -> str:
    parts = path.parts
    try:
        idx = parts.index("eval_runs")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ""


def source_turn(path: Path) -> str:
    m = KERNEL_T_RE.search(path.name)
    return m.group(1) if m else ""


def qualify_record(record: dict[str, Any], kernel_id: str) -> dict[str, Any]:
    out = dict(record)
    out["kernel_id"] = kernel_id
    return out


def analyze_kernel_path(
    path: Path,
    reference_names: set[str],
    record_cache: dict[Path, dict[str, Any]],
    kernel_id: str,
) -> dict[str, Any]:
    path = path.resolve()
    record = record_cache.get(path)
    if record is None:
        record = analyze_one((path, reference_names))
        record_cache[path] = record
    return qualify_record(record, kernel_id)


def analyze_generated_kernel(
    source: str,
    tmp_dir: Path,
    reference_names: set[str],
    trajectory_id: str,
    turn: int,
) -> dict[str, Any]:
    exp_dir = tmp_dir / trajectory_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / f"kernel_t{turn}.cu"
    path.write_text(source)
    record = analyze_one((path, reference_names))
    return qualify_record(record, f"fixed/{trajectory_id}/kernel_t{turn}.cu")


def compute_ted(
    original: dict[str, Any],
    fixed: dict[str, Any],
    cache: ZssCache,
) -> tuple[int, float]:
    za = cache.get(original["kernel_id"])
    zb = cache.get(fixed["kernel_id"])
    raw = int(simple_distance(za, zb))
    denom = max(int(original["cfg_node_count"]), int(fixed["cfg_node_count"]), 1)
    return raw, raw / denom


def fmt_json_list(value: Any) -> str:
    return json.dumps(value if value is not None else [], separators=(",", ":"))


def build_rows(
    run_dir: Path,
    metrics_rows: list[dict[str, str]],
    plan_by_exp: dict[str, dict[str, Any]],
    reference_names: set[str],
    tmp_dir: Path,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated_by_traj: dict[str, dict[int, str]] = {}
    analyzed_originals: dict[Path, dict[str, Any]] = {}
    analyzed_fixed: dict[tuple[str, int], dict[str, Any]] = {}
    records_for_ted: list[dict[str, Any]] = []
    pending_pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []

    for metric_index, metric in enumerate(metrics_rows, start=1):
        if progress_every and metric_index % progress_every == 0:
            print(
                f"[gemini-fixed-ted] prepared metadata for "
                f"{metric_index}/{len(metrics_rows)} rows",
                flush=True,
            )
        trajectory_id = metric["trajectory_id"]
        turn = int(metric["turn"])
        plan = plan_by_exp.get(trajectory_id)

        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "trajectory_id": trajectory_id,
            "turn": turn,
            "fixed_correctness": metric.get("correctness", ""),
            "fixed_speedup": metric.get("speedup", ""),
            "fixed_arch_tag": metric.get("arch_tag", ""),
            "definition": "",
            "prompt_tag": "",
            "test_path": "",
            "original_kernel_path": "",
            "original_log_path": "",
            "original_source_run": "",
            "original_trajectory_id": "",
            "original_turn": "",
            "ted_raw": "",
            "ted_norm": "",
            "ted_error": "",
            "fixed_kernel_extracted": "false",
            "original_num_global": "",
            "fixed_num_global": "",
            "original_cfg_node_count": "",
            "fixed_cfg_node_count": "",
            "original_cfg_depth": "",
            "fixed_cfg_depth": "",
            "original_cfg_branches": "",
            "fixed_cfg_branches": "",
            "original_cfg_loops": "",
            "fixed_cfg_loops": "",
            "original_cfg_hash": "",
            "fixed_cfg_hash": "",
            "original_global_kernel_names": "",
            "fixed_global_kernel_names": "",
            "original_helpers_count": "",
            "fixed_helpers_count": "",
        }

        if plan is None:
            row["ted_error"] = "missing plan entry"
            rows.append(row)
            continue

        original_path = Path(plan["error_kernel_path"])
        original_log_path = Path(plan.get("error_log_path", ""))
        row.update(
            {
                "definition": plan.get("definition", ""),
                "prompt_tag": plan.get("prompt_tag", ""),
                "test_path": plan.get("test_path", ""),
                "original_kernel_path": str(original_path),
                "original_log_path": str(original_log_path),
                "original_source_run": source_run_name(original_path),
                "original_trajectory_id": original_path.parent.name,
                "original_turn": source_turn(original_path),
            }
        )

        if trajectory_id not in generated_by_traj:
            traj_path = run_dir / "trajectories" / f"{trajectory_id}.json"
            if traj_path.is_file():
                generated_by_traj[trajectory_id] = extract_turn_kernels(
                    json.loads(traj_path.read_text())
                )
            else:
                generated_by_traj[trajectory_id] = {}

        generated_source = generated_by_traj[trajectory_id].get(turn)
        if not generated_source:
            row["ted_error"] = "no extracted cpp block for fixed turn"
            rows.append(row)
            continue
        row["fixed_kernel_extracted"] = "true"

        if not original_path.is_file():
            row["ted_error"] = "missing original kernel path"
            rows.append(row)
            continue

        try:
            original_rec = analyze_kernel_path(
                original_path,
                reference_names,
                analyzed_originals,
                f"original/{trajectory_id}/{original_path.parent.name}/{original_path.name}",
            )
            fixed_rec = analyzed_fixed.get((trajectory_id, turn))
            if fixed_rec is None:
                fixed_rec = analyze_generated_kernel(
                    generated_source,
                    tmp_dir,
                    reference_names,
                    trajectory_id,
                    turn,
                )
                analyzed_fixed[(trajectory_id, turn)] = fixed_rec
            row.update(
                {
                    "original_num_global": original_rec["num_global"],
                    "fixed_num_global": fixed_rec["num_global"],
                    "original_cfg_node_count": original_rec["cfg_node_count"],
                    "fixed_cfg_node_count": fixed_rec["cfg_node_count"],
                    "original_cfg_depth": original_rec["cfg_depth"],
                    "fixed_cfg_depth": fixed_rec["cfg_depth"],
                    "original_cfg_branches": original_rec["cfg_branches"],
                    "fixed_cfg_branches": fixed_rec["cfg_branches"],
                    "original_cfg_loops": original_rec["cfg_loops"],
                    "fixed_cfg_loops": fixed_rec["cfg_loops"],
                    "original_cfg_hash": original_rec["cfg_hash"],
                    "fixed_cfg_hash": fixed_rec["cfg_hash"],
                    "original_global_kernel_names": fmt_json_list(
                        original_rec.get("global_kernel_names")
                    ),
                    "fixed_global_kernel_names": fmt_json_list(
                        fixed_rec.get("global_kernel_names")
                    ),
                    "original_helpers_count": len(original_rec.get("helpers") or []),
                    "fixed_helpers_count": len(fixed_rec.get("helpers") or []),
                }
            )
            pair_index = len(rows)
            pending_pairs.append((pair_index, original_rec, fixed_rec))
            records_for_ted.extend([original_rec, fixed_rec])
        except Exception as exc:  # Keep the CSV rectangular for diagnostics.
            row["ted_error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    print(
        f"[gemini-fixed-ted] prepared {len(rows)} rows; "
        f"{len(pending_pairs)} rows are queued for TED computation",
        flush=True,
    )

    cache = ZssCache(records_for_ted)
    ted_by_cfg_pair: dict[tuple[str, str], tuple[int, float]] = {}
    for pair_number, (row_index, original_rec, fixed_rec) in enumerate(
        pending_pairs, start=1
    ):
        if rows[row_index]["ted_error"]:
            continue
        try:
            key = (original_rec["cfg_hash"], fixed_rec["cfg_hash"])
            if original_rec["cfg_hash"] == fixed_rec["cfg_hash"]:
                raw, norm = 0, 0.0
            elif key in ted_by_cfg_pair:
                raw, norm = ted_by_cfg_pair[key]
            else:
                if progress_every and len(ted_by_cfg_pair) % progress_every == 0:
                    print(
                        "[gemini-fixed-ted] computing TED "
                        f"pair {pair_number}/{len(pending_pairs)} "
                        f"(unique cfg-pair {len(ted_by_cfg_pair) + 1}; "
                        f"orig_nodes={original_rec['cfg_node_count']}, "
                        f"fixed_nodes={fixed_rec['cfg_node_count']}; "
                        f"row={rows[row_index]['trajectory_id']} "
                        f"turn={rows[row_index]['turn']})",
                        flush=True,
                    )
                raw, norm = compute_ted(original_rec, fixed_rec, cache)
                ted_by_cfg_pair[key] = (raw, norm)
            rows[row_index]["ted_raw"] = raw
            rows[row_index]["ted_norm"] = f"{norm:.9g}"
        except Exception as exc:
            rows[row_index]["ted_error"] = f"{type(exc).__name__}: {exc}"

    ted_values = [
        float(row["ted_norm"])
        for row in rows
        if row.get("ted_norm") not in ("", None)
    ]
    summary = {
        "run_dir": str(run_dir),
        "input_rows": len(metrics_rows),
        "output_rows": len(rows),
        "rows_with_extracted_fixed_kernel": sum(
            row["fixed_kernel_extracted"] == "true" for row in rows
        ),
        "rows_with_ted": len(ted_values),
        "rows_without_ted": len(rows) - len(ted_values),
        "unique_nonzero_cfg_pairs_computed": len(ted_by_cfg_pair),
        "ted_norm_min": min(ted_values) if ted_values else None,
        "ted_norm_median": statistics.median(ted_values) if ted_values else None,
        "ted_norm_mean": statistics.fmean(ted_values) if ted_values else None,
        "ted_norm_max": max(ted_values) if ted_values else None,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CFG TED between original failed kernels and fix-it run turns."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--turn-csv",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/figures/turn_correctness_arch.csv",
    )
    parser.add_argument(
        "--reference-fns",
        type=Path,
        default=DEFAULT_REFERENCE_FNS,
        help="Reference function-name YAML used by analyze_patterns_batch.py",
    )
    parser.add_argument(
        "--output-name",
        default="gemini_fixed_kernel_ted.csv",
        help="Output CSV filename inside --output-dir",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N rows/cache-misses; 0 disables progress output.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    turn_csv = args.turn_csv or (run_dir / "figures" / "turn_correctness_arch.csv")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_names = load_reference_fn_names(args.reference_fns)
    plan_by_exp = load_plan(run_dir)
    metrics_rows = load_turn_metrics(turn_csv)

    with tempfile.TemporaryDirectory(prefix="gemini-fixed-kernel-ted-") as tmp:
        rows, summary = build_rows(
            run_dir=run_dir,
            metrics_rows=metrics_rows,
            plan_by_exp=plan_by_exp,
            reference_names=reference_names,
            tmp_dir=Path(tmp),
            progress_every=args.progress_every,
        )

    out_csv = output_dir / args.output_name
    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("")

    out_summary = output_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote summary to {out_summary}")
    if summary["rows_with_ted"]:
        print(
            "TED norm: "
            f"median={summary['ted_norm_median']:.6g}, "
            f"mean={summary['ted_norm_mean']:.6g}, "
            f"min={summary['ted_norm_min']:.6g}, "
            f"max={summary['ted_norm_max']:.6g}"
        )
    else:
        print("No TED values computed")


if __name__ == "__main__":
    main()
