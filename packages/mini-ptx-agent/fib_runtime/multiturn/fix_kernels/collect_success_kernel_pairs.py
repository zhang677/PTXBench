#!/usr/bin/env python3
"""Collect <wrong, correct> kernel pairs from successful fixit eval runs.

For each success/<exp_id>/kernel_v*.cu artifact, the wrong kernel path is read
from the matching entry in plan.json. The per-run turn_correctness_arch.csv is
generated with AccRL's benchmark exporter when missing, then used for optional
arch-tag filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ACCRL_ROOT = Path("/home/ubuntu/AccRL")
sys.path.insert(0, str(ACCRL_ROOT))
from accrl.utils.code_utils import extract_code_block  # noqa: E402

EXPORT_TURN_CORRECTNESS = ACCRL_ROOT / "benchmark" / "export_turn_correctness_arch.py"
TURN_CSV_REL = Path("figures") / "turn_correctness_arch.csv"
EXP_ID_RE = re.compile(r"^exp_(\d+)$")
KERNEL_VERSION_RE = re.compile(r"kernel_v(\d+)\.cu$")
WRONG_KERNEL_TURN_RE = re.compile(r"kernel_t(\d+)\.cu$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--exp-dir",
        type=Path,
        action="append",
        help="Eval run directory. May be repeated.",
    )
    source.add_argument(
        "--selected-runs-csv",
        type=Path,
        help="CSV containing exp_dir rows and optional arch/definition metadata.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--arch-tag",
        action="append",
        dest="arch_tags",
        help="Required arch tag in turn_correctness_arch.csv, e.g. H or B. May be repeated.",
    )
    parser.add_argument(
        "--correct-kernel-mode",
        choices=["best", "latest", "all"],
        default="best",
        help="Which success kernel(s) to pair with each wrong kernel.",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        help=(
            "Only select correct kernels whose minimum "
            "evaluation.performance.speedup_factor meets this threshold."
        ),
    )
    parser.add_argument(
        "--force-turn-csv",
        action="store_true",
        help="Regenerate figures/turn_correctness_arch.csv even when it already exists.",
    )
    args = parser.parse_args()
    if args.min_speedup is not None and args.min_speedup < 0:
        parser.error("--min-speedup must be non-negative")
    return args


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def load_run_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.exp_dir is not None:
        rows = []
        for exp_dir in args.exp_dir:
            rows.append({"exp_dir": str(exp_dir.expanduser())})
        return rows

    fieldnames, rows = read_csv_rows(args.selected_runs_csv)
    if "exp_dir" not in fieldnames:
        raise ValueError(f"{args.selected_runs_csv} is missing required exp_dir column")
    return rows


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def temp_experiments_csv(row: dict[str, str]) -> Path:
    fields = ["model", "arch", "definition", "workload", "exp_dir"]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as f:
        path = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
    return path


def ensure_turn_csv(row: dict[str, str], *, force: bool) -> Path:
    exp_dir = Path(row["exp_dir"]).expanduser()
    turn_csv = exp_dir / TURN_CSV_REL
    if turn_csv.is_file() and not force:
        return turn_csv

    if not row.get("arch", "").strip():
        raise ValueError(f"{exp_dir} needs an arch column or inferable plan metadata to generate {TURN_CSV_REL}")

    experiments_csv = temp_experiments_csv(row)
    try:
        cmd = [
            sys.executable,
            str(EXPORT_TURN_CORRECTNESS),
            "--experiments-csv",
            str(experiments_csv),
        ]
        if force:
            cmd.append("--force")
        run_command(cmd)
    finally:
        experiments_csv.unlink(missing_ok=True)

    if not turn_csv.is_file():
        raise FileNotFoundError(f"expected generated CSV does not exist: {turn_csv}")
    return turn_csv


def load_plan_entries(run_dir: Path) -> dict[str, dict]:
    plan_path = run_dir / "plan.json"
    data = json.loads(plan_path.read_text())
    plan = data.get("plan")
    if not isinstance(plan, list):
        raise ValueError(f"{plan_path} has no list-valued plan field")

    by_exp_id = {}
    for entry in plan:
        exp_index = entry.get("exp_index")
        if exp_index is None:
            continue
        by_exp_id[f"exp_{int(exp_index):03d}"] = entry
    return by_exp_id


def wrong_kernel_path(entry: dict) -> str:
    for key in ("error_kernel_path", "wrong_kernel_path", "masked_kernel_path", "kernel_path"):
        value = str(entry.get(key, "")).strip()
        if value:
            return value
    return ""


def infer_wrong_turn(path_text: str) -> str:
    match = WRONG_KERNEL_TURN_RE.search(Path(path_text).name)
    return match.group(1) if match else ""


def infer_wrong_trajectory_path(path_text: str) -> str:
    path = Path(path_text).expanduser()
    try:
        kernels_dir = path.parents[1]
        exp_dir = path.parent.name
        if kernels_dir.name == "kernels" and exp_dir.startswith("exp_"):
            return str(kernels_dir.parent / "trajectories" / f"{exp_dir}.json")
    except IndexError:
        pass
    return ""


def infer_arch_from_plan_entry(entry: dict) -> str:
    text = " ".join(
        str(entry.get(key, ""))
        for key in ("arch", "gpu_arch", "prompt_tag", "test_path", "definition")
    ).lower()
    for arch in ("blackwell", "hopper", "ampere"):
        if arch in text:
            return arch
    return ""


def enrich_row_from_plan(row: dict[str, str], plan_entries: dict[str, dict]) -> dict[str, str]:
    enriched = dict(row)
    first_entry = next(iter(plan_entries.values()), {})
    if not enriched.get("definition", "").strip() and first_entry.get("definition"):
        enriched["definition"] = str(first_entry["definition"])
    if not enriched.get("arch", "").strip():
        inferred_arch = infer_arch_from_plan_entry(first_entry)
        if inferred_arch:
            enriched["arch"] = inferred_arch
    return enriched


def split_arch_tags(value: str) -> set[str]:
    return {
        tag.strip()
        for tag in value.split(",")
        if tag.strip()
    }


def load_arch_tags(turn_csv: Path) -> tuple[dict[str, set[str]], dict[tuple[str, int], set[str]]]:
    tags_by_trajectory: dict[str, set[str]] = {}
    tags_by_turn: dict[tuple[str, int], set[str]] = {}
    with turn_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory_id = row.get("trajectory_id", "").strip()
            if not trajectory_id:
                continue
            tags = split_arch_tags(row.get("arch_tag", ""))
            tags_by_trajectory.setdefault(trajectory_id, set()).update(tags)
            try:
                turn = int(row.get("turn", ""))
            except ValueError:
                continue
            tags_by_turn[(trajectory_id, turn)] = tags
    return tags_by_trajectory, tags_by_turn


def kernel_version(path: Path) -> int:
    match = KERNEL_VERSION_RE.search(path.name)
    return int(match.group(1)) if match else -1


def speedup_from_trace(trace: dict) -> float | None:
    speedup = (
        trace.get("evaluation", {})
        .get("performance", {})
        .get("speedup_factor")
    )
    if speedup is None:
        return None
    try:
        return float(speedup)
    except (TypeError, ValueError):
        return None


def record_version(entry: dict) -> int | None:
    version = entry.get("version")
    try:
        return int(version)
    except (TypeError, ValueError):
        return None


def load_record_entries(success_dir: Path) -> list[dict]:
    record_path = success_dir / "record.json"
    if not record_path.is_file():
        return []
    try:
        records = json.loads(record_path.read_text())
    except json.JSONDecodeError:
        return []
    return records if isinstance(records, list) else []


def turn_by_version_from_record(success_dir: Path) -> dict[int, int]:
    result = {}
    for entry in load_record_entries(success_dir):
        version = record_version(entry)
        if version is None:
            continue
        try:
            turn = int(entry.get("turn"))
        except (TypeError, ValueError):
            continue
        result[version] = turn
    return result


def normalize_kernel_source(source: str) -> str:
    return source.strip()


def extract_turn_kernels_from_trajectory(trajectory: dict) -> list[tuple[int, str]]:
    turns = []
    turn = 0
    for msg in trajectory.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        kernel_source = extract_code_block(content, languages=["cpp"], keep_separators=False)
        if kernel_source:
            turns.append((turn, normalize_kernel_source(kernel_source)))
        turn += 1
    return turns


def turn_by_version_from_trajectory(run_dir: Path, exp_id: str, success_dir: Path) -> dict[int, int]:
    traj_path = run_dir / "trajectories" / f"{exp_id}.json"
    if not traj_path.is_file():
        return {}
    try:
        trajectory = json.loads(traj_path.read_text())
    except json.JSONDecodeError:
        return {}

    turns_by_source: dict[str, list[int]] = {}
    for turn, kernel_source in extract_turn_kernels_from_trajectory(trajectory):
        turns_by_source.setdefault(kernel_source, []).append(turn)

    result = {}
    for kernel_path in sorted(success_dir.glob("kernel_v*.cu"), key=kernel_version):
        version = kernel_version(kernel_path)
        source = normalize_kernel_source(kernel_path.read_text(errors="replace"))
        turns = turns_by_source.get(source)
        if turns:
            result[version] = turns.pop(0)
    return result


def best_version_from_record(success_dir: Path, eligible_versions: set[int] | None = None) -> int | None:
    records = load_record_entries(success_dir)

    best_version = None
    best_speedup = None
    for entry in records:
        version = record_version(entry)
        if version is None or (eligible_versions is not None and version not in eligible_versions):
            continue
        for trace in entry.get("traces", []):
            speedup = speedup_from_trace(trace)
            if speedup is None:
                continue
            if best_speedup is None or speedup > best_speedup:
                best_speedup = speedup
                best_version = version
    return best_version


def min_speedup_by_version(success_dir: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for entry in load_record_entries(success_dir):
        version = record_version(entry)
        if version is None:
            continue
        speedups = [
            speedup
            for trace in entry.get("traces", [])
            if (speedup := speedup_from_trace(trace)) is not None
        ]
        if not speedups:
            continue
        version_min_speedup = min(speedups)
        if version in result:
            result[version] = min(result[version], version_min_speedup)
        else:
            result[version] = version_min_speedup
    return result


def select_correct_kernels(success_dir: Path, mode: str, min_speedup: float | None) -> list[Path]:
    kernels = sorted(success_dir.glob("kernel_v*.cu"), key=kernel_version)
    eligible_versions = None
    if min_speedup is not None:
        min_speedups = min_speedup_by_version(success_dir)
        eligible_versions = {
            version
            for version, version_min_speedup in min_speedups.items()
            if version_min_speedup >= min_speedup
        }
        kernels = [kernel for kernel in kernels if kernel_version(kernel) in eligible_versions]

    if mode == "all" or not kernels:
        return kernels
    if mode == "latest":
        return [kernels[-1]]

    best_version = best_version_from_record(success_dir, eligible_versions)
    if best_version is not None:
        best_path = success_dir / f"kernel_v{best_version}.cu"
        if best_path.is_file():
            return [best_path]
    return [kernels[-1]]


def collect_pairs_for_run(
    row: dict[str, str],
    plan_entries: dict[str, dict],
    required_arch_tags: set[str],
    correct_kernel_mode: str,
    min_speedup: float | None,
) -> tuple[list[dict[str, str]], list[str]]:
    run_dir = Path(row["exp_dir"]).expanduser()
    arch_tags_by_trajectory, arch_tags_by_turn = load_arch_tags(run_dir / TURN_CSV_REL)
    rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for success_dir in sorted((run_dir / "success").glob("exp_*")):
        if not success_dir.is_dir():
            continue
        exp_id = success_dir.name
        match = EXP_ID_RE.match(exp_id)
        if not match:
            continue
        correct_kernels = select_correct_kernels(success_dir, correct_kernel_mode, min_speedup)
        if not correct_kernels:
            continue

        fallback_tags = arch_tags_by_trajectory.get(exp_id, set())
        turn_by_version = turn_by_version_from_record(success_dir)
        turn_by_version.update(turn_by_version_from_trajectory(run_dir, exp_id, success_dir))

        plan_entry = plan_entries.get(exp_id)
        if plan_entry is None:
            warnings.append(f"missing plan entry for {run_dir}/{exp_id}")
            continue
        wrong_path = wrong_kernel_path(plan_entry)
        if not wrong_path:
            warnings.append(f"missing wrong kernel path in plan entry for {run_dir}/{exp_id}")
            continue

        for correct_path in correct_kernels:
            correct_version = kernel_version(correct_path)
            correct_turn = turn_by_version.get(correct_version)
            tags = (
                arch_tags_by_turn.get((exp_id, correct_turn), set())
                if correct_turn is not None
                else set()
            )
            if not tags:
                tags = fallback_tags
            if required_arch_tags and not required_arch_tags.issubset(tags):
                continue
            rows.append(
                {
                    "exp_dir": str(run_dir),
                    "arch": row.get("arch", ""),
                    "definition": plan_entry.get("definition", row.get("definition", "")),
                    "test_path": plan_entry.get("test_path", row.get("test_path", "")),
                    "trajectory_id": exp_id,
                    "prompt_tag": plan_entry.get("prompt_tag", ""),
                    "arch_tag": ", ".join(sorted(tags)),
                    "wrong_kernel_path": wrong_path,
                    "wrong_log_path": plan_entry.get("error_log_path", ""),
                    "wrong_trajectory_path": infer_wrong_trajectory_path(wrong_path),
                    "wrong_turn": infer_wrong_turn(wrong_path),
                    "correct_kernel_path": str(correct_path),
                    "correct_kernel_version": str(correct_version),
                    "plan_path": str(run_dir / "plan.json"),
                    "turn_csv": str(run_dir / TURN_CSV_REL),
                }
            )
    return rows, warnings


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "exp_dir",
        "arch",
        "definition",
        "test_path",
        "trajectory_id",
        "prompt_tag",
        "arch_tag",
        "wrong_kernel_path",
        "wrong_log_path",
        "wrong_trajectory_path",
        "wrong_turn",
        "correct_kernel_path",
        "correct_kernel_version",
        "plan_path",
        "turn_csv",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    required_arch_tags = {tag.strip() for tag in args.arch_tags or [] if tag.strip()}
    run_rows = load_run_rows(args)

    selected: list[dict[str, str]] = []
    warnings: list[str] = []
    for row in run_rows:
        exp_dir = Path(row["exp_dir"]).expanduser()
        if not (exp_dir / "plan.json").is_file():
            warnings.append(f"skipping missing plan.json: {exp_dir}")
            continue
        plan_entries = load_plan_entries(exp_dir)
        row = enrich_row_from_plan(row, plan_entries)
        ensure_turn_csv(row, force=args.force_turn_csv)
        rows, run_warnings = collect_pairs_for_run(
            row=row,
            plan_entries=plan_entries,
            required_arch_tags=required_arch_tags,
            correct_kernel_mode=args.correct_kernel_mode,
            min_speedup=args.min_speedup,
        )
        selected.extend(rows)
        warnings.extend(run_warnings)

    write_output(args.output_csv, selected)
    print(f"runs={len(run_rows)}")
    print(f"arch_tags={','.join(sorted(required_arch_tags)) if required_arch_tags else 'any'}")
    print(f"correct_kernel_mode={args.correct_kernel_mode}")
    print(f"min_speedup={args.min_speedup if args.min_speedup is not None else 'none'}")
    print(f"pairs={len(selected)}")
    print(f"wrote_csv={args.output_csv}")
    if warnings:
        print(f"warnings={len(warnings)}", file=sys.stderr)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
