#!/usr/bin/env python3
"""Export per-turn correctness, speedup, and arch tags for eval runs.

Reads a top-level experiments.csv with an exp_dir column, extracts trajectory
turn outcomes from each listed run, and writes one CSV per run under:

    <exp_dir>/figures/turn_correctness_arch.csv
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ACCRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACCRL_ROOT))
from accrl.utils.code_utils import extract_code_block  # noqa: E402

TVM_FFI_DIR = Path("/home/ubuntu/miniconda3/envs/acc/lib/python3.12/site-packages/tvm_ffi")
ARCH_TO_GPU_ARCH = {
    "hopper": "compute_90a",
    "blackwell": "compute_100a",
}

_ptx_cache: dict[str, str | None] = {}


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


def compile_to_ptx(cu_path: Path, gpu_arch: str, tvm_ffi_dir: Path) -> str | None:
    source = cu_path.read_text(errors="replace")
    key = hashlib.sha256(f"{source}\0{gpu_arch}".encode()).hexdigest()
    if key in _ptx_cache:
        return _ptx_cache[key]

    include_dir = tvm_ffi_dir / "include"
    if not include_dir.exists():
        _ptx_cache[key] = None
        return None

    with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as tmp:
        ptx_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                "nvcc",
                "--ptx",
                "-O3",
                "-gencode",
                f"arch={gpu_arch},code={gpu_arch}",
                str(cu_path),
                f"-I{include_dir}",
                "-std=c++17",
                "-o",
                str(ptx_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            _ptx_cache[key] = None
            return None
        ptx_text = ptx_path.read_text(errors="replace")
        _ptx_cache[key] = ptx_text
        return ptx_text
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        _ptx_cache[key] = None
        return None
    finally:
        ptx_path.unlink(missing_ok=True)


def check_arch_from_text(text: str) -> str:
    arch_tags = []
    if any(
        kw in text
        for kw in [
            "cp.async.cg.shared.global",
            "cp.async.ca.shared.global",
            "ldmatrix.sync.aligned",
            "wmma::load_matrix_sync",
            "wmma::store_matrix_sync",
            "mma.sync.aligned",
            "wmma::mma_sync",
        ]
    ):
        arch_tags.append("A")
    if any(
        kw in text
        for kw in [
            "stmatrix.sync.aligned",
            "cp.async.bulk.tensor",
            "barrier.cluster",
            "setmaxnreg.",
            "elect.sync",
            "mapa.",
            "mbarrier",
            "wgmma."
        ]
    ):
        arch_tags.append("H")
    if any(kw in text for kw in [
            "tcgen05", 
            "red.global.v4"
        ]):
        arch_tags.append("B")
    return ", ".join(arch_tags) if arch_tags else "G"


def check_arch(kernel_path: Path, gpu_arch: str, tvm_ffi_dir: Path) -> str:
    ptx = compile_to_ptx(kernel_path, gpu_arch, tvm_ffi_dir)
    if ptx is not None:
        return check_arch_from_text(ptx)
    return check_arch_from_text(kernel_path.read_text(errors="replace"))


def check_arch_from_source(source: str, gpu_arch: str, tvm_ffi_dir: Path) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as tmp:
        kernel_path = Path(tmp.name)
        tmp.write(source)
    try:
        return check_arch(kernel_path, gpu_arch, tvm_ffi_dir)
    finally:
        kernel_path.unlink(missing_ok=True)


def find_best_kernel_in_success(run_dir: Path, trajectory_id: str) -> tuple[Path | None, float | None]:
    record_path = run_dir / "success" / trajectory_id / "record.json"
    if not record_path.exists():
        return None, None

    records = json.loads(record_path.read_text())
    best_version = None
    best_speedup = None
    for entry in records:
        for trace in entry.get("traces", []):
            speedup = trace.get("evaluation", {}).get("performance", {}).get("speedup_factor")
            if speedup is not None and (best_speedup is None or speedup > best_speedup):
                best_speedup = speedup
                best_version = entry.get("version")

    if best_version is None:
        return None, None

    kernel_path = run_dir / "success" / trajectory_id / f"kernel_v{best_version}.cu"
    if kernel_path.exists():
        return kernel_path, best_speedup
    return None, best_speedup


def classify_turn(content: str) -> str:
    if "PASSED" in content:
        return "Correct"
    if "INCORRECT_NUMERICAL" in content or "Result is incorrect" in content:
        return "Numerical error"
    if "Failed to compile kernel" in content:
        return "Compilation error"
    if re.search(r"Timed out after \d+(?:\.\d+)?s waiting for sanitize", content):
        return "Sanitize Timeout"
    if "returncode 137" in content:
        return "Profiling Service Timeout"
    if (
        "Kernel execution timed out" in content
        or "Evaluation timeout after" in content
        or "memcheck timed out" in content
        or re.search(r"\]\s+TIMEOUT\b", content)
    ):
        return "Kernel Execution Timeout"
    if "Could not extract" in content:
        return "Extraction error"
    if "RUNTIME_ERROR" in content or "CUDA error" in content or "CU error" in content:
        return "Runtime error"
    return "Other error"


def extract_turn_sequence(traj: dict) -> list[str]:
    seq = []
    first_user = True
    for i, msg in enumerate(traj.get("messages", [])):
        if msg.get("role") != "user" or i == 0:
            continue
        if first_user:
            first_user = False
            continue
        seq.append(classify_turn(msg.get("content", "")))
    return seq


def speedup_from_eval_message(eval_message: dict) -> float | None:
    extra = eval_message.get("extra") or {}
    speedups = []
    for trace in extra.get("traces") or []:
        speedup = nested_get(trace, ("evaluation", "performance", "speedup_factor"))
        value = as_number(speedup)
        if value is not None:
            speedups.append(value)
    if speedups:
        return max(speedups)
    return as_number(extra.get("min_speedup"))


def extract_turn_speedups(traj: dict) -> dict[int, float]:
    speedups = {}
    first_user = True
    fallback_turn = -1
    for i, msg in enumerate(traj.get("messages", [])):
        if msg.get("role") != "user" or i == 0:
            continue
        if first_user:
            first_user = False
            continue
        fallback_turn += 1
        turn = as_number(nested_get(msg, ("extra", "rollout", "turn_idx")))
        turn = int(turn) if turn is not None else fallback_turn
        speedup = speedup_from_eval_message(msg)
        if speedup is not None:
            speedups[turn] = speedup
    return speedups


def assistant_eval_turns(traj: dict) -> list[tuple[int, dict, dict]]:
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
        elif role == "assistant" and saw_initial_user:
            pending_assistant = msg

    return turns


def best_arch_tag_from_trajectory(traj: dict) -> str:
    best_speedup = None
    best_content = ""
    for _turn, assistant_message, eval_message in assistant_eval_turns(traj):
        speedup = speedup_from_eval_message(eval_message)
        if speedup is None or speedup <= 0:
            continue
        if best_speedup is None or speedup > best_speedup:
            content = assistant_message.get("content", "")
            best_speedup = speedup
            best_content = content if isinstance(content, str) else ""
    return check_arch_from_text(best_content) if best_content else ""


def extract_turn_arch_tags(
    traj: dict,
    correctness_by_turn: list[str],
    gpu_arch: str,
    tvm_ffi_dir: Path,
) -> dict[int, str]:
    tags = {}
    for turn, assistant_message, _eval_message in assistant_eval_turns(traj):
        if turn >= len(correctness_by_turn) or correctness_by_turn[turn] != "Correct":
            tags[turn] = ""
            continue
        content = assistant_message.get("content", "")
        kernel_source = extract_code_block(
            content if isinstance(content, str) else "",
            languages=["cpp"],
            keep_separators=False,
        )
        tags[turn] = check_arch_from_source(kernel_source, gpu_arch, tvm_ffi_dir) if kernel_source else ""
    return tags


def gpu_arch_for_row(row: dict[str, str]) -> str:
    arch = row.get("arch", "").strip().lower()
    if arch in ARCH_TO_GPU_ARCH:
        return ARCH_TO_GPU_ARCH[arch]
    if arch == "comput_100a":
        return "compute_100a"
    if arch.startswith("compute_"):
        return arch
    raise ValueError(f"Unsupported arch value {row.get('arch')!r}")


def export_run(run_dir: Path, gpu_arch: str, out_name: str, tvm_ffi_dir: Path) -> int:
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        raise FileNotFoundError(f"Missing trajectories directory: {traj_dir}")

    rows = []
    for traj_path in sorted(traj_dir.glob("*.json")):
        trajectory_id = traj_path.stem
        traj = json.loads(traj_path.read_text())
        seq = extract_turn_sequence(traj)
        speedups_by_turn = extract_turn_speedups(traj)
        arch_tags_by_turn = extract_turn_arch_tags(traj, seq, gpu_arch, tvm_ffi_dir)

        for turn, correctness in enumerate(seq):
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "turn": turn,
                    "correctness": correctness,
                    "speedup": speedups_by_turn.get(turn),
                    "arch_tag": arch_tags_by_turn.get(turn, ""),
                }
            )

    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trajectory_id", "turn", "correctness", "speedup", "arch_tag"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_experiment_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"arch", "exp_dir"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments.csv",
        help="CSV with arch and exp_dir columns",
    )
    parser.add_argument(
        "--out-name",
        default="turn_correctness_arch.csv",
        help="Output filename inside each exp_dir/figures directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate CSVs even when the output file already exists",
    )
    parser.add_argument(
        "--tvm-ffi-dir",
        type=Path,
        default=TVM_FFI_DIR,
        help="Path to tvm_ffi package containing include/",
    )
    args = parser.parse_args()

    for row in load_experiment_rows(args.experiments_csv):
        run_dir = Path(row["exp_dir"])
        out_path = run_dir / "figures" / args.out_name
        if out_path.exists() and not args.force:
            print(f"{run_dir}: skipped existing figures/{args.out_name}")
            continue
        gpu_arch = gpu_arch_for_row(row)
        n_rows = export_run(
            run_dir=run_dir,
            gpu_arch=gpu_arch,
            out_name=args.out_name,
            tvm_ffi_dir=args.tvm_ffi_dir,
        )
        print(f"{run_dir}: wrote {n_rows} rows to figures/{args.out_name}")


if __name__ == "__main__":
    main()
