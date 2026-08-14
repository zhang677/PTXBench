#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MINI_PTX_AGENT_ROOT="${MINI_PTX_AGENT_ROOT:-$PTXBENCH_ROOT/packages/mini-ptx-agent}"
PTXBENCH_DATA_ROOT="${PTXBENCH_DATA_ROOT:-$PTXBENCH_ROOT/data}"
FIXIT_ROOT="$PTXBENCH_ROOT/experiments/fixit"
PROJECT="${PTXBENCH_FIXIT_PROJECT:-$PTXBENCH_DATA_ROOT/sft_experiments/test-fixit-qwen36-27b-gemini-glm}"
export PTXBENCH_ROOT MINI_PTX_AGENT_ROOT PTXBENCH_DATA_ROOT

declare -a SOURCE_STAGES=(
  source_00_watch_qwen36_linfo_mha.sh
  source_01_prepare_gemini_repairs.sh
  source_02_watch_gemini_repairs.sh
  source_03_collect_kernel_pairs.sh
)

declare -a STAGES=(
  00_synthesize_qwen36-27b_reasoning.sh
  01_resynthesize_filtered_reasonings.sh
  02_build_full_parquet.sh
  03_train_sft_full.sh
  04_serve_remote_full.sh
  05_watch_5defs_eval.sh
)

usage() {
  echo "Usage: scripts/reproduce_fixit.sh --check | --check-data | source-00..source-03 | source-all | 00..05 | all | from-scratch" >&2
}

check_source() {
  for script in "${SOURCE_STAGES[@]}" "${STAGES[@]}"; do
    test -x "$FIXIT_ROOT/$script" || {
      echo "missing executable stage: $FIXIT_ROOT/$script" >&2
      return 1
    }
    bash -n "$FIXIT_ROOT/$script"
  done
  local required=(
    "$MINI_PTX_AGENT_ROOT/accrl/distill/inspector.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/build_sft_dataset_fixit.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/tinker_download_weights.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/tinker_sft_train.py"
    "$MINI_PTX_AGENT_ROOT/accrl/utils/code_utils.py"
    "$MINI_PTX_AGENT_ROOT/benchmark/export_turn_correctness_arch.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/analyze_kernel_per_turn.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/analyze_pattern.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/build_doc_v2.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/common.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/launcher_utils.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/resume_utils.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_parallel_v2.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_v2.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/collect_notes/note_feedback.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/build_rebalanced_fixit_error_collection.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/filter_fixit_error_kernels_by_prompt_tag.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/fix_kernels/select_failed_kernels.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/fix_kernels/run_parallel_fix_v2.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/fix_kernels/collect_success_kernel_pairs.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/fix_kernels/synthesize_pair_reasoning_openrouter.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/fix_kernels/resynthesize_filtered_reasonings_openrouter.py"
  )
  for path in "${required[@]}"; do
    test -f "$path" || {
      echo "missing required source: $path" >&2
      return 1
    }
  done
  python -m compileall -q "${required[@]}"
  local support=(
    "$FIXIT_ROOT/source-runs.csv"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/user_template.txt"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/prompt_configs/hub.json"
    "$PTXBENCH_ROOT/configs/fixit/2026-0605-mha-p4.json"
    "$PTXBENCH_ROOT/configs/fixit/2026-0605-mha-bwd-p4.json"
    "$PTXBENCH_ROOT/configs/fixit/2026-0605-mha-p4-mha-patched.json"
    "$PTXBENCH_ROOT/configs/fixit/2026-0605-mha-bwd-p4-mha-patched.json"
    "$PTXBENCH_ROOT/configs/fixit/gemm-3-r8-p4.json"
  )
  for path in "${support[@]}"; do
    test -f "$path" || {
      echo "missing required source: $path" >&2
      return 1
    }
  done
  echo "Fixit source check passed: ${#SOURCE_STAGES[@]} source stages, ${#STAGES[@]} pipeline stages"
}

check_data() {
  check_source
  local pairs="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
  test -f "$pairs" || {
    echo "missing Fixit input bundle entry: $pairs" >&2
    return 1
  }
  python - "$pairs" <<'PY'
import csv
import os
import sys
from pathlib import Path

pairs = Path(sys.argv[1])
rows = list(csv.DictReader(pairs.open(newline="")))
if len(rows) != 258:
    raise SystemExit(f"{pairs}: expected 258 rows, found {len(rows)}")
required = (
    "test_path",
    "wrong_kernel_path",
    "wrong_log_path",
    "wrong_trajectory_path",
    "correct_kernel_path",
    "plan_path",
    "turn_csv",
)

def resolved(value):
    path = Path(os.path.expandvars(value)).expanduser()
    data_root = Path(os.environ["PTXBENCH_DATA_ROOT"]).expanduser()
    mini_root = Path(os.environ["MINI_PTX_AGENT_ROOT"]).expanduser()
    if "eval_runs" in path.parts:
        candidate = data_root.joinpath(*path.parts[path.parts.index("eval_runs"):])
        if candidate.exists():
            return candidate
    if "sft_experiments" in path.parts:
        candidate = data_root.joinpath(*path.parts[path.parts.index("sft_experiments"):])
        if candidate.exists():
            return candidate
    if "fib_runtime" in path.parts:
        candidate = mini_root.joinpath(*path.parts[path.parts.index("fib_runtime"):])
        if candidate.exists():
            return candidate
    return path

missing = [
    (index, key, row.get(key, ""))
    for index, row in enumerate(rows, 2)
    for key in required
    if not resolved(row.get(key, "")).is_file()
]
for index, row in enumerate(rows, 2):
    synthesis_trajectory = (
        resolved(row.get("exp_dir", ""))
        / "trajectories"
        / f"{row.get('trajectory_id', '')}.json"
    )
    success_record = resolved(row.get("correct_kernel_path", "")).parent / "record.json"
    if not synthesis_trajectory.is_file():
        missing.append((index, "synthesis_trajectory", str(synthesis_trajectory)))
    if not success_record.is_file():
        missing.append((index, "success_record", str(success_record)))
if missing:
    preview = "\n".join(f"row {index}: missing {key}={value}" for index, key, value in missing[:20])
    raise SystemExit(f"Fixit bundle has {len(missing)} missing referenced files:\n{preview}")
print(f"Fixit data closure passed: {len(rows)} pairs")
PY
}

mode="${1:---check}"
case "$mode" in
  --check)
    check_source
    ;;
  --check-data)
    check_data
    ;;
  source-all)
    check_source
    for script in "${SOURCE_STAGES[@]}"; do
      bash "$FIXIT_ROOT/$script"
    done
    ;;
  source-0[0-3])
    source_index="${mode#source-}"
    bash "$FIXIT_ROOT/${SOURCE_STAGES[10#$source_index]}"
    ;;
  all)
    check_data
    for script in "${STAGES[@]}"; do
      bash "$FIXIT_ROOT/$script"
    done
    ;;
  from-scratch)
    check_source
    for script in "${SOURCE_STAGES[@]}" "${STAGES[@]}"; do
      bash "$FIXIT_ROOT/$script"
    done
    ;;
  0[0-5])
    bash "$FIXIT_ROOT/${STAGES[10#$mode]}"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
