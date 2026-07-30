#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MINI_PTX_AGENT_ROOT="${MINI_PTX_AGENT_ROOT:-$PTXBENCH_ROOT/packages/mini-ptx-agent}"
PTXBENCH_DATA_ROOT="${PTXBENCH_DATA_ROOT:-$PTXBENCH_ROOT/data}"
EXPERIMENT_ROOT="$PTXBENCH_ROOT/experiments/kernelgen"
export PTXBENCH_ROOT MINI_PTX_AGENT_ROOT PTXBENCH_DATA_ROOT

declare -a STAGES=(
  00_enrich_correct_kernels.sh
  01_synthesize_reasoning.sh
  02_build_parquet.sh
  03_train_sft.sh
  04_serve_remote.sh
  05_watch_5defs_eval.sh
)

usage() {
  echo "Usage: scripts/reproduce_kernelgen.sh --check | --check-data | 00..05 | all" >&2
}

check_source() {
  for script in "${STAGES[@]}"; do
    test -x "$EXPERIMENT_ROOT/$script" || {
      echo "missing executable stage: $EXPERIMENT_ROOT/$script" >&2
      return 1
    }
    bash -n "$EXPERIMENT_ROOT/$script"
  done
  local required=(
    "$MINI_PTX_AGENT_ROOT/accrl/distill/inspector.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/build_sft_dataset_kernelgen.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/tinker_download_weights.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/tinker_sft_train.py"
    "$MINI_PTX_AGENT_ROOT/accrl/utils/code_utils.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/analyze_pattern.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/build_doc_v2.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/common.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/launcher_utils.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/resume_utils.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_parallel_v2.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_v2.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/collect_kernels/collect_correct_kernels.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/collect_notes/note_feedback.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/task_to_correct_kernels/enrich_correct_kernels_for_reasoning.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/task_to_correct_kernels/synthesize_correct_kernel_reasoning_openrouter.py"
  )
  for path in "${required[@]}"; do
    test -f "$path" || {
      echo "missing required source: $path" >&2
      return 1
    }
  done
  python -m compileall -q "${required[@]}"
  local support=(
    "$EXPERIMENT_ROOT/source-runs.csv"
    "$PTXBENCH_ROOT/experiments/fixit/05_watch_5defs_eval.sh"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/user_template.txt"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/prompt_configs/hub.json"
  )
  for path in "${support[@]}"; do
    test -f "$path" || {
      echo "missing required source: $path" >&2
      return 1
    }
  done
  echo "KernelGen source check passed: ${#STAGES[@]} pipeline stages"
}

check_data() {
  check_source
  python - "$EXPERIMENT_ROOT/source-runs.csv" <<'PY'
import csv
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
rows = list(csv.DictReader(source.open(newline="")))
if len(rows) != 12:
    raise SystemExit(f"{source}: expected 12 source runs, found {len(rows)}")
missing = []
selected_rows = 0
arch_tags = {"ampere": "A", "hopper": "H", "blackwell": "B"}
for row in rows:
    exp_dir = Path(os.path.expandvars(row["exp_dir"])).expanduser()
    test_path = Path(os.path.expandvars(row["test_path"])).expanduser()
    turn_csv = exp_dir / "figures" / "turn_correctness_arch.csv"
    for label, path in (
        ("eval run", exp_dir),
        ("test", test_path),
        ("turn correctness", turn_csv),
    ):
        if not path.exists():
            missing.append((row["exp_dir"], label, path))
    if not (exp_dir / "plan.json").is_file() and not (exp_dir / "summary.json").is_file():
        missing.append((row["exp_dir"], "plan or summary", exp_dir))
    if not turn_csv.is_file():
        continue
    expected_tag = arch_tags[row["arch"].strip().lower()]
    for turn_row in csv.DictReader(turn_csv.open(newline="")):
        try:
            speedup = float(turn_row.get("speedup", ""))
        except ValueError:
            continue
        tags = {tag.strip() for tag in turn_row.get("arch_tag", "").split(",")}
        if (
            turn_row.get("correctness") != "Correct"
            or speedup <= 0
            or expected_tag not in tags
        ):
            continue
        selected_rows += 1
        trajectory_id = turn_row["trajectory_id"]
        turn = int(turn_row["turn"])
        for label, path in (
            (
                "selected kernel",
                exp_dir / "kernels" / trajectory_id / f"kernel_t{turn}.cu",
            ),
            ("selected trajectory", exp_dir / "trajectories" / f"{trajectory_id}.json"),
        ):
            if not path.is_file():
                missing.append((row["exp_dir"], label, path))
if missing:
    preview = "\n".join(f"{run}: missing {label}: {path}" for run, label, path in missing[:20])
    raise SystemExit(f"KernelGen bundle has {len(missing)} missing paths:\n{preview}")
if selected_rows != 521:
    raise SystemExit(f"KernelGen bundle expected 521 selected rows, found {selected_rows}")
print(f"KernelGen source-run data closure passed: {len(rows)} runs, {selected_rows} rows")
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
  all)
    check_data
    for script in "${STAGES[@]}"; do
      bash "$EXPERIMENT_ROOT/$script"
    done
    ;;
  0[0-5])
    bash "$EXPERIMENT_ROOT/${STAGES[10#$mode]}"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
