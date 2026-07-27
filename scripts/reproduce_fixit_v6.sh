#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MINI_PTX_AGENT_ROOT="${MINI_PTX_AGENT_ROOT:-$PTXBENCH_ROOT/packages/mini-ptx-agent}"
PTXBENCH_DATA_ROOT="${PTXBENCH_DATA_ROOT:-$PTXBENCH_ROOT/data}"
FIXIT_ROOT="$PTXBENCH_ROOT/experiments/fixit-v6"
PROJECT="${PTXBENCH_FIXIT_PROJECT:-$PTXBENCH_DATA_ROOT/sft_experiments/test-fixit-qwen36-27b-gemini-glm}"
export PTXBENCH_ROOT MINI_PTX_AGENT_ROOT PTXBENCH_DATA_ROOT

declare -a STAGES=(
  00_synthesize_qwen36-27b_reasoning.sh
  01_resynthesize_filtered_reasonings.sh
  02_build_full_parquet.sh
  03_train_sft_full.sh
  04_serve_remote_full.sh
  05_watch_v6_full_5defs_eval.sh
  06_serve_patched_remote_full.sh
  07_watch_v6_full_5defs_eval.sh
)

usage() {
  echo "Usage: $0 --check | --check-data | 00..07 | all" >&2
}

check_source() {
  for script in "${STAGES[@]}"; do
    test -x "$FIXIT_ROOT/$script" || {
      echo "missing executable stage: $FIXIT_ROOT/$script" >&2
      return 1
    }
    bash -n "$FIXIT_ROOT/$script"
  done
  local required=(
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/fix_kernels/synthesize_pair_reasoning_openrouter.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/fix_kernels/resynthesize_filtered_reasonings_openrouter.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/build_sft_dataset_fixit.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/tinker_sft_train.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/tinker_download_weights.py"
    "$MINI_PTX_AGENT_ROOT/accrl/distill/inspector.py"
    "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_v2.py"
  )
  for path in "${required[@]}"; do
    test -f "$path" || {
      echo "missing required source: $path" >&2
      return 1
    }
  done
  python -m compileall -q "${required[@]}"
  python - "$PTXBENCH_ROOT" "$PTXBENCH_ROOT/experiments/fixit-v6/provenance.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
provenance_path = Path(sys.argv[2])
provenance = json.loads(provenance_path.read_text())
missing = [
    relative_path
    for relative_path in provenance["required_source_files"]
    if not (root / relative_path).is_file()
]
if missing:
    raise SystemExit("missing retained Fixit-v6 source:\n" + "\n".join(missing))

multiturn = root / "packages/mini-ptx-agent/fib_runtime/multiturn"
hub_path = multiturn / "prompt_configs/hub.json"
hub = json.loads(hub_path.read_text())
seen = set()
fragments = set()

def visit(tag):
    if tag in seen:
        return
    seen.add(tag)
    if tag not in hub:
        raise SystemExit(f"{hub_path}: missing required prompt tag {tag!r}")
    for item in hub[tag]:
        if "/" in item:
            fragments.add(item)
        else:
            visit(item)

for tag in provenance["required_prompt_tags"]:
    visit(tag)
missing_fragments = [
    relative_path
    for relative_path in sorted(fragments)
    if not (multiturn.parent / relative_path).is_file()
]
if missing_fragments:
    raise SystemExit("missing retained Fixit-v6 prompt fragments:\n" + "\n".join(missing_fragments))
print(
    "Fixit-v6 source closure passed: "
    f"{len(provenance['required_source_files'])} retained files, "
    f"{len(seen)} prompt tags, {len(fragments)} prompt fragments"
)
PY
}

check_data() {
  check_source
  local pairs="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
  test -f "$pairs" || {
    echo "missing Fixit-v6 input bundle entry: $pairs" >&2
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
    raise SystemExit(f"Fixit-v6 bundle has {len(missing)} missing referenced files:\n{preview}")
print(f"Fixit-v6 data closure passed: {len(rows)} pairs")
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
      bash "$FIXIT_ROOT/$script"
    done
    ;;
  0[0-7])
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
