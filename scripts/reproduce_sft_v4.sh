#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MINI_PTX_AGENT_ROOT="${MINI_PTX_AGENT_ROOT:-$PTXBENCH_ROOT/packages/mini-ptx-agent}"
PTXBENCH_DATA_ROOT="${PTXBENCH_DATA_ROOT:-$PTXBENCH_ROOT/data}"
EXPERIMENT_ROOT="$PTXBENCH_ROOT/experiments/sft-v4"
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
  echo "Usage: $0 --check | --check-data | 00..05 | all" >&2
}

check_source() {
  for script in "${STAGES[@]}"; do
    test -x "$EXPERIMENT_ROOT/$script" || {
      echo "missing executable stage: $EXPERIMENT_ROOT/$script" >&2
      return 1
    }
    bash -n "$EXPERIMENT_ROOT/$script"
  done
  python - "$PTXBENCH_ROOT" "$EXPERIMENT_ROOT/provenance.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
provenance = json.loads(Path(sys.argv[2]).read_text())
missing = [
    relative_path
    for relative_path in provenance["required_source_files"]
    if not (root / relative_path).is_file()
]
if missing:
    raise SystemExit("missing retained SFT-v4 source:\n" + "\n".join(missing))

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
    raise SystemExit("missing retained SFT-v4 prompt fragments:\n" + "\n".join(missing_fragments))
print(
    "SFT-v4 source closure passed: "
    f"{len(provenance['required_source_files'])} retained files, "
    f"{len(seen)} prompt tags, {len(fragments)} prompt fragments"
)
PY
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
    raise SystemExit(f"SFT-v4 bundle has {len(missing)} missing paths:\n{preview}")
if selected_rows != 521:
    raise SystemExit(f"SFT-v4 bundle expected 521 selected rows, found {selected_rows}")
print(f"SFT-v4 source-run data closure passed: {len(rows)} runs, {selected_rows} rows")
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
