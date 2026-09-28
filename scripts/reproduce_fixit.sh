#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/experiments/shared/paths.sh"
STAGES=(
  00_run_sources.sh
  01_select_failures.sh
  02_repair_failures.sh
  03_collect_pairs.sh
  04_synthesize_reasoning.sh
  05_resynthesize_reasoning.sh
  06_build_dataset.sh
  07_train.sh
  08_serve.sh
  09_evaluate.sh
)

check() {
  local stage
  python "$ROOT/multiturn/scripts/render.py" --check
  for stage in "${STAGES[@]}"; do
    bash -n "$ROOT/experiments/fixit/$stage"
  done
  python "$ROOT/experiments/shared/run_manifest.py" "$ROOT/experiments/fixit/source-runs.csv" --check
  python "$ROOT/experiments/shared/run_manifest.py" "$ROOT/experiments/fixit/eval-runs.csv" --check
  python - "$ROOT" <<'PY'
import ast
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in [*(root / 'experiments/shared').glob('*.py'), *(root / 'experiments/fixit').glob('*.py')]:
    ast.parse(path.read_text(), filename=str(path))
print('Fixit source and stage checks passed')
PY
}

mode="${1:---check}"
case "$mode" in
  --check) check ;;
  all)
    check
    for stage in "${STAGES[@]}"; do
      bash "$ROOT/experiments/fixit/$stage"
    done
    ;;
  0[0-9])
    index=$((10#$mode))
    bash "$ROOT/experiments/fixit/${STAGES[$index]}"
    ;;
  --help|-h)
    echo 'Usage: scripts/reproduce_fixit.sh --check | all | 00..09'
    ;;
  *)
    echo "Unknown Fixit stage: $mode" >&2
    exit 2
    ;;
esac
