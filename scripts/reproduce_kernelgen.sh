#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/experiments/shared/paths.sh"
STAGES=(
  00_run_sources.sh
  01_collect_correct.sh
  02_synthesize_reasoning.sh
  03_build_dataset.sh
  04_train.sh
  05_serve.sh
  06_evaluate.sh
)

check() {
  local stage
  python "$ROOT/multiturn/scripts/render.py" --check
  for stage in "${STAGES[@]}"; do
    bash -n "$ROOT/experiments/kernelgen/$stage"
  done
  python "$ROOT/experiments/shared/run_manifest.py" "$ROOT/experiments/kernelgen/source-runs.csv" --check
  python "$ROOT/experiments/shared/run_manifest.py" "$ROOT/experiments/kernelgen/eval-runs.csv" --check
  source "$ROOT/experiments/kernelgen/paths.sh"
  python - "$ROOT" <<'PY'
import os
import sys
from pathlib import Path
import yaml
root = Path(sys.argv[1])
config = yaml.safe_load((root / 'experiments/kernelgen/synthesize.yaml').read_text())
for key in ('input_csv', 'output_jsonl', 'provenance_json'):
    value = os.path.expandvars(config[key])
    if '$' in value or not Path(value).is_absolute():
        raise ValueError(f'KernelGen synthesis path is unresolved: {key}={value}')
PY
  python - "$ROOT" <<'PY'
import ast
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in [*(root / 'experiments/shared').glob('*.py'), *(root / 'experiments/kernelgen').glob('*.py')]:
    ast.parse(path.read_text(), filename=str(path))
print('KernelGen source and stage checks passed')
PY
}

mode="${1:---check}"
case "$mode" in
  --check) check ;;
  all)
    check
    for stage in "${STAGES[@]}"; do
      bash "$ROOT/experiments/kernelgen/$stage"
    done
    ;;
  0[0-6])
    index=$((10#$mode))
    bash "$ROOT/experiments/kernelgen/${STAGES[$index]}"
    ;;
  --help|-h)
    echo 'Usage: scripts/reproduce_kernelgen.sh --check | all | 00..06'
    ;;
  *)
    echo "Unknown KernelGen stage: $mode" >&2
    exit 2
    ;;
esac
