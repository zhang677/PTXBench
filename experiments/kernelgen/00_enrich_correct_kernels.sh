#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

python "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/collect_kernels/collect_correct_kernels.py" \
  "$SCRIPT_DIR/source-runs.csv" \
  --min-speedup 0 \
  --output "$KERNELGEN_PROJECT/correct-kernels.csv"

python "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/task_to_correct_kernels/enrich_correct_kernels_for_reasoning.py" \
  --input-csv "$KERNELGEN_PROJECT/correct-kernels.csv" \
  --output-csv "$KERNELGEN_PROJECT/correct-kernels.enriched.csv"
