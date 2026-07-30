#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PTXBENCH_FIXIT_REPAIR_ROOT}"
PAIRS_CSV="${PAIRS_CSV:-$PTXBENCH_FIXIT_PROJECT/fixit-v5-gemini-kernel-pairs.csv}"
COLLECTOR="$PTXBENCH_MULTITURN_ROOT/fix_kernels/collect_success_kernel_pairs.py"

test -d "$OUTPUT_ROOT" || {
  echo "Missing Gemini repair output root: $OUTPUT_ROOT" >&2
  exit 1
}
mkdir -p "$PTXBENCH_FIXIT_PROJECT"

python "$COLLECTOR" \
  --exp-dir "$OUTPUT_ROOT" \
  --output-csv "$PAIRS_CSV" \
  --correct-kernel-mode all \
  --min-speedup 0.0 \
  --arch-tag H \
  --force-turn-csv

echo "Fixit kernel pairs: $PAIRS_CSV"
