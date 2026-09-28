#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"
PROFILE_URL="${SASS_PROFILE_URL:-${SERVICE_URL:-http://localhost:10000}}"
python "$PTXBENCH_SHARED_ROOT/prepare_turn_data.py" "$SCRIPT_DIR/source-runs.csv" --profile-url "$PROFILE_URL"
mkdir -p "$KERNELGEN_PROJECT"
python "$SCRIPT_DIR/collect_correct_kernels.py" "$SCRIPT_DIR/source-runs.csv" \
  --min-speedup 0 --output "$KERNELGEN_PROJECT/correct-kernels.csv"
python "$SCRIPT_DIR/enrich_correct_kernels.py" \
  --input-csv "$KERNELGEN_PROJECT/correct-kernels.csv" \
  --output-csv "$KERNELGEN_PROJECT/correct-kernels.enriched.csv"
