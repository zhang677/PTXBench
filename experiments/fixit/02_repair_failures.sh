#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for Gemini repair generation}"
CONFIG="${CONFIG:-$PTXBENCH_FIXIT_PROJECT/gemini-source-prompt-config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PTXBENCH_FIXIT_REPAIR_ROOT}"
test -f "$CONFIG"
args=(
  --output-root "$OUTPUT_ROOT" --model gemini-3.1-pro-preview
  --service-url "${SERVICE_URL:-http://localhost:10000}"
  --gpu-arch hopper --without-local-gpu
  --max-parallel "${MAX_PARALLEL:-4}" --max-profiles "${MAX_PROFILES:-4}"
  --timeout "${TIMEOUT:-86400}" --turn-timeout "${TURN_TIMEOUT:-980}"
)
if [[ -d "$OUTPUT_ROOT" ]]; then
  test -f "$OUTPUT_ROOT/plan.json"
  args=(--resume "${args[@]}")
else
  args=(--config "$CONFIG" "${args[@]}")
fi
python "$PTXBENCH_MULTITURN_ROOT/run_parallel_v2.py" "${args[@]}"
