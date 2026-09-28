#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"
: "${PTXBENCH_MODEL_HOST:?Set PTXBENCH_MODEL_HOST to the served SFT endpoint host:port}"
python "$PTXBENCH_SHARED_ROOT/run_manifest.py" "$SCRIPT_DIR/eval-runs.csv" --run \
  --model Qwen3.6-27B --service-url "${SERVICE_URL:-http://localhost:10000}" \
  --max-parallel "${MAX_PARALLEL:-4}" --max-profiles "${MAX_PROFILES:-4}"
