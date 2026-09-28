#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for Gemini source runs}"
python "$PTXBENCH_SHARED_ROOT/run_manifest.py" "$SCRIPT_DIR/source-runs.csv" --run \
  --service-url "${SERVICE_URL:-http://localhost:10000}" \
  --max-parallel "${MAX_PARALLEL:-4}" --max-profiles "${MAX_PROFILES:-4}"
