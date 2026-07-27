#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

: "${TINKER_API_KEY:?TINKER_API_KEY must be set}"

python "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py" \
  --stages train \
  --parquet "$SFT_V4_PARQUET" \
  --runs-dir "$SFT_V4_PROJECT/runs" \
  --base-model Qwen/Qwen3.6-27B \
  --train-session train-mha-8def-glm52 \
  --train-run-tag "$SFT_V4_RUN_TAG" \
  --train-num-epochs 5 \
  --train-learning-rate 4.65e-4
