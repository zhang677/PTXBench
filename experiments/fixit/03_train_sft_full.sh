#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PARQUET="$PROJECT/data/qwen36-27b-fixit-full.parquet"
TRAIN_SESSION="train-fixit-full-qwen36"
TRAIN_RUN_TAG="qwen36-27b-qwen36-fixit-full-e5-lr4.65e-4-lora32"
PROCESS="$PTXBENCH_CONSTRUCT_EVAL_ROOT/fixit_downstream_process.py"

python "$PROCESS" \
  --stages train \
  --parquet "$PARQUET" \
  --train-session "$TRAIN_SESSION" \
  --train-run-tag "$TRAIN_RUN_TAG"
