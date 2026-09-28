#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PARQUET="$PROJECT/data/fixit.parquet"
TRAIN_SESSION="train-fixit"
TRAIN_RUN_TAG="qwen36-27b-fixit-example-e5-lr4.65e-4-lora32"
PROCESS="$PTXBENCH_SHARED_ROOT/downstream.py"

python "$PROCESS" \
  --stages train \
  --parquet "$PARQUET" \
  --runs-dir "$PROJECT/runs" \
  --train-session "$TRAIN_SESSION" \
  --train-run-tag "$TRAIN_RUN_TAG"
