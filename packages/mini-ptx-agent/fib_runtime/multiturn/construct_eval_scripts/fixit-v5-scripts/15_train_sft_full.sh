#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PARQUET="$PROJECT/data/glm-5.2-fixit-v5-full.parquet"
TRAIN_SESSION="train-fixit-v5-full-glm52"
TRAIN_RUN_TAG="qwen36-27b-glm52-fixit-v5-full-e5-lr4.65e-4-lora32"
PROCESS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"

python "$PROCESS" \
  --stages train \
  --parquet "$PARQUET" \
  --train-session "$TRAIN_SESSION" \
  --train-run-tag "$TRAIN_RUN_TAG"
