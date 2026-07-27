#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PARQUET="$PROJECT/data/glm-5.2-fixit-v2-clean-v5-full-d128.parquet"
PROCESS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"
V2_FINAL_CHECKPOINT="tinker://62e73b90-5995-56a8-98d9-f31536036be5:train:0/weights/final"
TRAIN_SESSION="train-fixit-v2-clean-v5-full-d128-from-v2-final"
TRAIN_RUN_TAG="qwen36-27b-glm52-fixit-v2-clean-v5-full-d128-from-v2-final-e1-lr4.65e-4-lora32"

if [[ ! -f "$PARQUET" ]]; then
    echo "Missing combined parquet: $PARQUET" >&2
    echo "Run 19_combine_v2_clean_v5_full_parquet.sh first." >&2
    exit 1
fi

python "$PROCESS" \
    --stages train \
    --parquet "$PARQUET" \
    --train-session "$TRAIN_SESSION" \
    --train-run-tag "$TRAIN_RUN_TAG" \
    --train-num-epochs 1 \
    --train-learning-rate 4.65e-4 \
    --train-load-checkpoint-path "$V2_FINAL_CHECKPOINT"
