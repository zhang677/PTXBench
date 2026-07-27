#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
REASONING_JSONL="$PROJECT/reasoning_pairs.glm-5.2-fixit-v5-full.jsonl"
PARQUET="$PROJECT/data/glm-5.2-fixit-v5-full.parquet"
TOKENIZER="Qwen/Qwen3.6-27B"
MAX_TOKENS="65536"
SHUFFLE_SEED="42"
PROCESS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"

python "$PROCESS" \
  --stages parquet \
  --pairs-csv "$PAIRS_CSV" \
  --reasoning-jsonl "$REASONING_JSONL" \
  --parquet "$PARQUET" \
  --tokenizer "$TOKENIZER" \
  --parquet-max-tokens "$MAX_TOKENS" \
  --source-label fixit-v5-glm52 \
  --shuffle \
  --shuffle-seed "$SHUFFLE_SEED"
