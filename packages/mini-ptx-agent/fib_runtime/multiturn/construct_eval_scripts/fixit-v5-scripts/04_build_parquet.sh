#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
BALANCED_PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.parquet-balanced.csv"
REASONING_JSONL="$PROJECT/reasoning_pairs.glm-5.2-fixit-v5.jsonl"
PARQUET="$PROJECT/data/glm-5.2-fixit-v5.parquet"
TOKENIZER="Qwen/Qwen3.6-27B"
MAX_TOKENS="65536"
SHUFFLE_SEED="42"
MIN_SPEEDUP="0.05"
CAP_PER_DEFINITION="28"
PROCESS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"
BALANCE_SCRIPT="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/balance_kernel_pairs_for_parquet.py"

python "$BALANCE_SCRIPT" \
  --input-csv "$PAIRS_CSV" \
  --output-csv "$BALANCED_PAIRS_CSV" \
  --min-speedup "$MIN_SPEEDUP" \
  --cap-per-definition "$CAP_PER_DEFINITION"

python "$PROCESS" \
  --stages parquet \
  --pairs-csv "$BALANCED_PAIRS_CSV" \
  --reasoning-jsonl "$REASONING_JSONL" \
  --parquet "$PARQUET" \
  --tokenizer "$TOKENIZER" \
  --parquet-max-tokens "$MAX_TOKENS" \
  --source-label fixit-v5-glm52 \
  --shuffle \
  --shuffle-seed "$SHUFFLE_SEED"
