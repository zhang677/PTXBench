#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="/home/ubuntu/AccRL-exps/eval_runs/fixit-v5-qwen36-linfo-gemini"
PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"

test -d "$OUTPUT_ROOT"
mkdir -p "$PROJECT"

python /home/ubuntu/AccRL/fib_runtime/multiturn/fix_kernels/collect_success_kernel_pairs.py \
  --exp-dir "$OUTPUT_ROOT" \
  --output-csv "$PAIRS_CSV" \
  --correct-kernel-mode all \
  --min-speedup 0.0 \
  --arch-tag H \
  --force-turn-csv
