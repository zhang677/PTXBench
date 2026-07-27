#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
SOURCE_RUNS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/fixit-v5-source-runs.csv"
PREBALANCED_CSV="$PROJECT/fixit-v5-error-kernels.prebalanced.csv"
UNFILTERED_CSV="$PROJECT/fixit-v5-error-kernels.unfiltered.csv"
BALANCED_CSV="$PROJECT/fixit-v5-error-kernels.csv"
ENRICHED_CSV="$PROJECT/fixit-v5-error-kernels.enriched.csv"
CONFIG_JSON="$PROJECT/fixit-v5-gemini-source-prompt-config.json"
MANIFEST_JSON="$PROJECT/fixit-v5-manifest.json"
BUILDER="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/build_rebalanced_fixit_error_collection.py"
PROMPT_TAG_FILTER="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/filter_fixit_error_kernels_by_prompt_tag.py"
DIST_COUNTER="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/count_error_kernel_distribution.py"
PREBALANCED_DIST="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/fixit-v5-error-type-distribution.prebalanced.csv"
BALANCED_DIST="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/fixit-v5-error-type-distribution.csv"

mkdir -p "$PROJECT"

SELECT_ARGS=(
  --selected-runs-csv "$SOURCE_RUNS"
  --output-csv "$UNFILTERED_CSV"
)

if [[ "${FORCE_DERIVED:-0}" == "1" ]]; then
  SELECT_ARGS+=(--force-turn-csv --force-kernels)
fi

echo "[1/4] Select failed source kernels"
python /home/ubuntu/AccRL/fib_runtime/multiturn/fix_kernels/select_failed_kernels.py "${SELECT_ARGS[@]}"

echo "[2/4] Filter to fixit-v5 prompt tags and source turns"
python "$PROMPT_TAG_FILTER" \
  --input-csv "$UNFILTERED_CSV" \
  --output-csv "$PREBALANCED_CSV"

echo "[3/4] Rebalance rows and write Gemini source prompt config"
python "$BUILDER" \
  --input-csv "$PREBALANCED_CSV" \
  --balanced-csv "$BALANCED_CSV" \
  --enriched-csv "$ENRICHED_CSV" \
  --config-json "$CONFIG_JSON" \
  --manifest-json "$MANIFEST_JSON" \
  --source-markdown "/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5.md" \
  --source-runs-csv "$SOURCE_RUNS" \
  --output-root "/home/ubuntu/AccRL-exps/eval_runs/fixit-v5-qwen36-linfo-gemini" \
  --per-definition-cap 33 \
  --num-turns 5 \
  --target-speedup 0.15 \
  --min-speedup-for-later-collection 0.0

echo "[4/4] Record prebalanced and balanced error-type distributions"
python "$DIST_COUNTER" \
  --source-runs-csv "$SOURCE_RUNS" \
  --input-csv "$PREBALANCED_CSV" \
  --manifest "$MANIFEST_JSON" \
  --skip-manifest-check \
  --output-csv "$PREBALANCED_DIST"

python "$DIST_COUNTER" \
  --source-runs-csv "$SOURCE_RUNS" \
  --input-csv "$ENRICHED_CSV" \
  --manifest "$MANIFEST_JSON" \
  --output-csv "$BALANCED_DIST"
