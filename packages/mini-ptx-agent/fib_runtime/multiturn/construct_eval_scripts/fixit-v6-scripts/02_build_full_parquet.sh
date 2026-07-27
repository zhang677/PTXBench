#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../ptxbench_paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
REASONING_JSONL="$PROJECT/reasoning_pairs.qwen36-27b-fixit-v6-full.jsonl"
PARQUET="$PROJECT/data/qwen36-27b-fixit-v6-full.parquet"
TOKENIZER="Qwen/Qwen3.6-27B"
MAX_TOKENS="65536"
SHUFFLE_SEED="42"
PROCESS="$PTXBENCH_CONSTRUCT_EVAL_ROOT/fixit_downstream_process.py"

if [[ ! -f "$REASONING_JSONL" ]]; then
  echo "Missing repaired reasoning JSONL: $REASONING_JSONL" >&2
  echo "Run 01_resynthesize_filtered_reasonings.sh before building the full parquet." >&2
  exit 1
fi

python "$PROCESS" \
  --stages parquet \
  --pairs-csv "$PAIRS_CSV" \
  --reasoning-jsonl "$REASONING_JSONL" \
  --parquet "$PARQUET" \
  --tokenizer "$TOKENIZER" \
  --parquet-max-tokens "$MAX_TOKENS" \
  --source-label fixit-v6-qwen36-27b \
  --shuffle \
  --shuffle-seed "$SHUFFLE_SEED"
