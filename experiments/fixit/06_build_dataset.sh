#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PAIRS_CSV="$PROJECT/kernel-pairs.csv"
REASONING_JSONL="$PROJECT/reasoning-pairs.filtered.jsonl"
PARQUET="$PROJECT/data/fixit.parquet"
TOKENIZER="Qwen/Qwen3.6-27B"
MAX_TOKENS="65536"
SHUFFLE_SEED="42"
PROCESS="$PTXBENCH_SHARED_ROOT/downstream.py"

if [[ ! -f "$REASONING_JSONL" ]]; then
  echo "Missing repaired reasoning JSONL: $REASONING_JSONL" >&2
  echo "Run stage 05 before building the parquet." >&2
  exit 1
fi

python "$PROCESS" \
  --stages parquet \
  --pairs-csv "$PAIRS_CSV" \
  --reasoning-jsonl "$REASONING_JSONL" \
  --parquet "$PARQUET" \
  --tokenizer "$TOKENIZER" \
  --parquet-max-tokens "$MAX_TOKENS" \
  --source-label fixit-qwen36-27b \
  --shuffle \
  --shuffle-seed "$SHUFFLE_SEED"
