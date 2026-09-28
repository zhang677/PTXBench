#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

python "$SCRIPT_DIR/build_dataset.py" \
  --pairs "$KERNELGEN_PROJECT/reasoning_pairs.jsonl" \
  --output "$KERNELGEN_PARQUET" \
  --reasoning-field reasoning \
  --tokenizer Qwen/Qwen3.6-27B \
  --max-tokens 65536 \
  --normalize-with-chat-template
