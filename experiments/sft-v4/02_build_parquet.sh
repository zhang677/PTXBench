#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

python "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/build_sft_dataset_sft_v4.py" \
  --pairs "$SFT_V4_PROJECT/reasoning_pairs.jsonl" \
  --output "$SFT_V4_PARQUET" \
  --reasoning-field reasoning \
  --tokenizer Qwen/Qwen3.6-27B \
  --max-tokens 65536 \
  --normalize-with-chat-template
