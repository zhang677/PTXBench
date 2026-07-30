#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

python "$MINI_PTX_AGENT_ROOT/accrl/distill/sft/build_sft_dataset_kernelgen.py" \
  --pairs "$KERNELGEN_PROJECT/reasoning_pairs.jsonl" \
  --output "$KERNELGEN_PARQUET" \
  --reasoning-field reasoning \
  --tokenizer Qwen/Qwen3.6-27B \
  --max-tokens 65536 \
  --normalize-with-chat-template
