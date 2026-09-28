#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PAIRS_CSV="$PROJECT/kernel-pairs.csv"
SYNTH_CONFIG="$PROJECT/reasoning-config.yaml"
REASONING_JSONL="$PROJECT/reasoning-pairs.jsonl"
PROVENANCE_JSON="$PROJECT/reasoning-provenance.json"
MAX_PASSES="${MAX_PASSES:-20}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
PROCESS="$PTXBENCH_SHARED_ROOT/downstream.py"

# Match common.py's Qwen3.6-27B routing: use LiteLLM's OpenAI-compatible
# provider and send requests to an explicitly selected, dedicated model serve.
: "${PTXBENCH_MODEL_HOST:?Set PTXBENCH_MODEL_HOST to the base Qwen endpoint host:port}"
export OPENAI_BASE_URL="http://${PTXBENCH_MODEL_HOST}/v1"

# The synthesis worker currently checks OPENROUTER_API_KEY even for models
# routed through LiteLLM's OpenAI-compatible provider. The local server does
# not authenticate this value.
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-dummy}"

python "$PROCESS" \
  --max-concurrent "$MAX_CONCURRENT" \
  --stages synthesize \
  --pairs-csv "$PAIRS_CSV" \
  --synth-config "$SYNTH_CONFIG" \
  --reasoning-jsonl "$REASONING_JSONL" \
  --provenance-json "$PROVENANCE_JSON" \
  --synth-name fixit-qwen36-27b \
  --synth-description "Synthesize Qwen3.6-27B reasoning for the Fixit Qwen3.6/Gemini repair pairs." \
  --reasoning-model openai/Qwen/Qwen3.6-27B \
  --max-tokens 81920 \
  --max-passes "$MAX_PASSES"
