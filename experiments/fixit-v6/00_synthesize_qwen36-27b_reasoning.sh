#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
SYNTH_CONFIG="$PROJECT/qwen36-27b-fixit-v6-config.yaml"
REASONING_JSONL="$PROJECT/reasoning_pairs.qwen36-27b-fixit-v6.jsonl"
PROVENANCE_JSON="$PROJECT/provenance.qwen36-27b-fixit-v6.json"
MAX_PASSES="${MAX_PASSES:-20}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
PROCESS="$PTXBENCH_CONSTRUCT_EVAL_ROOT/fixit_downstream_process.py"

# Match common.py's Qwen3.6-27B routing: use LiteLLM's OpenAI-compatible
# provider and send requests to an explicitly selected, dedicated model serve.
: "${ACCRL_MODEL_HOST:?Set ACCRL_MODEL_HOST to a dedicated Qwen3.6-27B endpoint}"
export ACCRL_MODEL_HOST
export OPENAI_BASE_URL="http://${ACCRL_MODEL_HOST}/v1"

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
  --synth-name fixit-v6-qwen36-27b \
  --synth-description "Synthesize Qwen3.6-27B reasoning for the fixit-v5 Qwen3.6 linfo Gemini repair pairs." \
  --reasoning-model openai/Qwen/Qwen3.6-27B \
  --max-tokens 81920 \
  --max-passes "$MAX_PASSES"
