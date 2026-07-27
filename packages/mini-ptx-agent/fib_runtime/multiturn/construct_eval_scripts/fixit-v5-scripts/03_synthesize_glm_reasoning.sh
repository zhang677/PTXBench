#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
SYNTH_CONFIG="$PROJECT/glm-52-fixit-v5-config.yaml"
REASONING_JSONL="$PROJECT/reasoning_pairs.glm-5.2-fixit-v5.jsonl"
PROVENANCE_JSON="$PROJECT/provenance.glm-5.2-fixit-v5.json"
MAX_PASSES="20"
PROCESS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"

python "$PROCESS" \
  --stages synthesize \
  --pairs-csv "$PAIRS_CSV" \
  --synth-config "$SYNTH_CONFIG" \
  --reasoning-jsonl "$REASONING_JSONL" \
  --provenance-json "$PROVENANCE_JSON" \
  --synth-name fixit-v5-glm52 \
  --synth-description "Synthesize GLM 5.2 reasoning for fixit-v5 Qwen3.6 linfo Gemini repair pairs." \
  --max-passes "$MAX_PASSES"
