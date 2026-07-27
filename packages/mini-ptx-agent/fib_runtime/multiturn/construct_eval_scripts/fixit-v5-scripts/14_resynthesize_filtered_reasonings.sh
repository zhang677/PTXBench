#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
CONFIG="${1:-$PROJECT/glm-52-fixit-v5-resynthesis-config.yaml}"
SCRIPT="/home/ubuntu/AccRL/fib_runtime/multiturn/fix_kernels/resynthesize_filtered_reasonings_openrouter.py"

python "$SCRIPT" "$CONFIG"
