#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set for GLM-5.2 synthesis}"

CONFIG="${KERNELGEN_CONFIG:-$SCRIPT_DIR/synthesize.yaml}"

python "$SCRIPT_DIR/synthesize_correct_kernel_reasoning_openrouter.py" \
  "$CONFIG"
