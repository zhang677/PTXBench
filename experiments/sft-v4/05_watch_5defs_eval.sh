#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

WATCHER="$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/fixit-v6-scripts/05_watch_v6_full_5defs_eval.sh"

export PROJECT="$SFT_V4_PROJECT"
export TRAIN_RUN_TAG="$SFT_V4_RUN_TAG"
export RUN_SUFFIX="${RUN_SUFFIX:-sft-v4}"
export ACCRL_MODEL_HOST="${ACCRL_MODEL_HOST:-localhost:30012}"
export SERVICE_URL="${SERVICE_URL:-http://localhost:10002}"
export PROFILE_HOST="${PROFILE_HOST:-p5-4}"

exec bash "$WATCHER"
