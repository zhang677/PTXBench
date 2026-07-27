#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PARQUET="$PROJECT/data/glm-5.2-fixit-v5-full.parquet"
TRAIN_RUN_TAG="qwen36-27b-glm52-fixit-v5-full-e5-lr4.65e-4-lora32"
REMOTE="hyper00"
CONTAINER="sglang-genghan"
REMOTE_PORT="9003"
LOCAL_PORT="30042"
SERVE_SESSION="serve-fixit-v5-full-glm52"
TUNNEL_SESSION="connect-sglang-fixit-v5-full"
SERVE_TIMEOUT_S="1800"
PROCESS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"

ARGS=(
  --execute-serve
  --stages serve
  --parquet "$PARQUET"
  --train-run-tag "$TRAIN_RUN_TAG"
  --remote "$REMOTE"
  --container "$CONTAINER"
  --remote-port "$REMOTE_PORT"
  --local-port "$LOCAL_PORT"
  --serve-session "$SERVE_SESSION"
  --tunnel-session "$TUNNEL_SESSION"
  --serve-timeout-s "$SERVE_TIMEOUT_S"
)

python "$PROCESS" "${ARGS[@]}"
