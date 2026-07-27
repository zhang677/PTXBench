#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../ptxbench_paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PARQUET="$PROJECT/data/qwen36-27b-fixit-v6-full.parquet"
TRAIN_RUN_TAG="qwen36-27b-qwen36-fixit-v6-full-e5-lr4.65e-4-lora32"
REMOTE="${REMOTE:-hyper01}"
CONTAINER="${CONTAINER:-sglang-genghan}"
REMOTE_PORT="${REMOTE_PORT:-9005}"
LOCAL_PORT="${LOCAL_PORT:-30052}"
SERVE_SESSION="${SERVE_SESSION:-serve-fixit-v6-full-qwen36-patched}"
TUNNEL_SESSION="${TUNNEL_SESSION:-connect-sglang-fixit-v6-full-patched}"
SERVE_TIMEOUT_S="${SERVE_TIMEOUT_S:-1800}"
POLL_S="${POLL_S:-120}"
PROCESS="$PTXBENCH_CONSTRUCT_EVAL_ROOT/fixit_downstream_process.py"

ARGS=(
  --execute-serve
  --stages serve
  --wait-for-checkpoint
  --parquet "$PARQUET"
  --train-run-tag "$TRAIN_RUN_TAG"
  --remote "$REMOTE"
  --container "$CONTAINER"
  --remote-port "$REMOTE_PORT"
  --local-port "$LOCAL_PORT"
  --serve-session "$SERVE_SESSION"
  --tunnel-session "$TUNNEL_SESSION"
  --serve-timeout-s "$SERVE_TIMEOUT_S"
  --poll-s "$POLL_S"
)

python "$PROCESS" "${ARGS[@]}"
