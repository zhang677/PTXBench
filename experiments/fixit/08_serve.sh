#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
PARQUET="$PROJECT/data/fixit.parquet"
TRAIN_RUN_TAG="qwen36-27b-fixit-example-e5-lr4.65e-4-lora32"
: "${REMOTE:?Set REMOTE to the SSH host running SGLang}"
: "${CONTAINER:?Set CONTAINER to the SGLang Docker container name}"
REMOTE_PORT="${REMOTE_PORT:-9005}"
LOCAL_PORT="${LOCAL_PORT:-30052}"
SERVE_SESSION="${SERVE_SESSION:-serve-fixit}"
TUNNEL_SESSION="${TUNNEL_SESSION:-connect-sglang-fixit}"
SERVE_TIMEOUT_S="${SERVE_TIMEOUT_S:-1800}"
POLL_S="${POLL_S:-120}"
PROCESS="$PTXBENCH_SHARED_ROOT/downstream.py"

ARGS=(
  --execute-serve
  --stages serve
  --wait-for-checkpoint
  --parquet "$PARQUET"
  --runs-dir "$PROJECT/runs"
  --train-run-tag "$TRAIN_RUN_TAG"
  --remote "$REMOTE"
  --container "$CONTAINER"
  --remote-port "$REMOTE_PORT"
  --local-port "$LOCAL_PORT"
  --serve-session "$SERVE_SESSION"
  --tunnel-session "$TUNNEL_SESSION"
  --serve-timeout-s "$SERVE_TIMEOUT_S"
  --poll-s "$POLL_S"
  --model-name Qwen/Qwen3.6-27B
)

python "$PROCESS" "${ARGS[@]}"
