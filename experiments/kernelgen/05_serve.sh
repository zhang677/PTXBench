#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"
: "${REMOTE:?Set REMOTE to the SSH host running SGLang}"
: "${CONTAINER:?Set CONTAINER to the SGLang Docker container name}"

python "$PTXBENCH_SHARED_ROOT/downstream.py" \
  --execute-serve \
  --stages serve \
  --wait-for-checkpoint \
  --parquet "$KERNELGEN_PARQUET" \
  --runs-dir "$KERNELGEN_PROJECT/runs" \
  --base-model Qwen/Qwen3.6-27B \
  --train-run-tag "$KERNELGEN_RUN_TAG" \
  --remote "$REMOTE" \
  --container "$CONTAINER" \
  --remote-port "${REMOTE_PORT:-9001}" \
  --local-port "${LOCAL_PORT:-30012}" \
  --serve-session "${SERVE_SESSION:-serve-kernelgen-glm52}" \
  --tunnel-session "${TUNNEL_SESSION:-connect-sglang-9001}" \
  --serve-timeout-s "${SERVE_TIMEOUT_S:-1800}" \
  --poll-s "${POLL_S:-20}" \
  --model-name Qwen/Qwen3.6-27B
