#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

python "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py" \
  --execute-serve \
  --stages serve \
  --wait-for-checkpoint \
  --parquet "$KERNELGEN_PARQUET" \
  --runs-dir "$KERNELGEN_PROJECT/runs" \
  --base-model Qwen/Qwen3.6-27B \
  --train-run-tag "$KERNELGEN_RUN_TAG" \
  --remote "${REMOTE:-ion-b200}" \
  --container "${CONTAINER:-sglang-genghan}" \
  --remote-port "${REMOTE_PORT:-9001}" \
  --local-port "${LOCAL_PORT:-30012}" \
  --serve-session "${SERVE_SESSION:-serve-mha-8def-glm52}" \
  --tunnel-session "${TUNNEL_SESSION:-connect-sglang-9001}" \
  --serve-timeout-s "${SERVE_TIMEOUT_S:-1800}" \
  --poll-s "${POLL_S:-20}"
