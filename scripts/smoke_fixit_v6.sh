#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MINI_PTX_AGENT_ROOT="${MINI_PTX_AGENT_ROOT:-$PTXBENCH_ROOT/packages/mini-ptx-agent}"
PTXBENCH_DATA_ROOT="${PTXBENCH_DATA_ROOT:-$PTXBENCH_ROOT/data}"
CONSTRUCT_ROOT="$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/construct_eval_scripts"
FIXIT_ROOT="$CONSTRUCT_ROOT/fixit-v6-scripts"
WATCHER="${FIXIT_V6_WATCHER:-$FIXIT_ROOT/05_watch_v6_full_5defs_eval.sh}"

usage() {
  cat <<'EOF'
Usage:
  scripts/smoke_fixit_v6.sh --check
  MODEL_NAME=... ACCRL_MODEL_HOST=host:port scripts/smoke_fixit_v6.sh --run

Environment:
  SERVICE_URL          FIBServe URL (default http://localhost:10000)
  PROFILE_CONTAINER    local FIBServe container (default ptxbench-fibserve)
  MODEL_NAME           exact ID returned by the OpenAI-compatible model endpoint
  ACCRL_MODEL_HOST     model endpoint host:port
  SMOKE_PATCHED=1      use the patched prompt lane and script 07
EOF
}

mode="${1:---check}"
case "$mode" in
  --check|--run) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "${SMOKE_PATCHED:-0}" == "1" ]]; then
  WATCHER="$FIXIT_ROOT/07_watch_v6_full_5defs_eval.sh"
  SMOKE_CONFIG="$PTXBENCH_ROOT/configs/fixit-v6/smoke-patched.json"
else
  SMOKE_CONFIG="$PTXBENCH_ROOT/configs/fixit-v6/smoke-base.json"
fi

required=(
  "$WATCHER"
  "$CONSTRUCT_ROOT/watch_eval_common.sh"
  "$CONSTRUCT_ROOT/watch_eval_audit.py"
  "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_parallel_v2.py"
  "$SMOKE_CONFIG"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required PTXBench path: $path" >&2
    exit 1
  fi
done

for script in "$FIXIT_ROOT"/*.sh "$CONSTRUCT_ROOT/watch_eval_common.sh"; do
  bash -n "$script"
done

python -m compileall -q \
  "$MINI_PTX_AGENT_ROOT/mini_ptx_agent" \
  "$MINI_PTX_AGENT_ROOT/accrl/distill/inspector.py" \
  "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_parallel_v2.py" \
  "$CONSTRUCT_ROOT/fixit_downstream_process.py" \
  "$CONSTRUCT_ROOT/watch_eval_audit.py"

echo "Fixit-v6 static smoke preflight passed"
echo "watcher=$WATCHER"
echo "config=$SMOKE_CONFIG"

if [[ "$mode" == "--check" ]]; then
  BASE_RUN_DATE=smoke \
  RUN_DATE=smoke-fixit-v6 \
  MODEL_NAME=smoke-model \
  ACCRL_MODEL_HOST=localhost:1 \
  SERVICE_URL=http://localhost:1 \
  PROFILE_LOCAL=1 \
  WATCH_PREFLIGHT_ONLY=1 \
  CONFIG_MHA_FWD="$SMOKE_CONFIG" \
  CONFIG_MHA_BWD="$SMOKE_CONFIG" \
  CONFIG_GEMM="$SMOKE_CONFIG" \
  OUTPUT_ROOT_BASE="$PTXBENCH_DATA_ROOT/eval_runs" \
  PTXBENCH_ROOT="$PTXBENCH_ROOT" \
  MINI_PTX_AGENT_ROOT="$MINI_PTX_AGENT_ROOT" \
  PTXBENCH_DATA_ROOT="$PTXBENCH_DATA_ROOT" \
    bash "$WATCHER"
  exit 0
fi

: "${MODEL_NAME:?MODEL_NAME must match the exact model ID at /v1/models}"
: "${ACCRL_MODEL_HOST:?ACCRL_MODEL_HOST must be host:port for the model endpoint}"

export PTXBENCH_ROOT MINI_PTX_AGENT_ROOT PTXBENCH_DATA_ROOT
export BASE_RUN_DATE="${BASE_RUN_DATE:-smoke}"
export RUN_DATE="${RUN_DATE:-smoke-fixit-v6}"
export SERVICE_URL="${SERVICE_URL:-http://localhost:10000}"
export PROFILE_LOCAL="${PROFILE_LOCAL:-1}"
export PROFILE_CONTAINER="${PROFILE_CONTAINER:-ptxbench-fibserve}"
export PROFILE_RESTART_MODE="${PROFILE_RESTART_MODE:-container}"
export PROFILE_MAX_GPUS="${PROFILE_MAX_GPUS:-2}"
export PROFILE_MIN_GPUS="${PROFILE_MIN_GPUS:-2}"
export MAX_PARALLEL="${MAX_PARALLEL:-2}"
export MAX_PROFILES="${MAX_PROFILES:-2}"
export TIMEOUT="${TIMEOUT:-3600}"
export TURN_TIMEOUT="${TURN_TIMEOUT:-980}"
export SLEEP_S="${SLEEP_S:-5}"
export CONFIG_MHA_FWD="$SMOKE_CONFIG"
export CONFIG_MHA_BWD="$SMOKE_CONFIG"
export CONFIG_GEMM="$SMOKE_CONFIG"
export OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$PTXBENCH_DATA_ROOT/eval_runs}"

mkdir -p "$OUTPUT_ROOT_BASE"
exec bash "$WATCHER"
