#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

CONFIG="${CONFIG:-$PTXBENCH_FIXIT_PROJECT/gemini-source-prompt-config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PTXBENCH_FIXIT_REPAIR_ROOT}"
SERVICE_URL="${SERVICE_URL:-http://localhost:10000}"
RUNNER="$PTXBENCH_MULTITURN_ROOT/fix_kernels/run_parallel_fix_v2.py"
WATCH_COMMON="$PTXBENCH_CONSTRUCT_EVAL_ROOT/watch_eval_common.sh"
WATCH_AUDIT="$PTXBENCH_CONSTRUCT_EVAL_ROOT/watch_eval_audit.py"
MODEL_NAME="${MODEL_NAME:-gemini-3.1-pro-preview}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
MAX_PROFILES="${MAX_PROFILES:-8}"
GPU_ARCH="${GPU_ARCH:-hopper}"
TIMEOUT="${TIMEOUT:-86400}"
TURN_TIMEOUT="${TURN_TIMEOUT:-980}"
SLEEP_S="${SLEEP_S:-300}"
PROFILE_LOCAL="${PROFILE_LOCAL:-1}"
PROFILE_CONTAINER="${PROFILE_CONTAINER:-ptxbench-fibserve}"
PROFILE_RESTART_MODE="${PROFILE_RESTART_MODE:-container}"
PROFILE_DEFINITION="${PROFILE_DEFINITION:-mha_bwd_d128}"
PROFILE_MIN_GPUS="${PROFILE_MIN_GPUS:-1}"
PROFILE_POLL_ATTEMPTS="${PROFILE_POLL_ATTEMPTS:-60}"
PROFILE_POLL_SLEEP="${PROFILE_POLL_SLEEP:-30}"
PROFILE_CONTAINER_POLL_ATTEMPTS="${PROFILE_CONTAINER_POLL_ATTEMPTS:-12}"
PROFILE_CONTAINER_POLL_SLEEP="${PROFILE_CONTAINER_POLL_SLEEP:-30}"
WATCH_SCRIPT="$(basename "$0")"

source "$WATCH_COMMON"

validate_fixit_static_inputs
log "Gemini repair watcher started OUTPUT_ROOT=$OUTPUT_ROOT CONFIG=$CONFIG MODEL_NAME=$MODEL_NAME SERVICE_URL=$SERVICE_URL"
if [[ "${WATCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  log "WATCH_PREFLIGHT_ONLY=1; static Gemini watcher preflight complete"
  exit 0
fi
if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini repairs." >&2
  exit 1
fi

while true; do
  if pids="$(running_fixit_pids 2>/dev/null)"; then
    log "Gemini repair runner still running pids=[$pids]"
    sleep "$SLEEP_S"
    continue
  fi

  if ! refresh_existing_plan_from_config; then
    log "Gemini repair plan refresh failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  if [[ -d "$OUTPUT_ROOT" ]]; then
    log "no active Gemini repair process found; auditing root"
    if audit_root; then
      log "Gemini repair root complete"
      exit 0
    fi
  else
    log "Gemini repair output root missing; will launch fresh"
  fi

  if ! restart_profile_service; then
    log "FIBServe restart failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  set +e
  run_fixit_once
  rc=$?
  set -e
  log "Gemini repair runner exited rc=$rc"
  sleep "$SLEEP_S"
done
