#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm/fixit-v5-gemini-source-prompt-config.json"
OUTPUT_ROOT="/home/ubuntu/AccRL-exps/eval_runs/fixit-v5-qwen36-linfo-gemini"
SERVICE_URL="http://localhost:10003"
RUNNER="/home/ubuntu/AccRL/fib_runtime/multiturn/fix_kernels/run_parallel_fix_v2.py"
WATCH_COMMON="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_common.sh"
WATCH_AUDIT="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_audit.py"
MODEL_NAME="gemini-3.1-pro-preview"
MAX_PARALLEL="6"
MAX_PROFILES="8"
GPU_ARCH="hopper"
TIMEOUT="86400"
TURN_TIMEOUT="980"
SLEEP_S="300"
PROFILE_HOST="p5-1"
PROFILE_CONTAINER="fib-profile"
PROFILE_DEFINITION="mha_bwd_d128"
PROFILE_MIN_GPUS="1"
PROFILE_POLL_ATTEMPTS="60"
PROFILE_POLL_SLEEP="30"
PROFILE_CONTAINER_POLL_ATTEMPTS="12"
PROFILE_CONTAINER_POLL_SLEEP="30"
WATCH_SCRIPT="$(basename "$0")"

source "$WATCH_COMMON"

validate_fixit_static_inputs
log "watcher started OUTPUT_ROOT=$OUTPUT_ROOT CONFIG=$CONFIG MODEL_NAME=$MODEL_NAME SERVICE_URL=$SERVICE_URL RUNNER=$RUNNER PROFILE_HOST=$PROFILE_HOST PROFILE_CONTAINER=$PROFILE_CONTAINER"

while true; do
  if pids="$(running_fixit_pids 2>/dev/null)"; then
    log "fixit runner still running pids=[$pids]"
    sleep "$SLEEP_S"
    continue
  fi

  if ! refresh_existing_plan_from_config; then
    log "plan refresh failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  if [[ -d "$OUTPUT_ROOT" ]]; then
    log "no active fixit process found; auditing root"
    if audit_root; then
      log "fixit root complete"
      exit 0
    fi
  else
    log "output root missing; will launch fresh"
  fi

  if ! restart_profile_service; then
    log "profile restart failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  set +e
  run_fixit_once
  rc=$?
  set -e
  log "fixit runner exited rc=$rc"
  sleep "$SLEEP_S"
done
