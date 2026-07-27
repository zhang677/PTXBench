#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="gemini-31-pro"
MODEL_NAME="gemini-3.1-pro-preview"
SERVICE_URL="http://localhost:10003"
MAX_PARALLEL="6"
MAX_PROFILES="8"
GPU_ARCH="hopper"
TIMEOUT="86400"
TURN_TIMEOUT="980"
MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="3"
SLEEP_S="300"
OUTPUT_ROOT_BASE="/home/ubuntu/AccRL-exps/eval_runs"
RUNNER="/home/ubuntu/AccRL/fib_runtime/multiturn/run_parallel_v2.py"
WATCH_COMMON="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_common.sh"
WATCH_AUDIT="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_audit.py"
CONFIG_MHA_BWD="/home/ubuntu/AccRL-exps/prompt_configs/2026-0504-fa3-bwd.json"
PROFILE_HOST="p5-1"
PROFILE_CONTAINER="fib-profile"
PROFILE_TUNNEL_LOCAL_PORT="10003"
PROFILE_TUNNEL_REMOTE_PORT="10000"
PROFILE_TUNNEL_SESSION="connect-fib-profile-p5-1-10003"
PROFILE_DEFINITION="mha_bwd_d128"
PROFILE_MIN_GPUS="1"
PROFILE_POLL_ATTEMPTS="60"
PROFILE_POLL_SLEEP="30"
PROFILE_CONTAINER_POLL_ATTEMPTS="12"
PROFILE_CONTAINER_POLL_SLEEP="30"
WATCH_SCRIPT="$(basename "$0")"

DEFINITIONS=(
  mha_bwd_d128
)

TEST_PATHS=(
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_38c3b07c-f006-5f5e-9860-ba214c805a6b.py
)

CONFIGS=(
  "$CONFIG_MHA_BWD"
)

OUTPUT_ROOTS=(
  "$OUTPUT_ROOT_BASE/$RUN_DATE-linfo-mha-bwd-d128"
)

source "$WATCH_COMMON"

validate_static_inputs
log "watcher started for RUN_DATE=$RUN_DATE MODEL_NAME=$MODEL_NAME SERVICE_URL=$SERVICE_URL RUNNER=$RUNNER PROFILE_HOST=$PROFILE_HOST PROFILE_CONTAINER=$PROFILE_CONTAINER PROFILE_TUNNEL_SESSION=$PROFILE_TUNNEL_SESSION CONFIG_MHA_BWD=$CONFIG_MHA_BWD"

while true; do
  if pids="$(running_eval_pids 2>/dev/null)"; then
    log "eval launcher still running pids=[$pids]"
    sleep "$SLEEP_S"
    continue
  fi

  if ! refresh_existing_plans_from_configs; then
    log "plan refresh failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  log "no active eval process found; auditing roots"
  if audit_roots; then
    log "all roots complete"
    exit 0
  fi

  if ! restart_profile_service; then
    log "profile restart failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  log "starting eval runner with mixed resume/fresh roots for RUN_DATE=$RUN_DATE"
  set +e
  run_eval_once
  rc=$?
  set -e
  log "eval launcher exited rc=$rc"
  sleep "$SLEEP_S"
done
