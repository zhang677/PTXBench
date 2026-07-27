#!/usr/bin/env bash
# Use nodebug tests, the 2026-0624-0939 checkpoint, and the same prompt tags as
# /home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-*.
set -euo pipefail

RUN_DATE="2026-0624-0939"
MODEL_NAME="qwen36-27b-SFT-2026-0624-0939"
ACCRL_MODEL_HOST="${ACCRL_MODEL_HOST:?Set ACCRL_MODEL_HOST to a dedicated model endpoint}"
SERVICE_URL="http://localhost:10003"
MAX_PARALLEL="16"
MAX_PROFILES="8"
GPU_ARCH="hopper"
TIMEOUT="86400"
TURN_TIMEOUT="980"
MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="1"
SLEEP_S="300"
OUTPUT_ROOT_BASE="/home/ubuntu/AccRL-exps/eval_runs"
RUNNER="/home/ubuntu/AccRL/fib_runtime/multiturn/run_parallel_v2.py"
WATCH_COMMON="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_common.sh"
WATCH_AUDIT="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_audit.py"
CONFIG_MHA_FWD="/home/ubuntu/AccRL-exps/prompt_configs/2026-0605-mha-p4-mha-patched.json"
CONFIG_MHA_BWD="/home/ubuntu/AccRL-exps/prompt_configs/2026-0605-mha-bwd-p4-mha-patched.json"
PROFILE_HOST="p5-1"
PROFILE_CONTAINER="fib-profile"
PROFILE_TUNNEL_LOCAL_PORT="10003"
PROFILE_TUNNEL_REMOTE_PORT="10000"
PROFILE_TUNNEL_SESSION="connect-fib-profile-p5-1-10003"
PROFILE_DEFINITION="mha_bwd_d96"
PROFILE_MIN_GPUS="1"
PROFILE_POLL_ATTEMPTS="60"
PROFILE_POLL_SLEEP="30"
PROFILE_CONTAINER_POLL_ATTEMPTS="12"
PROFILE_CONTAINER_POLL_SLEEP="30"
MODEL_POLL_ATTEMPTS="60"
MODEL_POLL_SLEEP="10"
WATCH_SCRIPT="$(basename "$0")"

DEFINITIONS=(
  mha_with_lse_d96
  mha_with_lse_d96_causal
  mha_bwd_d96
  mha_bwd_d96_causal
)

TEST_PATHS=(
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d96_e86112ff-2cd0-4e97-8efb-9ce6356ecb2b_nodebug.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d96_causal_982c9922-961f-452f-ab9e-574fdcd4e28c_nodebug.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d96_02cd8b5b-bb29-5737-ad07-f35e5f4020a5_nodebug.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d96_causal_c2474a6e-35d9-5e68-b249-c53aef561b99_nodebug.py
)

CONFIGS=(
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_BWD"
  "$CONFIG_MHA_BWD"
)

OUTPUT_ROOTS=(
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d96"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d96-causal"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d96"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d96-causal"
)

source "$WATCH_COMMON"

validate_static_inputs
log "watcher started for RUN_DATE=$RUN_DATE MODEL_NAME=$MODEL_NAME ACCRL_MODEL_HOST=$ACCRL_MODEL_HOST SERVICE_URL=$SERVICE_URL RUNNER=$RUNNER PROFILE_HOST=$PROFILE_HOST PROFILE_CONTAINER=$PROFILE_CONTAINER PROFILE_TUNNEL_SESSION=$PROFILE_TUNNEL_SESSION CONFIG_MHA_FWD=$CONFIG_MHA_FWD CONFIG_MHA_BWD=$CONFIG_MHA_BWD"

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

  if ! wait_for_model; then
    log "model endpoint verification failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
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
