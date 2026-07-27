#!/usr/bin/env bash
# Use nodebug tests, the 2026-0624-0939 checkpoint, and the same prompt tags as
# /home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-*.
set -euo pipefail

EXPERIMENT_START_EPOCH="$(date +%s)"
EXPERIMENT_START_TIME="$(date -u -d "@$EXPERIMENT_START_EPOCH" '+%Y-%m-%dT%H:%M:%SZ')"

RUN_DATE="2026-0624-0939-w8"
MODEL_NAME="qwen36-27b-SFT-2026-0624-0939"
ACCRL_MODEL_HOST="${ACCRL_MODEL_HOST:?Set ACCRL_MODEL_HOST to a dedicated model endpoint}"
SERVICE_URL="http://localhost:10002"
MAX_PARALLEL="8"
MAX_PROFILES="4"
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
PROFILE_HOST="p5-4"
PROFILE_CONTAINER="fib-profile"
PROFILE_TUNNEL_LOCAL_PORT="10002"
PROFILE_TUNNEL_REMOTE_PORT="10000"
PROFILE_TUNNEL_SESSION="connect-fib-profile-p5-4-10002"
PROFILE_DEFINITION="mha_bwd_d96"
PROFILE_MIN_GPUS="4"
PROFILE_MAX_GPUS="4"
PROFILE_POLL_ATTEMPTS="60"
PROFILE_POLL_SLEEP="30"
PROFILE_CONTAINER_POLL_ATTEMPTS="12"
PROFILE_CONTAINER_POLL_SLEEP="30"
MODEL_POLL_ATTEMPTS="60"
MODEL_POLL_SLEEP="10"
WATCH_SCRIPT="$(basename "$0")"

DEFINITIONS=(
  mha_with_lse_d64
  mha_with_lse_d64_causal
  mha_bwd_d64
  mha_bwd_d64_causal
)

TEST_PATHS=(
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d64_7d2575a0-bcc2-42a0-812f-6a7e9a57d97f.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d64_causal_b69f7675-568f-40f2-9a4b-8bbe374b4a59.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d64_d3bcb902-6a13-5ada-9251-fa841b10cd0b.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d64_causal_5799ea50-77aa-56cb-9f62-a4c1f5473770.py
)

CONFIGS=(
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_BWD"
  "$CONFIG_MHA_BWD"
)

OUTPUT_ROOTS=(
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d64"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d64-causal"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d64"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d64-causal"
)

source "$WATCH_COMMON"

report_experiment_timing() {
  local end_epoch end_time elapsed_seconds elapsed
  end_epoch="$(date +%s)"
  end_time="$(date -u -d "@$end_epoch" '+%Y-%m-%dT%H:%M:%SZ')"
  elapsed_seconds=$((end_epoch - EXPERIMENT_START_EPOCH))
  printf -v elapsed '%02d:%02d:%02d' \
    "$((elapsed_seconds / 3600))" \
    "$(((elapsed_seconds % 3600) / 60))" \
    "$((elapsed_seconds % 60))"
  log "experiment timing MAX_PARALLEL=$MAX_PARALLEL start=$EXPERIMENT_START_TIME end=$end_time elapsed=$elapsed elapsed_seconds=$elapsed_seconds"
}

validate_static_inputs
log "watcher started for RUN_DATE=$RUN_DATE MODEL_NAME=$MODEL_NAME ACCRL_MODEL_HOST=$ACCRL_MODEL_HOST SERVICE_URL=$SERVICE_URL RUNNER=$RUNNER PROFILE_HOST=$PROFILE_HOST PROFILE_CONTAINER=$PROFILE_CONTAINER PROFILE_TUNNEL_SESSION=$PROFILE_TUNNEL_SESSION PROFILE_MAX_GPUS=$PROFILE_MAX_GPUS CONFIG_MHA_FWD=$CONFIG_MHA_FWD CONFIG_MHA_BWD=$CONFIG_MHA_BWD"

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
    report_experiment_timing
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
