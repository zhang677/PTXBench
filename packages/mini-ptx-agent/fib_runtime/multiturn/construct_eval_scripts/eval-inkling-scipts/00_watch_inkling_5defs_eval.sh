#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="inkling"
MODEL_NAME="inkling"
SERVICE_URL="http://localhost:10003"
MAX_PARALLEL="16"
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
CONFIG_MHA_FWD="/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-5-r8-p4.json"
CONFIG_MHA_BWD="/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-bwd-5-r8-p4.json"
CONFIG_GEMM="/home/ubuntu/AccRL-exps/prompt_configs/hopper-gemm-5-r8-p4.json"
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
  mha_with_lse_d128
  mha_with_lse_d128_causal
  mha_bwd_d128
  mha_bwd_d128_causal
  gemm_n7168_k5120
)

TEST_PATHS=(
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d128_bc38b351-d595-451b-9153-8e225702e53b.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d128_causal_6d2f67a7-225a-4af5-87d3-cbb99b496325.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_38c3b07c-f006-5f5e-9860-ba214c805a6b.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_causal_c119b3f0-c051-5e96-9c2a-2268d992fe1a.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0413-1611/gemm_n7168_k5120_94920358-01a8-4c5b-9209-3103fd490e94.py
)

CONFIGS=(
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_BWD"
  "$CONFIG_MHA_BWD"
  "$CONFIG_GEMM"
)

OUTPUT_ROOTS=(
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d128"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d128-causal"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d128"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d128-causal"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-gemm"
)

source "$WATCH_COMMON"

validate_static_inputs
log "watcher started for RUN_DATE=$RUN_DATE MODEL_NAME=$MODEL_NAME SERVICE_URL=$SERVICE_URL RUNNER=$RUNNER PROFILE_HOST=$PROFILE_HOST PROFILE_CONTAINER=$PROFILE_CONTAINER PROFILE_TUNNEL_SESSION=$PROFILE_TUNNEL_SESSION CONFIG_MHA_FWD=$CONFIG_MHA_FWD CONFIG_MHA_BWD=$CONFIG_MHA_BWD CONFIG_GEMM=$CONFIG_GEMM"

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
