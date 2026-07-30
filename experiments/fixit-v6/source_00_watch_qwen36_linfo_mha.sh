#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

RUN_DATE="${RUN_DATE:-qwen36-27b-linfo}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B}"
ACCRL_MODEL_HOST="${ACCRL_MODEL_HOST:-localhost:30062}"
SERVICE_URL="${SERVICE_URL:-http://localhost:10000}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MAX_PROFILES="${MAX_PROFILES:-8}"
GPU_ARCH="${GPU_ARCH:-hopper}"
TIMEOUT="${TIMEOUT:-86400}"
TURN_TIMEOUT="${TURN_TIMEOUT:-980}"
SLEEP_S="${SLEEP_S:-300}"
MODEL_POLL_ATTEMPTS="${MODEL_POLL_ATTEMPTS:-60}"
MODEL_POLL_SLEEP="${MODEL_POLL_SLEEP:-30}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$PTXBENCH_EVAL_RUNS_ROOT}"
RUNNER="$PTXBENCH_MULTITURN_ROOT/run_parallel_v2.py"
WATCH_COMMON="$PTXBENCH_CONSTRUCT_EVAL_ROOT/watch_eval_common.sh"
WATCH_AUDIT="$PTXBENCH_CONSTRUCT_EVAL_ROOT/watch_eval_audit.py"
CONFIG_MHA_FWD="${CONFIG_MHA_FWD:-$PTXBENCH_CONFIG_ROOT/fixit-v6/2026-0605-mha-p4.json}"
CONFIG_MHA_BWD="${CONFIG_MHA_BWD:-$PTXBENCH_CONFIG_ROOT/fixit-v6/2026-0605-mha-bwd-p4.json}"
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

DEFINITIONS=(
  mha_with_lse_d64
  mha_with_lse_d64_causal
  mha_with_lse_d128
  mha_with_lse_d128_causal
  mha_bwd_d64
  mha_bwd_d64_causal
  mha_bwd_d128
  mha_bwd_d128_causal
)

TEST_PATHS=(
  "$PTXBENCH_MULTITURN_ROOT/2026-0426-1410/mha_with_lse_d64_7d2575a0-bcc2-42a0-812f-6a7e9a57d97f.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0426-1410/mha_with_lse_d64_causal_b69f7675-568f-40f2-9a4b-8bbe374b4a59.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0426-1410/mha_with_lse_d128_bc38b351-d595-451b-9153-8e225702e53b.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0426-1410/mha_with_lse_d128_causal_6d2f67a7-225a-4af5-87d3-cbb99b496325.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0427-1308/mha_bwd_d64_d3bcb902-6a13-5ada-9251-fa841b10cd0b.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0427-1308/mha_bwd_d64_causal_5799ea50-77aa-56cb-9f62-a4c1f5473770.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0427-1308/mha_bwd_d128_38c3b07c-f006-5f5e-9860-ba214c805a6b.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0427-1308/mha_bwd_d128_causal_c119b3f0-c051-5e96-9c2a-2268d992fe1a.py"
)

CONFIGS=(
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_FWD"
  "$CONFIG_MHA_BWD"
  "$CONFIG_MHA_BWD"
  "$CONFIG_MHA_BWD"
  "$CONFIG_MHA_BWD"
)

OUTPUT_ROOTS=(
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d64"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d64-causal"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d128"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d128-causal"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d64"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d64-causal"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d128"
  "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d128-causal"
)

source "$WATCH_COMMON"

validate_static_inputs
log "Fixit source watcher started RUN_DATE=$RUN_DATE MODEL_NAME=$MODEL_NAME ACCRL_MODEL_HOST=$ACCRL_MODEL_HOST SERVICE_URL=$SERVICE_URL"
if [[ "${WATCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  log "WATCH_PREFLIGHT_ONLY=1; static source watcher preflight complete"
  exit 0
fi

while true; do
  if pids="$(running_eval_pids 2>/dev/null)"; then
    log "source eval launcher still running pids=[$pids]"
    sleep "$SLEEP_S"
    continue
  fi

  if ! refresh_existing_plans_from_configs; then
    log "source plan refresh failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  log "no active source eval process found; auditing roots"
  if audit_roots; then
    log "all eight Fixit source roots complete"
    exit 0
  fi

  if ! wait_for_model; then
    log "base Qwen endpoint unavailable; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi
  if ! restart_profile_service; then
    log "FIBServe restart failed; retrying after sleep"
    sleep "$SLEEP_S"
    continue
  fi

  log "starting base-Qwen source evaluation"
  set +e
  run_eval_once
  rc=$?
  set -e
  log "source eval launcher exited rc=$rc"
  sleep "$SLEEP_S"
done
