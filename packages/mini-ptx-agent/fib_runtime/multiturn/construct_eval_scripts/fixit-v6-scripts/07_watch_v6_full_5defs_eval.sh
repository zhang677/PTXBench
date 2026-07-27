#!/usr/bin/env bash
set -euo pipefail

# Script 06 serves the matching checkpoint as MODEL_NAME on localhost:30052.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../ptxbench_paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
RUNS_DIR="$PROJECT/runs"
TRAIN_RUN_TAG="qwen36-27b-qwen36-fixit-v6-full-e5-lr4.65e-4-lora32"
MODEL_PREFIX="qwen36-27b-SFT"
if [[ -z "${BASE_RUN_DATE:-}" ]]; then
  BASE_RUN_DATE="$(python - "$RUNS_DIR" "$TRAIN_RUN_TAG" <<'PY'
import re
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
train_run_tag = sys.argv[2]
pattern = f"{train_run_tag}-Qwen-Qwen3.6-27B-*"
candidates = sorted(
    (path for path in runs_dir.glob(pattern) if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
)
if not candidates:
    raise SystemExit(f"no run dirs matching {pattern!r} under {runs_dir}")
match = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$", candidates[-1].name)
if not match:
    raise SystemExit(f"cannot derive RUN_DATE from {candidates[-1]}")
year, month, day, hour, minute = match.groups()
print(f"{year}-{month}{day}-{hour}{minute}")
PY
  )"
fi
RUN_DATE="${RUN_DATE:-$BASE_RUN_DATE-fixit-v6-full-patched}"
MODEL_NAME="${MODEL_NAME:-$MODEL_PREFIX-$BASE_RUN_DATE}"
SERVICE_URL="${SERVICE_URL:-http://localhost:10002}"
ACCRL_MODEL_HOST="${ACCRL_MODEL_HOST:-localhost:30052}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MAX_PROFILES="${MAX_PROFILES:-8}"
GPU_ARCH="${GPU_ARCH:-hopper}"
TIMEOUT="${TIMEOUT:-86400}"
TURN_TIMEOUT="${TURN_TIMEOUT:-980}"
SLEEP_S="${SLEEP_S:-300}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$PTXBENCH_EVAL_RUNS_ROOT}"
RUNNER="$PTXBENCH_MULTITURN_ROOT/run_parallel_v2.py"
WATCH_COMMON="$PTXBENCH_CONSTRUCT_EVAL_ROOT/watch_eval_common.sh"
WATCH_AUDIT="$PTXBENCH_CONSTRUCT_EVAL_ROOT/watch_eval_audit.py"
CONFIG_MHA_FWD="${CONFIG_MHA_FWD:-$PTXBENCH_CONFIG_ROOT/fixit-v6/2026-0605-mha-p4-mha-patched.json}"
CONFIG_MHA_BWD="${CONFIG_MHA_BWD:-$PTXBENCH_CONFIG_ROOT/fixit-v6/2026-0605-mha-bwd-p4-mha-patched.json}"
CONFIG_GEMM="${CONFIG_GEMM:-$PTXBENCH_CONFIG_ROOT/fixit-v6/gemm-5-r8-p4.json}"
PROFILE_HOST="${PROFILE_HOST:-p5-4}"
PROFILE_CONTAINER="${PROFILE_CONTAINER:-fib-profile}"
PROFILE_DEFINITION="${PROFILE_DEFINITION:-mha_bwd_d128}"
PROFILE_MIN_GPUS="${PROFILE_MIN_GPUS:-1}"
PROFILE_POLL_ATTEMPTS="${PROFILE_POLL_ATTEMPTS:-60}"
PROFILE_POLL_SLEEP="${PROFILE_POLL_SLEEP:-30}"
PROFILE_CONTAINER_POLL_ATTEMPTS="${PROFILE_CONTAINER_POLL_ATTEMPTS:-12}"
PROFILE_CONTAINER_POLL_SLEEP="${PROFILE_CONTAINER_POLL_SLEEP:-30}"
WATCH_SCRIPT="$(basename "$0")"

DEFINITIONS=(
  mha_with_lse_d128
  mha_with_lse_d128_causal
  mha_bwd_d128
  mha_bwd_d128_causal
  gemm_n7168_k5120
)

TEST_PATHS=(
  "$PTXBENCH_MULTITURN_ROOT/2026-0426-1410/mha_with_lse_d128_bc38b351-d595-451b-9153-8e225702e53b.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0426-1410/mha_with_lse_d128_causal_6d2f67a7-225a-4af5-87d3-cbb99b496325.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0427-1308/mha_bwd_d128_38c3b07c-f006-5f5e-9860-ba214c805a6b.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0427-1308/mha_bwd_d128_causal_c119b3f0-c051-5e96-9c2a-2268d992fe1a.py"
  "$PTXBENCH_MULTITURN_ROOT/2026-0413-1611/gemm_n7168_k5120_94920358-01a8-4c5b-9209-3103fd490e94.py"
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
log "watcher started for RUN_DATE=$RUN_DATE MODEL_NAME=$MODEL_NAME ACCRL_MODEL_HOST=$ACCRL_MODEL_HOST SERVICE_URL=$SERVICE_URL RUNNER=$RUNNER PROFILE_HOST=$PROFILE_HOST PROFILE_CONTAINER=$PROFILE_CONTAINER CONFIG_MHA_FWD=$CONFIG_MHA_FWD CONFIG_MHA_BWD=$CONFIG_MHA_BWD CONFIG_GEMM=$CONFIG_GEMM"
if [[ "${WATCH_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  log "WATCH_PREFLIGHT_ONLY=1; static watcher preflight complete"
  exit 0
fi

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
