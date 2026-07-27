#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
RUNS_DIR="$PROJECT/runs"
TRAIN_RUN_TAG="qwen36-27b-glm52-fixit-v5-e5-lr4.65e-4-lora32"
MODEL_PREFIX="qwen36-27b-SFT"
RUN_DATE="$(python - "$RUNS_DIR" "$TRAIN_RUN_TAG" <<'PY'
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
MODEL_NAME="$MODEL_PREFIX-$RUN_DATE"
SERVICE_URL="http://localhost:10003"
ACCRL_MODEL_HOST="localhost:30042"
MAX_PARALLEL="16"
MAX_PROFILES="8"
GPU_ARCH="hopper"
TIMEOUT="86400"
TURN_TIMEOUT="980"
SLEEP_S="300"
OUTPUT_ROOT_BASE="/home/ubuntu/AccRL-exps/eval_runs"
RUNNER="/home/ubuntu/AccRL/fib_runtime/multiturn/run_parallel_v2.py"
WATCH_COMMON="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_common.sh"
WATCH_AUDIT="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/watch_eval_audit.py"
CONFIG_MAIN="/home/ubuntu/AccRL-exps/prompt_configs/2026-0428-1632.json"
CONFIG_FP8="/home/ubuntu/AccRL-exps/prompt_configs/2026-0507-1340.json"
PROFILE_HOST="p5-1"
PROFILE_CONTAINER="fib-profile"
PROFILE_DEFINITION="mha_bwd_d96"
PROFILE_MIN_GPUS="1"
PROFILE_POLL_ATTEMPTS="60"
PROFILE_POLL_SLEEP="30"
PROFILE_CONTAINER_POLL_ATTEMPTS="12"
PROFILE_CONTAINER_POLL_SLEEP="30"
WATCH_SCRIPT="$(basename "$0")"

DEFINITIONS=(
  mha_with_lse_d96
  mha_with_lse_d96_causal
  mha_bwd_d96
  mha_bwd_d96_causal
  gqa_paged_decode_h32_kv8_d128_ps64
  gqa_paged_prefill_causal_h32_kv4_d128_ps64
  gqa_ragged_prefill_causal_h32_kv8_d128
  mla_paged_decode_h16_ckv512_kpe64_ps1
  mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
  mla_ragged_prefill_causal_h16_qk192_vo128
  fp8_mha_with_lse_d64
  fp8_mha_with_lse_d128
)

TEST_PATHS=(
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d96_e86112ff-2cd0-4e97-8efb-9ce6356ecb2b.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d96_causal_982c9922-961f-452f-ab9e-574fdcd4e28c.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d96_02cd8b5b-bb29-5737-ad07-f35e5f4020a5.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d96_causal_c2474a6e-35d9-5e68-b249-c53aef561b99.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gqa_paged_decode_h32_kv8_d128_ps64_ee6efcf9-bf19-432f-bcd3-33f1c9ec599e.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gqa_paged_prefill_causal_h32_kv4_d128_ps64_36916c21-db41-447b-bbc9-b88d5d6df89d.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gqa_ragged_prefill_causal_h32_kv8_d128_6f3e1bfe-2209-4921-9be4-beed5c9744cb.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/mla_paged_decode_h16_ckv512_kpe64_ps1_939f995a-1ab2-4d19-8d94-50f07e73542d.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_54187805-1b18-4d39-83ca-46332f85da9e.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/mla_ragged_prefill_causal_h16_qk192_vo128_1fe95283-ade9-4efa-8df0-8cd15dc8b09e.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0516-0609/fp8_mha_with_lse_d64_7834dd42-df9c-4cd4-a5a9-e1f3f8ad5fa1.py
  /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0516-0609/fp8_mha_with_lse_d128_8f46ada6-1617-49dc-b425-c6165e67e466.py
)

CONFIGS=(
  "$CONFIG_MAIN" "$CONFIG_MAIN" "$CONFIG_MAIN" "$CONFIG_MAIN"
  "$CONFIG_MAIN" "$CONFIG_MAIN" "$CONFIG_MAIN" "$CONFIG_MAIN"
  "$CONFIG_MAIN" "$CONFIG_MAIN" "$CONFIG_FP8" "$CONFIG_FP8"
)

OUTPUT_ROOTS=()
for definition in "${DEFINITIONS[@]}"; do
  OUTPUT_ROOTS+=("$OUTPUT_ROOT_BASE/$RUN_DATE-$definition")
done

source "$WATCH_COMMON"

validate_static_inputs
log "watcher started for RUN_DATE=$RUN_DATE MODEL_NAME=$MODEL_NAME ACCRL_MODEL_HOST=$ACCRL_MODEL_HOST SERVICE_URL=$SERVICE_URL RUNNER=$RUNNER PROFILE_HOST=$PROFILE_HOST PROFILE_CONTAINER=$PROFILE_CONTAINER CONFIG_MAIN=$CONFIG_MAIN CONFIG_FP8=$CONFIG_FP8"

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
