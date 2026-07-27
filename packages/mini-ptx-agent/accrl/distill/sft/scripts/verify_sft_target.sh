#!/usr/bin/env bash
# Verify Qwen thinking-channel SFT target rendering and Miles qwen3 loss mask.

set -euo pipefail

IMAGE="${IMAGE:-docker.io/radixark/miles:latest}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-sudo -n podman}"
WORK_DIR="${WORK_DIR:-/home/chengze/work}"
DATA_DIR="${DATA_DIR:-/data/local}"
ACCRL_DIR="${ACCRL_DIR:-${WORK_DIR}/AccRL}"
SFT_DATA="${SFT_DATA:-${WORK_DIR}/AccRL-exps/sft_experiments/glm_kimi_gemini_v2/data/mixed_intersection.parquet}"
TOKENIZER="${TOKENIZER:-/data/local/models/qwen3.5_9B}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
PREVIEW_RADIUS="${PREVIEW_RADIUS:-8}"
CHECK_ALL="${CHECK_ALL:-0}"
MAX_PREVIEWS="${MAX_PREVIEWS:-8}"

usage() {
  cat <<'EOF'
Usage:
  accrl/distill/sft/scripts/verify_sft_target.sh [run]

Env:
  SFT_DATA        SFT parquet/jsonl to verify.
  TOKENIZER       Qwen tokenizer path. Default: /data/local/models/qwen3.5_9B
  NUM_SAMPLES     Number of sampled rows to verify. Default: 8
  CHECK_ALL       Verify every row when 1. Default: 0
  MAX_PREVIEWS    Maximum row previews to print. Default: 8
  PREVIEW_RADIUS  Tokens around first-loss and </think> boundaries. Default: 8
EOF
}

cmd_run() {
  [[ -f "$SFT_DATA" ]] || { echo "missing SFT_DATA: $SFT_DATA" >&2; exit 2; }

  # shellcheck disable=SC2086
  local check_all_arg=()
  if [[ "$CHECK_ALL" = "1" ]]; then
    check_all_arg=(--check-all)
  fi

  ${CONTAINER_ENGINE} run --rm \
    --network host \
    --ipc host \
    --gpus all \
    --pids-limit=-1 \
    -v "${WORK_DIR}:${WORK_DIR}:rw" \
    -v "${DATA_DIR}:${DATA_DIR}:ro" \
    -v "${ACCRL_DIR}/accrl/distill/sft/miles:/root/miles:rw" \
    "$IMAGE" \
    bash -lc "
      set -euo pipefail
      cd '$ACCRL_DIR'
      python3 accrl/distill/sft/verify_sft_target.py \
        --data '$SFT_DATA' \
        --tokenizer '$TOKENIZER' \
        --num-samples '$NUM_SAMPLES' \
        --max-previews '$MAX_PREVIEWS' \
        --preview-radius '$PREVIEW_RADIUS' \
        ${check_all_arg[*]}
    "
}

case "${1:-run}" in
  run) cmd_run ;;
  -h|--help|help) usage ;;
  *) echo "unknown command: $1" >&2; exit 2 ;;
esac
