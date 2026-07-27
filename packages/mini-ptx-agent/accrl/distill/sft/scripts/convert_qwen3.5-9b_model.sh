#!/usr/bin/env bash
# Convert Qwen HuggingFace checkpoints to Megatron torch_dist for Miles.

set -euo pipefail

IMAGE="${IMAGE:-docker.io/radixark/miles:latest}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-sudo -n podman}"
WORK_DIR="${WORK_DIR:-/home/chengze/work}"
DATA_DIR="${DATA_DIR:-/data/local}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SFT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
MILES_SRC_DIR="${MILES_SRC_DIR:-${SFT_DIR}/miles}"

MODEL_PRESET="${MODEL_PRESET:-qwen35-9b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SCRIPT="${MODEL_SCRIPT:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REF_LOAD="${REF_LOAD:-}"
CONVERT_GPUS="${CONVERT_GPUS:-8}"

usage() {
  cat <<'EOF'
Usage:
  accrl/distill/sft/scripts/convert_qwen3.5-9b_model.sh [convert]

Converts a Qwen HuggingFace checkpoint to Megatron torch_dist format.
This is a one-time prerequisite for Miles Megatron SFT.

Env:
  MODEL_PRESET    qwen35-9b, qwen35-35b-a3b, or custom. Default: qwen35-9b
  MODEL_NAME      Default resolved from MODEL_PRESET.
  MODEL_SCRIPT    Miles scripts/models file. Default resolved from MODEL_PRESET.
  IMAGE           Default: docker.io/radixark/miles:latest
  WORK_DIR        Mounted host work dir. Default: /home/chengze/work
  DATA_DIR        Mounted host data dir. Default: /data/local
  MILES_SRC_DIR   Developable Miles source. Default: accrl/distill/sft/miles
  HF_CHECKPOINT   Default resolved from MODEL_PRESET.
  REF_LOAD        Default: ${HF_CHECKPOINT}_torch_dist
  CONVERT_GPUS    Default: 8
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

apply_model_preset() {
  case "$MODEL_PRESET" in
    qwen35-9b|qwen3.5-9b|qwen3.5_9b|9b)
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
      MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3.5-9B.sh}"
      HF_CHECKPOINT="${HF_CHECKPOINT:-/data/local/models/qwen3.5_9B}"
      ;;
    qwen35-35b-a3b|qwen3.5-35b-a3b|qwen3.5_35b_a3b|35b|35b-a3b)
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
      MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3.5-35B-A3B.sh}"
      HF_CHECKPOINT="${HF_CHECKPOINT:-/data/local/models/qwen35-35B-A3B}"
      ;;
    custom)
      : "${MODEL_NAME:?MODEL_NAME is required when MODEL_PRESET=custom}"
      : "${MODEL_SCRIPT:?MODEL_SCRIPT is required when MODEL_PRESET=custom}"
      : "${HF_CHECKPOINT:?HF_CHECKPOINT is required when MODEL_PRESET=custom}"
      ;;
    *)
      die "Unknown MODEL_PRESET: ${MODEL_PRESET}"
      ;;
  esac
  REF_LOAD="${REF_LOAD:-${HF_CHECKPOINT}_torch_dist}"
}

apply_model_preset

cmd_convert() {
  [[ -d "$HF_CHECKPOINT" ]] || die "HF_CHECKPOINT not found: $HF_CHECKPOINT"
  [[ -f "${MILES_SRC_DIR}/tools/convert_hf_to_torch_dist.py" ]] || die "MILES_SRC_DIR does not look like Miles source: $MILES_SRC_DIR"
  [[ -f "${MILES_SRC_DIR}/scripts/models/${MODEL_SCRIPT}" ]] || die "MODEL_SCRIPT not found under Miles source: ${MODEL_SCRIPT}"
  mkdir -p "$(dirname "$REF_LOAD")"

  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} run --rm \
    --network host \
    --ipc host \
    --gpus all \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864 \
    -v "${WORK_DIR}:${WORK_DIR}:rw" \
    -v "${DATA_DIR}:${DATA_DIR}:rw" \
    -v "${MILES_SRC_DIR}:/root/miles:rw" \
    -e MODEL_PRESET="$MODEL_PRESET" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e MODEL_SCRIPT="$MODEL_SCRIPT" \
    -e HF_CHECKPOINT="$HF_CHECKPOINT" \
    -e REF_LOAD="$REF_LOAD" \
    -e CONVERT_GPUS="$CONVERT_GPUS" \
    "$IMAGE" \
    bash -lc "$(cat <<'EOS'
set -euo pipefail
cd /root/miles
echo "Converting MODEL_PRESET=${MODEL_PRESET} MODEL_NAME=${MODEL_NAME} MODEL_SCRIPT=${MODEL_SCRIPT}"
[ -f "/root/miles/scripts/models/${MODEL_SCRIPT}" ] || { echo "MODEL_SCRIPT not found: /root/miles/scripts/models/${MODEL_SCRIPT}"; exit 1; }
source "/root/miles/scripts/models/${MODEL_SCRIPT}"

if [ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
  echo "torch_dist checkpoint already exists: ${REF_LOAD}"
  cat "${REF_LOAD}/latest_checkpointed_iteration.txt"
  exit 0
fi

PYTHONPATH=/root/miles:/root/Megatron-LM torchrun --nproc-per-node "${CONVERT_GPUS}" \
  tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --save "${REF_LOAD}" \
  "${MODEL_ARGS[@]}"
EOS
)"
}

case "${1:-convert}" in
  convert) cmd_convert ;;
  -h|--help|help) usage ;;
  *) die "Unknown command: $1" ;;
esac
