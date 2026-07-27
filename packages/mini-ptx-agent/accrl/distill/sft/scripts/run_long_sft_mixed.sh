#!/usr/bin/env bash
# Long AccRL/Miles SFT defaults for controlled GLM/Kimi intersection data.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

timestamp="$(date +%Y%m%d-%H%M%S)"

export DATASET="${DATASET:-mixed}"
export MODE="${MODE:-full}"
export MODEL_PRESET="${MODEL_PRESET:-qwen35-9b}"

# With eval enabled, the harness uses 4 GPUs for Megatron training and 4 GPUs
# for SGLang eval. The long reasoning samples have OOMed at CP2/batch4, so keep
# the default long-run profile conservative and override upward only after a
# short stability run.
export NUM_EPOCH="${NUM_EPOCH:-20}"
export TP_SIZE="${TP_SIZE:-1}"
if [[ "$MODEL_PRESET" = "qwen35-35b-a3b" || "$MODEL_PRESET" = "qwen3.5-35b-a3b" || "$MODEL_PRESET" = "35b" || "$MODEL_PRESET" = "35b-a3b" ]]; then
  export LR="${LR:-1e-7}"
  export CP_SIZE="${CP_SIZE:-4}"
  export SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-1}"
  if [[ "$SFT_ENABLE_EVAL" = "1" ]]; then
    export EP_SIZE="${EP_SIZE:-4}"
  else
    export EP_SIZE="${EP_SIZE:-8}"
  fi
  export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
  export ALLGATHER_CP="${ALLGATHER_CP:-0}"
  export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"
  export ENABLE_MTP_TRAINING="${ENABLE_MTP_TRAINING:-0}"
  export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
  export RUN_TAG="${RUN_TAG:-qwen35-35b-a3b-lr${LR}-cp${CP_SIZE}-zigzagcp-offload-${timestamp}}"
else
  export LR="${LR:-1e-6}"
  export CP_SIZE="${CP_SIZE:-4}"
  export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
  export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
  export RUN_TAG="${RUN_TAG:-lr1e-6-cp${CP_SIZE}-nooffload-${timestamp}}"
fi
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
export SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-1}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE}}"

# Keep NaN checking enabled for the main scientific run. If this trips, inspect
# the failing batch/loss instead of silently skipping it.
export SKIP_NAN_STEPS="${SKIP_NAN_STEPS:-0}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export WANDB_PROJECT="${WANDB_PROJECT:-accrl-sft}"
export WANDB_TEAM="${WANDB_TEAM:-${WANDB_ENTITY:-}}"
export WANDB_ENTITY="${WANDB_ENTITY:-${WANDB_TEAM:-}}"
export WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"

command="${1:-run}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$command" in
  run|prepare|print|-h|--help|help)
    exec "${SCRIPT_DIR}/run_sft_experiment.sh" "$command" "$@"
    ;;
  *)
    echo "ERROR: unknown command: ${command}" >&2
    echo "Usage: $0 [run|prepare|print|--help]" >&2
    exit 2
    ;;
esac
