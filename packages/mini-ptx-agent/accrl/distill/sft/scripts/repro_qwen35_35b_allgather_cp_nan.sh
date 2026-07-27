#!/usr/bin/env bash
# Minimal known repro for Qwen3.5-35B-A3B NaN gradients with allgather CP.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

mode="${1:-fail}"
case "$mode" in
  fail)
    allgather_cp=1
    default_tag="repro-allgather-cp-nan"
    ;;
  pass)
    allgather_cp=0
    default_tag="repro-noallgather-cp-finite"
    ;;
  print)
    allgather_cp="${ALLGATHER_CP:-1}"
    default_tag="repro-allgather-cp-print"
    ;;
  *)
    echo "Usage: $0 [fail|pass|print]" >&2
    echo "  fail: run with ALLGATHER_CP=1; expected to hit NaN local grad norm" >&2
    echo "  pass: run with ALLGATHER_CP=0; expected to finish finite" >&2
    echo "  print: print resolved harness paths without launching" >&2
    exit 2
    ;;
esac

repro_data="${REPRO_DATA:-/home/chengze/work/tmp/repro_nan_35b_cp4/first6_rollouts_exact_shuffle_seed42.parquet}"
if [[ ! -f "$repro_data" ]]; then
  echo "ERROR: repro parquet not found: $repro_data" >&2
  echo "Set REPRO_DATA=/path/to/first6_rollouts_exact_shuffle_seed42.parquet" >&2
  exit 2
fi

command="run"
if [[ "$mode" = "print" ]]; then
  command="print"
fi

export MODEL_PRESET="${MODEL_PRESET:-qwen35-35b-a3b}"
export SFT_DATA="$repro_data"
export DATASET="${DATASET:-mixed}"
export MODE="${MODE:-full}"
export NUM_EPOCH="${NUM_EPOCH:-1}"
export TP_SIZE="${TP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-4}"
export EP_SIZE="${EP_SIZE:-8}"
export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE}}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
export ALLGATHER_CP="${ALLGATHER_CP:-${allgather_cp}}"

# Keep optimizer updates off so the failure is isolated to forward/backward.
export DEBUG_DISABLE_OPTIMIZER="${DEBUG_DISABLE_OPTIMIZER:-1}"
export SFT_NAN_GRAD_DIAGNOSTICS="${SFT_NAN_GRAD_DIAGNOSTICS:-1}"
export SKIP_NAN_STEPS="${SKIP_NAN_STEPS:-0}"
export LR="${LR:-1e-8}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0}"
export MIN_LR="${MIN_LR:-0}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
export SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen35-35b-nan-repro}"
export WANDB_GROUP="${WANDB_GROUP:-${default_tag}}"
export WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"
export RUN_TAG="${RUN_TAG:-${default_tag}}"

exec "${SCRIPT_DIR}/run_sft_experiment.sh" "$command"
