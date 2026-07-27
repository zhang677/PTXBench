#!/usr/bin/env bash
# Run AccRL distill SFT with the Miles Docker/Podman image.

set -euo pipefail

IMAGE="${IMAGE:-docker.io/radixark/miles:latest}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-sudo -n podman}"
CONTAINER_NAME="${CONTAINER_NAME:-accrl-miles-sft}"
CONTAINER_LABEL="${CONTAINER_LABEL:-accrl.sft=1}"
SFT_CLEANUP_PREVIOUS_CONTAINER="${SFT_CLEANUP_PREVIOUS_CONTAINER:-1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SFT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORK_DIR="${WORK_DIR:-/home/chengze/work}"
DATA_DIR="${DATA_DIR:-/data/local}"
HOST_HOME="${HOST_HOME:-/home/chengze}"
MILES_SRC_DIR="${MILES_SRC_DIR:-${SFT_DIR}/miles}"

MODEL_PRESET="${MODEL_PRESET:-qwen35-9b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SCRIPT="${MODEL_SCRIPT:-}"
MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-}"
SFT_DATA="${SFT_DATA:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REF_LOAD="${REF_LOAD:-}"
WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-}"
WANDB_GROUP_DEFAULT="${WANDB_GROUP_DEFAULT:-}"
MODEL_IS_MOE="${MODEL_IS_MOE:-}"

apply_model_preset() {
  case "$MODEL_PRESET" in
    qwen35-9b|qwen3.5-9b|qwen3.5_9b|9b)
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
      MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3.5-9B.sh}"
      MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-Qwen3.5-9B_miles}"
      HF_CHECKPOINT="${HF_CHECKPOINT:-/data/local/models/qwen3.5_9B}"
      WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-qwen35-9b-sft-glm-v3}"
      WANDB_GROUP_DEFAULT="${WANDB_GROUP_DEFAULT:-glm-v3-9b}"
      MODEL_IS_MOE="${MODEL_IS_MOE:-0}"
      ;;
    qwen35-35b-a3b|qwen3.5-35b-a3b|qwen3.5_35b_a3b|35b|35b-a3b)
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
      MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3.5-35B-A3B.sh}"
      MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-Qwen3.5-35B-A3B_miles}"
      HF_CHECKPOINT="${HF_CHECKPOINT:-/data/local/models/qwen35-35B-A3B}"
      WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-qwen35-35b-a3b-sft-glm-v3}"
      WANDB_GROUP_DEFAULT="${WANDB_GROUP_DEFAULT:-glm-v3-qwen35-35b-a3b}"
      MODEL_IS_MOE="${MODEL_IS_MOE:-1}"
      TP_SIZE="${TP_SIZE:-1}"
      CP_SIZE="${CP_SIZE:-4}"
      SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-1}"
      if [[ "$SFT_ENABLE_EVAL" = "1" ]]; then
        EP_SIZE="${EP_SIZE:-4}"
      else
        EP_SIZE="${EP_SIZE:-8}"
      fi
      SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
      GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE}}"
      LR="${LR:-1e-7}"
      MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
      LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
      ALLGATHER_CP="${ALLGATHER_CP:-0}"
      SKIP_NAN_STEPS="${SKIP_NAN_STEPS:-0}"
      OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"
      OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-1}"
      ENABLE_MTP_TRAINING="${ENABLE_MTP_TRAINING:-0}"
      MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
      MTP_LOSS_SCALING_FACTOR="${MTP_LOSS_SCALING_FACTOR:-0.2}"
      MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-flex}"
      SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
      SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
      SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-triton}"
      SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-512}"
      ;;
    custom)
      : "${MODEL_NAME:?MODEL_NAME is required when MODEL_PRESET=custom}"
      : "${MODEL_SCRIPT:?MODEL_SCRIPT is required when MODEL_PRESET=custom}"
      : "${HF_CHECKPOINT:?HF_CHECKPOINT is required when MODEL_PRESET=custom}"
      MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-${MODEL_NAME}_miles}"
      WANDB_PROJECT_DEFAULT="${WANDB_PROJECT_DEFAULT:-accrl-sft}"
      WANDB_GROUP_DEFAULT="${WANDB_GROUP_DEFAULT:-${MODEL_NAME}}"
      ;;
    *)
      die "Unknown MODEL_PRESET: ${MODEL_PRESET}"
      ;;
  esac
  REF_LOAD="${REF_LOAD:-${HF_CHECKPOINT}_torch_dist}"
}

usage() {
  cat <<'EOF'
Usage:
  accrl/distill/sft/scripts/qwen3.5-9b_sft.sh <command>

Commands:
  images    Show root Podman images.
  smoke     Verify the Miles image can see GPUs.
  shell     Start an interactive shell in the Miles image.
  run       Launch Qwen SFT through Ray inside the Miles image.

Required for run:
  HF_CHECKPOINT Host path to the HF checkpoint.
  REF_LOAD      Host path to converted torch_dist checkpoint.
  OUT_FOLDER    Host output folder for checkpoints and eval prompts.
  MASTER_ADDR   Ray head IP, usually 127.0.0.1 for single-node.

Useful env:
  MODEL_PRESET      qwen35-9b, qwen35-35b-a3b, or custom. Default: qwen35-9b
  MODEL_NAME        Default resolved from MODEL_PRESET.
  MODEL_SCRIPT      Miles scripts/models file. Default resolved from MODEL_PRESET.
  IMAGE             Default: docker.io/radixark/miles:latest
  CONTAINER_NAME    Stable run container name. Default: accrl-miles-sft
  SFT_CLEANUP_PREVIOUS_CONTAINER Force-remove prior AccRL SFT containers before run. Default: 1
  WORK_DIR          Mounted host work dir. Default: /home/chengze/work
  DATA_DIR          Mounted host data dir. Default: /data/local
  MILES_SRC_DIR     Developable Miles source mounted at /root/miles.
                    Default: accrl/distill/sft/miles
  HOST_HOME         Mounted credential source. Default: /home/chengze
  HF_CHECKPOINT     Default: /data/local/models/qwen3.5_9B for 9B,
                    /data/local/models/qwen35-35B-A3B for 35B-A3B.
  REF_LOAD          Default: ${HF_CHECKPOINT}_torch_dist
  ACTOR_CKPT        Default: $OUT_FOLDER/${MODEL_NAME}_miles
  SFT_DATA          Input SFT parquet. Default: $OUT_FOLDER/glm_v3_sft.parquet
  RAY_NUM_CPUS      CPU resources advertised to Ray. Default: 16
  SFT_BATCH_SIZE    Default: 1; 35B-A3B preset: 2.
  GLOBAL_BATCH_SIZE Default: SFT_BATCH_SIZE. Must be divisible by data parallel size.
  NUM_EPOCH         Default: 20
  LR                Default: 5e-6; 35B-A3B preset: 1e-7.
  WEIGHT_DECAY      Default: 0.1
  LR_DECAY_STYLE    Default: cosine
  LR_WARMUP_FRACTION Default: 0.1
  MIN_LR            Default: 1e-7
  TP_SIZE           Default: 1
  CP_SIZE           Default: 1
  EP_SIZE           Expert-model parallel size. Default: 1; 35B-A3B preset: 8.
  OPTIMIZER_CPU_OFFLOAD Enable optimizer CPU offload. Default: 0; 35B-A3B preset: 1.
  ALLGATHER_CP      Add --allgather-cp when CP_SIZE>1. Default: 1; 35B-A3B preset: 0.
  RECOMPUTE_LOSS_FUNCTION Add --recompute-loss-function. Default: 1.
  DEBUG_DISABLE_OPTIMIZER Add --debug-disable-optimizer for backward diagnostics. Default: 0.
  CLIP_GRAD         Megatron --clip-grad. Default: 1.0.
  ENABLE_MTP_TRAINING Add Miles MTP training args. Default: 0.
  SAVE_CHECKPOINTS  Set 1 to enable periodic/final checkpoint saves. Default: 0.
  NO_SAVE_OPTIM     Add --no-save-optim when saving. Default: 0.
  MAX_TOKENS_PER_GPU Default: 32768
  WANDB_API_KEY, WANDB_HOST, WANDB_PROJECT, WANDB_TEAM, WANDB_ENTITY, WANDB_GROUP, WANDB_DIR
  WANDB_RANDOM_SUFFIX Disable with 0 for stable W&B run/group names. Default: 1
  SFT_WANDB_TOKEN_TABLE_EVERY Log token/mask W&B table every N rollouts. Default: 0
  SFT_WANDB_TOKEN_TABLE_RADIUS Tokens before/after each boundary in table. Default: 24
  SKIP_NAN_STEPS    Add --no-check-for-nan-in-loss-and-grad. Default: 1; 35B-A3B preset: 0.
  SFT_ENABLE_EVAL    1 to run Miles/SGLang eval in the same W&B run; set 0 for train-only debug. Default: 1
  SFT_EVAL_DATA      Eval JSONL prompts. Auto-built from SFT_DATA if unset.
  SFT_EVAL_TURNS     Optional Gemini turns JSONL used to rebuild policy-style eval prompts.
  SFT_EVAL_INTERVAL  Eval every N rollouts. Default: 50
  SFT_EVAL_NUM_SAMPLES Number of prompts when auto-building eval data. Default: 16
  SFT_EVAL_SEED      Eval prompt sampling seed. Default: 0
  SFT_EVAL_MAX_RESPONSE_LEN Generation cap. Default: 8192
  SFT_EVAL_TEMPERATURE Default: 0
  SFT_EVAL_TOP_P     Default: 1.0
  SGLANG_MOE_RUNNER_BACKEND Override SGLang MoE runner backend for eval. 35B-A3B default: triton.
  ACTOR_NUM_GPUS_PER_NODE Training GPUs. Default: 4 with eval, 8 when SFT_ENABLE_EVAL=0.
  ROLLOUT_NUM_GPUS Eval/SGLang GPUs. Default: 4 with eval, 0 when SFT_ENABLE_EVAL=0.
  ROLLOUT_NUM_GPUS_PER_ENGINE SGLang TP size. Default: ROLLOUT_NUM_GPUS.
  RAY_NUM_GPUS      Ray GPUs advertised. Default: actor + rollout.
  MILES_HOST_IP     Host IP advertised by Miles/SGLang. Default: MASTER_ADDR.
                    Keep as 127.0.0.1 for this single-node host-network run.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

need_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    die "Missing required env var: ${name}"
  fi
}

apply_model_preset

container_base_args=(
  run --rm
  --network host
  --ipc host
  --gpus all
  --label "$CONTAINER_LABEL"
  --pids-limit=-1
  --ulimit memlock=-1:-1
  --ulimit stack=67108864
  -v "${WORK_DIR}:${WORK_DIR}:rw"
  -v "${DATA_DIR}:${DATA_DIR}:rw"
  -v "${MILES_SRC_DIR}:/root/miles:rw"
  -v "${HOST_HOME}/.netrc:/root/.netrc:ro"
  -v "${HOST_HOME}/.config/wandb:/root/.config/wandb:ro"
)

cleanup_previous_container() {
  [[ "$SFT_CLEANUP_PREVIOUS_CONTAINER" = "1" ]] || return 0

  # Because runs use host networking, stale containers can leave Ray/SGLang
  # ports alive and make the next Ray head attach to old GCS metadata.
  # Scope cleanup to this launcher's stable name/label instead of killing all
  # host Python processes.
  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  local ids
  # shellcheck disable=SC2086
  ids="$(${CONTAINER_ENGINE} ps -aq --filter "label=${CONTAINER_LABEL}" 2>/dev/null || true)"
  if [[ -n "$ids" ]]; then
    # shellcheck disable=SC2086
    ${CONTAINER_ENGINE} rm -f $ids >/dev/null 2>&1 || true
  fi
}

container_exec() {
  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} "${container_base_args[@]}" "$IMAGE" "$@"
}

cmd_images() {
  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} images
}

cmd_smoke() {
  container_exec nvidia-smi -L
}

cmd_shell() {
  cleanup_previous_container
  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} "${container_base_args[@]}" -it --name "$CONTAINER_NAME" "$IMAGE" bash
}

cmd_run() {
  need_env OUT_FOLDER
  need_env MASTER_ADDR
  if [[ -z "$SFT_DATA" ]]; then
    SFT_DATA="${OUT_FOLDER%/}/glm_v3_sft.parquet"
  fi
  [[ -d "$HF_CHECKPOINT" ]] || die "HF_CHECKPOINT not found: $HF_CHECKPOINT"
  [[ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]] || die "REF_LOAD torch_dist checkpoint not found: $REF_LOAD (run convert_qwen3.5-9b_model.sh first)"
  [[ -f "$SFT_DATA" ]] || die "SFT data not found: $SFT_DATA (run convert first)"
  [[ -f "${MILES_SRC_DIR}/train_async.py" ]] || die "MILES_SRC_DIR does not look like Miles source: $MILES_SRC_DIR"
  [[ -f "${MILES_SRC_DIR}/scripts/models/${MODEL_SCRIPT}" ]] || die "MODEL_SCRIPT not found under Miles source: ${MODEL_SCRIPT}"
  ACTOR_CKPT="${ACTOR_CKPT:-${OUT_FOLDER%/}/${MODEL_CKPT_DIRNAME}}"
  SFT_ENABLE_EVAL_RESOLVED="${SFT_ENABLE_EVAL:-1}"
  if [[ "$SFT_ENABLE_EVAL_RESOLVED" = "1" ]]; then
    ACTOR_NUM_GPUS_PER_NODE_RESOLVED="${ACTOR_NUM_GPUS_PER_NODE:-4}"
    ROLLOUT_NUM_GPUS_RESOLVED="${ROLLOUT_NUM_GPUS:-4}"
  else
    ACTOR_NUM_GPUS_PER_NODE_RESOLVED="${ACTOR_NUM_GPUS_PER_NODE:-8}"
    ROLLOUT_NUM_GPUS_RESOLVED="${ROLLOUT_NUM_GPUS:-0}"
  fi
  ROLLOUT_NUM_GPUS_PER_ENGINE_RESOLVED="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${ROLLOUT_NUM_GPUS_RESOLVED}}"
  RAY_NUM_GPUS_RESOLVED="${RAY_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE_RESOLVED + ROLLOUT_NUM_GPUS_RESOLVED))}"
  TP_SIZE_RESOLVED="${TP_SIZE:-1}"
  CP_SIZE_RESOLVED="${CP_SIZE:-1}"
  SFT_BATCH_SIZE_RESOLVED="${SFT_BATCH_SIZE:-1}"
  GLOBAL_BATCH_SIZE_RESOLVED="${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE_RESOLVED}}"
  model_parallel_size=$((TP_SIZE_RESOLVED * CP_SIZE_RESOLVED))
  if (( TP_SIZE_RESOLVED <= 0 || CP_SIZE_RESOLVED <= 0 || model_parallel_size <= 0 )); then
    die "TP_SIZE and CP_SIZE must be positive; got TP_SIZE=${TP_SIZE_RESOLVED} CP_SIZE=${CP_SIZE_RESOLVED}"
  fi
  if (( ACTOR_NUM_GPUS_PER_NODE_RESOLVED <= 0 )); then
    die "ACTOR_NUM_GPUS_PER_NODE must be positive; got ${ACTOR_NUM_GPUS_PER_NODE_RESOLVED}"
  fi
  if [[ "$SFT_ENABLE_EVAL_RESOLVED" = "1" && "$ROLLOUT_NUM_GPUS_RESOLVED" -le 0 ]]; then
    die "SFT_ENABLE_EVAL=1 requires ROLLOUT_NUM_GPUS > 0"
  fi
  if (( ROLLOUT_NUM_GPUS_RESOLVED > 0 && ROLLOUT_NUM_GPUS_PER_ENGINE_RESOLVED <= 0 )); then
    die "ROLLOUT_NUM_GPUS_PER_ENGINE must be positive when rollout GPUs are used; got ${ROLLOUT_NUM_GPUS_PER_ENGINE_RESOLVED}"
  fi
  if (( ACTOR_NUM_GPUS_PER_NODE_RESOLVED % model_parallel_size != 0 )); then
    die "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE_RESOLVED} must be divisible by TP_SIZE*CP_SIZE=${model_parallel_size}"
  fi
  data_parallel_size=$((ACTOR_NUM_GPUS_PER_NODE_RESOLVED / model_parallel_size))
  if (( GLOBAL_BATCH_SIZE_RESOLVED <= 0 || SFT_BATCH_SIZE_RESOLVED <= 0 )); then
    die "SFT_BATCH_SIZE and GLOBAL_BATCH_SIZE must be positive; got SFT_BATCH_SIZE=${SFT_BATCH_SIZE_RESOLVED} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE_RESOLVED}"
  fi
  if (( GLOBAL_BATCH_SIZE_RESOLVED % data_parallel_size != 0 )); then
    die "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE_RESOLVED} must be divisible by data_parallel_size=${data_parallel_size} (actor_gpus / (TP_SIZE*CP_SIZE))"
  fi
  if [[ -z "${EP_SIZE:-}" ]]; then
    if [[ "$MODEL_IS_MOE" = "1" ]]; then
      EP_SIZE_RESOLVED="${data_parallel_size}"
    else
      EP_SIZE_RESOLVED="1"
    fi
  else
    EP_SIZE_RESOLVED="${EP_SIZE}"
  fi
  if (( EP_SIZE_RESOLVED <= 0 )); then
    die "EP_SIZE must be positive; got ${EP_SIZE_RESOLVED}"
  fi
  if (( ACTOR_NUM_GPUS_PER_NODE_RESOLVED % EP_SIZE_RESOLVED != 0 )); then
    die "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE_RESOLVED} must be divisible by EP_SIZE=${EP_SIZE_RESOLVED}"
  fi
  if (( ROLLOUT_NUM_GPUS_RESOLVED > 0 && ROLLOUT_NUM_GPUS_RESOLVED % ROLLOUT_NUM_GPUS_PER_ENGINE_RESOLVED != 0 )); then
    die "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS_RESOLVED} must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE_RESOLVED}"
  fi
  if (( RAY_NUM_GPUS_RESOLVED < ACTOR_NUM_GPUS_PER_NODE_RESOLVED + ROLLOUT_NUM_GPUS_RESOLVED )); then
    die "RAY_NUM_GPUS=${RAY_NUM_GPUS_RESOLVED} is less than actor+rollout GPUs=$((ACTOR_NUM_GPUS_PER_NODE_RESOLVED + ROLLOUT_NUM_GPUS_RESOLVED))"
  fi
  SGLANG_EP_SIZE_RESOLVED="${SGLANG_EP_SIZE:-}"
  if [[ -z "$SGLANG_EP_SIZE_RESOLVED" && "$MODEL_IS_MOE" = "1" && "$ROLLOUT_NUM_GPUS_RESOLVED" -gt 0 ]]; then
    SGLANG_EP_SIZE_RESOLVED="${ROLLOUT_NUM_GPUS_RESOLVED}"
  fi
  cleanup_previous_container

  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} "${container_base_args[@]}" \
    --name "$CONTAINER_NAME" \
    -e MODEL_PRESET="$MODEL_PRESET" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e MODEL_SCRIPT="$MODEL_SCRIPT" \
    -e MODEL_IS_MOE="$MODEL_IS_MOE" \
    -e HF_CHECKPOINT="$HF_CHECKPOINT" \
    -e REF_LOAD="$REF_LOAD" \
    -e ACTOR_CKPT="$ACTOR_CKPT" \
    -e SAVE_INTERVAL="${SAVE_INTERVAL:-50}" \
    -e SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}" \
    -e NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}" \
    -e OUT_FOLDER="$OUT_FOLDER" \
    -e MASTER_ADDR="$MASTER_ADDR" \
    -e MILES_HOST_IP="${MILES_HOST_IP:-${MASTER_ADDR}}" \
    -e SFT_DATA="$SFT_DATA" \
    -e SFT_BATCH_SIZE="${SFT_BATCH_SIZE_RESOLVED}" \
    -e GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE_RESOLVED}" \
    -e NUM_EPOCH="${NUM_EPOCH:-20}" \
    -e LR="${LR:-5e-6}" \
    -e WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}" \
    -e LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}" \
    -e LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.1}" \
    -e MIN_LR="${MIN_LR:-1e-7}" \
    -e TP_SIZE="${TP_SIZE_RESOLVED}" \
    -e CP_SIZE="${CP_SIZE_RESOLVED}" \
    -e EP_SIZE="${EP_SIZE_RESOLVED}" \
    -e MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}" \
    -e LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}" \
    -e ALLGATHER_CP="${ALLGATHER_CP:-1}" \
    -e RECOMPUTE_LOSS_FUNCTION="${RECOMPUTE_LOSS_FUNCTION:-1}" \
    -e DEBUG_DISABLE_OPTIMIZER="${DEBUG_DISABLE_OPTIMIZER:-0}" \
    -e CLIP_GRAD="${CLIP_GRAD:-1.0}" \
    -e OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-0}" \
    -e OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-${OPTIMIZER_CPU_OFFLOAD:-0}}" \
    -e USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-1}" \
    -e ENABLE_MTP_TRAINING="${ENABLE_MTP_TRAINING:-0}" \
    -e MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}" \
    -e MTP_LOSS_SCALING_FACTOR="${MTP_LOSS_SCALING_FACTOR:-0.2}" \
    -e MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-}" \
    -e RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}" \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
    -e OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
    -e NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_HOST="${WANDB_HOST:-https://meta.wandb.io}" \
    -e WANDB_BASE_URL="${WANDB_BASE_URL:-${WANDB_HOST:-https://meta.wandb.io}}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-${WANDB_PROJECT_DEFAULT}}" \
    -e WANDB_TEAM="${WANDB_TEAM:-${WANDB_ENTITY:-}}" \
    -e WANDB_ENTITY="${WANDB_ENTITY:-${WANDB_TEAM:-}}" \
    -e WANDB_GROUP="${WANDB_GROUP:-${WANDB_GROUP_DEFAULT}}" \
    -e WANDB_DIR="${WANDB_DIR:-}" \
    -e WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-1}" \
    -e SFT_WANDB_TOKEN_TABLE_EVERY="${SFT_WANDB_TOKEN_TABLE_EVERY:-0}" \
    -e SFT_WANDB_TOKEN_TABLE_RADIUS="${SFT_WANDB_TOKEN_TABLE_RADIUS:-24}" \
    -e SFT_NAN_GRAD_DIAGNOSTICS="${SFT_NAN_GRAD_DIAGNOSTICS:-0}" \
    -e SFT_WEIGHT_SYNC_DIAGNOSTICS="${SFT_WEIGHT_SYNC_DIAGNOSTICS:-0}" \
    -e SKIP_NAN_STEPS="${SKIP_NAN_STEPS:-1}" \
    -e SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL_RESOLVED}" \
    -e SFT_EVAL_DATA="${SFT_EVAL_DATA:-}" \
    -e SFT_EVAL_TURNS="${SFT_EVAL_TURNS:-}" \
    -e SFT_EVAL_INTERVAL="${SFT_EVAL_INTERVAL:-50}" \
    -e SFT_EVAL_NUM_SAMPLES="${SFT_EVAL_NUM_SAMPLES:-16}" \
    -e SFT_EVAL_SEED="${SFT_EVAL_SEED:-0}" \
    -e SFT_EVAL_MAX_RESPONSE_LEN="${SFT_EVAL_MAX_RESPONSE_LEN:-8192}" \
    -e SFT_EVAL_TEMPERATURE="${SFT_EVAL_TEMPERATURE:-0}" \
    -e SFT_EVAL_TOP_P="${SFT_EVAL_TOP_P:-1.0}" \
    -e ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE_RESOLVED}" \
    -e ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS_RESOLVED}" \
    -e ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE_RESOLVED}" \
    -e RAY_NUM_GPUS="${RAY_NUM_GPUS_RESOLVED}" \
    -e SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-}" \
    -e SGLANG_EP_SIZE="${SGLANG_EP_SIZE_RESOLVED}" \
    -e SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}" \
    -e SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-}" \
    -e SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-}" \
    -e SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-}" \
    -e SGLANG_FLASHINFER_PREWARM="${SGLANG_FLASHINFER_PREWARM:-1}" \
    -e ACCRL_DIR="${WORK_DIR}/AccRL" \
    -e HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
    -e HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
    -e http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
    -e https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
    -e NO_PROXY="${NO_PROXY:-${no_proxy:-}}" \
    -e no_proxy="${no_proxy:-${NO_PROXY:-}}" \
    "$IMAGE" \
    bash -lc "$(cat <<'EOS'
set -euo pipefail
cd /root/miles

pkill -9 sglang || true; sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

for path in \
  "${HF_CHECKPOINT}" \
  "${REF_LOAD}" \
  "${SFT_DATA}"
do
  [ -e "$path" ] || { echo "missing required path: $path"; exit 1; }
done
mkdir -p "${ACTOR_CKPT}"

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

echo "MODEL_PRESET=${MODEL_PRESET} MODEL_NAME=${MODEL_NAME} MODEL_SCRIPT=${MODEL_SCRIPT}"
[ -f "/root/miles/scripts/models/${MODEL_SCRIPT}" ] || { echo "MODEL_SCRIPT not found: /root/miles/scripts/models/${MODEL_SCRIPT}"; exit 1; }
source "/root/miles/scripts/models/${MODEL_SCRIPT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load      "${REF_LOAD}"
   --load          "${ACTOR_CKPT}/"
)
if [ "${SAVE_CHECKPOINTS:-0}" = "1" ]; then
   CKPT_ARGS+=(
      --save          "${ACTOR_CKPT}/"
      --save-interval "${SAVE_INTERVAL}"
   )
   if [ "${NO_SAVE_OPTIM:-0}" = "1" ]; then
      CKPT_ARGS+=(--no-save-optim)
   fi
fi

SFT_ARGS=(
   --rollout-function-path miles.rollout.sft_rollout.generate_rollout
   --prompt-data "${SFT_DATA}"
   --input-key messages
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --rollout-batch-size "${SFT_BATCH_SIZE}"
   --global-batch-size  "${GLOBAL_BATCH_SIZE}"

   --loss-type sft_loss
   --loss-mask-type qwen3
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
)

EVAL_ARGS=()
if [ "${SFT_ENABLE_EVAL}" = "1" ]; then
   SFT_EVAL_DATA_AUTO=0
   if [ -z "${SFT_EVAL_DATA}" ]; then
      SFT_EVAL_DATA="${OUT_FOLDER%/}/eval_prompts.jsonl"
      SFT_EVAL_DATA_AUTO=1
   fi
   if [ "${SFT_EVAL_DATA_AUTO}" = "1" ] || [ ! -f "${SFT_EVAL_DATA}" ]; then
      EVAL_PROMPT_ARGS=()
      if [ -n "${SFT_EVAL_TURNS:-}" ]; then
         [ -f "${SFT_EVAL_TURNS}" ] || { echo "SFT_EVAL_TURNS not found: ${SFT_EVAL_TURNS}"; exit 1; }
         EVAL_PROMPT_ARGS+=(--turns "${SFT_EVAL_TURNS}")
      fi
      python3 "${ACCRL_DIR}/accrl/distill/sft/prepare_sft_eval_prompts.py" \
        --data "${SFT_DATA}" \
        --output "${SFT_EVAL_DATA}" \
        --num-samples "${SFT_EVAL_NUM_SAMPLES}" \
        --seed "${SFT_EVAL_SEED}" \
        --output-key prompt \
        "${EVAL_PROMPT_ARGS[@]}"
   fi
   EVAL_ARGS=(
      --eval-interval "${SFT_EVAL_INTERVAL}"
      --eval-function-path miles.rollout.sglang_rollout.generate_rollout
      --custom-generate-function-path miles.rollout.sft_eval.generate_with_zero_reward
      --custom-eval-rollout-log-function-path miles.rollout.sft_eval.log_generated_thinking_metrics
      --eval-prompt-data sft "${SFT_EVAL_DATA}"
      --eval-input-key prompt
      --eval-label-key target
      --n-samples-per-eval-prompt 1
      --eval-max-response-len "${SFT_EVAL_MAX_RESPONSE_LEN}"
      --eval-temperature "${SFT_EVAL_TEMPERATURE}"
      --eval-top-p "${SFT_EVAL_TOP_P}"
      --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
      --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
      --num-gpus-per-node "${RAY_NUM_GPUS}"
   )
   if [ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]; then
      EVAL_ARGS+=(--sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}")
   fi
   if [ -n "${SGLANG_EP_SIZE:-}" ]; then
      EVAL_ARGS+=(--sglang-ep-size "${SGLANG_EP_SIZE}")
   fi
   if [ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]; then
      EVAL_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
   fi
   if [ -n "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ]; then
      EVAL_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
   fi
   if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
      EVAL_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
   fi
   if [ -n "${SGLANG_MOE_RUNNER_BACKEND:-}" ]; then
      EVAL_ARGS+=(--sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}")
   fi
else
   SFT_ARGS+=(--debug-train-only)
fi

if [ "${SKIP_NAN_STEPS}" = "1" ]; then
   SFT_ARGS+=(--no-check-for-nan-in-loss-and-grad)
fi
if [ "${DEBUG_DISABLE_OPTIMIZER:-0}" = "1" ]; then
   SFT_ARGS+=(--debug-disable-optimizer)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"

   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
   --clip-grad "${CLIP_GRAD}"
)

if [ "${CP_SIZE}" != "1" ] && [ "${ALLGATHER_CP:-1}" = "1" ]; then
   PERF_ARGS+=(--allgather-cp)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style "${LR_DECAY_STYLE:-cosine}"
   --min-lr "${MIN_LR:-1e-7}"
   --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.1}"
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 0.9
   --adam-beta2 0.98
)

if [ "${OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload)
   if [ "${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-1}" = "1" ]; then
      OPTIMIZER_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
   fi
fi
if [ "${USE_PRECISION_AWARE_OPTIMIZER:-1}" = "1" ]; then
   OPTIMIZER_ARGS+=(--use-precision-aware-optimizer)
fi

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ] || [ -f /root/.netrc ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group   "${WANDB_GROUP}"
      --wandb-host    "${WANDB_HOST}"
   )
   if [ -n "${WANDB_API_KEY:-}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
   fi
   if [ -n "${WANDB_TEAM:-${WANDB_ENTITY:-}}" ]; then
      WANDB_ARGS+=(--wandb-team "${WANDB_TEAM:-${WANDB_ENTITY}}")
   fi
   if [ -n "${WANDB_DIR:-}" ]; then
      WANDB_ARGS+=(--wandb-dir "${WANDB_DIR}")
   fi
   if [ "${WANDB_RANDOM_SUFFIX:-1}" = "0" ]; then
      WANDB_ARGS+=(--disable-wandb-random-suffix)
   fi
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --cross-entropy-loss-fusion
)
if [ "${RECOMPUTE_LOSS_FUNCTION:-1}" = "1" ]; then
   MISC_ARGS+=(--recompute-loss-function)
fi

MTP_ARGS=()
if [ "${ENABLE_MTP_TRAINING:-0}" = "1" ]; then
   MTP_ARGS=(
      --enable-mtp-training
      --mtp-num-layers "${MTP_NUM_LAYERS:-1}"
      --mtp-loss-scaling-factor "${MTP_LOSS_SCALING_FACTOR:-0.2}"
   )
fi

if [ -n "${MOE_TOKEN_DISPATCHER_TYPE:-}" ]; then
   MISC_ARGS+=(--moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}")
fi

export MILES_HOST_IP="${MILES_HOST_IP:-${MASTER_ADDR}}"
export no_proxy="127.0.0.1,${MASTER_ADDR},${MILES_HOST_IP}${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
export PYTHONPATH=/root/miles:/root/Megatron-LM/
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export MASTER_ADDR="${MASTER_ADDR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "${SFT_ENABLE_EVAL}" = "1" ] && [ "${SGLANG_FLASHINFER_PREWARM:-1}" = "1" ]; then
   python3 - <<'PY'
from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

get_trtllm_moe_sm100_module()
print("prewarmed flashinfer trtllm MoE module for SGLang eval", flush=True)
PY
fi

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" \
  --num-cpus "${RAY_NUM_CPUS:-16}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

for i in $(seq 1 60); do
  if ray status --address=auto >/dev/null 2>&1; then
    echo "ray cluster ready after ${i}s"
    break
  fi
  sleep 1
done

# Run as a local Ray driver instead of using Ray Jobs. In this Podman+host-net
# setup Ray Jobs can route through a dashboard JobAgent IPv6 address that refuses
# connections, while the local driver path only needs the GCS address.
python3 -c 'import os, ray, runpy; ray.init(address="auto", runtime_env={"env_vars": {k: os.environ[k] for k in ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "no_proxy", "NO_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "MASTER_ADDR", "MILES_HOST_IP", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "WANDB_BASE_URL", "WANDB_HOST", "WANDB_TEAM", "WANDB_ENTITY", "SFT_WANDB_TOKEN_TABLE_EVERY", "SFT_WANDB_TOKEN_TABLE_RADIUS", "SFT_NAN_GRAD_DIAGNOSTICS", "SFT_WEIGHT_SYNC_DIAGNOSTICS") if k in os.environ}}); runpy.run_path("train_async.py", run_name="__main__")' \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${MISC_ARGS[@]}"
EOS
)"
}

main() {
  case "${1:-}" in
    images) cmd_images ;;
    smoke) cmd_smoke ;;
    shell) cmd_shell ;;
    run) cmd_run ;;
    -h|--help|help|"") usage ;;
    *) die "Unknown command: $1" ;;
  esac
}

main "$@"
