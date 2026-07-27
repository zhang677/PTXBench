#!/usr/bin/env bash
# Reproducible AccRL/Miles SFT experiment harness.

set -euo pipefail

IMAGE="${IMAGE:-docker.io/radixark/miles:latest}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-sudo -n podman}"
WORK_DIR="${WORK_DIR:-/home/chengze/work}"
ACCRL_DIR="${ACCRL_DIR:-${WORK_DIR}/AccRL}"
EXPS_DIR="${EXPS_DIR:-${WORK_DIR}/AccRL-exps}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-glm_kimi_intersection}"
SFT_EXPERIMENT_ROOT="${SFT_EXPERIMENT_ROOT:-${EXPS_DIR}/sft_experiments/${SFT_EXPERIMENT_NAME}}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${SFT_EXPERIMENT_ROOT}/data}"
SFT_EVAL_TURNS="${SFT_EVAL_TURNS:-${EXPS_DIR}/distill/gemini_turns_0422.jsonl}"
OUT_ROOT="${OUT_ROOT:-${SFT_EXPERIMENT_ROOT}/runs}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MILES_SRC_DIR="${MILES_SRC_DIR:-${ACCRL_DIR}/accrl/distill/sft/miles}"
GROUP_RUNS_BY_MODEL="${GROUP_RUNS_BY_MODEL:-1}"

DATASET="${DATASET:-mixed}"       # glm, kimi, mixed, or explicit SFT_DATA
MODE="${MODE:-full}"              # full or smoke
EXP_NAME="${EXP_NAME:-}"
SMOKE_ROWS="${SMOKE_ROWS:-16}"
SMOKE_MODE="${SMOKE_MODE:-head}"  # head, sample, shortest, longest, or synthetic
SMOKE_SEED="${SMOKE_SEED:-1234}"
RUN_TAG="${RUN_TAG:-}"
MODEL_PRESET="${MODEL_PRESET:-qwen35-9b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SCRIPT="${MODEL_SCRIPT:-}"
MODEL_EXP_PREFIX="${MODEL_EXP_PREFIX:-}"
MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REF_LOAD="${REF_LOAD:-}"
DEFAULT_LR="${DEFAULT_LR:-}"

usage() {
  cat <<'EOF'
Usage:
  accrl/distill/sft/scripts/run_sft_experiment.sh <command>

Commands:
  print     Print resolved experiment settings.
  prepare   Prepare experiment directory, smoke subset if needed, manifest, and run metadata.
  run       Prepare, then launch qwen3.5-9b_sft.sh.
  overfit   Run a small fixed-row SFT overfit sanity check.
  suite     Run DATASETS sequentially with the long-run defaults.

Env:
  MODEL_PRESET  qwen35-9b, qwen35-35b-a3b, or custom. Default: qwen35-9b
  MODEL_NAME    Default resolved from MODEL_PRESET.
  MODEL_SCRIPT  Miles scripts/models file. Default resolved from MODEL_PRESET.
  DATASET       glm, kimi, mixed, or set SFT_DATA explicitly. Default: mixed
  MODE          full or smoke. Default: full
  EXP_NAME      Experiment name. Default: ${MODEL_EXP_PREFIX}-sft-${DATASET}-${MODE}
  RUN_TAG       Optional suffix for EXP_NAME.
  SFT_EXPERIMENT_NAME Default: glm_kimi_intersection
  SFT_EXPERIMENT_ROOT Default: /home/chengze/work/AccRL-exps/sft_experiments/$SFT_EXPERIMENT_NAME
  SFT_DATA_DIR  Default: $SFT_EXPERIMENT_ROOT/data
  SFT_EVAL_TURNS Default: /home/chengze/work/AccRL-exps/distill/gemini_turns_0422.jsonl
  OUT_ROOT      Default: $SFT_EXPERIMENT_ROOT/runs
  GROUP_RUNS_BY_MODEL Put runs under $OUT_ROOT/${MODEL_EXP_PREFIX}/. Default: 1
  MASTER_ADDR   Default: 127.0.0.1
  TRAIN_LOG     Default: $OUT_FOLDER/experiment_harness/train.log
  SMOKE_ROWS    Default: 16
  SMOKE_MODE    head, sample, shortest, longest, or synthetic. Default: head.
                overfit default: synthetic for 35B, shortest otherwise
  SMOKE_SEED    Default: 1234
  OVERFIT_ROWS  For overfit only. Default: 4
  OVERFIT_SOURCE synthetic or dataset. Default: synthetic for 35B, dataset otherwise
  OVERFIT_SMOKE_MODE Override the smoke mode used by overfit.
  OVERFIT_CP_SIZE Override CP_SIZE for synthetic 35B overfit. Default: 1
  OVERFIT_BATCH_SIZE Override SFT/GLOBAL batch for synthetic 35B overfit. Default: OVERFIT_ROWS
  DATASETS      For suite only. Default: glm kimi mixed
  BASE_TAG      For suite only. Default: thinkwrap-e30-cp4-b1-lr1e-6-<timestamp>

Training env is passed through to qwen3.5-9b_sft.sh. Direct run defaults
match qwen3.5-9b_sft.sh; use run_long_sft_mixed.sh or suite for long-run
defaults:
  LR, WEIGHT_DECAY, LR_DECAY_STYLE, LR_WARMUP_FRACTION, MIN_LR,
  NUM_EPOCH, TP_SIZE, CP_SIZE, EP_SIZE, SFT_BATCH_SIZE, GLOBAL_BATCH_SIZE,
  MAX_TOKENS_PER_GPU,
  OPTIMIZER_CPU_OFFLOAD, ENABLE_MTP_TRAINING,
  SKIP_NAN_STEPS, WANDB_HOST, WANDB_PROJECT, WANDB_TEAM/WANDB_ENTITY, WANDB_GROUP, WANDB_DIR, etc.
  Eval is enabled by default. Set SFT_ENABLE_EVAL=0 for train-only debugging.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

apply_model_preset() {
  case "$MODEL_PRESET" in
    qwen35-9b|qwen3.5-9b|qwen3.5_9b|9b)
      MODEL_PRESET="qwen35-9b"
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
      MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3.5-9B.sh}"
      MODEL_EXP_PREFIX="${MODEL_EXP_PREFIX:-qwen35-9b}"
      MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-Qwen3.5-9B_miles}"
      HF_CHECKPOINT="${HF_CHECKPOINT:-/data/local/models/qwen3.5_9B}"
      DEFAULT_LR="${DEFAULT_LR:-5e-6}"
      ;;
    qwen35-35b-a3b|qwen3.5-35b-a3b|qwen3.5_35b_a3b|35b|35b-a3b)
      MODEL_PRESET="qwen35-35b-a3b"
      MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
      MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3.5-35B-A3B.sh}"
      MODEL_EXP_PREFIX="${MODEL_EXP_PREFIX:-qwen35-35b-a3b}"
      MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-Qwen3.5-35B-A3B_miles}"
      HF_CHECKPOINT="${HF_CHECKPOINT:-/data/local/models/qwen35-35B-A3B}"
      DEFAULT_LR="${DEFAULT_LR:-1e-7}"
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
      SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
      SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-512}"
      ;;
    custom)
      : "${MODEL_NAME:?MODEL_NAME is required when MODEL_PRESET=custom}"
      : "${MODEL_SCRIPT:?MODEL_SCRIPT is required when MODEL_PRESET=custom}"
      : "${HF_CHECKPOINT:?HF_CHECKPOINT is required when MODEL_PRESET=custom}"
      MODEL_EXP_PREFIX="${MODEL_EXP_PREFIX:-custom}"
      MODEL_CKPT_DIRNAME="${MODEL_CKPT_DIRNAME:-${MODEL_NAME}_miles}"
      ;;
    *)
      die "Unknown MODEL_PRESET: ${MODEL_PRESET}"
      ;;
  esac
  REF_LOAD="${REF_LOAD:-${HF_CHECKPOINT}_torch_dist}"
}

apply_model_preset
export MODEL_PRESET MODEL_NAME MODEL_SCRIPT MODEL_EXP_PREFIX MODEL_CKPT_DIRNAME HF_CHECKPOINT REF_LOAD
export DEFAULT_LR
export SFT_EXPERIMENT_NAME SFT_EXPERIMENT_ROOT SFT_DATA_DIR OUT_ROOT
export MILES_SRC_DIR SFT_ENABLE_EVAL
export TP_SIZE CP_SIZE SFT_BATCH_SIZE GLOBAL_BATCH_SIZE MAX_TOKENS_PER_GPU LOG_PROBS_CHUNK_SIZE ALLGATHER_CP
export OPTIMIZER_CPU_OFFLOAD OVERLAP_CPU_OPTIMIZER_D2H_H2D
export ENABLE_MTP_TRAINING MTP_NUM_LAYERS MTP_LOSS_SCALING_FACTOR MOE_TOKEN_DISPATCHER_TYPE
export SGLANG_MEM_FRACTION_STATIC SGLANG_MAX_RUNNING_REQUESTS

dataset_path() {
  if [[ -n "${SFT_DATA:-}" ]]; then
    printf '%s\n' "$SFT_DATA"
    return
  fi
  case "$DATASET" in
    glm) printf '%s\n' "${SFT_DATA_DIR}/glm_intersection.parquet" ;;
    kimi) printf '%s\n' "${SFT_DATA_DIR}/kimi_intersection.parquet" ;;
    mixed) printf '%s\n' "${SFT_DATA_DIR}/mixed_intersection.parquet" ;;
    *) die "Unknown DATASET: ${DATASET}. Use glm, kimi, mixed, or set SFT_DATA." ;;
  esac
}

resolve_exp_name() {
  if [[ -n "$EXP_NAME" ]]; then
    printf '%s\n' "$EXP_NAME"
    return
  fi
  local name="${MODEL_EXP_PREFIX}-sft-${DATASET}-${MODE}"
  if [[ -n "$RUN_TAG" ]]; then
    name="${name}-${RUN_TAG}"
  fi
  printf '%s\n' "$name"
}

container_python() {
  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} run --rm \
    --network host \
    -v "${WORK_DIR}:${WORK_DIR}:rw" \
    "$IMAGE" \
    python3 "$@"
}

write_manifest() {
  local exp_name="$1"
  local source_data="$2"
  local train_data="$3"
  local out_folder="$4"
  local manifest="$5"

  /home/chengze/micromamba/envs/dev/bin/python - "$manifest" <<PY
import json, os, pathlib, subprocess, sys, time

manifest_path = pathlib.Path(sys.argv[1])
manifest_path.parent.mkdir(parents=True, exist_ok=True)

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="${ACCRL_DIR}", text=True).strip()
    except Exception:
        return None

def git_info(path):
    repo = pathlib.Path(path)
    if not (repo / ".git").exists():
        return None
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
        return {"path": str(repo), "branch": branch, "sha": sha, "dirty": bool(status.strip())}
    except Exception:
        return {"path": str(repo), "branch": None, "sha": None, "dirty": None}

def sha256_file(path):
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

data = {
    "created_unix": time.time(),
    "experiment_name": "${exp_name}",
    "model_preset": "${MODEL_PRESET}",
    "model_name": "${MODEL_NAME}",
    "model_script": "${MODEL_SCRIPT}",
    "dataset": "${DATASET}",
    "mode": "${MODE}",
    "source_sft_data": "${source_data}",
    "train_sft_data": "${train_data}",
    "sft_eval_turns": "${SFT_EVAL_TURNS}",
    "out_folder": "${out_folder}",
    "master_addr": "${MASTER_ADDR}",
    "git_sha": git_sha(),
    "accrl_git": git_info("${ACCRL_DIR}"),
    "miles_git": git_info("${MILES_SRC_DIR}"),
    "sha256": {
        "source_sft_data": sha256_file("${source_data}"),
        "train_sft_data": sha256_file("${train_data}"),
        "sft_eval_turns": sha256_file("${SFT_EVAL_TURNS}"),
    },
    "env": {
        key: os.environ.get(key)
        for key in [
            "MODEL_PRESET", "MODEL_NAME", "MODEL_SCRIPT", "MODEL_CKPT_DIRNAME",
            "HF_CHECKPOINT", "REF_LOAD", "MILES_SRC_DIR",
            "SFT_EXPERIMENT_NAME", "SFT_EXPERIMENT_ROOT", "SFT_DATA_DIR",
            "OUT_ROOT", "GROUP_RUNS_BY_MODEL",
            "SMOKE_ROWS", "SMOKE_MODE", "SMOKE_SEED",
            "OVERFIT_SOURCE", "OVERFIT_SMOKE_MODE", "OVERFIT_CP_SIZE", "OVERFIT_BATCH_SIZE",
            "DEFAULT_LR", "LR", "WEIGHT_DECAY", "NUM_EPOCH", "CP_SIZE", "TP_SIZE", "SFT_BATCH_SIZE",
            "GLOBAL_BATCH_SIZE",
            "EP_SIZE", "OPTIMIZER_CPU_OFFLOAD", "OVERLAP_CPU_OPTIMIZER_D2H_H2D",
            "USE_PRECISION_AWARE_OPTIMIZER", "ENABLE_MTP_TRAINING", "MTP_NUM_LAYERS",
            "MTP_LOSS_SCALING_FACTOR", "MOE_TOKEN_DISPATCHER_TYPE",
            "LR_DECAY_STYLE", "LR_WARMUP_FRACTION", "MIN_LR",
            "MAX_TOKENS_PER_GPU", "LOG_PROBS_CHUNK_SIZE", "ALLGATHER_CP",
            "RECOMPUTE_LOSS_FUNCTION", "DEBUG_DISABLE_OPTIMIZER", "CLIP_GRAD",
            "SFT_NAN_GRAD_DIAGNOSTICS", "RAY_NUM_CPUS",
            "SKIP_NAN_STEPS", "MILES_HOST_IP", "SAVE_CHECKPOINTS",
            "SAVE_INTERVAL", "NO_SAVE_OPTIM",
            "WANDB_PROJECT", "WANDB_TEAM", "WANDB_ENTITY", "WANDB_GROUP", "WANDB_HOST",
            "WANDB_DIR", "WANDB_RANDOM_SUFFIX",
            "SFT_ENABLE_EVAL", "SFT_EVAL_INTERVAL", "SFT_EVAL_NUM_SAMPLES",
            "SFT_EVAL_SEED", "SFT_EVAL_TURNS", "SFT_EVAL_MAX_RESPONSE_LEN",
            "SFT_EVAL_TEMPERATURE", "SFT_EVAL_TOP_P", "ACTOR_NUM_GPUS_PER_NODE",
            "ROLLOUT_NUM_GPUS", "ROLLOUT_NUM_GPUS_PER_ENGINE",
            "RAY_NUM_GPUS", "ACTOR_CKPT", "SGLANG_MEM_FRACTION_STATIC",
            "SGLANG_EP_SIZE", "SGLANG_MAX_RUNNING_REQUESTS",
            "SGLANG_CUDA_GRAPH_MAX_BS", "SGLANG_ATTENTION_BACKEND",
        ]
        if os.environ.get(key) is not None
    },
}
manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n")
print(manifest_path)
PY
}

write_resolved_env() {
  local exp_name="$1"
  local source_data="$2"
  local train_data="$3"
  local out_folder="$4"
  local manifest_dir="$5"
  local file="${manifest_dir}/resolved_env.txt"

  {
    printf 'EXP_NAME=%q\n' "$exp_name"
    printf 'DATASET=%q\n' "$DATASET"
    printf 'MODE=%q\n' "$MODE"
    printf 'SOURCE_SFT_DATA=%q\n' "$source_data"
    printf 'TRAIN_SFT_DATA=%q\n' "$train_data"
    printf 'OUT_FOLDER=%q\n' "$out_folder"
    printf 'HARNESS_DIR=%q\n' "$manifest_dir"
    printf 'TRAIN_LOG=%q\n' "${TRAIN_LOG:-${manifest_dir}/train.log}"
    printf 'SFT_EXPERIMENT_NAME=%q\n' "$SFT_EXPERIMENT_NAME"
    printf 'SFT_EXPERIMENT_ROOT=%q\n' "$SFT_EXPERIMENT_ROOT"
    printf 'SFT_DATA_DIR=%q\n' "$SFT_DATA_DIR"
    printf 'OUT_ROOT=%q\n' "$OUT_ROOT"
    printf 'GROUP_RUNS_BY_MODEL=%q\n' "${GROUP_RUNS_BY_MODEL:-1}"
    printf 'MASTER_ADDR=%q\n' "$MASTER_ADDR"
    printf 'MODEL_PRESET=%q\n' "$MODEL_PRESET"
    printf 'MODEL_NAME=%q\n' "$MODEL_NAME"
    printf 'MODEL_SCRIPT=%q\n' "$MODEL_SCRIPT"
    printf 'MODEL_CKPT_DIRNAME=%q\n' "$MODEL_CKPT_DIRNAME"
    printf 'SMOKE_ROWS=%q\n' "$SMOKE_ROWS"
    printf 'SMOKE_MODE=%q\n' "$SMOKE_MODE"
    printf 'SMOKE_SEED=%q\n' "$SMOKE_SEED"
    printf 'OVERFIT_SOURCE=%q\n' "${OVERFIT_SOURCE:-}"
    printf 'OVERFIT_SMOKE_MODE=%q\n' "${OVERFIT_SMOKE_MODE:-}"
    printf 'OVERFIT_CP_SIZE=%q\n' "${OVERFIT_CP_SIZE:-}"
    printf 'OVERFIT_BATCH_SIZE=%q\n' "${OVERFIT_BATCH_SIZE:-}"
    printf 'WANDB_PROJECT=%q\n' "${WANDB_PROJECT:-accrl-sft}"
    printf 'WANDB_TEAM=%q\n' "${WANDB_TEAM:-${WANDB_ENTITY:-}}"
    printf 'WANDB_ENTITY=%q\n' "${WANDB_ENTITY:-${WANDB_TEAM:-}}"
    printf 'WANDB_GROUP=%q\n' "${WANDB_GROUP:-${exp_name}}"
    printf 'WANDB_DIR=%q\n' "${WANDB_DIR:-${manifest_dir}/wandb}"
    printf 'WANDB_RANDOM_SUFFIX=%q\n' "${WANDB_RANDOM_SUFFIX:-0}"
    printf 'HF_CHECKPOINT=%q\n' "$HF_CHECKPOINT"
    printf 'REF_LOAD=%q\n' "$REF_LOAD"
    printf 'MILES_SRC_DIR=%q\n' "$MILES_SRC_DIR"
    printf 'ACTOR_CKPT=%q\n' "${ACTOR_CKPT:-${out_folder%/}/${MODEL_CKPT_DIRNAME}}"
    printf 'SAVE_CHECKPOINTS=%q\n' "${SAVE_CHECKPOINTS:-0}"
    printf 'SAVE_INTERVAL=%q\n' "${SAVE_INTERVAL:-50}"
    printf 'NO_SAVE_OPTIM=%q\n' "${NO_SAVE_OPTIM:-0}"
    printf 'LR=%q\n' "${LR:-${DEFAULT_LR}}"
    printf 'WEIGHT_DECAY=%q\n' "${WEIGHT_DECAY:-0.1}"
    printf 'LR_DECAY_STYLE=%q\n' "${LR_DECAY_STYLE:-cosine}"
    printf 'LR_WARMUP_FRACTION=%q\n' "${LR_WARMUP_FRACTION:-0.1}"
    printf 'MIN_LR=%q\n' "${MIN_LR:-1e-7}"
    printf 'NUM_EPOCH=%q\n' "${NUM_EPOCH:-20}"
    printf 'SFT_BATCH_SIZE=%q\n' "${SFT_BATCH_SIZE:-1}"
    printf 'GLOBAL_BATCH_SIZE=%q\n' "${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE:-1}}"
    printf 'TP_SIZE=%q\n' "${TP_SIZE:-1}"
    printf 'CP_SIZE=%q\n' "${CP_SIZE:-1}"
    printf 'EP_SIZE=%q\n' "${EP_SIZE:-}"
    printf 'OPTIMIZER_CPU_OFFLOAD=%q\n' "${OPTIMIZER_CPU_OFFLOAD:-0}"
    printf 'OVERLAP_CPU_OPTIMIZER_D2H_H2D=%q\n' "${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-${OPTIMIZER_CPU_OFFLOAD:-0}}"
    printf 'ENABLE_MTP_TRAINING=%q\n' "${ENABLE_MTP_TRAINING:-0}"
    printf 'MTP_NUM_LAYERS=%q\n' "${MTP_NUM_LAYERS:-1}"
    printf 'MTP_LOSS_SCALING_FACTOR=%q\n' "${MTP_LOSS_SCALING_FACTOR:-0.2}"
    printf 'MOE_TOKEN_DISPATCHER_TYPE=%q\n' "${MOE_TOKEN_DISPATCHER_TYPE:-}"
    printf 'MAX_TOKENS_PER_GPU=%q\n' "${MAX_TOKENS_PER_GPU:-32768}"
    printf 'LOG_PROBS_CHUNK_SIZE=%q\n' "${LOG_PROBS_CHUNK_SIZE:-1024}"
    printf 'ALLGATHER_CP=%q\n' "${ALLGATHER_CP:-1}"
    printf 'RECOMPUTE_LOSS_FUNCTION=%q\n' "${RECOMPUTE_LOSS_FUNCTION:-1}"
    printf 'DEBUG_DISABLE_OPTIMIZER=%q\n' "${DEBUG_DISABLE_OPTIMIZER:-0}"
    printf 'CLIP_GRAD=%q\n' "${CLIP_GRAD:-1.0}"
    printf 'SFT_NAN_GRAD_DIAGNOSTICS=%q\n' "${SFT_NAN_GRAD_DIAGNOSTICS:-0}"
    printf 'RAY_NUM_CPUS=%q\n' "${RAY_NUM_CPUS:-16}"
    printf 'SKIP_NAN_STEPS=%q\n' "${SKIP_NAN_STEPS:-1}"
    printf 'MILES_HOST_IP=%q\n' "${MILES_HOST_IP:-${MASTER_ADDR}}"
    printf 'SFT_ENABLE_EVAL=%q\n' "${SFT_ENABLE_EVAL:-1}"
    printf 'SFT_EVAL_TURNS=%q\n' "${SFT_EVAL_TURNS:-}"
    printf 'SFT_EVAL_INTERVAL=%q\n' "${SFT_EVAL_INTERVAL:-50}"
    printf 'SFT_EVAL_NUM_SAMPLES=%q\n' "${SFT_EVAL_NUM_SAMPLES:-16}"
    printf 'SFT_EVAL_SEED=%q\n' "${SFT_EVAL_SEED:-0}"
    printf 'SFT_EVAL_MAX_RESPONSE_LEN=%q\n' "${SFT_EVAL_MAX_RESPONSE_LEN:-8192}"
    printf 'SFT_EVAL_TEMPERATURE=%q\n' "${SFT_EVAL_TEMPERATURE:-0}"
    printf 'SFT_EVAL_TOP_P=%q\n' "${SFT_EVAL_TOP_P:-1.0}"
    printf 'ACTOR_NUM_GPUS_PER_NODE=%q\n' "${ACTOR_NUM_GPUS_PER_NODE:-}"
    printf 'ROLLOUT_NUM_GPUS=%q\n' "${ROLLOUT_NUM_GPUS:-}"
    printf 'ROLLOUT_NUM_GPUS_PER_ENGINE=%q\n' "${ROLLOUT_NUM_GPUS_PER_ENGINE:-}"
    printf 'RAY_NUM_GPUS=%q\n' "${RAY_NUM_GPUS:-}"
    printf 'SGLANG_MEM_FRACTION_STATIC=%q\n' "${SGLANG_MEM_FRACTION_STATIC:-}"
    printf 'SGLANG_EP_SIZE=%q\n' "${SGLANG_EP_SIZE:-}"
    printf 'SGLANG_MAX_RUNNING_REQUESTS=%q\n' "${SGLANG_MAX_RUNNING_REQUESTS:-}"
    printf 'SGLANG_CUDA_GRAPH_MAX_BS=%q\n' "${SGLANG_CUDA_GRAPH_MAX_BS:-}"
    printf 'SGLANG_ATTENTION_BACKEND=%q\n' "${SGLANG_ATTENTION_BACKEND:-}"
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      printf 'WANDB_API_KEY=<set; omitted>\n'
    fi
  } > "$file"
}

write_run_command() {
  local exp_name="$1"
  local train_data="$2"
  local out_folder="$3"
  local manifest_dir="$4"
  local file="${manifest_dir}/run_command.sh"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n\n'
    printf 'cd %q\n\n' "$ACCRL_DIR"
    printf 'DATASET=%q \\\n' "$DATASET"
    printf 'MODE=%q \\\n' "$MODE"
    printf 'EXP_NAME=%q \\\n' "$exp_name"
    printf 'MODEL_PRESET=%q \\\n' "$MODEL_PRESET"
    printf 'MODEL_NAME=%q \\\n' "$MODEL_NAME"
    printf 'MODEL_SCRIPT=%q \\\n' "$MODEL_SCRIPT"
    printf 'MODEL_CKPT_DIRNAME=%q \\\n' "$MODEL_CKPT_DIRNAME"
    printf 'SMOKE_ROWS=%q \\\n' "$SMOKE_ROWS"
    printf 'SMOKE_MODE=%q \\\n' "$SMOKE_MODE"
    printf 'SMOKE_SEED=%q \\\n' "$SMOKE_SEED"
    printf 'OVERFIT_SOURCE=%q \\\n' "${OVERFIT_SOURCE:-}"
    printf 'OVERFIT_SMOKE_MODE=%q \\\n' "${OVERFIT_SMOKE_MODE:-}"
    printf 'OVERFIT_CP_SIZE=%q \\\n' "${OVERFIT_CP_SIZE:-}"
    printf 'OVERFIT_BATCH_SIZE=%q \\\n' "${OVERFIT_BATCH_SIZE:-}"
    printf 'SFT_EXPERIMENT_NAME=%q \\\n' "$SFT_EXPERIMENT_NAME"
    printf 'SFT_EXPERIMENT_ROOT=%q \\\n' "$SFT_EXPERIMENT_ROOT"
    printf 'SFT_DATA_DIR=%q \\\n' "$SFT_DATA_DIR"
    printf 'SFT_DATA=%q \\\n' "$train_data"
    printf 'OUT_ROOT=%q \\\n' "$OUT_ROOT"
    printf 'GROUP_RUNS_BY_MODEL=%q \\\n' "${GROUP_RUNS_BY_MODEL:-1}"
    printf 'MASTER_ADDR=%q \\\n' "$MASTER_ADDR"
    printf 'WANDB_PROJECT=%q \\\n' "${WANDB_PROJECT:-accrl-sft}"
    printf 'WANDB_TEAM=%q \\\n' "${WANDB_TEAM:-${WANDB_ENTITY:-}}"
    printf 'WANDB_ENTITY=%q \\\n' "${WANDB_ENTITY:-${WANDB_TEAM:-}}"
    printf 'WANDB_GROUP=%q \\\n' "${WANDB_GROUP:-${exp_name}}"
    printf 'WANDB_DIR=%q \\\n' "${WANDB_DIR:-${manifest_dir}/wandb}"
    printf 'WANDB_RANDOM_SUFFIX=%q \\\n' "${WANDB_RANDOM_SUFFIX:-0}"
    printf 'HF_CHECKPOINT=%q \\\n' "$HF_CHECKPOINT"
    printf 'REF_LOAD=%q \\\n' "$REF_LOAD"
    printf 'MILES_SRC_DIR=%q \\\n' "$MILES_SRC_DIR"
    printf 'ACTOR_CKPT=%q \\\n' "${ACTOR_CKPT:-${out_folder%/}/${MODEL_CKPT_DIRNAME}}"
    printf 'SAVE_CHECKPOINTS=%q \\\n' "${SAVE_CHECKPOINTS:-0}"
    printf 'SAVE_INTERVAL=%q \\\n' "${SAVE_INTERVAL:-50}"
    printf 'NO_SAVE_OPTIM=%q \\\n' "${NO_SAVE_OPTIM:-0}"
    printf 'LR=%q \\\n' "${LR:-${DEFAULT_LR}}"
    printf 'WEIGHT_DECAY=%q \\\n' "${WEIGHT_DECAY:-0.1}"
    printf 'LR_DECAY_STYLE=%q \\\n' "${LR_DECAY_STYLE:-cosine}"
    printf 'LR_WARMUP_FRACTION=%q \\\n' "${LR_WARMUP_FRACTION:-0.1}"
    printf 'MIN_LR=%q \\\n' "${MIN_LR:-1e-7}"
    printf 'NUM_EPOCH=%q \\\n' "${NUM_EPOCH:-20}"
    printf 'SFT_BATCH_SIZE=%q \\\n' "${SFT_BATCH_SIZE:-1}"
    printf 'GLOBAL_BATCH_SIZE=%q \\\n' "${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE:-1}}"
    printf 'TP_SIZE=%q \\\n' "${TP_SIZE:-1}"
    printf 'CP_SIZE=%q \\\n' "${CP_SIZE:-1}"
    printf 'EP_SIZE=%q \\\n' "${EP_SIZE:-}"
    printf 'OPTIMIZER_CPU_OFFLOAD=%q \\\n' "${OPTIMIZER_CPU_OFFLOAD:-0}"
    printf 'OVERLAP_CPU_OPTIMIZER_D2H_H2D=%q \\\n' "${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-${OPTIMIZER_CPU_OFFLOAD:-0}}"
    printf 'ENABLE_MTP_TRAINING=%q \\\n' "${ENABLE_MTP_TRAINING:-0}"
    printf 'MTP_NUM_LAYERS=%q \\\n' "${MTP_NUM_LAYERS:-1}"
    printf 'MTP_LOSS_SCALING_FACTOR=%q \\\n' "${MTP_LOSS_SCALING_FACTOR:-0.2}"
    printf 'MOE_TOKEN_DISPATCHER_TYPE=%q \\\n' "${MOE_TOKEN_DISPATCHER_TYPE:-}"
    printf 'MAX_TOKENS_PER_GPU=%q \\\n' "${MAX_TOKENS_PER_GPU:-32768}"
    printf 'LOG_PROBS_CHUNK_SIZE=%q \\\n' "${LOG_PROBS_CHUNK_SIZE:-1024}"
    printf 'ALLGATHER_CP=%q \\\n' "${ALLGATHER_CP:-1}"
    printf 'RECOMPUTE_LOSS_FUNCTION=%q \\\n' "${RECOMPUTE_LOSS_FUNCTION:-1}"
    printf 'DEBUG_DISABLE_OPTIMIZER=%q \\\n' "${DEBUG_DISABLE_OPTIMIZER:-0}"
    printf 'CLIP_GRAD=%q \\\n' "${CLIP_GRAD:-1.0}"
    printf 'SFT_NAN_GRAD_DIAGNOSTICS=%q \\\n' "${SFT_NAN_GRAD_DIAGNOSTICS:-0}"
    printf 'RAY_NUM_CPUS=%q \\\n' "${RAY_NUM_CPUS:-16}"
    printf 'SKIP_NAN_STEPS=%q \\\n' "${SKIP_NAN_STEPS:-1}"
    printf 'MILES_HOST_IP=%q \\\n' "${MILES_HOST_IP:-${MASTER_ADDR}}"
    printf 'SFT_ENABLE_EVAL=%q \\\n' "${SFT_ENABLE_EVAL:-1}"
    printf 'SFT_EVAL_TURNS=%q \\\n' "${SFT_EVAL_TURNS:-}"
    printf 'SFT_EVAL_INTERVAL=%q \\\n' "${SFT_EVAL_INTERVAL:-50}"
    printf 'SFT_EVAL_NUM_SAMPLES=%q \\\n' "${SFT_EVAL_NUM_SAMPLES:-16}"
    printf 'SFT_EVAL_SEED=%q \\\n' "${SFT_EVAL_SEED:-0}"
    printf 'SFT_EVAL_MAX_RESPONSE_LEN=%q \\\n' "${SFT_EVAL_MAX_RESPONSE_LEN:-8192}"
    printf 'SFT_EVAL_TEMPERATURE=%q \\\n' "${SFT_EVAL_TEMPERATURE:-0}"
    printf 'SFT_EVAL_TOP_P=%q \\\n' "${SFT_EVAL_TOP_P:-1.0}"
    printf 'ACTOR_NUM_GPUS_PER_NODE=%q \\\n' "${ACTOR_NUM_GPUS_PER_NODE:-}"
    printf 'ROLLOUT_NUM_GPUS=%q \\\n' "${ROLLOUT_NUM_GPUS:-}"
    printf 'ROLLOUT_NUM_GPUS_PER_ENGINE=%q \\\n' "${ROLLOUT_NUM_GPUS_PER_ENGINE:-}"
    printf 'RAY_NUM_GPUS=%q \\\n' "${RAY_NUM_GPUS:-}"
    printf 'SGLANG_MEM_FRACTION_STATIC=%q \\\n' "${SGLANG_MEM_FRACTION_STATIC:-}"
    printf 'SGLANG_EP_SIZE=%q \\\n' "${SGLANG_EP_SIZE:-}"
    printf 'SGLANG_MAX_RUNNING_REQUESTS=%q \\\n' "${SGLANG_MAX_RUNNING_REQUESTS:-}"
    printf 'SGLANG_CUDA_GRAPH_MAX_BS=%q \\\n' "${SGLANG_CUDA_GRAPH_MAX_BS:-}"
    printf 'SGLANG_ATTENTION_BACKEND=%q \\\n' "${SGLANG_ATTENTION_BACKEND:-}"
    printf 'OUT_FOLDER=%q \\\n' "$out_folder"
    printf '  %q run\n' "${ACCRL_DIR}/accrl/distill/sft/scripts/qwen3.5-9b_sft.sh"
  } > "$file"
  chmod +x "$file"
}

resolve() {
  local exp_name source_data out_folder train_data manifest_dir
  exp_name="$(resolve_exp_name)"
  source_data="$(dataset_path)"
  [[ -f "$source_data" ]] || die "SFT data not found: $source_data"
  local out_root="${OUT_ROOT%/}"
  if [[ "${GROUP_RUNS_BY_MODEL:-1}" = "1" && "$(basename "$out_root")" != "$MODEL_EXP_PREFIX" ]]; then
    out_root="${out_root}/${MODEL_EXP_PREFIX}"
  fi
  out_folder="${out_root}/${exp_name}"
  manifest_dir="${out_folder}/experiment_harness"
  train_data="$source_data"
  if [[ "$MODE" = "smoke" ]]; then
    train_data="${manifest_dir}/smoke_${DATASET}_${SMOKE_ROWS}.parquet"
  elif [[ "$MODE" != "full" ]]; then
    die "Unknown MODE: ${MODE}. Use full or smoke."
  fi

  printf '%s\n%s\n%s\n%s\n%s\n' "$exp_name" "$source_data" "$train_data" "$out_folder" "$manifest_dir"
}

cmd_print() {
  mapfile -t resolved < <(resolve)
  cat <<EOF
EXP_NAME=${resolved[0]}
SOURCE_SFT_DATA=${resolved[1]}
TRAIN_SFT_DATA=${resolved[2]}
OUT_FOLDER=${resolved[3]}
HARNESS_DIR=${resolved[4]}
TRAIN_LOG=${TRAIN_LOG:-${resolved[4]}/train.log}
MODE=${MODE}
DATASET=${DATASET}
MODEL_PRESET=${MODEL_PRESET}
MODEL_NAME=${MODEL_NAME}
MODEL_SCRIPT=${MODEL_SCRIPT}
HF_CHECKPOINT=${HF_CHECKPOINT}
REF_LOAD=${REF_LOAD}
SFT_EVAL_TURNS=${SFT_EVAL_TURNS}
SFT_EXPERIMENT_NAME=${SFT_EXPERIMENT_NAME}
SFT_EXPERIMENT_ROOT=${SFT_EXPERIMENT_ROOT}
SFT_DATA_DIR=${SFT_DATA_DIR}
OUT_ROOT=${OUT_ROOT}
EOF
}

cmd_prepare() {
  mapfile -t resolved < <(resolve)
  local exp_name="${resolved[0]}"
  local source_data="${resolved[1]}"
  local train_data="${resolved[2]}"
  local out_folder="${resolved[3]}"
  local manifest_dir="${resolved[4]}"

  mkdir -p "$manifest_dir"
  if [[ "$MODE" = "smoke" ]]; then
    container_python "${ACCRL_DIR}/accrl/distill/sft/make_sft_subset.py" \
      --input "$source_data" \
      --output "$train_data" \
      --rows "$SMOKE_ROWS" \
      --mode "$SMOKE_MODE" \
      --seed "$SMOKE_SEED" \
      --manifest "${manifest_dir}/smoke_subset_manifest.json"
  fi

  write_manifest "$exp_name" "$source_data" "$train_data" "$out_folder" "${manifest_dir}/experiment_manifest.json"
  write_resolved_env "$exp_name" "$source_data" "$train_data" "$out_folder" "$manifest_dir"
  write_run_command "$exp_name" "$train_data" "$out_folder" "$manifest_dir"
  cmd_print
}

cmd_run() {
  mapfile -t resolved < <(resolve)
  local exp_name="${resolved[0]}"
  local train_data="${resolved[2]}"
  local out_folder="${resolved[3]}"
  local manifest_dir="${resolved[4]}"
  local log_file="${TRAIN_LOG:-${manifest_dir}/train.log}"

  mkdir -p "$manifest_dir"
  : > "$log_file"

  {
    cmd_prepare
    MODEL_PRESET="$MODEL_PRESET" \
    MODEL_NAME="$MODEL_NAME" \
    MODEL_SCRIPT="$MODEL_SCRIPT" \
    MODEL_CKPT_DIRNAME="$MODEL_CKPT_DIRNAME" \
    HF_CHECKPOINT="$HF_CHECKPOINT" \
    REF_LOAD="$REF_LOAD" \
    MILES_SRC_DIR="$MILES_SRC_DIR" \
    ACTOR_CKPT="${ACTOR_CKPT:-${out_folder%/}/${MODEL_CKPT_DIRNAME}}" \
    SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}" \
    SAVE_INTERVAL="${SAVE_INTERVAL:-50}" \
    NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}" \
    LR="${LR:-${DEFAULT_LR}}" \
    LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}" \
    LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.1}" \
    MIN_LR="${MIN_LR:-1e-7}" \
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}" \
    TP_SIZE="${TP_SIZE:-1}" \
    CP_SIZE="${CP_SIZE:-1}" \
    SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE:-1}}" \
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}" \
    LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}" \
    ALLGATHER_CP="${ALLGATHER_CP:-1}" \
    RECOMPUTE_LOSS_FUNCTION="${RECOMPUTE_LOSS_FUNCTION:-1}" \
    DEBUG_DISABLE_OPTIMIZER="${DEBUG_DISABLE_OPTIMIZER:-0}" \
    CLIP_GRAD="${CLIP_GRAD:-1.0}" \
    SFT_NAN_GRAD_DIAGNOSTICS="${SFT_NAN_GRAD_DIAGNOSTICS:-0}" \
    RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}" \
    SKIP_NAN_STEPS="${SKIP_NAN_STEPS:-1}" \
    MILES_HOST_IP="${MILES_HOST_IP:-${MASTER_ADDR}}" \
    EP_SIZE="${EP_SIZE:-}" \
    OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-0}" \
    OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-${OPTIMIZER_CPU_OFFLOAD:-0}}" \
    ENABLE_MTP_TRAINING="${ENABLE_MTP_TRAINING:-0}" \
    MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}" \
    MTP_LOSS_SCALING_FACTOR="${MTP_LOSS_SCALING_FACTOR:-0.2}" \
    MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-}" \
    SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-}" \
    SGLANG_EP_SIZE="${SGLANG_EP_SIZE:-}" \
    SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}" \
    SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-}" \
    SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-}" \
    WANDB_HOST="${WANDB_HOST:-https://meta.wandb.io}" \
    WANDB_PROJECT="${WANDB_PROJECT:-accrl-sft}" \
    WANDB_TEAM="${WANDB_TEAM:-${WANDB_ENTITY:-}}" \
    WANDB_ENTITY="${WANDB_ENTITY:-${WANDB_TEAM:-}}" \
    WANDB_GROUP="${WANDB_GROUP:-${exp_name}}" \
    WANDB_DIR="${WANDB_DIR:-${manifest_dir}/wandb}" \
    WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}" \
    SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-1}" \
    SFT_EVAL_TURNS="${SFT_EVAL_TURNS:-}" \
    SFT_EVAL_INTERVAL="${SFT_EVAL_INTERVAL:-50}" \
    SFT_EVAL_NUM_SAMPLES="${SFT_EVAL_NUM_SAMPLES:-16}" \
    SFT_EVAL_SEED="${SFT_EVAL_SEED:-0}" \
    SFT_EVAL_MAX_RESPONSE_LEN="${SFT_EVAL_MAX_RESPONSE_LEN:-8192}" \
    SFT_EVAL_TEMPERATURE="${SFT_EVAL_TEMPERATURE:-0}" \
    SFT_EVAL_TOP_P="${SFT_EVAL_TOP_P:-1.0}" \
    ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-}" \
    ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-}" \
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-}" \
    RAY_NUM_GPUS="${RAY_NUM_GPUS:-}" \
    SFT_DATA="$train_data" \
    OUT_FOLDER="$out_folder" \
    MASTER_ADDR="$MASTER_ADDR" \
      "${ACCRL_DIR}/accrl/distill/sft/scripts/qwen3.5-9b_sft.sh" run
  } 2>&1 | tee "$log_file"
}

cmd_overfit() {
  local timestamp
  timestamp="$(date +%Y%m%d-%H%M%S)"
  local overfit_source="${OVERFIT_SOURCE:-}"
  if [[ -z "$overfit_source" ]]; then
    if [[ "$MODEL_PRESET" = "qwen35-35b-a3b" ]]; then
      overfit_source="synthetic"
    else
      overfit_source="dataset"
    fi
  fi
  export OVERFIT_SOURCE="$overfit_source"

  export MODE="${OVERFIT_MODE:-smoke}"
  if [[ -n "${OVERFIT_ROWS:-}" ]]; then
    export SMOKE_ROWS="${OVERFIT_ROWS}"
  elif [[ "$MODEL_PRESET" = "qwen35-35b-a3b" && "$overfit_source" = "synthetic" ]]; then
    export SMOKE_ROWS=8
  elif [[ "$MODEL_PRESET" = "qwen35-35b-a3b" ]]; then
    export SMOKE_ROWS=2
  else
    export SMOKE_ROWS=4
  fi
  case "$overfit_source" in
    synthetic) export SMOKE_MODE="${OVERFIT_SMOKE_MODE:-synthetic}" ;;
    dataset) export SMOKE_MODE="${OVERFIT_SMOKE_MODE:-${SMOKE_MODE:-shortest}}" ;;
    *) die "Unknown OVERFIT_SOURCE: ${overfit_source}. Use synthetic or dataset." ;;
  esac
  if [[ "$MODEL_PRESET" = "qwen35-35b-a3b" && "$overfit_source" = "synthetic" ]]; then
    export NUM_EPOCH="${NUM_EPOCH:-20}"
    export LR="${LR:-1e-5}"
  elif [[ "$MODEL_PRESET" = "qwen35-35b-a3b" ]]; then
    export NUM_EPOCH="${NUM_EPOCH:-30}"
    export LR="${LR:-3e-6}"
  else
    export NUM_EPOCH="${NUM_EPOCH:-80}"
    export LR="${LR:-1e-5}"
  fi
  export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
  export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
  export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0}"
  export MIN_LR="${MIN_LR:-0}"
  export TP_SIZE="${TP_SIZE:-1}"
  if [[ "$MODEL_PRESET" = "qwen35-35b-a3b" ]]; then
    if [[ "$overfit_source" = "synthetic" ]]; then
      # Tiny sequences should not use context parallelism: CP can leave ranks
      # with no supervised tokens and produce NaN grads before the sanity check
      # tells us anything useful.
      export CP_SIZE="${OVERFIT_CP_SIZE:-1}"
      export SFT_BATCH_SIZE="${OVERFIT_BATCH_SIZE:-${SMOKE_ROWS}}"
    else
      export CP_SIZE="${CP_SIZE:-4}"
      export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    fi
    export EP_SIZE="${EP_SIZE:-8}"
    export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
    export RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
  else
    export CP_SIZE="${CP_SIZE:-4}"
    export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
    export RAY_NUM_GPUS="${RAY_NUM_GPUS:-4}"
    export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
  fi
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${SFT_BATCH_SIZE}}"
  export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
  export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
  export SKIP_NAN_STEPS="${SKIP_NAN_STEPS:-0}"
  export SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-0}"
  export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
  export WANDB_PROJECT="${WANDB_PROJECT:-accrl-sft}"
  export WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"
  if [[ -z "$RUN_TAG" ]]; then
    export RUN_TAG="overfit-r${SMOKE_ROWS}-lr${LR}-nowarmup-${timestamp}"
  fi

  echo "SFT overfit sanity: source=${overfit_source} dataset=${DATASET} rows=${SMOKE_ROWS} epochs=${NUM_EPOCH} lr=${LR} warmup=${LR_WARMUP_FRACTION}"
  echo "Eval disabled by default for overfit; this run should drive train/loss down on the fixed smoke subset."
  cmd_run
}

cmd_suite() {
  local datasets="${DATASETS:-glm kimi mixed}"
  local base_tag="${BASE_TAG:-thinkwrap-e30-cp4-b1-lr1e-6-$(date +%Y%m%d-%H%M%S)}"
  local script_dir
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

  export MODE="${MODE:-full}"
  export NUM_EPOCH="${NUM_EPOCH:-30}"
  export LR="${LR:-1e-6}"
  export TP_SIZE="${TP_SIZE:-1}"
  if [[ "$MODEL_PRESET" = "qwen35-35b-a3b" ]]; then
    export CP_SIZE="${CP_SIZE:-4}"
    export EP_SIZE="${EP_SIZE:-8}"
    export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
    export SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-0}"
  else
    export CP_SIZE="${CP_SIZE:-4}"
    export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
  fi
  export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
  export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
  export SKIP_NAN_STEPS="${SKIP_NAN_STEPS:-0}"
  export SFT_ENABLE_EVAL="${SFT_ENABLE_EVAL:-1}"
  export WANDB_PROJECT="${WANDB_PROJECT:-accrl-sft}"
  export WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"
  export SFT_WANDB_TOKEN_TABLE_EVERY="${SFT_WANDB_TOKEN_TABLE_EVERY:-50}"
  export SFT_WANDB_TOKEN_TABLE_RADIUS="${SFT_WANDB_TOKEN_TABLE_RADIUS:-24}"

  echo "SFT suite base tag: ${base_tag}"
  echo "Datasets: ${datasets}"
  echo "Epochs: ${NUM_EPOCH}"
  echo "TP=${TP_SIZE} CP=${CP_SIZE} batch=${SFT_BATCH_SIZE} max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"

  local dataset
  for dataset in ${datasets}; do
    echo
    echo "===== starting dataset=${dataset} ====="
    DATASET="${dataset}" RUN_TAG="${base_tag}" "${script_dir}/run_long_sft_mixed.sh" run
    echo "===== completed dataset=${dataset} ====="
  done
}

case "${1:-print}" in
  print) cmd_print ;;
  prepare) cmd_prepare ;;
  run) cmd_run ;;
  overfit) cmd_overfit ;;
  suite) cmd_suite ;;
  -h|--help|help) usage ;;
  *) die "Unknown command: $1" ;;
esac
