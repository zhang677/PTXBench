#!/usr/bin/env bash
# Build controlled GLM/Kimi intersection SFT datasets.

set -euo pipefail

IMAGE="${IMAGE:-docker.io/radixark/miles:latest}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-sudo -n podman}"
WORK_DIR="${WORK_DIR:-/home/chengze/work}"
DATA_DIR="${DATA_DIR:-/data/local}"
ACCRL_DIR="${ACCRL_DIR:-${WORK_DIR}/AccRL}"
EXPS_DIR="${EXPS_DIR:-${WORK_DIR}/AccRL-exps}"

BUILDER="${BUILDER:-v2}" # v2 or legacy
GLM_DIR="${GLM_DIR:-${EXPS_DIR}/distill/experiments/trajectory_reasoning_glm5_1_6a3ddbm_thinking_0403}"
KIMI_DIR="${KIMI_DIR:-${EXPS_DIR}/distill/experiments/trajectory_reasoning_kimi2.6——1228}"
GEMINI_TURNS="${GEMINI_TURNS:-${EXPS_DIR}/distill/gemini_turns_0422.jsonl}"
if [[ -z "${OUT_DIR:-}" ]]; then
  case "$BUILDER" in
    v2) OUT_DIR="${EXPS_DIR}/sft_experiments/glm_kimi_gemini_v2/data" ;;
    legacy) OUT_DIR="${EXPS_DIR}/sft_data/glm_kimi_intersection" ;;
    *) OUT_DIR="${EXPS_DIR}/sft_experiments/glm_kimi_gemini_v2/data" ;;
  esac
fi
TOKENIZER="${TOKENIZER:-/data/local/models/qwen3.5_9B}"
REASONING_FIELD="${REASONING_FIELD:-reasoning}"
MIN_REASONING_CHARS="${MIN_REASONING_CHARS:-200}"
WRITE_JSONL="${WRITE_JSONL:-0}"

usage() {
  cat <<'EOF'
Usage:
  accrl/distill/sft/scripts/build_intersection_sft.sh [build]

Builds three controlled SFT datasets from GLM/Kimi distill experiment overlap.

Default BUILDER=v2 builds next-turn Gemini imitation data:
  context = original Gemini multi-turn history up to turn N
  target  = <think>GLM/Kimi reasoning_N</think> + Gemini raw assistant response_N

The legacy builder is still available with BUILDER=legacy and trains think-only
targets:
  <think>
  row[REASONING_FIELD].strip()
  </think>

  glm_intersection.parquet
  kimi_intersection.parquet
  mixed_intersection.parquet

Also writes:
  manifest.json
  length_stats.json

Env:
  IMAGE                 Default: docker.io/radixark/miles:latest
  WORK_DIR              Default: /home/chengze/work
  DATA_DIR              Default: /data/local
  ACCRL_DIR             Default: $WORK_DIR/AccRL
  EXPS_DIR              Default: $WORK_DIR/AccRL-exps
  BUILDER               v2 or legacy. Default: v2
  GLM_DIR               GLM experiment dir
  KIMI_DIR              Kimi experiment dir
  GEMINI_TURNS          Required for BUILDER=v2. Default: $EXPS_DIR/distill/gemini_turns_0422.jsonl
  OUT_DIR               Default for v2: $EXPS_DIR/sft_experiments/glm_kimi_gemini_v2/data
                        Default for legacy: $EXPS_DIR/sft_data/glm_kimi_intersection
  TOKENIZER             Default: /data/local/models/qwen3.5_9B
  REASONING_FIELD       Source field wrapped into <think>...</think>. Default: reasoning
  MIN_REASONING_CHARS   Default: 200
  WRITE_JSONL           Legacy only: also write jsonl files when 1. Default: 0
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

cmd_build() {
  [[ -f "${GLM_DIR}/reasoning_pairs.jsonl" ]] || die "missing GLM reasoning_pairs.jsonl: ${GLM_DIR}"
  [[ -f "${KIMI_DIR}/reasoning_pairs.jsonl" ]] || die "missing Kimi reasoning_pairs.jsonl: ${KIMI_DIR}"
  [[ "$BUILDER" = "legacy" || "$BUILDER" = "v2" ]] || die "BUILDER must be v2 or legacy, got: ${BUILDER}"
  if [[ "$BUILDER" = "v2" ]]; then
    [[ -f "${GEMINI_TURNS}" ]] || die "missing GEMINI_TURNS: ${GEMINI_TURNS}"
  fi

  local jsonl_arg=()
  if [[ "$WRITE_JSONL" = "1" ]]; then
    jsonl_arg=(--write-jsonl)
  fi

  local build_cmd
  if [[ "$BUILDER" = "v2" ]]; then
    build_cmd="
      python3 accrl/distill/sft/build_sft_dataset_v2.py \
        --gemini-turns '$GEMINI_TURNS' \
        --glm-dir '$GLM_DIR' \
        --kimi-dir '$KIMI_DIR' \
        --output-dir '$OUT_DIR' \
        --tokenizer '$TOKENIZER' \
        --reasoning-field '$REASONING_FIELD' \
        --min-reasoning-chars '$MIN_REASONING_CHARS'
    "
  else
    build_cmd="
      python3 -m accrl.distill.sft.build_sft_dataset \
        --glm-dir '$GLM_DIR' \
        --kimi-dir '$KIMI_DIR' \
        --output-dir '$OUT_DIR' \
        --tokenizer '$TOKENIZER' \
        --reasoning-field '$REASONING_FIELD' \
        --min-reasoning-chars '$MIN_REASONING_CHARS' \
        ${jsonl_arg[*]}
    "
  fi

  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} run --rm \
    --network host \
    --ipc host \
    --gpus all \
    --pids-limit=-1 \
    -v "${WORK_DIR}:${WORK_DIR}:rw" \
    -v "${DATA_DIR}:${DATA_DIR}:ro" \
    "$IMAGE" \
    bash -lc "
      set -euo pipefail
      cd '$ACCRL_DIR'
      $build_cmd
    "
}

case "${1:-build}" in
  build) cmd_build ;;
  -h|--help|help) usage ;;
  *) die "Unknown command: $1" ;;
esac
