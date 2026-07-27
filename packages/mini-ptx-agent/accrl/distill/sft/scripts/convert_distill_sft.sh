#!/usr/bin/env bash
# Convert AccRL distill reasoning pairs to Miles SFT data.

set -euo pipefail

IMAGE="${IMAGE:-docker.io/radixark/miles:latest}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-sudo -n podman}"
WORK_DIR="${WORK_DIR:-/home/chengze/work}"
ACCRL_DIR="${ACCRL_DIR:-${WORK_DIR}/AccRL}"
EXPS_DIR="${EXPS_DIR:-${WORK_DIR}/AccRL-exps}"

PAIRS="${PAIRS:-${EXPS_DIR}/distill/experiments/trajectory_reasoning_glm_v3/reasoning_pairs.jsonl}"
SFT_DATA="${SFT_DATA:-${EXPS_DIR}/distill/glm_v3_sft.parquet}"
REASONING_FIELD="${REASONING_FIELD:-reasoning}"
SFT_STYLE="${SFT_STYLE:-reverse_cot}"

usage() {
  cat <<'EOF'
Usage:
  accrl/distill/sft/scripts/convert_distill_sft.sh [convert]

Converts reasoning_pairs.jsonl to Miles SFT parquet/jsonl.

Env:
  IMAGE             Default: docker.io/radixark/miles:latest
  WORK_DIR          Mounted host work dir. Default: /home/chengze/work
  ACCRL_DIR         AccRL repo path. Default: $WORK_DIR/AccRL
  EXPS_DIR          AccRL experiments path. Default: $WORK_DIR/AccRL-exps
  PAIRS             Input reasoning_pairs.jsonl
  SFT_DATA          Output .parquet or .jsonl
  REASONING_FIELD   reasoning or thinking. Default: reasoning
  SFT_STYLE         reverse_cot or kernel_sft. Default: reverse_cot

reverse_cot matches the parquet Chengze's friend used:
  columns: id, messages
  assistant content: reasoning only
  user content: full distill prompt with current kernel
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

container_exec() {
  # shellcheck disable=SC2086
  ${CONTAINER_ENGINE} run --rm \
    --network host \
    --ipc host \
    --gpus all \
    -v "${WORK_DIR}:${WORK_DIR}:rw" \
    "$IMAGE" "$@"
}

cmd_convert() {
  [[ -f "$PAIRS" ]] || die "Pairs file not found: $PAIRS"
  container_exec bash -lc "
    set -euo pipefail
    cd '$ACCRL_DIR'
    python3 -m accrl.distill.sft.export_sft \
      --pairs '$PAIRS' \
      --output '$SFT_DATA' \
      --reasoning-field '$REASONING_FIELD' \
      --style '$SFT_STYLE'
  "
}

case "${1:-convert}" in
  convert) cmd_convert ;;
  -h|--help|help) usage ;;
  *) die "Unknown command: $1" ;;
esac
