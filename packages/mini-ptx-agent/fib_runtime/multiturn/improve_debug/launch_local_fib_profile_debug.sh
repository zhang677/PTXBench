#!/usr/bin/env bash
set -euo pipefail

# Start fib-profile for debug iteration with the local flashinfer-bench checkout
# mounted over /workspace/flashinfer-bench-private and exactly one visible GPU.
#
# This launches a long-lived container and starts one direct flashinfer-bench serve
# backend inside it; no dispatcher is used.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./launch_local_fib_profile_debug.sh

Environment overrides:
  CONTAINER=fib-profile
  IMAGE=fib-profile:latest
  HOST_FLASHINFER_BENCH=/home/ubuntu/flashinfer-bench
  HOST_ACCRL_TRAINING=/home/ubuntu/accrl-training
  GPU_ID=0
  PROFILE_PORT=10000
  FIB_DEVICE=cuda:0
  TVM_FFI_CUDA_ARCH_LIST=9.0a
  FIB_TIMEOUT=120
EOF
  exit 0
fi

CONTAINER="${CONTAINER:-fib-profile}"
IMAGE="${IMAGE:-fib-profile:latest}"
HOST_FLASHINFER_BENCH="${HOST_FLASHINFER_BENCH:-/home/ubuntu/flashinfer-bench}"
HOST_ACCRL_TRAINING="${HOST_ACCRL_TRAINING:-/home/ubuntu/accrl-training}"
GPU_ID="${GPU_ID:-0}"
PROFILE_PORT="${PROFILE_PORT:-10000}"
FIB_DEVICE="${FIB_DEVICE:-cuda:0}"
TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-9.0a}"
FIB_TIMEOUT="${FIB_TIMEOUT:-30}"

if [[ ! -d "${HOST_FLASHINFER_BENCH}" ]]; then
  echo "missing HOST_FLASHINFER_BENCH: ${HOST_FLASHINFER_BENCH}" >&2
  exit 1
fi
if [[ ! -d "${HOST_ACCRL_TRAINING}" ]]; then
  echo "missing HOST_ACCRL_TRAINING: ${HOST_ACCRL_TRAINING}" >&2
  exit 1
fi

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker run -d --init --name "${CONTAINER}" \
  --gpus "device=${GPU_ID}" \
  -v "${HOST_FLASHINFER_BENCH}:/workspace/flashinfer-bench-private" \
  -v "${HOST_ACCRL_TRAINING}:/workspace/accrl-training" \
  -e FIB_SKIP_DATASET_DOWNLOAD=1 \
  -e TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST}" \
  -e FIB_DEVICE="${FIB_DEVICE}" \
  -e PROFILE_PORT="${PROFILE_PORT}" \
  -e FIB_TIMEOUT="${FIB_TIMEOUT}" \
  -p "${PROFILE_PORT}:${PROFILE_PORT}" \
  "${IMAGE}" bash -lc 'sleep infinity'

docker exec "${CONTAINER}" bash -lc '
  set -euo pipefail
  VENV_DIR="${VENV_DIR:-/workspace/acc}"
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    if command -v uv >/dev/null 2>&1; then
      uv venv "${VENV_DIR}"
    else
      python3 -m venv "${VENV_DIR}"
    fi
  fi
  source "${VENV_DIR}/bin/activate"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e "/workspace/flashinfer-bench-private[serve]"
  else
    python -m pip install -e "/workspace/flashinfer-bench-private[serve]"
  fi
  export BENCH_ROOT=/workspace/flashinfer-bench-private
  export DATASET_ROOT=/workspace/accrl-training
  export CONFIG_PATH=/workspace/flashinfer-bench-private/acc_config.yaml
  export HOST=0.0.0.0
  export PORT="${PROFILE_PORT:-10000}"
  export DEVICE="${FIB_DEVICE:-cuda:0}"
  export TIMEOUT="${FIB_TIMEOUT:-120}"
  export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-9.0a}"
  /workspace/flashinfer-bench-private/scripts/launch_fib_serve_direct.sh /workspace/accrl-training
'

echo "direct debug service: http://localhost:${PROFILE_PORT}"
echo "container: ${CONTAINER}"
echo "GPU: host device ${GPU_ID} as ${FIB_DEVICE}"
