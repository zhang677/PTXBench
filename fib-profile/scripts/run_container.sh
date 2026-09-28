#!/usr/bin/env bash
# Recreate the idle container with the local bench, dataset, and scripts mounted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_FIB_ROOT="${HOST_FIB_ROOT:-$(cd "$HERE/.." && pwd)}"
IMAGE="${IMAGE:-fib-profile:latest}"
NAME="${NAME:-fib-profile}"
PROFILE_PORT="${PROFILE_PORT:-10000}"
HOST_PROFILE_PORT="${HOST_PROFILE_PORT:-$PROFILE_PORT}"
GPUS="${GPUS:-\"device=0,1,2,3\"}"
FIB_DEVICES="${FIB_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
FIB_GPU_MAX_TEMP_C="${FIB_GPU_MAX_TEMP_C:-45}"
TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-9.0a}"

for required in flashinfer-bench-private/pyproject.toml dataset/definitions dataset/workloads scripts/restart_profiling.sh; do
    if [ ! -e "$HOST_FIB_ROOT/$required" ]; then
        echo "[run_container] ERROR: missing $HOST_FIB_ROOT/$required" >&2
        exit 1
    fi
done

# shellcheck disable=SC1091
source "$HERE/gpu_preflight.sh"
if command -v nvidia-smi >/dev/null 2>&1; then
    FIB_DEVICES="$(profile_service_filter_cool_gpu_devices "[run_container]" "$FIB_DEVICES" nvidia-smi)"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --init --name "$NAME" --gpus "$GPUS" \
    -v "$HOST_FIB_ROOT/flashinfer-bench-private:/workspace/flashinfer-bench-private" \
    -v "$HOST_FIB_ROOT/dataset:/workspace/dataset" \
    -v "$HOST_FIB_ROOT/scripts:/workspace/scripts:ro" \
    -e "FIB_DATASET_DIR=/workspace/dataset" \
    -e "FIB_DATASET_DIRS=/workspace/dataset" \
    -e "FIB_DEVICES=$FIB_DEVICES" \
    -e "FIB_GPU_MAX_TEMP_C=$FIB_GPU_MAX_TEMP_C" \
    -e "PROFILE_PORT=$PROFILE_PORT" \
    -e "TVM_FFI_CUDA_ARCH_LIST=$TVM_FFI_CUDA_ARCH_LIST" \
    -p "$HOST_PROFILE_PORT:$PROFILE_PORT" \
    "$IMAGE" bash -lc 'sleep infinity'

echo "[run_container] $NAME is up with $FIB_DEVICES; run /workspace/scripts/entrypoint.sh to install and start the service" >&2
