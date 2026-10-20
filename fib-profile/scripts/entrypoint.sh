#!/usr/bin/env bash
# Install the mounted bench into a local venv, then start the service.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
FIB_DIR="${FIB_DIR:-flashinfer-bench-private}"
BENCH_ROOT="$WORKDIR/$FIB_DIR"
DATASET_ROOT="${FIB_DATASET_DIR:-$WORKDIR/dataset}"
VENV_DIR="${VENV_DIR:-$WORKDIR/acc}"
FIB_DATASET_MODE="${FIB_DATASET_MODE:-existing}"
FLASH_KDA_REPO_URL="https://github.com/MoonshotAI/FlashKDA.git"
FLASH_KDA_DIR="${FLASH_KDA_DIR:-$WORKDIR/flash-kda}"

case "$FIB_DATASET_MODE" in
    existing) ;;
    bundled)
        if [ ! -f "$DATASET_ROOT/generate_blobs.py" ]; then
            echo "[entrypoint] ERROR: bundled mode requires $DATASET_ROOT/generate_blobs.py" >&2
            exit 1
        fi
        ;;
    *) echo "[entrypoint] ERROR: FIB_DATASET_MODE must be existing or bundled" >&2; exit 1 ;;
esac

for required in "$BENCH_ROOT/pyproject.toml" "$DATASET_ROOT/definitions" "$DATASET_ROOT/workloads"; do
    if [ ! -e "$required" ]; then
        echo "[entrypoint] ERROR: missing mounted input $required" >&2
        exit 1
    fi
done

mkdir -p "$WORKDIR" "${UV_CACHE_DIR:-$WORKDIR/.cache/uv}"
[ -d "$VENV_DIR" ] || uv venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Keep the profiling runtime fixed across clean installs.
uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -e "$BENCH_ROOT[serve,profile-cu130]"

# Build FlashKDA from the default branch on a fresh clone. Reuse an existing
# checkout as-is; there is no commit pin or automatic branch change.
if [ ! -e "$FLASH_KDA_DIR" ]; then
    git clone "$FLASH_KDA_REPO_URL" "$FLASH_KDA_DIR"
elif [ ! -e "$FLASH_KDA_DIR/.git" ]; then
    echo "[entrypoint] ERROR: $FLASH_KDA_DIR exists but is not a git checkout" >&2
    exit 1
fi
git -C "$FLASH_KDA_DIR" submodule update --init --recursive
(
    cd "$FLASH_KDA_DIR"
    uv pip install -v --no-build-isolation .
)

python - <<'PY'
import torch
import triton
import flash_kda

assert torch.__version__ == "2.12.1+cu130", torch.__version__
assert triton.__version__ == "3.7.1", triton.__version__
PY

if [ "$FIB_DATASET_MODE" = bundled ]; then
    # Use the same UUID-based thermal filter as service startup, before generation.
    # shellcheck disable=SC1091
    source "$HERE/gpu_preflight.sh"
    cool_devices="$(profile_service_filter_cool_gpu_devices "[entrypoint]" \
        "${FIB_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}" nvidia-smi)"
    blob_device="${cool_devices%%,*}"
    echo "[entrypoint] Preparing bundled blobs on $blob_device" >&2
    env LD_LIBRARY_PATH= "$VENV_DIR/bin/python" \
        "$DATASET_ROOT/generate_blobs.py" --dataset-root "$DATASET_ROOT" --device "$blob_device"
fi

exec "$HERE/restart_profiling.sh"
