#!/usr/bin/env bash
# Install the mounted bench into a local venv, then start the service.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
FIB_DIR="${FIB_DIR:-flashinfer-bench-private}"
BENCH_ROOT="$WORKDIR/$FIB_DIR"
DATASET_ROOT="${FIB_DATASET_DIR:-$WORKDIR/dataset}"
VENV_DIR="${VENV_DIR:-$WORKDIR/acc}"

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
    -e "$BENCH_ROOT[serve]" httpx cupti-python "flash-linear-attention==0.5.2" \
    "torch==2.12.1+cu130" "triton==3.7.1"
python - <<'PY'
import torch
import triton

assert torch.__version__ == "2.12.1+cu130", torch.__version__
assert triton.__version__ == "3.7.1", triton.__version__
PY

exec "$HERE/restart_profiling.sh"
