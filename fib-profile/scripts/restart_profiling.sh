#!/usr/bin/env bash
# Restart only the profiling service sessions created by entrypoint.sh.
#
# entrypoint.sh delegates service lifetime to flashinfer-bench's tmux launcher,
# using TMUX_PREFIX (default: fib-serve). This wrapper removes all sessions with
# that prefix, including leftovers from older device counts, then reruns only
# the dispatcher launcher. It intentionally skips clone/install/dataset download.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
FIB_DIR="${FIB_DIR:-flashinfer-bench-private}"
FIB_DATASET_DIR="${FIB_DATASET_DIR:-$WORKDIR/dataset}"
FIB_DATASET_DIRS="${FIB_DATASET_DIRS:-$FIB_DATASET_DIR}"
PORT="${PROFILE_PORT:-10000}"
FIB_TIMEOUT="${FIB_TIMEOUT:-30}"
VENV_DIR="${VENV_DIR:-$WORKDIR/acc}"
BASE_PORT="${BASE_PORT:-40001}"
LAUNCH_VERIFY_TIMEOUT="${LAUNCH_VERIFY_TIMEOUT:-10}"
FIB_GPU_COOLDOWN_SLEEP_S="${FIB_GPU_COOLDOWN_SLEEP_S:-30}"
PROFILE_MAX_GPUS="${PROFILE_MAX_GPUS:-}"
export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-9.0a}"
TMUX_PREFIX="${TMUX_PREFIX:-fib-serve}"
DEFAULT_FIB_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
SERVICE_DEVICES="${FIB_DEVICES:-${DEVICES:-$DEFAULT_FIB_DEVICES}}"

launcher="${LAUNCHER:-$WORKDIR/$FIB_DIR/scripts/launch_fib_serve_dispatcher.sh}"

# shellcheck disable=SC1091
source "$HERE/gpu_preflight.sh"

if [ ! -x "$launcher" ]; then
    echo "[restart_profiling] ERROR: dispatcher launcher is missing or not executable: $launcher" >&2
    echo "[restart_profiling] Run /workspace/scripts/entrypoint.sh once for setup, or set LAUNCHER=/path/to/launch_fib_serve_dispatcher.sh." >&2
    exit 1
fi
IFS=':' read -r -a FIB_DATASET_DIR_LIST <<< "$FIB_DATASET_DIRS"
for dataset_dir in "${FIB_DATASET_DIR_LIST[@]}"; do
    if [ ! -d "$dataset_dir" ]; then
        echo "[restart_profiling] ERROR: dataset dir is missing: $dataset_dir" >&2
        echo "[restart_profiling] Mount the local dataset at /workspace/dataset." >&2
        exit 1
    fi
done
if [ ! -f "$WORKDIR/$FIB_DIR/acc_config.yaml" ]; then
    echo "[restart_profiling] ERROR: config is missing: $WORKDIR/$FIB_DIR/acc_config.yaml" >&2
    echo "[restart_profiling] Mount flashinfer-bench-private at /workspace/flashinfer-bench-private." >&2
    exit 1
fi
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[restart_profiling] ERROR: venv is missing: $VENV_DIR" >&2
    echo "[restart_profiling] Run /workspace/scripts/entrypoint.sh once to create/install the runtime env." >&2
    exit 1
fi

# Match entrypoint.sh's launcher environment without rerunning setup/downloads.
# The tmux launcher starts shell commands that expect this venv on PATH.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
if ! command -v flashinfer-bench >/dev/null 2>&1; then
    echo "[restart_profiling] ERROR: flashinfer-bench is not installed in $VENV_DIR" >&2
    echo "[restart_profiling] Run /workspace/scripts/entrypoint.sh once to install runtime dependencies." >&2
    exit 1
fi
if ! python -c 'import fastapi, httpx, uvicorn' >/dev/null 2>&1; then
    echo "[restart_profiling] ERROR: dispatcher dependencies are missing in $VENV_DIR" >&2
    echo "[restart_profiling] Run: source $VENV_DIR/bin/activate && uv pip install \"flashinfer-bench[serve]\" httpx" >&2
    exit 1
fi

port_is_listening() {
    local port="$1"
    (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

wait_port_open() {
    local port="$1"
    local deadline="$2"
    local i
    for ((i = 0; i < deadline; i++)); do
        port_is_listening "$port" && return 0
        sleep 1
    done
    return 1
}

sleep_before_thermal_preflight() {
    if ! [[ "$FIB_GPU_COOLDOWN_SLEEP_S" =~ ^[0-9]+$ ]]; then
        echo "[restart_profiling] ERROR: FIB_GPU_COOLDOWN_SLEEP_S must be a non-negative integer, got: $FIB_GPU_COOLDOWN_SLEEP_S" >&2
        exit 1
    fi
    if [ "$FIB_GPU_COOLDOWN_SLEEP_S" -gt 0 ]; then
        echo "[restart_profiling] waiting ${FIB_GPU_COOLDOWN_SLEEP_S}s for GPUs to cool before thermal preflight" >&2
        sleep "$FIB_GPU_COOLDOWN_SLEEP_S"
    fi
}

validate_profile_max_gpus() {
    if [ -z "$PROFILE_MAX_GPUS" ]; then
        return
    fi
    if ! [[ "$PROFILE_MAX_GPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "[restart_profiling] ERROR: PROFILE_MAX_GPUS must be a positive integer, got: $PROFILE_MAX_GPUS" >&2
        exit 1
    fi
}

limit_profile_gpus() {
    if [ -z "$PROFILE_MAX_GPUS" ]; then
        return
    fi
    local devices=()
    IFS=',' read -r -a devices <<< "$SERVICE_DEVICES"
    if [ "${#devices[@]}" -le "$PROFILE_MAX_GPUS" ]; then
        echo "[restart_profiling] PROFILE_MAX_GPUS=$PROFILE_MAX_GPUS leaves ${#devices[@]} selected device(s) unchanged: $SERVICE_DEVICES" >&2
        return
    fi

    devices=("${devices[@]:0:PROFILE_MAX_GPUS}")
    SERVICE_DEVICES="$(IFS=,; echo "${devices[*]}")"
    echo "[restart_profiling] limiting profile service to $PROFILE_MAX_GPUS GPU(s): $SERVICE_DEVICES" >&2
}

cuda_preflight() {
    if [ "${FIB_SKIP_CUDA_PREFLIGHT:-0}" = "1" ]; then
        echo "[restart_profiling] WARNING: skipping CUDA/NVML preflight because FIB_SKIP_CUDA_PREFLIGHT=1" >&2
        limit_profile_gpus
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[restart_profiling] ERROR: nvidia-smi is not available inside the container" >&2
        echo "[restart_profiling] Recreate fib-profile with Docker --gpus and --init before restarting the service." >&2
        exit 1
    fi

    local smi_output
    if ! smi_output="$(nvidia-smi -L 2>&1)"; then
        echo "[restart_profiling] ERROR: nvidia-smi failed inside the container:" >&2
        echo "$smi_output" >&2
        echo "[restart_profiling] Refusing to kill existing service sessions while fresh CUDA/NVML is broken." >&2
        echo "[restart_profiling] Recreate the container with /workspace/scripts/run_container.sh on the host." >&2
        exit 1
    fi

    local filtered_devices
    if filtered_devices="$(profile_service_filter_cool_gpu_devices "[restart_profiling]" "$SERVICE_DEVICES" nvidia-smi)"; then
        SERVICE_DEVICES="$filtered_devices"
    else
        local filter_status=$?
        if [ "$filter_status" -eq 1 ]; then
            exit 1
        fi
        [ -n "$filtered_devices" ] && SERVICE_DEVICES="$filtered_devices"
    fi

    limit_profile_gpus
    FIB_PREFLIGHT_DEVICES="$SERVICE_DEVICES" python - <<'PY'
import os
import sys

import torch

raw_devices = os.environ.get("FIB_PREFLIGHT_DEVICES", "")
devices = [item.strip() for item in raw_devices.split(",") if item.strip()]

if not torch.cuda.is_available():
    print("[restart_profiling] ERROR: torch.cuda.is_available() is false inside the container", file=sys.stderr)
    sys.exit(1)

count = torch.cuda.device_count()
bad = []
for device in devices:
    if not device.startswith("cuda:"):
        continue
    try:
        index = int(device.split(":", 1)[1])
    except ValueError:
        bad.append((device, "invalid CUDA device syntax"))
        continue
    if index >= count:
        bad.append((device, f"index is outside torch.cuda.device_count()={count}"))
        continue
    try:
        torch.cuda.set_device(index)
        torch.cuda.get_device_name(index)
    except Exception as exc:
        bad.append((device, f"{type(exc).__name__}: {exc}"))

if bad:
    print("[restart_profiling] ERROR: CUDA preflight failed for requested devices:", file=sys.stderr)
    for device, error in bad:
        print(f"  {device}: {error}", file=sys.stderr)
    sys.exit(1)

print(
    f"[restart_profiling] CUDA preflight ok: torch sees {count} device(s); requested={','.join(devices)}",
    file=sys.stderr,
)
PY
}

validate_profile_max_gpus
PROFILE_PORT="$PORT" TMUX_PREFIX="$TMUX_PREFIX" "$HERE/stop_profiling.sh"

sleep_before_thermal_preflight
cuda_preflight

export BENCH_ROOT="$WORKDIR/$FIB_DIR"
export DATASET_ROOT="$FIB_DATASET_DIR"
export DATASET_ROOTS="$FIB_DATASET_DIRS"
export CONFIG_PATH="$WORKDIR/$FIB_DIR/acc_config.yaml"
export DISPATCH_PORT="$PORT"
export HOST="${FIB_HOST:-0.0.0.0}"
export TIMEOUT="$FIB_TIMEOUT"
export TMUX_PREFIX
export DEVICES="$SERVICE_DEVICES"

# If another tmux session keeps the server alive, refresh the environment that
# new service sessions inherit. If no server exists, the launcher-created server
# inherits this already-activated process environment.
tmux set-environment -g PATH "$PATH" 2>/dev/null || true
tmux set-environment -g VIRTUAL_ENV "${VIRTUAL_ENV:-}" 2>/dev/null || true
tmux set-environment -g TVM_FFI_CUDA_ARCH_LIST "$TVM_FFI_CUDA_ARCH_LIST" 2>/dev/null || true

echo "[restart_profiling] relaunching dispatcher on 0.0.0.0:$PORT via $launcher" >&2
"$launcher" "$@"

dispatcher_session="${TMUX_PREFIX}-dispatcher"
if ! tmux has-session -t "$dispatcher_session" 2>/dev/null; then
    echo "[restart_profiling] ERROR: dispatcher tmux session exited during startup: $dispatcher_session" >&2
    echo "[restart_profiling] Backends may still be running. Check backend sessions with: tmux ls" >&2
    echo "[restart_profiling] To see the dispatcher error directly, run:" >&2
    echo "  cd $BENCH_ROOT && python $BENCH_ROOT/flashinfer_bench/serve/dispatcher.py --urls localhost:$BASE_PORT ... --host $HOST --port $PORT" >&2
    exit 1
fi

if ! wait_port_open "$PORT" "$LAUNCH_VERIFY_TIMEOUT"; then
    echo "[restart_profiling] ERROR: dispatcher session exists but port did not open: 127.0.0.1:$PORT" >&2
    echo "[restart_profiling] Inspect it with: tmux capture-pane -pt $dispatcher_session -S -200" >&2
    exit 1
fi

echo "[restart_profiling] launcher returned; service sessions are managed by tmux" >&2
tmux list-sessions 2>/dev/null || true
