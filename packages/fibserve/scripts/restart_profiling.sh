#!/usr/bin/env bash
# Restart only the profiling service sessions created by PTXBench's launcher.
#
# The Docker entrypoint delegates service lifetime to the tmux launcher using
# TMUX_PREFIX (default: fib-serve). This wrapper removes all sessions with that
# prefix, including leftovers from older device counts, then reruns only the
# dispatcher launcher. It does not install dependencies or change datasets.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$HERE/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-$PTXBENCH_ROOT/packages/fibserve}"
FIB_DATASET_DIR="${FIB_DATASET_DIR:-${DATASET_ROOT:-/workspace/accrl-training}}"
FIB_HEAVY_DATASET_DIR="${FIB_HEAVY_DATASET_DIR:-/workspace/accrl-training-heavy}"
FIB_DATASET_DIRS="${FIB_DATASET_DIRS:-${DATASET_ROOTS:-$FIB_DATASET_DIR:$FIB_HEAVY_DATASET_DIR}}"
PORT="${DISPATCH_PORT:-${PROFILE_PORT:-10000}}"
FIB_TIMEOUT="${FIB_TIMEOUT:-30}"
VENV_DIR="${VENV_DIR:-$PTXBENCH_ROOT/.venv}"
BASE_PORT="${BASE_PORT:-40001}"
LAUNCH_VERIFY_TIMEOUT="${LAUNCH_VERIFY_TIMEOUT:-10}"
FIB_GPU_COOLDOWN_SLEEP_S="${FIB_GPU_COOLDOWN_SLEEP_S:-30}"
PROFILE_MAX_GPUS="${PROFILE_MAX_GPUS:-}"
export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-9.0a}"
TMUX_PREFIX="${TMUX_PREFIX:-fib-serve}"
DEFAULT_FIB_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
SERVICE_DEVICES="${FIB_DEVICES:-${DEVICES:-$DEFAULT_FIB_DEVICES}}"
CONFIG_PATH="${CONFIG_PATH:-$BENCH_ROOT/acc_config.yaml}"
SUPERVISOR_STATE_DIR="${SUPERVISOR_STATE_DIR:-/tmp/fibserve-supervisor}"
restart_marker="$SUPERVISOR_STATE_DIR/restart-in-progress"

launcher="${LAUNCHER:-$BENCH_ROOT/scripts/launch_fib_serve_dispatcher.sh}"

# shellcheck disable=SC1091
source "$HERE/gpu_preflight.sh"

if [ ! -x "$launcher" ]; then
    echo "[restart_profiling] ERROR: dispatcher launcher is missing or not executable: $launcher" >&2
    echo "[restart_profiling] Rebuild the PTXBench FIBServe image or set LAUNCHER explicitly." >&2
    exit 1
fi
IFS=':' read -r -a FIB_DATASET_DIR_LIST <<< "$FIB_DATASET_DIRS"
for dataset_dir in "${FIB_DATASET_DIR_LIST[@]}"; do
    if [ ! -d "$dataset_dir" ]; then
        echo "[restart_profiling] ERROR: dataset dir is missing: $dataset_dir" >&2
        echo "[restart_profiling] Recreate the container with both PTXBench trace-set mounts." >&2
        exit 1
    fi
done
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[restart_profiling] ERROR: config is missing: $CONFIG_PATH" >&2
    echo "[restart_profiling] Rebuild the PTXBench FIBServe image." >&2
    exit 1
fi
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[restart_profiling] ERROR: venv is missing: $VENV_DIR" >&2
    echo "[restart_profiling] Rebuild the PTXBench FIBServe image." >&2
    exit 1
fi

# Match the Docker entrypoint's launcher environment without rerunning setup.
# The tmux launcher starts shell commands that expect this venv on PATH.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
if ! command -v fibserve >/dev/null 2>&1; then
    echo "[restart_profiling] ERROR: fibserve is not installed in $VENV_DIR" >&2
    echo "[restart_profiling] Rebuild the PTXBench FIBServe image." >&2
    exit 1
fi
if ! python -c 'import fastapi, httpx, uvicorn' >/dev/null 2>&1; then
    echo "[restart_profiling] ERROR: dispatcher dependencies are missing in $VENV_DIR" >&2
    echo "[restart_profiling] Rebuild the PTXBench FIBServe image." >&2
    exit 1
fi

port_is_listening() {
    local port="$1"
    (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

wait_port_closed() {
    local port="$1"
    local deadline="$2"
    local i
    for ((i = 0; i < deadline; i++)); do
        port_is_listening "$port" || return 0
        sleep 1
    done
    return 1
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
        echo "[restart_profiling] Recreate ptxbench-fibserve with Docker GPU access." >&2
        exit 1
    fi

    local smi_output
    if ! smi_output="$(nvidia-smi -L 2>&1)"; then
        echo "[restart_profiling] ERROR: nvidia-smi failed inside the container:" >&2
        echo "$smi_output" >&2
        echo "[restart_profiling] Refusing to kill existing service sessions while fresh CUDA/NVML is broken." >&2
        echo "[restart_profiling] Recreate the container with docker/compose.yaml." >&2
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

kill_tvm_ffi_compilers() {
    local pgids=()
    mapfile -t pgids < <(
        ps -eo pgid=,comm=,args= \
            | awk '$2 ~ /^(sh|nvcc|ptxas|cicc|ninja)$/ && $0 ~ /flashinfer_bench\/cache\/tvm_ffi/ {print $1}' \
            | sort -u
    )

    if [ "${#pgids[@]}" -eq 0 ]; then
        return
    fi

    echo "[restart_profiling] killing stale TVM-FFI compiler process groups: ${pgids[*]}" >&2
    for pgid in "${pgids[@]}"; do
        kill -KILL -- "-$pgid" 2>/dev/null || true
    done
}

kill_profile_tmux_sessions() {
    local sessions=()
    mapfile -t sessions < <(
        tmux list-sessions -F '#S' 2>/dev/null \
            | awk -v prefix="$TMUX_PREFIX" '$0 == prefix || index($0, prefix "-") == 1'
    )

    if [ "${#sessions[@]}" -eq 0 ]; then
        echo "[restart_profiling] no tmux sessions found for TMUX_PREFIX=$TMUX_PREFIX" >&2
    else
        echo "[restart_profiling] removing tmux sessions for TMUX_PREFIX=$TMUX_PREFIX: ${sessions[*]}" >&2
        for session in "${sessions[@]}"; do
            tmux kill-session -t "$session" 2>/dev/null || true
        done
    fi
}

kill_container_gpu_processes() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[restart_profiling] WARNING: nvidia-smi is unavailable; cannot enumerate leftover GPU processes" >&2
        return 0
    fi

    local pids=()
    mapfile -t pids < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | awk 'NF {print $1}' \
            | sort -n -u
    )

    if [ "${#pids[@]}" -eq 0 ]; then
        echo "[restart_profiling] no leftover GPU processes visible inside ptxbench-fibserve" >&2
        return 0
    fi

    echo "[restart_profiling] terminating leftover GPU processes before thermal preflight: ${pids[*]}" >&2
    for pid in "${pids[@]}"; do
        if [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    local deadline=10
    local i
    for ((i = 0; i < deadline; i++)); do
        local alive=()
        for pid in "${pids[@]}"; do
            if [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            fi
        done
        if [ "${#alive[@]}" -eq 0 ]; then
            return 0
        fi
        sleep 1
    done

    for pid in "${pids[@]}"; do
        if [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

validate_profile_max_gpus
mkdir -p "$SUPERVISOR_STATE_DIR"
touch "$restart_marker"
trap 'rm -f "$restart_marker"' EXIT
kill_profile_tmux_sessions
kill_tvm_ffi_compilers
kill_container_gpu_processes

if ! wait_port_closed "$PORT" 5; then
    echo "[restart_profiling] ERROR: dispatcher port is still listening after tmux cleanup: 127.0.0.1:$PORT" >&2
    echo "[restart_profiling] A non-tmux process is probably holding the port. Inside the container, check: ss -ltnp | grep ':$PORT'" >&2
    exit 1
fi

sleep_before_thermal_preflight
cuda_preflight

export BENCH_ROOT
export DATASET_ROOT="$FIB_DATASET_DIR"
export DATASET_ROOTS="$FIB_DATASET_DIRS"
export CONFIG_PATH
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
    echo "  cd $BENCH_ROOT && python -m flashinfer_bench.serve.dispatcher --urls localhost:$BASE_PORT ... --host $HOST --port $PORT" >&2
    exit 1
fi

if ! wait_port_open "$PORT" "$LAUNCH_VERIFY_TIMEOUT"; then
    echo "[restart_profiling] ERROR: dispatcher session exists but port did not open: 127.0.0.1:$PORT" >&2
    echo "[restart_profiling] Inspect it with: tmux capture-pane -pt $dispatcher_session -S -200" >&2
    exit 1
fi

echo "[restart_profiling] launcher returned; service sessions are managed by tmux" >&2
tmux list-sessions 2>/dev/null || true
