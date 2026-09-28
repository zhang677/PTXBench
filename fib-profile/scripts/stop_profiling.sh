#!/usr/bin/env bash
# Stop only the flashinfer-bench profiling service processes in fib-profile.
#
# The service launcher owns tmux sessions named with TMUX_PREFIX (default:
# fib-serve). Removing those sessions should stop the dispatcher and backends.
# The remaining cleanup matches restart_profiling.sh's historical stop phase so
# stale compiler or GPU children cannot keep the service port or GPUs occupied.
set -euo pipefail

PORT="${PROFILE_PORT:-10000}"
TMUX_PREFIX="${TMUX_PREFIX:-fib-serve}"

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

    echo "[stop_profiling] killing stale TVM-FFI compiler process groups: ${pgids[*]}" >&2
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
        echo "[stop_profiling] no tmux sessions found for TMUX_PREFIX=$TMUX_PREFIX" >&2
    else
        echo "[stop_profiling] removing tmux sessions for TMUX_PREFIX=$TMUX_PREFIX: ${sessions[*]}" >&2
        for session in "${sessions[@]}"; do
            tmux kill-session -t "$session" 2>/dev/null || true
        done
    fi
}

kill_container_gpu_processes() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[stop_profiling] WARNING: nvidia-smi is unavailable; cannot enumerate leftover GPU processes" >&2
        return 0
    fi

    local pids=()
    mapfile -t pids < <(
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | awk 'NF {print $1}' \
            | sort -n -u
    )

    if [ "${#pids[@]}" -eq 0 ]; then
        echo "[stop_profiling] no leftover GPU processes visible inside fib-profile" >&2
        return 0
    fi

    echo "[stop_profiling] terminating leftover GPU processes: ${pids[*]}" >&2
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

kill_profile_tmux_sessions
kill_tvm_ffi_compilers
kill_container_gpu_processes

if ! wait_port_closed "$PORT" 5; then
    echo "[stop_profiling] ERROR: dispatcher port is still listening after tmux cleanup: 127.0.0.1:$PORT" >&2
    echo "[stop_profiling] A non-tmux process is probably holding the port. Inside the container, check: ss -ltnp | grep ':$PORT'" >&2
    exit 1
fi

echo "[stop_profiling] profiling service stopped; dispatcher port is closed: 127.0.0.1:$PORT" >&2
