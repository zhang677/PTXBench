#!/usr/bin/env bash
set -euo pipefail

# Launch exactly one flashinfer-bench serve backend, without the dispatcher.
#
# Defaults:
#   service:    localhost:10000
#   device:     cuda:0
#   bench root: /workspace/flashinfer-bench-private if present, otherwise this
#               script's parent directory
#   config:     ${BENCH_ROOT}/acc_config.yaml
#
# Override examples:
#   ./scripts/launch_fib_serve_direct.sh /path/to/data
#   DATASET_ROOT=/path/to/data DEVICE=cuda:1 PORT=10001 ./scripts/launch_fib_serve_direct.sh
#   DATASET_ROOTS=/path/to/data:/path/to/heavy ./scripts/launch_fib_serve_direct.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/launch_fib_serve_direct.sh /path/to/trace-set

Environment overrides:
  BENCH_ROOT=/workspace/flashinfer-bench-private
  DATASET_ROOT=/workspace/accrl-training
  DATASET_ROOTS=/workspace/accrl-training:/workspace/accrl-training-heavy
  CONFIG_PATH=${BENCH_ROOT}/acc_config.yaml
  HOST=0.0.0.0
  PORT=10000
  DEVICE=cuda:0
  TIMEOUT=30
  TMUX_SESSION=fib-serve-direct
  RESTART_DELAY=2
EOF
  exit 0
fi

if [[ -d /workspace/flashinfer-bench-private ]]; then
  DEFAULT_BENCH_ROOT=/workspace/flashinfer-bench-private
else
  DEFAULT_BENCH_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi
BENCH_ROOT="${BENCH_ROOT:-${DEFAULT_BENCH_ROOT}}"
CONFIG_PATH="${CONFIG_PATH:-${BENCH_ROOT}/acc_config.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${PROFILE_PORT:-10000}}"
DEVICE="${DEVICE:-${FIB_DEVICE:-cuda:0}}"
TIMEOUT="${TIMEOUT:-${FIB_TIMEOUT:-30}}"
TMUX_SESSION="${TMUX_SESSION:-${TMUX_PREFIX:-fib-serve}-direct}"
RESTART_DELAY="${RESTART_DELAY:-2}"

DATASET_ROOT_LIST=()
if [[ -n "${DATASET_ROOTS:-}" ]]; then
  IFS=':' read -r -a DATASET_ROOT_LIST <<< "${DATASET_ROOTS}"
elif [[ -n "${DATASET_ROOT:-}" ]]; then
  DATASET_ROOT_LIST=("${DATASET_ROOT}")
elif [[ "$#" -gt 0 ]]; then
  DATASET_ROOT_LIST=("$@")
fi

if [[ "${#DATASET_ROOT_LIST[@]}" -eq 0 ]]; then
  echo "Usage: $0 /path/to/trace-set" >&2
  echo "Or set DATASET_ROOT=/path/to/trace-set" >&2
  echo "Or set DATASET_ROOTS=/path/to/trace-set:/path/to/another-trace-set" >&2
  exit 1
fi
if [[ ! -d "${BENCH_ROOT}" ]]; then
  echo "BENCH_ROOT does not exist: ${BENCH_ROOT}" >&2
  exit 1
fi
for dataset_root in "${DATASET_ROOT_LIST[@]}"; do
  if [[ ! -d "${dataset_root}" ]]; then
    echo "DATASET_ROOT does not exist: ${dataset_root}" >&2
    exit 1
  fi
done
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

if python3 - "${HOST}" "${PORT}" <<'PY'
import socket
import sys

host, port_text = sys.argv[1], sys.argv[2]
port = int(port_text)

bind_hosts = [host]
if host in {"0.0.0.0", "::"}:
    bind_hosts = ["127.0.0.1", "::1"]

for bind_host in bind_hosts:
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        if sock.connect_ex((bind_host, port)) == 0:
            sys.exit(0)
sys.exit(1)
PY
then
  echo "Port ${PORT} is already in use on ${HOST}; not starting ${TMUX_SESSION}" >&2
  exit 1
fi

shell_quote() {
  printf "%q" "$1"
}

local_args=""
for dataset_root in "${DATASET_ROOT_LIST[@]}"; do
  local_args+=" $(shell_quote "${dataset_root}")"
done

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

if tmux_has_session "${TMUX_SESSION}"; then
  tmux kill-session -t "${TMUX_SESSION}"
fi

cmd="cd $(shell_quote "${BENCH_ROOT}")"
cmd+=" && while true; do"
cmd+=" flashinfer-bench serve"
cmd+=" --local${local_args}"
cmd+=" --port $(shell_quote "${PORT}")"
cmd+=" --host $(shell_quote "${HOST}")"
cmd+=" --timeout $(shell_quote "${TIMEOUT}")"
cmd+=" --config $(shell_quote "${CONFIG_PATH}")"
cmd+=" --devices $(shell_quote "${DEVICE}")"
cmd+="; status=\$?"
cmd+="; echo \"[launch_fib_serve_direct] ${TMUX_SESSION} exited with status \${status}; restarting in ${RESTART_DELAY}s\" >&2"
cmd+="; sleep $(shell_quote "${RESTART_DELAY}")"
cmd+="; done"

echo "Starting ${TMUX_SESSION}: ${DEVICE} -> ${PORT}"
tmux new-session -d -s "${TMUX_SESSION}" -c "${BENCH_ROOT}" "${cmd}"

echo
echo "Launched direct backend: http://localhost:${PORT}"
echo "Device: ${DEVICE}"
echo "Use:"
echo "  PROFILE_BASE_URL=http://localhost:${PORT} <your verifier or client command>"
echo "Logs:"
echo "  tmux attach -t ${TMUX_SESSION}"
