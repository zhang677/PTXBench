#!/usr/bin/env bash
set -euo pipefail

# Launch one flashinfer-bench backend per GPU and front them with the dispatcher.
#
# Defaults:
#   backends:   localhost:40001 ... localhost:40008
#   dispatcher: localhost:40000
#   bench root: parent directory of this script's directory
#   config:     ${BENCH_ROOT}/acc_config.yaml
#
# Override examples:
#   ./scripts/launch_fib_serve_dispatcher.sh /path/to/data
#   DATASET_ROOT=/path/to/data ./scripts/launch_fib_serve_dispatcher.sh
#   DATASET_ROOTS=/path/to/data:/path/to/heavy ./scripts/launch_fib_serve_dispatcher.sh
#   DEVICES=cuda:0,cuda:1 BASE_PORT=41001 DISPATCH_PORT=41000 ./scripts/launch_fib_serve_dispatcher.sh /path/to/data

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${BENCH_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${BENCH_ROOT}/acc_config.yaml}"
BASE_PORT="${BASE_PORT:-40001}"
DISPATCH_PORT="${DISPATCH_PORT:-40000}"
HOST="${HOST:-0.0.0.0}"
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
TIMEOUT="${TIMEOUT:-30}"
TMUX_PREFIX="${TMUX_PREFIX:-fib-serve}"
BACKEND_RESTART_DELAY="${BACKEND_RESTART_DELAY:-2}"

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

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
if [[ "${#DEVICE_LIST[@]}" -eq 0 ]]; then
  echo "DEVICES is empty" >&2
  exit 1
fi

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

tmux_replace_session() {
  local session="$1"
  local command="$2"
  if tmux_has_session "${session}"; then
    tmux kill-session -t "${session}"
  fi
  tmux new-session -d -s "${session}" -c "${BENCH_ROOT}" "${command}"
}

shell_quote() {
  printf "%q" "$1"
}

local_args=""
for dataset_root in "${DATASET_ROOT_LIST[@]}"; do
  local_args+=" $(shell_quote "${dataset_root}")"
done

backend_urls=()
for idx in "${!DEVICE_LIST[@]}"; do
  device="${DEVICE_LIST[${idx}]}"
  port="$((BASE_PORT + idx))"
  session="${TMUX_PREFIX}-backend-${idx}"
  backend_urls+=("localhost:${port}")

  cmd="cd $(shell_quote "${BENCH_ROOT}")"
  cmd+=" && while true; do"
  cmd+=" flashinfer-bench serve"
  cmd+=" --local${local_args}"
  cmd+=" --port $(shell_quote "${port}")"
  cmd+=" --host $(shell_quote "${HOST}")"
  cmd+=" --timeout $(shell_quote "${TIMEOUT}")"
  cmd+=" --config $(shell_quote "${CONFIG_PATH}")"
  cmd+=" --devices $(shell_quote "${device}")"
  cmd+="; status=\$?"
  cmd+="; echo \"[launch_fib_serve_dispatcher] ${session} exited with status \${status}; restarting in ${BACKEND_RESTART_DELAY}s\" >&2"
  cmd+="; sleep $(shell_quote "${BACKEND_RESTART_DELAY}")"
  cmd+="; done"

  echo "Starting ${session}: ${device} -> ${port}"
  tmux_replace_session "${session}" "${cmd}"
done

dispatcher_session="${TMUX_PREFIX}-dispatcher"
dispatcher_cmd="cd $(shell_quote "${BENCH_ROOT}")"
dispatcher_cmd+=" && exec python $(shell_quote "${BENCH_ROOT}/flashinfer_bench/serve/dispatcher.py")"
dispatcher_cmd+=" --urls ${backend_urls[*]}"
dispatcher_cmd+=" --host $(shell_quote "${HOST}")"
dispatcher_cmd+=" --port $(shell_quote "${DISPATCH_PORT}")"

echo "Starting ${dispatcher_session}: dispatcher -> ${DISPATCH_PORT}"
tmux_replace_session "${dispatcher_session}" "${dispatcher_cmd}"

echo
echo "Launched ${#DEVICE_LIST[@]} backend service(s):"
for idx in "${!DEVICE_LIST[@]}"; do
  printf "  %-24s %s\n" "${backend_urls[${idx}]}" "${DEVICE_LIST[${idx}]}"
done
echo "Dispatcher: http://localhost:${DISPATCH_PORT}"
echo
echo "Use:"
echo "  PROFILE_BASE_URL=http://localhost:${DISPATCH_PORT} <your verifier or client command>"
echo
echo "Logs:"
echo "  tmux attach -t ${dispatcher_session}"
echo "  tmux attach -t ${TMUX_PREFIX}-backend-0"
