#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMUX_PREFIX="${TMUX_PREFIX:-fib-serve}"
DISPATCH_SESSION="${TMUX_PREFIX}-dispatcher"
SUPERVISOR_STATE_DIR="${SUPERVISOR_STATE_DIR:-/tmp/fibserve-supervisor}"
RESTART_MARKER="${SUPERVISOR_STATE_DIR}/restart-in-progress"

cleanup() {
  local sessions=()
  mapfile -t sessions < <(
    tmux list-sessions -F '#S' 2>/dev/null \
      | awk -v prefix="${TMUX_PREFIX}" '$0 == prefix || index($0, prefix "-") == 1'
  )
  for session in "${sessions[@]}"; do
    tmux kill-session -t "${session}" 2>/dev/null || true
  done
}

terminate() {
  exit 0
}

trap cleanup EXIT
trap terminate INT TERM

mkdir -p "${SUPERVISOR_STATE_DIR}"
rm -f "${RESTART_MARKER}"
"${SCRIPT_DIR}/launch_fib_serve_dispatcher.sh" "$@"

# Keep the container alive while the public dispatcher exists or an explicit
# in-container restart is replacing the sessions. Backend sessions restart
# their FIBServe processes independently inside the launcher.
while tmux has-session -t "${DISPATCH_SESSION}" 2>/dev/null \
    || [[ -f "${RESTART_MARKER}" ]]; do
  sleep 5
done

echo "Dispatcher tmux session exited: ${DISPATCH_SESSION}" >&2
exit 1
