#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMUX_PREFIX="${TMUX_PREFIX:-fib-serve}"
DISPATCH_SESSION="${TMUX_PREFIX}-dispatcher"

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

trap cleanup EXIT INT TERM

"${SCRIPT_DIR}/launch_fib_serve_dispatcher.sh" "$@"

# Keep the container alive while the public dispatcher session exists. Backend
# sessions restart their FIBServe processes independently inside the launcher.
while tmux has-session -t "${DISPATCH_SESSION}" 2>/dev/null; do
  sleep 5
done

echo "Dispatcher tmux session exited: ${DISPATCH_SESSION}" >&2
exit 1
