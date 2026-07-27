#!/usr/bin/env bash
# Verify a FIBServe deployment through its single public dispatcher API.
#
# Local Docker service:
#   FIBSERVE_PORT=10000 ./packages/fibserve/scripts/run_verify.sh
#
# Existing URL:
#   PROFILE_BASE_URL=http://localhost:11000 ./packages/fibserve/scripts/run_verify.sh
#
# Remote service through an SSH tunnel:
#   REMOTE=hyper00 REMOTE_PORT=10000 LOCAL_PORT=20000 \
#     ./packages/fibserve/scripts/run_verify.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd -- "$HERE/../../.." && pwd)}"
VERIFY="${VERIFY:-$PTXBENCH_ROOT/packages/mini-ptx-agent/fib_runtime/multiturn/2026-0426-1410/scripts/verify_via_service.py}"
PERF_CSV="${PERF_CSV:-/tmp/ptxbench-verify-perf.csv}"
PYTHON="${PYTHON:-$PTXBENCH_ROOT/.venv/bin/python}"
HEALTH_RETRIES="${HEALTH_RETRIES:-60}"
TUNNEL_PID=""

cleanup() {
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi
}

terminate() {
  exit 130
}

trap cleanup EXIT
trap terminate INT TERM

if [[ ! -f "$VERIFY" ]]; then
  echo "[run_verify] verifier does not exist: $VERIFY" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "[run_verify] Python does not exist or is not executable: $PYTHON" >&2
  exit 1
fi

if [[ -n "${PROFILE_BASE_URL:-}" ]]; then
  verify_url="${PROFILE_BASE_URL%/}"
elif [[ -n "${REMOTE:-}" ]]; then
  REMOTE_PORT="${REMOTE_PORT:-10000}"
  LOCAL_PORT="${LOCAL_PORT:-20000}"
  ssh \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=30 \
    -o ExitOnForwardFailure=yes \
    -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    "$REMOTE" &
  TUNNEL_PID=$!
  verify_url="http://localhost:${LOCAL_PORT}"
else
  verify_url="http://localhost:${FIBSERVE_PORT:-10000}"
fi

echo "[run_verify] checking ${verify_url}/health" >&2
curl \
  --fail \
  --silent \
  --show-error \
  --retry "$HEALTH_RETRIES" \
  --retry-connrefused \
  --retry-delay 1 \
  --max-time 120 \
  "${verify_url}/health"
echo

export PROFILE_BASE_URL="$verify_url"
export PYTHONPATH="$PTXBENCH_ROOT/packages/mini-ptx-agent${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$VERIFY" --perf-csv "$PERF_CSV" "$@"
