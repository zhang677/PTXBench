#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper for the host-side direct /debug client.
# This intentionally does not docker cp or docker exec: the kernel source is
# read on the host and submitted in the /debug JSON payload.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
  cat <<'EOF'
Usage: ./debug_error_kernel_in_fib_profile.sh /path/to/kernel_t*.cu [debug_error_kernel.py args...]

Environment overrides:
  BASE_URL=http://localhost:10000

Example:
  ./debug_error_kernel_in_fib_profile.sh \
    /home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128/kernels/exp_001/kernel_t5.cu \
    --timeout 120 --wait-timeout 180
EOF
  exit 0
fi

BASE_URL="${BASE_URL:-http://localhost:10000}"
KERNEL_PATH="$1"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/debug_error_kernel.py" "${KERNEL_PATH}" --base-url "${BASE_URL}" "$@"
