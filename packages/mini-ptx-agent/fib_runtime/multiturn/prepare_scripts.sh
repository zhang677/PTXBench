#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  "$SCRIPT_DIR/gemm-problems/create_test.py"
  "$SCRIPT_DIR/mha-with-lse-problems/create_test.py"
  "$SCRIPT_DIR/mha-bwd-problems/create_test.py"
  "$SCRIPT_DIR/fp8-mha-with-lse-problems/scripts/create_test.py"
  "$SCRIPT_DIR/single_op_eval/create_test.py"
)

for script in "${scripts[@]}"; do
  echo "Running ${script}"
  "${PYTHON_BIN}" "${script}"
done
