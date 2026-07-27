#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

scripts=(
  "/home/ubuntu/AccRL/fib_runtime/multiturn/2026-0413-1611/create_test.py"
  "/home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/create_test.py"
  "/home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/create_test.py"
  "/home/ubuntu/AccRL/fib_runtime/multiturn/2026-0516-0609/scripts/create_test.py"
  "/home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/create_test.py"
)

for script in "${scripts[@]}"; do
  echo "Running ${script}"
  "${PYTHON_BIN}" "${script}"
done
