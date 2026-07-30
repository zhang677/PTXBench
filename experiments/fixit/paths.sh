#!/usr/bin/env bash
# Resolve the repository once for every Fixit stage.

_FIXIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$_FIXIT_DIR/../.." && pwd)}"
source "$PTXBENCH_ROOT/packages/mini-ptx-agent/fib_runtime/multiturn/construct_eval_scripts/ptxbench_paths.sh"
unset _FIXIT_DIR
