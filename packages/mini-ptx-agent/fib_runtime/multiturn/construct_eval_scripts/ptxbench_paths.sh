#!/usr/bin/env bash
# Shared portable roots for scripts sourced or executed from construct_eval_scripts.

_PTXBENCH_PATHS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MINI_PTX_AGENT_ROOT="${MINI_PTX_AGENT_ROOT:-$(cd "$_PTXBENCH_PATHS_DIR/../../.." && pwd)}"
export PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$MINI_PTX_AGENT_ROOT/../.." && pwd)}"
export PTXBENCH_DATA_ROOT="${PTXBENCH_DATA_ROOT:-$PTXBENCH_ROOT/data}"
export PTXBENCH_CONFIG_ROOT="${PTXBENCH_CONFIG_ROOT:-$PTXBENCH_ROOT/configs}"
export PTXBENCH_MULTITURN_ROOT="${PTXBENCH_MULTITURN_ROOT:-$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn}"
export PTXBENCH_CONSTRUCT_EVAL_ROOT="${PTXBENCH_CONSTRUCT_EVAL_ROOT:-$PTXBENCH_MULTITURN_ROOT/construct_eval_scripts}"
export PTXBENCH_FIXIT_PROJECT="${PTXBENCH_FIXIT_PROJECT:-$PTXBENCH_DATA_ROOT/sft_experiments/test-fixit-qwen36-27b-gemini-glm}"
export PTXBENCH_EVAL_RUNS_ROOT="${PTXBENCH_EVAL_RUNS_ROOT:-$PTXBENCH_DATA_ROOT/eval_runs}"

unset _PTXBENCH_PATHS_DIR

