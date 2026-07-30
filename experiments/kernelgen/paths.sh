#!/usr/bin/env bash

_KERNELGEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTXBENCH_ROOT="${PTXBENCH_ROOT:-$(cd "$_KERNELGEN_DIR/../.." && pwd)}"
export MINI_PTX_AGENT_ROOT="${MINI_PTX_AGENT_ROOT:-$PTXBENCH_ROOT/packages/mini-ptx-agent}"
export PTXBENCH_DATA_ROOT="${PTXBENCH_DATA_ROOT:-$PTXBENCH_ROOT/data}"
export KERNELGEN_PROJECT="${KERNELGEN_PROJECT:-$PTXBENCH_DATA_ROOT/sft_experiments/mha-8def-single-turn-qwen36-27b-gemini-glm}"
export KERNELGEN_RUN_TAG="${KERNELGEN_RUN_TAG:-qwen36-27b-mha-8def-single-turn-glm52-e5-lr4.65e-4-lora32}"
export KERNELGEN_PARQUET="${KERNELGEN_PARQUET:-$KERNELGEN_PROJECT/data/glm52-mha-8def-single-turn-gemini-output.parquet}"

unset _KERNELGEN_DIR
