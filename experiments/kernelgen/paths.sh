#!/usr/bin/env bash

_KERNELGEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_KERNELGEN_DIR/../shared/paths.sh"
export KERNELGEN_PROJECT="${KERNELGEN_PROJECT:-$PTXBENCH_DATA_ROOT/kernelgen}"
export KERNELGEN_RUN_TAG="${KERNELGEN_RUN_TAG:-qwen36-27b-kernelgen-example-e5-lr4.65e-4-lora32}"
export KERNELGEN_PARQUET="${KERNELGEN_PARQUET:-$KERNELGEN_PROJECT/data/kernelgen.parquet}"

unset _KERNELGEN_DIR
