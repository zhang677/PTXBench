#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

SOURCE_RUNS="${SOURCE_RUNS:-$SCRIPT_DIR/source-runs.csv}"
UNFILTERED_CSV="${UNFILTERED_CSV:-$PTXBENCH_FIXIT_PROJECT/fixit-error-kernels.unfiltered.csv}"
PREBALANCED_CSV="${PREBALANCED_CSV:-$PTXBENCH_FIXIT_PROJECT/fixit-error-kernels.prebalanced.csv}"
BALANCED_CSV="${BALANCED_CSV:-$PTXBENCH_FIXIT_PROJECT/fixit-error-kernels.csv}"
ENRICHED_CSV="${ENRICHED_CSV:-$PTXBENCH_FIXIT_PROJECT/fixit-error-kernels.enriched.csv}"
CONFIG_JSON="${CONFIG_JSON:-$PTXBENCH_FIXIT_PROJECT/gemini-source-prompt-config.json}"
MANIFEST_JSON="${MANIFEST_JSON:-$PTXBENCH_FIXIT_PROJECT/fixit-source-manifest.json}"
SELECTOR="$PTXBENCH_MULTITURN_ROOT/fix_kernels/select_failed_kernels.py"
FILTER="$PTXBENCH_CONSTRUCT_EVAL_ROOT/filter_fixit_error_kernels_by_prompt_tag.py"
BUILDER="$PTXBENCH_CONSTRUCT_EVAL_ROOT/build_rebalanced_fixit_error_collection.py"
PER_DEFINITION_CAP="${PER_DEFINITION_CAP:-33}"
NUM_FIX_TURNS="${NUM_FIX_TURNS:-5}"
TARGET_SPEEDUP="${TARGET_SPEEDUP:-0.15}"

mkdir -p "$PTXBENCH_FIXIT_PROJECT"

select_args=(
  --selected-runs-csv "$SOURCE_RUNS"
  --output-csv "$UNFILTERED_CSV"
)
if [[ "${FORCE_DERIVED:-0}" == "1" ]]; then
  select_args+=(--force-turn-csv --force-kernels)
fi

echo "[1/3] Select failed kernels from the eight base-Qwen source roots"
python "$SELECTOR" "${select_args[@]}"

echo "[2/3] Keep the historical Fixit prompt families and first five source turns"
python "$FILTER" \
  --input-csv "$UNFILTERED_CSV" \
  --output-csv "$PREBALANCED_CSV"

builder_args=(
  --input-csv "$PREBALANCED_CSV"
  --balanced-csv "$BALANCED_CSV"
  --enriched-csv "$ENRICHED_CSV"
  --config-json "$CONFIG_JSON"
  --manifest-json "$MANIFEST_JSON"
  --source-markdown "$SCRIPT_DIR/README.md"
  --source-runs-csv "$SOURCE_RUNS"
  --output-root "$PTXBENCH_FIXIT_REPAIR_ROOT"
  --per-definition-cap "$PER_DEFINITION_CAP"
  --num-turns "$NUM_FIX_TURNS"
  --target-speedup "$TARGET_SPEEDUP"
  --min-speedup-for-later-collection 0.0
)
if [[ "${ALLOW_UNDERFILLED_DEFINITIONS:-0}" == "1" ]]; then
  builder_args+=(--allow-underfilled-definitions)
fi

echo "[3/3] Balance failures and write the Gemini repair configuration"
python "$BUILDER" "${builder_args[@]}"
