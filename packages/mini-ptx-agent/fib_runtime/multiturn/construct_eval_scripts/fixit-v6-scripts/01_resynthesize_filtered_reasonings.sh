#!/usr/bin/env bash
set -euo pipefail

# Repair filtered reasoning rows before building the final full parquet.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../ptxbench_paths.sh"

PROJECT="$PTXBENCH_FIXIT_PROJECT"
SYNTH_CONFIG="$PROJECT/qwen36-27b-fixit-v6-config.yaml"
DEFAULT_CONFIG="$PROJECT/qwen36-27b-fixit-v6-resynthesis-config.yaml"
PAIRS_CSV="$PROJECT/fixit-v5-gemini-kernel-pairs.csv"
INPUT_JSONL="$PROJECT/reasoning_pairs.qwen36-27b-fixit-v6.jsonl"
PROCESS="$PTXBENCH_CONSTRUCT_EVAL_ROOT/fixit_downstream_process.py"
SCRIPT="$PTXBENCH_MULTITURN_ROOT/fix_kernels/resynthesize_filtered_reasonings_openrouter.py"

CONFIG="$DEFAULT_CONFIG"
GENERATE_CONFIG="1"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  GENERATE_CONFIG="0"
  shift
fi

if [[ "$GENERATE_CONFIG" == "1" ]]; then
  if [[ ! -f "$SYNTH_CONFIG" ]]; then
    echo "Missing synthesis config: $SYNTH_CONFIG" >&2
    echo "Run 00_synthesize_qwen36-27b_reasoning.sh first." >&2
    exit 1
  fi
  if [[ ! -f "$INPUT_JSONL" ]]; then
    echo "Missing synthesized reasoning JSONL: $INPUT_JSONL" >&2
    echo "Run 00_synthesize_qwen36-27b_reasoning.sh first." >&2
    exit 1
  fi

  read -r COVERED TOTAL < <(
    python - "$PROCESS" "$PAIRS_CSV" "$INPUT_JSONL" <<'PY'
from pathlib import Path
import sys

process_path = Path(sys.argv[1])
sys.path.insert(0, str(process_path.parent))
from fixit_downstream_process import reasoning_coverage

print(*reasoning_coverage(Path(sys.argv[2]), Path(sys.argv[3])))
PY
  )
  if [[ "$COVERED" != "$TOTAL" ]]; then
    echo "Reasoning synthesis is incomplete: $COVERED/$TOTAL pairs are covered." >&2
    echo "Wait for 00_synthesize_qwen36-27b_reasoning.sh to finish." >&2
    exit 1
  fi

  python - "$SYNTH_CONFIG" "$CONFIG" "$PROJECT" <<'PY'
from pathlib import Path
import sys

import yaml

synth_config = Path(sys.argv[1])
output_config = Path(sys.argv[2])
project = Path(sys.argv[3])

config = yaml.safe_load(synth_config.read_text()) or {}
config["description"] = (
    "Resynthesize filtered Qwen3.6-27B reasoning for the full fixit-v6 dataset."
)
config["resynthesis"] = {
    "input_jsonl": str(project / "reasoning_pairs.qwen36-27b-fixit-v6.jsonl"),
    "kernel_pairs_csv": str(project / "fixit-v5-gemini-kernel-pairs.csv"),
    "output_jsonl": str(project / "reasoning_pairs.qwen36-27b-fixit-v6-full.jsonl"),
    "checkpoint_jsonl": str(
        project / "reasoning_pairs.qwen36-27b-fixit-v6-full.repairs.jsonl"
    ),
    "provenance_json": str(
        project / "reasoning_pairs.qwen36-27b-fixit-v6-full.provenance.json"
    ),
    "tokenizer": "Qwen/Qwen3.6-27B",
    "source_label": "fixit-v6-qwen36-27b",
    "max_sequence_tokens": 65536,
    "safety_tokens": 1024,
    "min_target_ratio": 0.95,
    "overwrite": False,
    "dry_run": False,
}
output_config.parent.mkdir(parents=True, exist_ok=True)
output_config.write_text(yaml.safe_dump(config, sort_keys=False))
PY
elif [[ ! -f "$CONFIG" ]]; then
  echo "Missing resynthesis config: $CONFIG" >&2
  exit 1
fi

# Use the same local Qwen3.6-27B routing as the synthesis wrapper.
: "${ACCRL_MODEL_HOST:?Set ACCRL_MODEL_HOST to a dedicated Qwen3.6-27B endpoint}"
export ACCRL_MODEL_HOST
export OPENAI_BASE_URL="http://${ACCRL_MODEL_HOST}/v1"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-dummy}"

python "$SCRIPT" "$CONFIG" "$@"
