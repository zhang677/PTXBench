#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
PARQUET="$PROJECT/data/glm-5.2-fixit-v2-clean-v5-full-d128.parquet"
RUNS_DIR="$PROJECT/runs"
TRAIN_RUN_TAG="qwen36-27b-glm52-fixit-v2-clean-v5-full-d128-from-v2-final-e1-lr4.65e-4-lora32"
MODEL_PREFIX="qwen36-27b-SFT"
BASE_RUN_DATE="$(python - "$RUNS_DIR" "$TRAIN_RUN_TAG" <<'PY'
import re
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
train_run_tag = sys.argv[2]
pattern = f"{train_run_tag}-Qwen-Qwen3.6-27B-*"
candidates = sorted(
    (path for path in runs_dir.glob(pattern) if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
)
if not candidates:
    raise SystemExit(f"no run dirs matching {pattern!r} under {runs_dir}")
match = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$", candidates[-1].name)
if not match:
    raise SystemExit(f"cannot derive RUN_DATE from {candidates[-1]}")
year, month, day, hour, minute = match.groups()
print(f"{year}-{month}{day}-{hour}{minute}")
PY
)"
RUN_DATE="$BASE_RUN_DATE-v2-clean-v5-full-d128-from-v2-final"
MODEL_NAME="$MODEL_PREFIX-$RUN_DATE"
REMOTE="hyper00"
CONTAINER="sglang-genghan"
REMOTE_PORT="9002"
LOCAL_PORT="30032"
SERVE_SESSION="serve-fixit-v2-clean-v5-full-d128-from-v2-final"
TUNNEL_SESSION="connect-sglang-fixit-v2-clean-v5-full-d128-from-v2-final"
SERVE_TIMEOUT_S="1800"
PROCESS="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit_downstream_process.py"

ARGS=(
  --execute-serve
  --stages serve
  --parquet "$PARQUET"
  --train-run-tag "$TRAIN_RUN_TAG"
  --run-date "$RUN_DATE"
  --model-name "$MODEL_NAME"
  --remote "$REMOTE"
  --container "$CONTAINER"
  --remote-port "$REMOTE_PORT"
  --local-port "$LOCAL_PORT"
  --serve-session "$SERVE_SESSION"
  --tunnel-session "$TUNNEL_SESSION"
  --serve-timeout-s "$SERVE_TIMEOUT_S"
)

python "$PROCESS" "${ARGS[@]}"
