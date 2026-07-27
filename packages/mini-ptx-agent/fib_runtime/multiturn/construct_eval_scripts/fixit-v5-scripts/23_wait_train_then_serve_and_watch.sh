#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
RUNS_DIR="$PROJECT/runs"
TRAIN_RUN_TAG="qwen36-27b-glm52-fixit-v2-clean-v5-full-d128-from-v2-final-e1-lr4.65e-4-lora32"
SERVE_SCRIPT="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/21_serve_remote_v2_clean_v5_full.sh"
WATCH_SCRIPT="/home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/22_watch_v2_clean_v5_full_5defs_eval.sh"
POLL_S="30"
TRAIN_TIMEOUT_S="172800"

for script in "$SERVE_SCRIPT" "$WATCH_SCRIPT"; do
  if [[ ! -x "$script" ]]; then
    echo "Missing executable script: $script" >&2
    exit 1
  fi
done

latest_final_run() {
  python - "$RUNS_DIR" "$TRAIN_RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
train_run_tag = sys.argv[2]
pattern = f"{train_run_tag}-Qwen-Qwen3.6-27B-*"
candidates = sorted(
    (path for path in runs_dir.glob(pattern) if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not candidates:
    raise SystemExit(1)

run_dir = candidates[0]
checkpoints = run_dir / "checkpoints.jsonl"
if not checkpoints.is_file():
    raise SystemExit(1)

for line in checkpoints.read_text().splitlines():
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("name") == "final":
        print(run_dir)
        raise SystemExit(0)
    if str(record.get("state_path", "")).endswith("/weights/final"):
        print(run_dir)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

training_is_running() {
  ps -eo args= | grep -F "tinker_sft_train.py" | grep -F "run_tag=$TRAIN_RUN_TAG" >/dev/null
}

start_time="$(date +%s)"
last_status_time=0
training_seen=0
inactive_checks=0

echo "Waiting for final checkpoint for TRAIN_RUN_TAG=$TRAIN_RUN_TAG"
while true; do
  if run_dir="$(latest_final_run)"; then
    echo "Training complete: $run_dir"
    break
  fi

  now="$(date +%s)"
  if training_is_running; then
    training_seen=1
    inactive_checks=0
  elif (( training_seen == 1 )); then
    inactive_checks=$((inactive_checks + 1))
  fi
  if (( inactive_checks >= 3 )); then
    echo "Training process exited without a final checkpoint for $TRAIN_RUN_TAG" >&2
    exit 1
  fi

  elapsed=$((now - start_time))
  if (( elapsed >= TRAIN_TIMEOUT_S )); then
    echo "Timed out after ${TRAIN_TIMEOUT_S}s waiting for final checkpoint" >&2
    exit 1
  fi
  if (( now - last_status_time >= 600 )); then
    echo "Still waiting: elapsed=${elapsed}s training_seen=$training_seen"
    last_status_time="$now"
  fi
  sleep "$POLL_S"
done

echo "Launching serve stage: $SERVE_SCRIPT"
"$SERVE_SCRIPT"

echo "Serve stage is ready; launching evaluation watcher: $WATCH_SCRIPT"
exec "$WATCH_SCRIPT"
