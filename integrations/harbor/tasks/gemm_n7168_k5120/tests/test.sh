#!/bin/bash
set -u

result_path=/logs/verifier/final-evaluation.json
reward_path=/logs/verifier/reward.txt

ptxbench eval /workspace/kernel.cu --task-config /tests/task.json --json >"$result_path"
command_status=$?

python - "$result_path" "$reward_path" "$command_status" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
reward_path = Path(sys.argv[2])
command_status = int(sys.argv[3])

reward = 0
if command_status == 0 and result_path.is_file():
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        result = {}
    if result.get("schema") == "ptxbench.eval.v1" and result.get("all_passed") is True:
        reward = 1

reward_path.write_text(f"{reward}\n")
PY

exit 0
