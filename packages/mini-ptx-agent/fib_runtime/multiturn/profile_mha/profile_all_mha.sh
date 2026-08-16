#!/usr/bin/env bash
# Collect perf.csv for every MHA workload in /workspace/accrl-training through
# the heavy-mounted fib-profile container.
set -euo pipefail

CONTAINER="${CONTAINER:-fib-profile-heavy}"
DATASET_DIR="${DATASET_DIR:-/home/ubuntu/accrl-training-heavy}"
CONTAINER_DATASET_DIR="${CONTAINER_DATASET_DIR:-/workspace/accrl-training}"
MULTITURN_DIR="${MULTITURN_DIR:-/home/ubuntu/AccRL/fib_runtime/multiturn}"
PERF_DIR="${PERF_DIR:-$DATASET_DIR/perf_parts}"
OUT_CSV="${OUT_CSV:-$DATASET_DIR/perf.csv}"
DEADLINE_S="${DEADLINE_S:-1200}"

mkdir -p "$PERF_DIR"

docker exec "$CONTAINER" bash -lc 'curl -fsS --max-time 10 http://localhost:10000/health >/dev/null'

docker cp "$MULTITURN_DIR/mha-with-lse-problems/scripts/verify_via_service.py" "$CONTAINER:/tmp/verify_mha_with_lse.py"
docker cp "$MULTITURN_DIR/mha-bwd-problems/scripts/verify_via_service.py" "$CONTAINER:/tmp/verify_mha_bwd.py"
docker cp "$MULTITURN_DIR/fp8-mha-with-lse-problems/scripts/verify_via_service.py" "$CONTAINER:/tmp/verify_fp8_mha_with_lse.py"

run_suite() {
    local script="$1"
    local out="$2"
    docker exec "$CONTAINER" bash -lc \
        "PYTHONPATH=/workspace/AccRL PROFILE_BASE_URL=http://localhost:10000 /workspace/acc/bin/python '$script' --deadline-s '$DEADLINE_S' --perf-csv '$out'"
}

run_suite /tmp/verify_mha_with_lse.py "$CONTAINER_DATASET_DIR/perf_parts/mha_with_lse_perf.csv"
run_suite /tmp/verify_mha_bwd.py "$CONTAINER_DATASET_DIR/perf_parts/mha_bwd_perf.csv"
run_suite /tmp/verify_fp8_mha_with_lse.py "$CONTAINER_DATASET_DIR/perf_parts/fp8_mha_with_lse_perf.csv"

python - <<'PY'
import csv
from pathlib import Path

root = Path("/home/ubuntu/accrl-training-heavy")
parts = [
    root / "perf_parts" / "mha_with_lse_perf.csv",
    root / "perf_parts" / "mha_bwd_perf.csv",
    root / "perf_parts" / "fp8_mha_with_lse_perf.csv",
]
out = root / "perf.csv"
rows = []
for part in parts:
    with part.open(newline="") as f:
        for row in csv.DictReader(f):
            row["suite"] = part.stem.removesuffix("_perf")
            rows.append(row)

fieldnames = ["suite", "definition_name", "seq_len", "latency_ms", "tflops", "workload_uuid"]
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out} rows={len(rows)}")
PY

python - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

root = Path("/home/ubuntu/accrl-training-heavy")
patterns = ["mha_with_lse*.jsonl", "mha_bwd*.jsonl", "fp8_mha_with_lse*.jsonl"]
expected = {}
for pattern in patterns:
    for workload_file in sorted((root / "workloads" / "attention").glob(pattern)):
        for line in workload_file.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            workload = item["workload"]
            seq_len = workload["axes"].get("S", workload["axes"].get("L"))
            expected[workload["uuid"]] = (item["definition"], str(seq_len))

with (root / "perf.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

seen = {row["workload_uuid"]: (row["definition_name"], row["seq_len"]) for row in rows}
duplicates = [(uuid, n) for uuid, n in Counter(row["workload_uuid"] for row in rows).items() if n != 1]
missing = sorted(set(expected) - set(seen))
extra = sorted(set(seen) - set(expected))
mismatched = sorted(
    (uuid, expected[uuid], seen[uuid])
    for uuid in set(expected) & set(seen)
    if expected[uuid] != seen[uuid]
)

print(f"expected={len(expected)} rows={len(rows)} unique={len(seen)}")
if missing or extra or mismatched or duplicates:
    print(f"missing={missing[:5]}")
    print(f"extra={extra[:5]}")
    print(f"mismatched={mismatched[:5]}")
    print(f"duplicates={duplicates[:5]}")
    raise SystemExit(1)
PY

echo "done: $OUT_CSV"
