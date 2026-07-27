#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm"
V2_CLEAN="$PROJECT/data/glm-5.2-mha-d128-4def-full-no-myreasoning.parquet"
V5_FULL="$PROJECT/data/glm-5.2-fixit-v5-full.parquet"
OUTPUT="$PROJECT/data/glm-5.2-fixit-v2-clean-v5-full-d128.parquet"
MANIFEST="$OUTPUT.combine_manifest.json"
SHUFFLE_SEED="42"

python - "$V2_CLEAN" "$V5_FULL" "$OUTPUT" "$MANIFEST" "$SHUFFLE_SEED" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


v2_path, v5_path, output_path, manifest_path = map(Path, sys.argv[1:5])
shuffle_seed = int(sys.argv[5])
expected_columns = ["id", "messages", "metadata"]
expected_roles = ("system", "user", "assistant", "user", "assistant")
expected_masks = (0, 0, 0, 0, 1)
selected_definitions = {
    "mha_with_lse_d128",
    "mha_with_lse_d128_causal",
    "mha_bwd_d128",
    "mha_bwd_d128_causal",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(frame: pd.DataFrame, label: str) -> None:
    if list(frame.columns) != expected_columns:
        raise SystemExit(
            f"{label}: expected columns {expected_columns}, got {list(frame.columns)}"
        )
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].tolist()[:10]
        raise SystemExit(f"{label}: missing or duplicate ids: {duplicates}")

    bad_roles = []
    bad_masks = []
    forbidden_reasoning = []
    for row_index, messages in enumerate(frame["messages"]):
        roles = tuple(str(message.get("role")) for message in messages)
        masks = tuple(int(message.get("step_loss_mask", -1)) for message in messages)
        if roles != expected_roles:
            bad_roles.append((row_index, roles))
        if masks != expected_masks:
            bad_masks.append((row_index, masks))
        if any("<my_reasoning>" in str(message.get("content", "")) for message in messages):
            forbidden_reasoning.append(row_index)
    if bad_roles:
        raise SystemExit(f"{label}: invalid message-role sequences: {bad_roles[:10]}")
    if bad_masks:
        raise SystemExit(f"{label}: invalid step_loss_mask sequences: {bad_masks[:10]}")
    if forbidden_reasoning:
        raise SystemExit(
            f"{label}: found forbidden <my_reasoning> delimiters in rows "
            f"{forbidden_reasoning[:10]}"
        )


for source in (v2_path, v5_path):
    if not source.is_file():
        raise SystemExit(f"missing source parquet: {source}")

v2_input = pd.read_parquet(v2_path)
v5_input = pd.read_parquet(v5_path)
validate(v2_input, "fixit-v2-clean input")
validate(v5_input, "fixit-v5-full input")


def select_d128(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        frame["metadata"].map(
            lambda metadata: str(metadata.get("definition", ""))
            in selected_definitions
        )
    ]
    return selected.reset_index(drop=True)


v2 = select_d128(v2_input)
v5 = select_d128(v5_input)
validate(v2, "fixit-v2-clean selected d128 rows")
validate(v5, "fixit-v5-full selected d128 rows")

selected_definition_values = {
    str(metadata.get("definition", ""))
    for frame in (v2, v5)
    for metadata in frame["metadata"]
}
if selected_definition_values != selected_definitions:
    raise SystemExit(
        "selected rows do not cover exactly the four d128 definitions: "
        f"expected={sorted(selected_definitions)} "
        f"got={sorted(selected_definition_values)}"
    )

id_overlap = sorted(set(v2["id"]) & set(v5["id"]))
if id_overlap:
    raise SystemExit(f"source parquet id overlap: {id_overlap[:10]}")

combined = pd.concat([v2, v5], ignore_index=True)
combined = combined.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
validate(combined, "combined")

output_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = output_path.with_name(f".{output_path.name}.tmp")
combined.to_parquet(temporary_path, index=False)
os.replace(temporary_path, output_path)

readback = pd.read_parquet(output_path)
validate(readback, "combined readback")
if set(readback["id"]) != set(combined["id"]):
    raise SystemExit("combined parquet readback changed the id set")

definition_counts = Counter(
    str(metadata.get("definition", "")) for metadata in readback["metadata"]
)
source_label_counts = Counter(
    str(metadata.get("source_label", "")) for metadata in readback["metadata"]
)
manifest = {
    "format_version": 1,
    "operation": "filter_four_d128_definitions_then_concatenate_and_shuffle",
    "shuffle_seed": shuffle_seed,
    "selected_definitions": sorted(selected_definitions),
    "sources": [
        {
            "label": "fixit-v2-clean",
            "path": str(v2_path),
            "input_rows": len(v2_input),
            "selected_rows": len(v2),
            "dropped_rows": len(v2_input) - len(v2),
            "sha256": sha256(v2_path),
        },
        {
            "label": "fixit-v5-full",
            "path": str(v5_path),
            "input_rows": len(v5_input),
            "selected_rows": len(v5),
            "dropped_rows": len(v5_input) - len(v5),
            "sha256": sha256(v5_path),
        },
    ],
    "output": {
        "path": str(output_path),
        "rows": len(readback),
        "unique_ids": int(readback["id"].nunique()),
        "sha256": sha256(output_path),
    },
    "message_roles": list(expected_roles),
    "step_loss_masks": list(expected_masks),
    "definition_counts": dict(sorted(definition_counts.items())),
    "source_label_counts": dict(sorted(source_label_counts.items())),
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

print(json.dumps(manifest, indent=2, sort_keys=True))
PY
