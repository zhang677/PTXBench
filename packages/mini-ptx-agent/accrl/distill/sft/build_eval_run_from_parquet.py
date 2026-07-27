#!/usr/bin/env python3
"""Build a benchmark-compatible eval-run subset from an SFT parquet.

The parquet rows are turn-level training examples. This script turns them back
into an eval-run-like directory by keeping only the target trajectory turns and
success kernels referenced by fix-it rows, or by using run_id/exp_id/turn
metadata for standard SFT rows.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PARQUET = Path(
    "/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm/data/"
    "kimi-k2.7-code-65536.parquet"
)
DEFAULT_SFT_MAPPING = Path("/home/ubuntu/AccRL/benchmark/sft_mapping.csv")
DEFAULT_EVAL_RUNS_ROOT = Path("/home/ubuntu/AccRL-exps/eval_runs")
DEFAULT_BENCHMARK_EXPERIMENTS = Path("/home/ubuntu/AccRL/benchmark/experiments.csv")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read the parquet") from exc
    return pq.read_table(path).to_pylist()


def load_sft_mapping(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"tag", "data", "checkpoint"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
    return {row["tag"]: row for row in rows}


def load_benchmark_workload_index(path: Path) -> dict[str, Any]:
    by_exp_dir = {}
    workloads_by_definition: dict[str, set[str]] = defaultdict(set)
    if not path.is_file():
        return {
            "by_exp_dir": by_exp_dir,
            "workloads_by_definition": {},
        }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_dir = row.get("exp_dir")
            definition = row.get("definition")
            workload = row.get("workload")
            if exp_dir:
                by_exp_dir[str(Path(exp_dir))] = row
            if definition and workload:
                workloads_by_definition[definition].add(workload)

    return {
        "by_exp_dir": by_exp_dir,
        "workloads_by_definition": {
            definition: sorted(workloads)
            for definition, workloads in workloads_by_definition.items()
        },
    }


def checkpoint_paths(mapping_row: dict[str, str]) -> list[Path]:
    return [
        Path(item)
        for item in mapping_row.get("checkpoint", "").split(";")
        if item.strip()
    ]


def tinker_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.name == "checkpoints.jsonl":
        return checkpoint_path.parent
    if checkpoint_path.is_dir():
        return checkpoint_path
    return checkpoint_path.parent


def load_tinker_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing Tinker run config: {config_path}")
    config = load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: expected JSON object")
    return config


def tinker_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = config.get("dataset_builder")
    if not isinstance(dataset_config, dict):
        raise ValueError("Tinker config is missing dataset_builder")
    return dataset_config


def tinker_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = tinker_dataset_config(config)
    common_config = dataset_config.get("common_config")
    if not isinstance(common_config, dict):
        raise ValueError("Tinker config is missing dataset_builder.common_config")
    return {
        "enabled": bool(dataset_config.get("filter_over_max_length", True)),
        "max_length": common_config.get("max_length"),
        "tokenizer": common_config.get("model_name_for_tokenizer") or config.get("model_name"),
    }


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required to reproduce the Tinker length filter") from exc
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def tinker_row_token_count(tokenizer: Any, row: dict[str, Any]) -> int:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{row.get('id', '<unknown>')}: missing messages list")
    return sum(len(tokenizer.encode(message["content"])) for message in messages)


def apply_tinker_filter(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    filter_config = tinker_filter_config(config)
    max_length = filter_config["max_length"]
    tokenizer_name = filter_config["tokenizer"]
    filter_info = {
        "enabled": filter_config["enabled"],
        "max_length": max_length,
        "tokenizer": tokenizer_name,
        "input_rows": len(rows),
        "filtered_rows": 0,
        "kept_rows": len(rows),
    }
    if not filter_config["enabled"] or max_length is None:
        return rows, [], filter_info
    if not tokenizer_name:
        raise ValueError("Tinker config does not specify a tokenizer")

    tokenizer = load_tokenizer(str(tokenizer_name))
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        token_count = tinker_row_token_count(tokenizer, row)
        if token_count <= int(max_length):
            kept.append(row)
            continue
        metadata = row.get("metadata") or {}
        filtered.append(
            {
                "id": row.get("id", ""),
                "tokens": token_count,
                "max_length": int(max_length),
                "exp_id": metadata.get("exp_id") or metadata.get("trajectory_id", ""),
                "correct_kernel_path": metadata.get("correct_kernel_path", ""),
                "reason": "tinker_filter_over_max_length",
            }
        )

    filter_info.update(
        {
            "filtered_rows": len(filtered),
            "kept_rows": len(kept),
        }
    )
    return kept, filtered, filter_info


def exp_index(exp_id: str) -> int:
    if not exp_id.startswith("exp_"):
        raise ValueError(f"expected exp_NNN id, got {exp_id!r}")
    return int(exp_id.removeprefix("exp_"))


def metadata_path(metadata: dict[str, Any], key: str) -> Path:
    value = metadata.get(key)
    if not value:
        raise ValueError(f"row metadata is missing {key}")
    return Path(str(value))


def metadata_str(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if value is None or value == "":
        raise ValueError(f"row metadata is missing {key}")
    return str(value)


def optional_metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    return str(value)


def slug_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._=-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def output_dir_for_group(output_run_dir: Path, definition: str, workload: str) -> Path:
    suffix = f"{slug_component(definition)}-{slug_component(workload)}"
    return output_run_dir.with_name(f"{output_run_dir.name}-{suffix}")


def workload_from_test_path(test_path: object, definition: str | None) -> str | None:
    if not test_path:
        return None
    stem = Path(str(test_path)).stem
    if definition and stem.startswith(f"{definition}_"):
        return stem.removeprefix(f"{definition}_")

    uuid_match = re.search(
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$",
        stem,
    )
    if uuid_match:
        return uuid_match.group(1)
    return None


def resolve_definition_workload(
    *,
    definition: str | None,
    source_run_dir: Path | None,
    plan_entry: dict[str, Any] | None,
    benchmark_index: dict[str, Any],
    row_id: str,
) -> tuple[str, str]:
    benchmark_row = None
    if source_run_dir is not None:
        benchmark_row = benchmark_index["by_exp_dir"].get(str(source_run_dir))

    resolved_definition = (
        definition
        or (str(plan_entry["definition"]) if plan_entry and plan_entry.get("definition") else None)
        or (benchmark_row.get("definition") if benchmark_row else None)
    )
    if not resolved_definition:
        raise ValueError(f"{row_id}: cannot infer definition for workload split")

    workload = None
    if plan_entry:
        workload = plan_entry.get("workload")
        if not workload:
            workload = workload_from_test_path(plan_entry.get("test_path"), resolved_definition)
    if not workload and benchmark_row and benchmark_row.get("definition") == resolved_definition:
        workload = benchmark_row.get("workload")
    if not workload:
        workloads = benchmark_index["workloads_by_definition"].get(resolved_definition, [])
        if len(workloads) == 1:
            workload = workloads[0]

    if not workload:
        raise ValueError(
            f"{row_id}: cannot infer workload for definition {resolved_definition!r}; "
            f"add the source run to {DEFAULT_BENCHMARK_EXPERIMENTS} or include workload/test_path in plan.json"
        )
    return resolved_definition, str(workload)


def detect_row_format(rows: list[dict[str, Any]]) -> str:
    metadata = rows[0].get("metadata") or {}
    if metadata.get("correct_kernel_path") and metadata.get("correct_kernel_version") is not None:
        return "fixit"
    if metadata.get("run_id") and metadata.get("exp_id") and metadata.get("turn") is not None:
        return "standard"
    if metadata.get("exp_id") and metadata.get("turn") is not None:
        return "legacy_standard"
    raise ValueError(
        "cannot infer row format: expected fix-it metadata with correct_kernel_path "
        "or standard metadata with run_id/exp_id/turn or legacy exp_id/turn"
    )


def companion_manifest_path(parquet_path: Path) -> Path | None:
    candidates = [
        parquet_path.with_suffix(".manifest.json"),
        parquet_path.with_suffix(parquet_path.suffix + ".manifest.json"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def infer_source_run_id_from_manifest(parquet_path: Path) -> tuple[str | None, Path | None, Path | None]:
    manifest_path = companion_manifest_path(parquet_path)
    if manifest_path is None:
        return None, None, None

    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: expected JSON object")

    turns_path_value = manifest.get("gemini_turns_path")
    if not turns_path_value:
        return None, manifest_path, None
    turns_path = Path(str(turns_path_value))
    rows = load_jsonl(turns_path)
    run_ids = sorted(
        {
            str(row["run_id"])
            for row in rows
            if isinstance(row, dict) and row.get("run_id") not in (None, "")
        }
    )
    if len(run_ids) == 1:
        return run_ids[0], manifest_path, turns_path
    if not run_ids:
        return None, manifest_path, turns_path
    raise ValueError(
        f"{turns_path}: found multiple run_id values in legacy manifest source: "
        f"{', '.join(run_ids)}; pass --source-run-id"
    )


def resolve_legacy_source_run_id(parquet_path: Path, source_run_id: str | None) -> dict[str, Any]:
    if source_run_id:
        return {
            "source_run_id": source_run_id,
            "source_run_id_source": "cli",
            "source_manifest_path": None,
            "source_turns_path": None,
        }

    inferred_run_id, manifest_path, turns_path = infer_source_run_id_from_manifest(parquet_path)
    if not inferred_run_id:
        detail = f" from {manifest_path}" if manifest_path else ""
        raise ValueError(
            f"legacy standard rows are missing metadata.run_id and no unique run_id "
            f"could be inferred{detail}; pass --source-run-id"
        )
    return {
        "source_run_id": inferred_run_id,
        "source_run_id_source": "manifest",
        "source_manifest_path": str(manifest_path) if manifest_path else None,
        "source_turns_path": str(turns_path) if turns_path else None,
    }


def infer_source_run_dir(row: dict[str, Any]) -> Path:
    metadata = row.get("metadata") or {}
    plan_path = metadata.get("plan_path")
    if plan_path:
        return Path(str(plan_path)).parent

    correct_kernel_path = metadata_path(metadata, "correct_kernel_path")
    # .../<run>/success/exp_NNN/kernel_vM.cu
    try:
        return correct_kernel_path.parents[2]
    except IndexError as exc:
        raise ValueError(f"cannot infer run dir from {correct_kernel_path}") from exc


def row_selection(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    exp_id = metadata.get("exp_id") or metadata.get("trajectory_id")
    if not exp_id:
        raise ValueError(f"{row.get('id', '<unknown>')}: missing exp_id/trajectory_id")
    version = metadata_str(metadata, "correct_kernel_version")
    correct_kernel_path = metadata_path(metadata, "correct_kernel_path")
    return {
        "row_id": row.get("id", ""),
        "exp_id": str(exp_id),
        "version": int(version),
        "correct_kernel_path": correct_kernel_path,
        "wrong_trajectory_path": str(metadata.get("wrong_trajectory_path", "")),
        "wrong_turn": metadata.get("wrong_turn", ""),
    }


def standard_row_selection(row: dict[str, Any], *, default_run_id: str | None = None) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    run_id = optional_metadata_str(metadata, "run_id") or default_run_id
    if not run_id:
        raise ValueError(f"{row.get('id', '<unknown>')}: missing run_id")
    return {
        "row_id": row.get("id", ""),
        "run_id": run_id,
        "exp_id": metadata_str(metadata, "exp_id"),
        "turn": int(metadata_str(metadata, "turn")),
        "definition_name": metadata.get("definition_name", ""),
        "kernel_passed": metadata.get("kernel_passed"),
        "kernel_speedup": metadata.get("kernel_speedup"),
    }


def standard_row_split_key(
    row: dict[str, Any],
    *,
    default_run_id: str | None,
    benchmark_index: dict[str, Any],
) -> tuple[str, str]:
    selection = standard_row_selection(row, default_run_id=default_run_id)
    source_run_dir = DEFAULT_EVAL_RUNS_ROOT / selection["run_id"]
    plan_entry = source_plan_entry(source_run_dir, selection["exp_id"])
    return resolve_definition_workload(
        definition=selection["definition_name"] or None,
        source_run_dir=source_run_dir,
        plan_entry=plan_entry,
        benchmark_index=benchmark_index,
        row_id=selection["row_id"],
    )


def assistant_eval_turns(traj: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    turns: list[tuple[dict[str, Any], dict[str, Any]]] = []
    saw_initial_user = False
    pending_assistant: dict[str, Any] | None = None

    for msg in traj.get("messages", []):
        role = msg.get("role")
        if role == "user":
            if not saw_initial_user:
                saw_initial_user = True
                continue
            if pending_assistant is not None:
                turns.append((pending_assistant, msg))
                pending_assistant = None
        elif role == "assistant" and saw_initial_user:
            pending_assistant = msg
    return turns


def trajectory_prefix_messages(traj: dict[str, Any]) -> list[dict[str, Any]]:
    messages = traj.get("messages")
    if not isinstance(messages, list):
        raise ValueError("trajectory is missing messages list")

    out: list[dict[str, Any]] = []
    saw_initial_user = False
    for msg in messages:
        out.append(copy.deepcopy(msg))
        if msg.get("role") == "user":
            saw_initial_user = True
            break
    if not saw_initial_user:
        raise ValueError("trajectory has no initial user message")
    return out


def set_compact_turn_idx(message: dict[str, Any], *, compact_turn: int, original_turn: int) -> None:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return
    rollout = extra.get("rollout")
    if not isinstance(rollout, dict):
        return
    rollout.setdefault("original_turn_idx", rollout.get("turn_idx"))
    rollout["turn_idx"] = compact_turn
    extra.setdefault("parquet_subset", {})["original_turn_idx"] = original_turn


def build_trimmed_trajectory(
    source_trajectory_path: Path,
    selected_original_turns: list[int],
) -> dict[str, Any]:
    traj = load_json(source_trajectory_path)
    turns = assistant_eval_turns(traj)
    trimmed = copy.deepcopy(traj)
    trimmed["messages"] = trajectory_prefix_messages(traj)

    for compact_turn, original_turn in enumerate(selected_original_turns):
        if original_turn < 0 or original_turn >= len(turns):
            raise ValueError(
                f"{source_trajectory_path}: selected turn {original_turn} out of range "
                f"for {len(turns)} turns"
            )
        assistant_msg, eval_msg = copy.deepcopy(turns[original_turn])
        set_compact_turn_idx(
            assistant_msg,
            compact_turn=compact_turn,
            original_turn=original_turn,
        )
        set_compact_turn_idx(
            eval_msg,
            compact_turn=compact_turn,
            original_turn=original_turn,
        )
        trimmed["messages"].extend([assistant_msg, eval_msg])

    trimmed.setdefault("extra", {})
    if isinstance(trimmed["extra"], dict):
        trimmed["extra"]["parquet_subset"] = {
            "source_trajectory_path": str(source_trajectory_path),
            "selected_original_turns": selected_original_turns,
        }
    return trimmed


def filter_plan(source_plan_path: Path, selected_turn_counts: dict[str, int]) -> dict[str, Any]:
    plan = load_json(source_plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("plan"), list):
        return plan

    selected_indices = {exp_index(exp_id): count for exp_id, count in selected_turn_counts.items()}
    filtered_plan = copy.deepcopy(plan)
    filtered_entries = []
    for entry in plan["plan"]:
        if not isinstance(entry, dict):
            continue
        index = int(entry.get("exp_index", -1))
        if index not in selected_indices:
            continue
        new_entry = copy.deepcopy(entry)
        new_entry.setdefault("original_num_turns", new_entry.get("num_turns"))
        new_entry["num_turns"] = selected_indices[index]
        filtered_entries.append(new_entry)
    filtered_plan["plan"] = filtered_entries
    filtered_plan["parquet_subset"] = {
        "source_plan_path": str(source_plan_path),
        "selected_exp_ids": sorted(selected_turn_counts),
    }
    return filtered_plan


def prepare_output_dir(output_run_dir: Path, *, overwrite: bool) -> None:
    if output_run_dir.exists():
        if not overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace: {output_run_dir}")
        shutil.rmtree(output_run_dir)
    output_run_dir.mkdir(parents=True)


def output_trajectory_id(run_id: str, exp_id: str, *, namespace_run_id: bool) -> str:
    if namespace_run_id:
        return f"{run_id}__{exp_id}"
    return exp_id


def source_plan_entry(source_run_dir: Path, exp_id: str) -> dict[str, Any] | None:
    plan_path = source_run_dir / "plan.json"
    if not plan_path.is_file():
        return None
    plan = load_json(plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("plan"), list):
        return None
    target_index = exp_index(exp_id)
    for entry in plan["plan"]:
        if isinstance(entry, dict) and int(entry.get("exp_index", -1)) == target_index:
            return copy.deepcopy(entry)
    return None


def aggregate_split_manifests(
    *,
    parquet_path: Path,
    output_run_dir: Path,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "builder": "accrl.distill.sft.build_eval_run_from_parquet",
        "source_parquet": str(parquet_path),
        "output_run_dir_prefix": str(output_run_dir),
        "split_by_workload": True,
        "num_output_run_dirs": len(manifests),
        "selected_rows": sum(int(manifest["selected_rows"]) for manifest in manifests),
        "output_trajectories": sum(int(manifest["output_trajectories"]) for manifest in manifests),
        "output_turns": sum(int(manifest["output_turns"]) for manifest in manifests),
        "outputs": [
            {
                "definition": manifest.get("split_group", {}).get("definition"),
                "workload": manifest.get("split_group", {}).get("workload"),
                "output_run_dir": manifest["output_run_dir"],
                "selected_rows": manifest["selected_rows"],
                "output_trajectories": manifest["output_trajectories"],
                "output_turns": manifest["output_turns"],
                "manifest": str(
                    Path(manifest["output_run_dir"]) / "parquet_subset_manifest.json"
                ),
            }
            for manifest in manifests
        ],
    }


def build_standard_subset(
    *,
    rows: list[dict[str, Any]],
    parquet_path: Path,
    output_run_dir: Path,
    overwrite: bool,
    input_rows: int,
    tinker_filtered: list[dict[str, Any]],
    tinker_filter_info: dict[str, Any] | None,
    sft_mapping_tag: str | None,
    sft_mapping_path: Path | None,
    checkpoint_path: Path | None,
    tinker_config_path: Path | None,
    row_format: str,
    legacy_source: dict[str, Any] | None,
    split_group: dict[str, str] | None = None,
) -> dict[str, Any]:
    prepare_output_dir(output_run_dir, overwrite=overwrite)

    selections_by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_source_turn: set[tuple[str, str, int]] = set()
    default_run_id = legacy_source["source_run_id"] if legacy_source else None
    for row in rows:
        selection = standard_row_selection(row, default_run_id=default_run_id)
        key = (selection["run_id"], selection["exp_id"], selection["turn"])
        if key in seen_source_turn:
            raise ValueError(f"duplicate parquet rows for {key}")
        seen_source_turn.add(key)
        selections_by_source[(selection["run_id"], selection["exp_id"])].append(selection)

    run_ids = {run_id for run_id, _exp_id in selections_by_source}
    namespace_run_id = len(run_ids) > 1
    manifest_records = []
    plan_entries = []
    total_selected_turns = 0

    for plan_index, ((run_id, exp_id), selections) in enumerate(
        sorted(selections_by_source.items(), key=lambda item: (item[0][0], exp_index(item[0][1])))
    ):
        source_run_dir = DEFAULT_EVAL_RUNS_ROOT / run_id
        source_trajectory_path = source_run_dir / "trajectories" / f"{exp_id}.json"
        if not source_trajectory_path.is_file():
            raise FileNotFoundError(f"missing source trajectory: {source_trajectory_path}")

        selected_turns = sorted(int(selection["turn"]) for selection in selections)
        output_id = output_trajectory_id(run_id, exp_id, namespace_run_id=namespace_run_id)
        trimmed = build_trimmed_trajectory(source_trajectory_path, selected_turns)
        trimmed.setdefault("extra", {})
        if isinstance(trimmed["extra"], dict):
            trimmed["extra"]["parquet_subset"].update(
                {
                    "source_run_id": run_id,
                    "source_exp_id": exp_id,
                    "output_trajectory_id": output_id,
                }
            )
        write_json(output_run_dir / "trajectories" / f"{output_id}.json", trimmed)
        total_selected_turns += len(selected_turns)

        original_plan_entry = source_plan_entry(source_run_dir, exp_id)
        plan_entry = {
            "exp_index": plan_index,
            "num_turns": len(selected_turns),
            "output_trajectory_id": output_id,
            "source_run_id": run_id,
            "source_exp_id": exp_id,
            "selected_original_turns": selected_turns,
        }
        if original_plan_entry:
            plan_entry.update(
                {
                    "num_trajectories": original_plan_entry.get("num_trajectories"),
                    "target_speedup": original_plan_entry.get("target_speedup"),
                    "prompt_tag": original_plan_entry.get("prompt_tag"),
                    "original_plan_entry": original_plan_entry,
                }
            )
        plan_entries.append(plan_entry)

        selections_by_turn = {int(selection["turn"]): selection for selection in selections}
        for compact_turn, original_turn in enumerate(selected_turns):
            selection = selections_by_turn[original_turn]
            manifest_records.append(
                {
                    "row_id": selection["row_id"],
                    "run_id": run_id,
                    "exp_id": exp_id,
                    "output_trajectory_id": output_id,
                    "definition_name": selection["definition_name"],
                    "original_turn": original_turn,
                    "compact_turn": compact_turn,
                    "kernel_passed": selection["kernel_passed"],
                    "kernel_speedup": selection["kernel_speedup"],
                }
            )

    plan = {
        "config": {
            "source": "standard_sft_parquet_subset",
            "source_run_ids": sorted(run_ids),
        },
        "plan": plan_entries,
        "parquet_subset": {
            "source_parquet": str(parquet_path),
            "trajectory_ids_namespaced_by_run_id": namespace_run_id,
            "split_group": split_group,
        },
    }
    write_json(output_run_dir / "plan.json", plan)

    manifest = {
        "builder": "accrl.distill.sft.build_eval_run_from_parquet",
        "row_format": row_format,
        "source_parquet": str(parquet_path),
        "source_parquet_sha256": file_sha256(parquet_path),
        "split_group": split_group,
        "legacy_source": legacy_source,
        "sft_mapping_tag": sft_mapping_tag,
        "sft_mapping_path": str(sft_mapping_path) if sft_mapping_path else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "tinker_config_path": str(tinker_config_path) if tinker_config_path else None,
        "tinker_filter": tinker_filter_info,
        "tinker_filtered": tinker_filtered,
        "source_run_dir": None,
        "source_run_dirs": [str(DEFAULT_EVAL_RUNS_ROOT / run_id) for run_id in sorted(run_ids)],
        "source_plan_path": None,
        "output_run_dir": str(output_run_dir),
        "parquet_rows": input_rows,
        "selected_rows": len(rows),
        "output_trajectories": len(selections_by_source),
        "output_turns": total_selected_turns,
        "records": manifest_records,
    }
    write_json(output_run_dir / "parquet_subset_manifest.json", manifest)
    return manifest


def build_fixit_subset(
    *,
    rows: list[dict[str, Any]],
    parquet_path: Path,
    output_run_dir: Path,
    overwrite: bool,
    input_rows: int,
    tinker_filtered: list[dict[str, Any]],
    tinker_filter_info: dict[str, Any] | None,
    sft_mapping_tag: str | None,
    sft_mapping_path: Path | None,
    checkpoint_path: Path | None,
    tinker_config_path: Path | None,
    split_group: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_run_dir = infer_source_run_dir(rows[0])
    source_plan_path = source_run_dir / "plan.json"
    if not source_plan_path.is_file():
        raise FileNotFoundError(f"missing source plan.json: {source_plan_path}")

    selections_by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_exp_version: set[tuple[str, int]] = set()
    for row in rows:
        selection = row_selection(row)
        key = (selection["exp_id"], selection["version"])
        if key in seen_exp_version:
            raise ValueError(f"duplicate parquet rows for {key}")
        seen_exp_version.add(key)
        selections_by_exp[selection["exp_id"]].append(selection)

    prepare_output_dir(output_run_dir, overwrite=overwrite)

    plan = filter_plan(
        source_plan_path,
        {exp_id: len(selections) for exp_id, selections in selections_by_exp.items()},
    )
    if isinstance(plan, dict):
        plan.setdefault("parquet_subset", {})["split_group"] = split_group
    write_json(output_run_dir / "plan.json", plan)

    manifest_records = []
    total_selected_turns = 0
    for exp_id in sorted(selections_by_exp, key=exp_index):
        selections = selections_by_exp[exp_id]
        source_trajectory_path = source_run_dir / "trajectories" / f"{exp_id}.json"
        source_record_path = source_run_dir / "success" / exp_id / "record.json"
        if not source_trajectory_path.is_file():
            raise FileNotFoundError(f"missing source trajectory: {source_trajectory_path}")
        if not source_record_path.is_file():
            raise FileNotFoundError(f"missing source success record: {source_record_path}")

        source_records = load_json(source_record_path)
        if not isinstance(source_records, list):
            raise ValueError(f"{source_record_path}: expected list")
        records_by_version = {int(record["version"]): record for record in source_records}

        selected_entries = []
        selected_turns = []
        for selection in selections:
            version = int(selection["version"])
            if version not in records_by_version:
                raise ValueError(f"{source_record_path}: missing version {version}")
            record = copy.deepcopy(records_by_version[version])
            original_turn = int(record["turn"])
            selected_entries.append((original_turn, version, selection, record))

        selected_entries.sort(key=lambda item: (item[0], item[1]))
        compact_records = []
        for compact_turn, (original_turn, version, selection, record) in enumerate(selected_entries):
            record.setdefault("original_turn", original_turn)
            record["turn"] = compact_turn
            compact_records.append(record)
            selected_turns.append(original_turn)

            source_kernel = selection["correct_kernel_path"]
            if not source_kernel.is_file():
                raise FileNotFoundError(f"missing source kernel: {source_kernel}")
            dest_kernel = output_run_dir / "success" / exp_id / f"kernel_v{version}.cu"
            dest_kernel.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_kernel, dest_kernel)

            manifest_records.append(
                {
                    "row_id": selection["row_id"],
                    "exp_id": exp_id,
                    "version": version,
                    "original_turn": original_turn,
                    "compact_turn": compact_turn,
                    "source_kernel_path": str(source_kernel),
                    "output_kernel_path": str(dest_kernel),
                    "wrong_trajectory_path": selection["wrong_trajectory_path"],
                    "wrong_turn": selection["wrong_turn"],
                }
            )

        trimmed = build_trimmed_trajectory(source_trajectory_path, selected_turns)
        write_json(output_run_dir / "trajectories" / f"{exp_id}.json", trimmed)
        write_json(output_run_dir / "success" / exp_id / "record.json", compact_records)
        total_selected_turns += len(selected_turns)

    manifest = {
        "builder": "accrl.distill.sft.build_eval_run_from_parquet",
        "row_format": "fixit",
        "source_parquet": str(parquet_path),
        "source_parquet_sha256": file_sha256(parquet_path),
        "split_group": split_group,
        "sft_mapping_tag": sft_mapping_tag,
        "sft_mapping_path": str(sft_mapping_path) if sft_mapping_path else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "tinker_config_path": str(tinker_config_path) if tinker_config_path else None,
        "tinker_filter": tinker_filter_info,
        "tinker_filtered": tinker_filtered,
        "source_run_dir": str(source_run_dir),
        "source_plan_path": str(source_plan_path),
        "output_run_dir": str(output_run_dir),
        "parquet_rows": input_rows,
        "selected_rows": len(rows),
        "output_trajectories": len(selections_by_exp),
        "output_turns": total_selected_turns,
        "records": manifest_records,
    }
    write_json(output_run_dir / "parquet_subset_manifest.json", manifest)
    return manifest


def build_subset(
    *,
    parquet_path: Path,
    output_run_dir: Path,
    overwrite: bool,
    tinker_config_path: Path | None = None,
    apply_filter: bool = False,
    sft_mapping_tag: str | None = None,
    sft_mapping_path: Path | None = None,
    checkpoint_path: Path | None = None,
    source_run_id: str | None = None,
    split_by_workload: bool = True,
    benchmark_experiments_path: Path = DEFAULT_BENCHMARK_EXPERIMENTS,
) -> dict[str, Any]:
    rows = load_parquet_rows(parquet_path)
    if not rows:
        raise SystemExit(f"no rows in {parquet_path}")

    tinker_config = load_tinker_config(tinker_config_path.parent) if tinker_config_path else None
    tinker_filtered: list[dict[str, Any]] = []
    tinker_filter_info: dict[str, Any] | None = None
    input_rows = len(rows)
    if apply_filter:
        if tinker_config is None:
            raise ValueError("--apply-tinker-filter requires a Tinker run config")
        rows, tinker_filtered, tinker_filter_info = apply_tinker_filter(
            rows,
            config=tinker_config,
        )
        if not rows:
            raise SystemExit("no rows left after applying Tinker filter")

    row_format = detect_row_format(rows)
    if row_format in {"standard", "legacy_standard"}:
        legacy_source = (
            resolve_legacy_source_run_id(parquet_path, source_run_id)
            if row_format == "legacy_standard"
            else None
        )
        if split_by_workload:
            default_run_id = legacy_source["source_run_id"] if legacy_source else None
            benchmark_index = load_benchmark_workload_index(benchmark_experiments_path)
            grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                key = standard_row_split_key(
                    row,
                    default_run_id=default_run_id,
                    benchmark_index=benchmark_index,
                )
                grouped_rows[key].append(row)

            manifests = []
            for definition, workload in sorted(grouped_rows):
                group_output_run_dir = output_dir_for_group(output_run_dir, definition, workload)
                manifests.append(
                    build_standard_subset(
                        rows=grouped_rows[(definition, workload)],
                        parquet_path=parquet_path,
                        output_run_dir=group_output_run_dir,
                        overwrite=overwrite,
                        input_rows=input_rows,
                        tinker_filtered=tinker_filtered,
                        tinker_filter_info=tinker_filter_info,
                        sft_mapping_tag=sft_mapping_tag,
                        sft_mapping_path=sft_mapping_path,
                        checkpoint_path=checkpoint_path,
                        tinker_config_path=tinker_config_path,
                        row_format=row_format,
                        legacy_source=legacy_source,
                        split_group={
                            "definition": definition,
                            "workload": workload,
                        },
                    )
                )
            return aggregate_split_manifests(
                parquet_path=parquet_path,
                output_run_dir=output_run_dir,
                manifests=manifests,
            )

        return build_standard_subset(
            rows=rows,
            parquet_path=parquet_path,
            output_run_dir=output_run_dir,
            overwrite=overwrite,
            input_rows=input_rows,
            tinker_filtered=tinker_filtered,
            tinker_filter_info=tinker_filter_info,
            sft_mapping_tag=sft_mapping_tag,
            sft_mapping_path=sft_mapping_path,
            checkpoint_path=checkpoint_path,
            tinker_config_path=tinker_config_path,
            row_format=row_format,
            legacy_source=legacy_source,
            split_group=None,
        )

    if split_by_workload:
        benchmark_index = load_benchmark_workload_index(benchmark_experiments_path)
        grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            selection = row_selection(row)
            source_run_dir = infer_source_run_dir(row)
            exp_id = selection["exp_id"]
            plan_entry = source_plan_entry(source_run_dir, exp_id)
            key = resolve_definition_workload(
                definition=None,
                source_run_dir=source_run_dir,
                plan_entry=plan_entry,
                benchmark_index=benchmark_index,
                row_id=selection["row_id"] or exp_id,
            )
            grouped_rows[key].append(row)

        manifests = []
        for definition, workload in sorted(grouped_rows):
            group_output_run_dir = output_dir_for_group(output_run_dir, definition, workload)
            manifests.append(
                build_fixit_subset(
                    rows=grouped_rows[(definition, workload)],
                    parquet_path=parquet_path,
                    output_run_dir=group_output_run_dir,
                    overwrite=overwrite,
                    input_rows=input_rows,
                    tinker_filtered=tinker_filtered,
                    tinker_filter_info=tinker_filter_info,
                    sft_mapping_tag=sft_mapping_tag,
                    sft_mapping_path=sft_mapping_path,
                    checkpoint_path=checkpoint_path,
                    tinker_config_path=tinker_config_path,
                    split_group={
                        "definition": definition,
                        "workload": workload,
                    },
                )
            )
        return aggregate_split_manifests(
            parquet_path=parquet_path,
            output_run_dir=output_run_dir,
            manifests=manifests,
        )

    return build_fixit_subset(
        rows=rows,
        parquet_path=parquet_path,
        output_run_dir=output_run_dir,
        overwrite=overwrite,
        input_rows=input_rows,
        tinker_filtered=tinker_filtered,
        tinker_filter_info=tinker_filter_info,
        sft_mapping_tag=sft_mapping_tag,
        sft_mapping_path=sft_mapping_path,
        checkpoint_path=checkpoint_path,
        tinker_config_path=tinker_config_path,
        split_group=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument(
        "--sft-mapping",
        type=Path,
        default=DEFAULT_SFT_MAPPING,
        help="CSV with tag,data,checkpoint columns",
    )
    parser.add_argument(
        "--sft-tag",
        default=None,
        help="Resolve --parquet and Tinker run from --sft-mapping",
    )
    parser.add_argument(
        "--checkpoint-index",
        type=int,
        default=0,
        help="Checkpoint entry to use when a mapping row has semicolon-separated checkpoints",
    )
    parser.add_argument(
        "--apply-tinker-filter",
        action="store_true",
        help="Reproduce the selected Tinker run's filter_over_max_length prefilter.",
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Source eval-run id for legacy standard parquets that lack metadata.run_id.",
    )
    parser.add_argument(
        "--benchmark-experiments",
        type=Path,
        default=DEFAULT_BENCHMARK_EXPERIMENTS,
        help="Benchmark experiments.csv used to infer workload UUIDs.",
    )
    parser.add_argument(
        "--single-output-dir",
        action="store_true",
        help="Write exactly --output-run-dir instead of splitting by definition/workload.",
    )
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = args.parquet
    tinker_config_path = None
    checkpoint_path = None

    if args.sft_tag:
        mapping = load_sft_mapping(args.sft_mapping)
        if args.sft_tag not in mapping:
            raise SystemExit(f"{args.sft_mapping}: unknown SFT tag {args.sft_tag!r}")
        mapping_row = mapping[args.sft_tag]
        if parquet_path is None:
            parquet_path = Path(mapping_row["data"])
        checkpoints = checkpoint_paths(mapping_row)
        if not checkpoints:
            raise SystemExit(f"{args.sft_mapping}: tag {args.sft_tag!r} has no checkpoint path")
        if args.checkpoint_index < 0 or args.checkpoint_index >= len(checkpoints):
            raise SystemExit(
                f"--checkpoint-index {args.checkpoint_index} out of range for "
                f"{len(checkpoints)} checkpoint path(s)"
            )
        checkpoint_path = checkpoints[args.checkpoint_index]
        tinker_config_path = tinker_run_dir_from_checkpoint(checkpoint_path) / "config.json"

    if parquet_path is None:
        parquet_path = DEFAULT_PARQUET

    manifest = build_subset(
        parquet_path=parquet_path,
        output_run_dir=args.output_run_dir,
        overwrite=args.overwrite,
        tinker_config_path=tinker_config_path,
        apply_filter=args.apply_tinker_filter,
        sft_mapping_tag=args.sft_tag,
        sft_mapping_path=args.sft_mapping if args.sft_tag else None,
        checkpoint_path=checkpoint_path,
        source_run_id=args.source_run_id,
        split_by_workload=not args.single_output_dir,
        benchmark_experiments_path=args.benchmark_experiments,
    )
    print(f"source parquet:      {manifest['source_parquet']}")
    if manifest.get("split_by_workload"):
        print(f"output run prefix:   {manifest['output_run_dir_prefix']}")
        print(f"output run dirs:     {manifest['num_output_run_dirs']}")
        print(f"selected rows:       {manifest['selected_rows']}")
        print(f"output trajectories: {manifest['output_trajectories']}")
        print(f"output turns:        {manifest['output_turns']}")
        for output in manifest["outputs"]:
            print(
                "  "
                f"{output['definition']} {output['workload']}: "
                f"{output['output_run_dir']} "
                f"({output['selected_rows']} rows, {output['output_trajectories']} trajectories)"
            )
        return

    if manifest["sft_mapping_tag"]:
        print(f"sft mapping tag:     {manifest['sft_mapping_tag']}")
    if manifest["tinker_config_path"]:
        print(f"tinker config:       {manifest['tinker_config_path']}")
    print(f"output run dir:      {manifest['output_run_dir']}")
    print(f"parquet rows:        {manifest['parquet_rows']}")
    print(f"selected rows:       {manifest['selected_rows']}")
    if manifest["tinker_filter"]:
        print(f"tinker filtered:     {manifest['tinker_filter']['filtered_rows']}")
    print(f"output trajectories: {manifest['output_trajectories']}")
    print(f"output turns:        {manifest['output_turns']}")
    print(f"manifest:            {Path(manifest['output_run_dir']) / 'parquet_subset_manifest.json'}")


if __name__ == "__main__":
    main()
