"""Shared helpers for multiturn parallel launchers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


TINKER_BASE_MODEL_NAMES = {"inkling"}
TINKER_CHECKPOINT_MODEL_NAMES = {"Qwen3.5-35B-A3B-tinker"}
TINKER_MODEL_NAMES = TINKER_BASE_MODEL_NAMES | TINKER_CHECKPOINT_MODEL_NAMES


class InfraAbort(RuntimeError):
    pass


def kill_container(workspace: Path) -> None:
    cid_file = workspace / ".container_id"
    try:
        if not cid_file.exists():
            return
        container_id = cid_file.read_text().strip()
        if container_id:
            subprocess.run(["docker", "kill", container_id], capture_output=True, timeout=30)
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=30)
    except Exception:
        pass


def load_config(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        config_items = json.load(f)
    if not isinstance(config_items, list) or not config_items:
        raise ValueError(f"{path} must be a non-empty JSON list.")
    return config_items


def run_targets_from_args(args: Any) -> list[dict[str, str]] | None:
    definitions = getattr(args, "definitions", None)
    test_paths = getattr(args, "test_paths", None)
    configs = getattr(args, "configs", None)
    output_roots = getattr(args, "output_roots", None)
    single_definition = getattr(args, "definition", None)
    single_test_path = getattr(args, "test_path", None)

    plural_args = [definitions, test_paths, configs, output_roots]
    if not any(value for value in plural_args):
        if single_definition or single_test_path:
            if not single_definition or not single_test_path:
                raise ValueError("--definition and --test-path must be provided together")
            target = {
                "definition": str(single_definition),
                "test_path": str(single_test_path),
            }
            return [target]
        return None

    if single_definition or single_test_path:
        raise ValueError(
            "--definition/--test-path cannot be mixed with --definitions/--test-paths"
        )
    if not definitions or not test_paths:
        raise ValueError("--definitions and --test-paths are required for multi-target runs")
    if configs is None:
        raise ValueError("--configs is required for multi-target runs")
    if output_roots is None:
        raise ValueError("--output-roots is required for multi-target runs")
    if len(definitions) != len(test_paths):
        raise ValueError(
            f"--definitions has {len(definitions)} entries but --test-paths has {len(test_paths)}"
        )
    if len(configs) != len(definitions):
        raise ValueError(
            f"--configs has {len(configs)} entries but --definitions has {len(definitions)}"
        )
    if len(output_roots) != len(definitions):
        raise ValueError(
            f"--output-roots has {len(output_roots)} entries but --definitions has {len(definitions)}"
        )

    targets = []
    for index, (definition, test_path) in enumerate(zip(definitions, test_paths)):
        if not str(definition).strip():
            raise ValueError(f"--definitions entry {index} is empty")
        if not str(test_path).strip():
            raise ValueError(f"--test-paths entry {index} is empty")
        target = {
            "definition": str(definition),
            "test_path": str(test_path),
        }
        if not str(configs[index]).strip():
            raise ValueError(f"--configs entry {index} is empty")
        target["config"] = str(configs[index])
        if not str(output_roots[index]).strip():
            raise ValueError(f"--output-roots entry {index} is empty")
        target["output_root"] = str(output_roots[index])
        targets.append(target)
    return targets


def materialize_run_fields(
    experiments: list[tuple[int, dict[str, Any]]],
    targets: list[dict[str, str]] | None,
) -> list[tuple[int, dict[str, Any]]]:
    materialized: list[tuple[int, dict[str, Any]]] = []

    if targets and len(targets) > 1:
        raise ValueError("multi-target runs must be expanded with each target's own config")

    fallback = targets[0] if targets else {}
    for exp_index, item in experiments:
        item = dict(item)
        item_definition = item.get("definition") or fallback.get("definition")
        item_test_path = item.get("test_path") or fallback.get("test_path")
        if not item_definition or not item_test_path:
            raise ValueError(
                f"exp_{exp_index:03d} is missing definition/test_path; "
                "provide them in the config or via --definition/--test-path or "
                "--definitions/--test-paths"
            )
        item["definition"] = str(item_definition)
        item["test_path"] = str(item_test_path)
        materialized.append((exp_index, item))
    return materialized


def clean_plan_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def plan_from_experiments(experiments: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"exp_index": index, **clean_plan_entry(item)} for index, item in experiments]


def experiments_from_plan(plan: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    experiments: list[tuple[int, dict[str, Any]]] = []
    for entry in plan:
        exp_index = entry.get("exp_index")
        if not isinstance(exp_index, int):
            raise ValueError(f"plan entry has invalid exp_index: {entry}")
        item = clean_plan_entry({key: value for key, value in entry.items() if key != "exp_index"})
        experiments.append((exp_index, item))
    return experiments


def build_child_env(args: Any) -> dict[str, str] | None:
    if args.model not in TINKER_CHECKPOINT_MODEL_NAMES:
        return None
    if not args.tinker_checkpoints_jsonl and not args.tinker_checkpoint_name:
        return None
    env = dict(os.environ)
    if args.tinker_checkpoints_jsonl:
        env["TINKER_CHECKPOINTS_JSONL"] = str(Path(args.tinker_checkpoints_jsonl).resolve())
    if args.tinker_checkpoint_name:
        env["TINKER_CHECKPOINT_NAME"] = args.tinker_checkpoint_name
    return env


def validate_tinker_args(args: Any) -> None:
    if args.model not in TINKER_MODEL_NAMES:
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("--model is a Tinker model but TINKER_API_KEY is not set")
    if args.model in TINKER_BASE_MODEL_NAMES:
        if args.tinker_checkpoints_jsonl or args.tinker_checkpoint_name:
            raise ValueError(
                "--tinker-checkpoints-jsonl/--tinker-checkpoint-name cannot be used "
                "with a Tinker base model"
            )
        return
    if (
        not args.tinker_checkpoints_jsonl
        and not os.environ.get("TINKER_MODEL_PATH")
        and not os.environ.get("TINKER_CHECKPOINTS_JSONL")
    ):
        raise ValueError(
            "--model is a Tinker model but no checkpoint is configured. "
            "Pass --tinker-checkpoints-jsonl <path> (with optional "
            "--tinker-checkpoint-name <name>), or set TINKER_MODEL_PATH / "
            "TINKER_CHECKPOINTS_JSONL in the environment."
        )
    if args.tinker_checkpoints_jsonl and not Path(args.tinker_checkpoints_jsonl).exists():
        raise ValueError(f"--tinker-checkpoints-jsonl {args.tinker_checkpoints_jsonl!r} does not exist.")


def combined_output(stdout: str | None, stderr: str | None) -> str:
    return f"{stdout or ''}\n{stderr or ''}"


def check_service_health(service_url: str, definition: str, attempts: int = 2) -> tuple[bool, str | None]:
    url = f"{service_url}/definitions/{definition}"
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, timeout=(3, 10))
            resp.raise_for_status()
            return True, None
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(1)
    return False, last_error


def load_current_plan_data(output_root: Path) -> dict[str, Any]:
    plan_path = output_root / "plan.json"
    try:
        data = json.loads(plan_path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{plan_path} does not exist") from None
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"failed to read {plan_path}: {exc}") from exc
    plan = data.get("plan")
    if not isinstance(plan, list):
        raise ValueError(f"{plan_path} must contain a top-level list field named 'plan'")
    if not all(isinstance(entry, dict) for entry in plan):
        raise ValueError(f"{plan_path} contains non-object plan entries")
    return data
