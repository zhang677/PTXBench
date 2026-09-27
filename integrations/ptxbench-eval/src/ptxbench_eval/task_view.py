from __future__ import annotations

import copy
from typing import Any

from .models import WorkloadManifest


def validate_task_view(value: Any) -> str:
    if not isinstance(value, str) or value not in ("full", "multiturn"):
        raise ValueError("task_view must be 'full' or 'multiturn'")
    return value


def task_view_from_config(config: dict[str, Any]) -> str:
    view = validate_task_view(config.get("task_view", "full"))
    runs = config.get("runs")
    if isinstance(runs, list) and any(
        isinstance(run, dict) and "task_view" in run for run in runs
    ):
        raise ValueError("task_view is experiment-wide; per-run overrides are not supported")
    return view


def agent_task_payload(manifest: WorkloadManifest, task_view: str) -> dict[str, Any]:
    """Return the agent-visible task without changing the evaluator manifest."""
    if validate_task_view(task_view) == "full":
        return copy.deepcopy(manifest.data)
    # Match the multiturn prompt builder's definition projection.
    definition = copy.deepcopy(manifest.data["definition_payload"])
    definition.pop("tags", None)
    return definition
