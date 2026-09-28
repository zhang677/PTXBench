from __future__ import annotations

import hmac
import hashlib
import secrets
import time
from pathlib import Path
from typing import Any

from .feedback import validate_feedback_style
from .models import ContractError, WorkloadManifest, normalized_usage
from .task_view import agent_task_payload, validate_task_view
from .util import atomic_write_json, file_lock, json_digest, read_json, token_hash, utc_now


TERMINAL_STATES = {
    "completed",
    "failed",
    "infra_failed",
    "budget_exhausted",
    "turn_limit_exhausted",
    "cancelled",
}


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        for name in (
            "runs",
            "usage",
            "trajectories",
            "raw_profile_events",
            "harness_events",
            "actions",
            "workspaces",
            "success",
            "secrets",
            "prompts",
            "manifests",
            "tasks",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"{run_id}.json"

    def usage_path(self, run_id: str) -> Path:
        return self.root / "usage" / f"{run_id}.json"

    def trajectory_path(self, run_id: str) -> Path:
        return self.root / "trajectories" / f"{run_id}.json"

    def raw_events_path(self, run_id: str) -> Path:
        return self.root / "raw_profile_events" / f"{run_id}.jsonl"

    def harness_events_path(self, run_id: str) -> Path:
        return self.root / "harness_events" / f"{run_id}.jsonl"

    def prompt_path(self, run_id: str) -> Path:
        return self.root / "prompts" / f"{run_id}.md"

    def manifest_path(self, run_id: str) -> Path:
        return self.root / "manifests" / f"{run_id}.json"

    def task_path(self, run_id: str) -> Path:
        return self.root / "tasks" / f"{run_id}.json"

    def action_path(self, run_id: str, action_id: str) -> Path:
        return self.root / "actions" / run_id / f"{action_id}.json"

    def action_lock_path(self, run_id: str, action_id: str) -> Path:
        return self.root / "actions" / run_id / f"{action_id}.lock"

    def run_action_lock_path(self, run_id: str) -> Path:
        return self.root / "actions" / run_id / ".run.lock"

    def create_run(
        self,
        *,
        run_id: str,
        manifest: dict[str, Any],
        agent_prompt: str,
        token_budget: int,
        harness: str,
        model: str,
        prompt_tag: str,
        max_turns: int | None = None,
        artifact_provenance: dict[str, Any] | None = None,
        token: str | None = None,
        feedback_style: str = "default",
        task_view: str = "full",
    ) -> str:
        feedback_style = validate_feedback_style(feedback_style)
        task_view = validate_task_view(task_view)
        if not run_id or "/" in run_id or ".." in run_id:
            raise ContractError("invalid run_id")
        workload = WorkloadManifest.parse(manifest)
        if token_budget <= 0:
            raise ContractError("token_budget must be positive")
        if max_turns is not None and (
            isinstance(max_turns, bool)
            or not isinstance(max_turns, int)
            or max_turns <= 0
        ):
            raise ContractError("max_turns must be a positive integer")
        token = token or secrets.token_urlsafe(32)
        prompt_sha256 = hashlib.sha256(agent_prompt.encode()).hexdigest()
        artifact_provenance = dict(artifact_provenance or {})
        path = self.run_path(run_id)
        with file_lock(path.with_suffix(".lock")):
            if path.exists():
                existing = read_json(path)
                if existing.get("manifest_digest") != workload.digest:
                    raise ContractError(f"run {run_id} is already bound to another task")
                raise ContractError(f"run {run_id} already exists")
            now = utc_now()
            record = {
                "schema_version": 1,
                "run_id": run_id,
                "trajectory_id": run_id,
                "task_id": workload.data["task_id"],
                "manifest": workload.data,
                "manifest_digest": workload.digest,
                "auth_token_sha256": token_hash(token),
                "token_budget": token_budget,
                "max_turns": max_turns,
                "harness": harness,
                "model": model,
                "prompt_tag": prompt_tag,
                "feedback_style": feedback_style,
                "task_view": task_view,
                "agent_prompt_sha256": prompt_sha256,
                "artifact_provenance": artifact_provenance,
                "status": "prepared",
                "created_at": now,
                "updated_at": now,
                "process": {},
                "budget_crossing": None,
                "turn_limit_crossing": None,
                "stop_request": None,
                "active_action": None,
                "active_actions": [],
            }
            atomic_write_json(path, record)
            atomic_write_json(
                self.usage_path(run_id),
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "semantics": "run_cumulative_provider_reported",
                    "provider_reported": True,
                    "usage": normalized_usage({}),
                    "updated_at": now,
                    "source_event": None,
                },
            )
            atomic_write_json(
                self.trajectory_path(run_id),
                {
                    "schema_version": 1,
                    "trajectory_id": run_id,
                    "run_id": run_id,
                    "task_id": workload.data["task_id"],
                    "manifest_digest": workload.digest,
                    "harness": harness,
                    "model": model,
                    "prompt_tag": prompt_tag,
                    "messages": [
                        {
                            "role": "system",
                            "content": "",
                            "extra": {
                                "ptxbench": {
                                    "synthetic_placeholder": True,
                                    "delivered_to_agent": False,
                                    "purpose": "canonical_export_compatibility",
                                }
                            },
                        },
                        {"role": "user", "content": agent_prompt},
                    ],
                    "ptxbench": {
                        "feedback_style": feedback_style,
                        "task_view": task_view,
                        "token_semantics": "run_cumulative_at_kernel_request",
                        "provider_reported": True,
                        "agent_prompt_sha256": prompt_sha256,
                        "system_message_placeholder": {
                            "synthetic": True,
                            "delivered_to_agent": False,
                            "content_sha256": hashlib.sha256(b"").hexdigest(),
                        },
                        "artifact_provenance": artifact_provenance,
                    },
                },
            )
        return token

    def get_run(self, run_id: str) -> dict[str, Any]:
        value = read_json(self.run_path(run_id))
        if not isinstance(value, dict):
            raise ContractError(f"unknown run_id: {run_id}")
        return value

    def authenticate(self, run_id: str, token: str) -> dict[str, Any]:
        record = self.get_run(run_id)
        if not hmac.compare_digest(record["auth_token_sha256"], token_hash(token)):
            raise PermissionError("invalid run token")
        return record

    def validate_manifest(self, record: dict[str, Any], manifest: Any) -> WorkloadManifest:
        if validate_task_view(record.get("task_view", "full")) == "multiturn":
            bound = WorkloadManifest.parse(record["manifest"])
            if not hmac.compare_digest(record["manifest_digest"], bound.digest):
                raise ContractError("stored manifest digest changed for this run")
            expected = agent_task_payload(bound, "multiturn")
            if not isinstance(manifest, dict) or not hmac.compare_digest(
                json_digest(expected), json_digest(manifest)
            ):
                raise ContractError("task definition mismatch for this run")
            # Workload selection and compilation settings come only from the
            # authenticated run, never from the client-visible definition.
            return bound
        supplied = WorkloadManifest.parse(manifest)
        if not hmac.compare_digest(record["manifest_digest"], supplied.digest):
            raise ContractError(
                "workload mismatch: this run is permanently bound to "
                f"task_id={record['task_id']} digest={record['manifest_digest']}"
            )
        return supplied

    def update_run(self, run_id: str, **changes: Any) -> dict[str, Any]:
        path = self.run_path(run_id)
        with file_lock(path.with_suffix(".lock")):
            record = self.get_run(run_id)
            record.update(changes)
            record["updated_at"] = utc_now()
            atomic_write_json(path, record)
            return record

    def set_process(self, run_id: str, *, pid: int, container_name: str) -> None:
        self.update_run(
            run_id,
            status="running",
            process={"pid": pid, "container_name": container_name, "started_at": utc_now()},
        )

    def set_active_action(self, run_id: str, action_id: str | None) -> None:
        action = {"action_id": action_id, "started_at": utc_now()} if action_id else None
        self.update_run(
            run_id,
            active_action=action,
            active_actions=[action] if action else [],
        )

    def begin_action(self, run_id: str, action_id: str) -> None:
        path = self.run_path(run_id)
        with file_lock(path.with_suffix(".lock")):
            record = self.get_run(run_id)
            actions = list(record.get("active_actions") or [])
            action = {"action_id": action_id, "started_at": utc_now()}
            actions.append(action)
            record["active_actions"] = actions
            record["active_action"] = actions[0]
            record["updated_at"] = utc_now()
            atomic_write_json(path, record)

    def end_action(self, run_id: str, action_id: str) -> None:
        path = self.run_path(run_id)
        with file_lock(path.with_suffix(".lock")):
            record = self.get_run(run_id)
            actions = list(record.get("active_actions") or [])
            remaining: list[dict[str, Any]] = []
            removed = False
            for action in actions:
                if not removed and action.get("action_id") == action_id:
                    removed = True
                    continue
                remaining.append(action)
            record["active_actions"] = remaining
            record["active_action"] = remaining[0] if remaining else None
            ended_at = utc_now()
            record["last_action_ended_at"] = ended_at
            record["updated_at"] = ended_at
            atomic_write_json(path, record)

    def request_stop(
        self, run_id: str, *, reason: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        path = self.run_path(run_id)
        with file_lock(path.with_suffix(".lock")):
            record = self.get_run(run_id)
            if (
                record.get("stop_request") is None
                and record.get("status") not in TERMINAL_STATES
            ):
                record["stop_request"] = {
                    "reason": reason,
                    "requested_at": utc_now(),
                    "details": dict(details or {}),
                }
                record["updated_at"] = utc_now()
                atomic_write_json(path, record)
            return record

    def update_usage(
        self, run_id: str, usage: Any, *, source_event: str | int | None
    ) -> dict[str, Any]:
        path = self.usage_path(run_id)
        candidate = normalized_usage(usage)
        with file_lock(path.with_suffix(".lock")):
            current = read_json(path, {})
            old = normalized_usage(current.get("usage"))
            if candidate["total_tokens"] < old["total_tokens"]:
                return current
            current = {
                "schema_version": 1,
                "run_id": run_id,
                "semantics": "run_cumulative_provider_reported",
                "provider_reported": True,
                "usage": candidate,
                "updated_at": utc_now(),
                "source_event": source_event,
            }
            atomic_write_json(path, current)
            return current

    def usage_snapshot(self, run_id: str) -> dict[str, Any]:
        value = read_json(self.usage_path(run_id))
        if not isinstance(value, dict):
            raise ContractError(f"missing usage registry for {run_id}")
        return value

    def canonical_turn_count(self, run_id: str) -> int:
        trajectory = read_json(self.trajectory_path(run_id), {})
        return sum(
            1
            for message in trajectory.get("messages", [])
            if message.get("role") == "user" and (message.get("extra") or {}).get("raw_output") is not None
        )

    def last_canonical_total(self, run_id: str) -> int:
        checkpoint = self.last_canonical_usage_checkpoint(run_id)
        return normalized_usage(checkpoint).get("total_tokens", 0)

    def last_canonical_usage_checkpoint(self, run_id: str) -> dict[str, Any]:
        trajectory = read_json(self.trajectory_path(run_id), {})
        for message in reversed(trajectory.get("messages", [])):
            checkpoint = ((message.get("extra") or {}).get("ptxbench") or {}).get(
                "cumulative_usage_at_kernel_request"
            )
            if isinstance(checkpoint, dict):
                return dict(checkpoint)
        return {}

    def wait_for_new_usage(
        self,
        run_id: str,
        *,
        after_source_event: str | int | None = None,
        after_updated_at: str | None = None,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        stable_since: float | None = None
        last_identity: tuple[Any, Any] | None = None
        while True:
            snapshot = self.usage_snapshot(run_id)
            source_event = snapshot.get("source_event")
            updated_at = snapshot.get("updated_at")
            if self._usage_is_newer(
                source_event=source_event,
                updated_at=updated_at,
                after_source_event=after_source_event,
                after_updated_at=after_updated_at,
            ):
                identity = (source_event, updated_at)
                if identity != last_identity:
                    last_identity = identity
                    stable_since = time.monotonic()
                elif stable_since is not None and time.monotonic() - stable_since >= 0.25:
                    return snapshot
            else:
                last_identity = None
                stable_since = None
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no newer stable provider usage checkpoint available for run {run_id}; "
                    f"last committed source_event={after_source_event!r}, "
                    f"current source_event={source_event!r}"
                )
            time.sleep(0.05)

    @staticmethod
    def _usage_is_newer(
        *,
        source_event: Any,
        updated_at: Any,
        after_source_event: Any,
        after_updated_at: Any,
    ) -> bool:
        if source_event is None:
            return False
        if after_source_event is None:
            if after_updated_at is None:
                return True
            return (
                isinstance(updated_at, str)
                and isinstance(after_updated_at, str)
                and updated_at > after_updated_at
            )
        if isinstance(source_event, int) and isinstance(after_source_event, int):
            return source_event > after_source_event
        return (
            source_event != after_source_event
            and isinstance(updated_at, str)
            and (
                after_updated_at is None
                or (
                    isinstance(after_updated_at, str)
                    and updated_at > after_updated_at
                )
            )
        )

    def append_turn(
        self,
        *,
        run_id: str,
        action_id: str,
        source: str,
        feedback: str,
        traces: list[dict[str, Any]] | None,
        all_passed: bool,
        min_speedup: float,
        usage_snapshot: dict[str, Any],
        replay_count: int,
        recovery_count: int = 0,
    ) -> None:
        feedback_style = validate_feedback_style(
            self.get_run(run_id).get("feedback_style", "default")
        )
        path = self.trajectory_path(run_id)
        with file_lock(path.with_suffix(".lock")):
            trajectory = read_json(path)
            for message in trajectory.get("messages", []):
                ptxbench = (message.get("extra") or {}).get("ptxbench") or {}
                if ptxbench.get("action_id") == action_id:
                    return
            turn_idx = self.canonical_turn_count(run_id)
            usage = normalized_usage(usage_snapshot.get("usage"))
            checkpoint = {
                "schema_version": 1,
                "semantics": "run_cumulative_at_kernel_request",
                "provider_reported": True,
                **usage,
                "checkpoint_updated_at": usage_snapshot.get("updated_at"),
                "source_event": usage_snapshot.get("source_event"),
            }
            assistant = {
                "role": "assistant",
                "content": f"```cpp\n{source.rstrip()}\n```",
                "extra": {
                    "ptxbench": {
                        "action_id": action_id,
                        "cumulative_usage_at_kernel_request": checkpoint,
                    }
                },
            }
            evaluation = {
                "role": "user",
                "content": feedback,
                "extra": {
                    "raw_output": feedback,
                    "traces": traces,
                    "all_passed": all_passed,
                    "min_speedup": min_speedup,
                    "rollout": {"turn_idx": turn_idx},
                    "ptxbench": {
                        "action_id": action_id,
                        "replay_count": replay_count,
                        "feedback_style": feedback_style,
                        "recovery_count": recovery_count,
                        "cumulative_usage_at_kernel_request": checkpoint,
                    },
                },
            }
            trajectory["messages"].extend([assistant, evaluation])
            atomic_write_json(path, trajectory)
