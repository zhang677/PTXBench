from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .models import normalized_usage
from .registry import Registry, TERMINAL_STATES
from .util import utc_now


def _age_seconds(timestamp: str) -> float:
    value = datetime.fromisoformat(timestamp)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - value).total_seconds(), 0.0)


def _stop_container(record: dict) -> str | None:
    process = record.get("process") or {}
    container = process.get("container_name")
    if not container:
        return None
    try:
        result = subprocess.run(
            ["docker", "stop", "--time", "30", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45,
        )
        if result.returncode != 0 and "No such container" not in result.stdout:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return repr(exc)
    return None


def _stop_bound_process(registry: Registry, record: dict) -> None:
    error = _stop_container(record)
    crossing = record.get("budget_crossing") or {}
    registry.update_run(
        record["run_id"],
        status="budget_exhausted",
        budget_crossing={**crossing, "stopped_at": utc_now(), "stop_error": error},
    )


def _stop_at_turn_limit(registry: Registry, record: dict) -> None:
    error = _stop_container(record)
    crossing = record.get("turn_limit_crossing") or {}
    registry.update_run(
        record["run_id"],
        status="turn_limit_exhausted",
        turn_limit_crossing={
            **crossing,
            "stopped_at": utc_now(),
            "stop_error": error,
        },
    )


def _fail_bound_process(registry: Registry, record: dict, failure: dict) -> None:
    error = _stop_container(record)
    status = (
        "infra_failed"
        if failure.get("kind") == "profile_recovery_failed"
        else "failed"
    )
    registry.update_run(
        record["run_id"],
        status=status,
        failure={**failure, "stopped_at": utc_now(), "stop_error": error},
    )


def _has_active_action(record: dict) -> bool:
    active_actions = record.get("active_actions")
    return (
        bool(active_actions)
        if isinstance(active_actions, list)
        else bool(record.get("active_action"))
    )


def _last_activity_at(
    registry: Registry, record: dict, usage_snapshot: dict
) -> str | None:
    timestamps = [
        (record.get("process") or {}).get("started_at"),
        record.get("last_action_ended_at"),
        usage_snapshot.get("updated_at"),
    ]
    harness_events = registry.harness_events_path(record["run_id"])
    if harness_events.exists():
        timestamps.append(
            datetime.fromtimestamp(
                harness_events.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        )
    parsed: list[datetime] = []
    for timestamp in timestamps:
        if not isinstance(timestamp, str):
            continue
        try:
            value = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        parsed.append(
            value
            if value.tzinfo is not None
            else value.replace(tzinfo=timezone.utc)
        )
    return max(parsed).isoformat() if parsed else None


def watch_once(
    registry: Registry,
    *,
    tool_start_grace: float,
    stream_stall_timeout: float = 600.0,
) -> tuple[bool, bool]:
    saw_run = False
    all_terminal = True
    for path in sorted((registry.root / "runs").glob("*.json")):
        saw_run = True
        record = registry.get_run(path.stem)
        usage_snapshot = registry.usage_snapshot(record["run_id"])
        usage = normalized_usage(usage_snapshot.get("usage"))
        canonical_turns = registry.canonical_turn_count(record["run_id"])
        max_turns = record.get("max_turns")
        turn_limit_reached = (
            max_turns is not None and canonical_turns == int(max_turns)
        )
        if record["status"] in TERMINAL_STATES:
            if (
                record["status"] != "budget_exhausted"
                and usage["total_tokens"] >= int(record["token_budget"])
            ):
                crossing = record.get("budget_crossing") or {
                    "observed_at": utc_now(),
                    "total_tokens": usage["total_tokens"],
                    "budget": record["token_budget"],
                    "overshoot": usage["total_tokens"] - int(record["token_budget"]),
                    "canonical_turns": canonical_turns,
                    "reconciled_after_process_exit": True,
                }
                record = registry.update_run(
                    record["run_id"],
                    status="budget_exhausted",
                    budget_crossing=crossing,
                )
            if (
                record["status"] not in {"budget_exhausted", "turn_limit_exhausted"}
                and turn_limit_reached
            ):
                crossing = record.get("turn_limit_crossing") or {
                    "observed_at": utc_now(),
                    "canonical_turns": canonical_turns,
                    "max_turns": max_turns,
                    "overshoot": canonical_turns - int(max_turns),
                    "reconciled_after_process_exit": True,
                }
                record = registry.update_run(
                    record["run_id"],
                    status="turn_limit_exhausted",
                    turn_limit_crossing=crossing,
                )
            if _has_active_action(record):
                all_terminal = False
            continue
        all_terminal = False
        if record["status"] != "running":
            continue
        if usage["total_tokens"] >= int(record["token_budget"]):
            crossing = record.get("budget_crossing")
            if not crossing:
                crossing = {
                    "observed_at": utc_now(),
                    "total_tokens": usage["total_tokens"],
                    "budget": record["token_budget"],
                    "overshoot": usage["total_tokens"] - int(record["token_budget"]),
                    "canonical_turns": canonical_turns,
                }
                record = registry.update_run(record["run_id"], budget_crossing=crossing)
            active_action = _has_active_action(record)
            canonical_total = registry.last_canonical_total(record["run_id"])
            feedback_committed = canonical_total >= int(crossing["total_tokens"])
            if active_action:
                _stop_bound_process(registry, record)
                continue
            if (
                not feedback_committed
                and _age_seconds(crossing["observed_at"]) < tool_start_grace
            ):
                continue
            _stop_bound_process(registry, record)
            continue

        if turn_limit_reached:
            crossing = record.get("turn_limit_crossing")
            if not crossing:
                crossing = {
                    "observed_at": utc_now(),
                    "canonical_turns": canonical_turns,
                    "max_turns": max_turns,
                    "overshoot": canonical_turns - int(max_turns),
                }
                record = registry.update_run(
                    record["run_id"], turn_limit_crossing=crossing
                )
            _stop_at_turn_limit(registry, record)
            continue

        stop_request = record.get("stop_request")
        if isinstance(stop_request, dict):
            _fail_bound_process(
                registry,
                record,
                {
                    "kind": str(stop_request.get("reason") or "stop_requested"),
                    "scope": "requested",
                    "observed_at": utc_now(),
                    "request": stop_request,
                },
            )
            continue

        if (
            record.get("harness") != "antigravity"
            or stream_stall_timeout <= 0
            or _has_active_action(record)
        ):
            continue
        last_activity_at = _last_activity_at(registry, record, usage_snapshot)
        if (
            last_activity_at is None
            or _age_seconds(last_activity_at) < stream_stall_timeout
        ):
            continue
        _fail_bound_process(
            registry,
            record,
            {
                "kind": "usage_stream_stalled",
                "scope": "agent_stream",
                "observed_at": utc_now(),
                "last_activity_at": last_activity_at,
                "timeout_seconds": stream_stall_timeout,
                "canonical_turns": registry.canonical_turn_count(record["run_id"]),
                "source_event": usage_snapshot.get("source_event"),
            },
        )
    return saw_run, all_terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ptxbench-token-watch")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--tool-start-grace", type=float, default=30.0)
    parser.add_argument("--stream-stall-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    registry = Registry(args.root)
    while True:
        saw_run, all_terminal = watch_once(
            registry,
            tool_start_grace=args.tool_start_grace,
            stream_stall_timeout=args.stream_stall_timeout,
        )
        if saw_run and all_terminal:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
