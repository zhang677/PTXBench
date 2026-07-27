#!/usr/bin/env python3
"""Shared watcher audit helpers for multiturn eval roots.

This intentionally treats summary.json as reporting metadata. Completion and
restart decisions are based on the same current trajectory predicates used by
run_parallel_v2.py resume selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MULTITURN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MULTITURN_DIR))

from resume_utils import (  # noqa: E402
    completed_turns_for_trajectory,
    is_api_failure_output,
    is_context_window_failure_output,
    load_previous_statuses,
    trajectory_infra_failure_turn,
    trajectory_submitted_successfully,
)


SUMMARY_KEYS = ("infra_failed", "api_failed", "timeout", "error")


@dataclass
class RootAudit:
    label: str
    root: Path
    plan_total: int
    trajectory_total: int
    summary_state: str
    summary_total: int
    summary_success: int
    summary_failed: int
    summary_bad: dict[str, int]
    missing: list[int]
    needs_resume: list[int]
    terminal_context_window: list[int]
    current_infra: list[int]
    bad_plan_rows: int

    @property
    def full_artifacts(self) -> bool:
        return self.plan_total > 0 and not self.missing and self.bad_plan_rows == 0

    @property
    def ok(self) -> bool:
        return self.full_artifacts and not self.needs_resume


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _exp_index(pos: int, item: dict[str, Any]) -> int | None:
    exp_index = item.get("exp_index")
    if exp_index is None:
        exp_name = item.get("exp_name")
        if isinstance(exp_name, str) and exp_name.startswith("exp_"):
            try:
                exp_index = int(exp_name.split("_", 1)[1])
            except ValueError:
                exp_index = None
    if exp_index is None:
        exp_index = pos
    try:
        return int(exp_index)
    except (TypeError, ValueError):
        return None


def audit_root(label: str, root: Path) -> RootAudit:
    summary_path = root / "summary.json"
    summary = None
    summary_state = "missing"
    if summary_path.exists():
        summary = _read_json(summary_path)
        summary_state = "present" if isinstance(summary, dict) else "bad"

    if isinstance(summary, dict):
        summary_total = int(summary.get("total", 0))
        summary_success = int(summary.get("success", 0))
        summary_failed = int(summary.get("failed", 0))
        summary_bad = {key: int(summary.get(key, 0)) for key in SUMMARY_KEYS}
    else:
        summary_total = summary_success = summary_failed = 0
        summary_bad = {key: 0 for key in SUMMARY_KEYS}

    plan_data = _read_json(root / "plan.json")
    plan = plan_data.get("plan", []) if isinstance(plan_data, dict) else []
    plan_total = len(plan) if isinstance(plan, list) else 0
    trajectory_dir = root / "trajectories"
    trajectory_total = len(list(trajectory_dir.glob("exp_*.json")))

    experiments: list[tuple[int, dict[str, Any]]] = []
    missing: list[int] = []
    bad_plan_rows = 0
    if isinstance(plan, list):
        for pos, item in enumerate(plan):
            if not isinstance(item, dict):
                bad_plan_rows += 1
                continue
            exp_index = _exp_index(pos, item)
            if exp_index is None:
                bad_plan_rows += 1
                continue
            experiments.append((exp_index, item))
            if not (trajectory_dir / f"exp_{exp_index:03d}.json").exists():
                missing.append(exp_index)

    previous_statuses = load_previous_statuses(root)
    needs_resume: list[int] = []
    terminal_context_window: list[int] = []
    current_infra: list[int] = []
    for exp_index, item in experiments:
        exp_name = f"exp_{exp_index:03d}"
        trajectory = trajectory_dir / f"{exp_name}.json"
        if not trajectory.exists():
            continue
        if trajectory_submitted_successfully(trajectory):
            continue
        if trajectory_infra_failure_turn(trajectory) is not None:
            needs_resume.append(exp_index)
            current_infra.append(exp_index)
            continue

        previous_status = previous_statuses.get(exp_name)
        log_file = root / "logs" / f"{exp_name}.log"
        try:
            log_text = log_file.read_text() if log_file.exists() else ""
        except OSError:
            log_text = ""
        try:
            trajectory_text = trajectory.read_text()
        except OSError:
            trajectory_text = ""
        failure_text = f"{log_text}\n{trajectory_text}"
        if is_context_window_failure_output(failure_text):
            terminal_context_window.append(exp_index)
            continue
        if previous_status != "success" and is_api_failure_output(failure_text):
            needs_resume.append(exp_index)
            continue

        try:
            planned_turns = int(item["num_turns"])
        except (KeyError, TypeError, ValueError):
            bad_plan_rows += 1
            continue
        completed_turns = completed_turns_for_trajectory(trajectory)
        if completed_turns is not None and completed_turns < planned_turns:
            needs_resume.append(exp_index)

    return RootAudit(
        label=label,
        root=root,
        plan_total=plan_total,
        trajectory_total=trajectory_total,
        summary_state=summary_state,
        summary_total=summary_total,
        summary_success=summary_success,
        summary_failed=summary_failed,
        summary_bad=summary_bad,
        missing=missing,
        needs_resume=needs_resume,
        terminal_context_window=terminal_context_window,
        current_infra=current_infra,
        bad_plan_rows=bad_plan_rows,
    )


def parse_root_pairs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if len(args.labels) != len(args.output_roots):
        raise SystemExit("--labels and --output-roots must have the same length")
    return [(label, Path(root)) for label, root in zip(args.labels, args.output_roots)]


def command_audit_roots(args: argparse.Namespace) -> int:
    all_ok = True
    for label, root in parse_root_pairs(args):
        audit = audit_root(label, root)
        print(
            f"{audit.label}: {'ok' if audit.ok else 'not-ok'} "
            f"plan={audit.plan_total} trajectories={audit.trajectory_total} "
            f"summary_state={audit.summary_state} "
            f"summary_total={audit.summary_total} success={audit.summary_success} "
            f"failed={audit.summary_failed} bad={audit.summary_bad} "
            f"missing={len(audit.missing)} resume={len(audit.needs_resume)} "
            f"terminal_context_window={len(audit.terminal_context_window)} "
            f"current_infra={len(audit.current_infra)} bad_plan_rows={audit.bad_plan_rows}"
        )
        all_ok = all_ok and audit.ok
    return 0 if all_ok else 1


def command_profile_restart_needed(args: argparse.Namespace) -> int:
    for label, root in parse_root_pairs(args):
        audit = audit_root(label, root)
        if audit.current_infra:
            indices = ",".join(f"exp_{idx:03d}" for idx in audit.current_infra)
            print(f"{audit.label}: profile restart requested for current infra trajectories [{indices}]")
            return 0
    print("no current profile-restart-triggering trajectory failures found")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, func in (
        ("audit-roots", command_audit_roots),
        ("profile-restart-needed", command_profile_restart_needed),
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--labels", nargs="+", required=True)
        sub.add_argument("--output-roots", nargs="+", required=True)
        sub.set_defaults(func=func)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
