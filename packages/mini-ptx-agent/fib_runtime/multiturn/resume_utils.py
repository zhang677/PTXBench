"""Shared resume helpers for multiturn parallel launchers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


INFRA_FAILED = "infra_failed"
API_FAILED = "api_failed"
API_FAILURE_PATTERNS = (
    "GeminiException",
    "VertexAIError",
    "litellm.exceptions.APIConnectionError",
    "litellm.exceptions.ServiceUnavailableError",
    "httpx.RemoteProtocolError",
    "httpcore.RemoteProtocolError",
    "503 Service Unavailable",
    "The service is currently unavailable",
    "peer closed connection without sending complete message body",
    "incomplete chunked read",
    "OpenAIException - Connection error",
)
CONTEXT_WINDOW_FAILURE_PATTERNS = (
    "ContextWindowExceededError",
    "Requested token count exceeds the model's maximum context length",
    "is longer than the model's context length",
)
MEMORY_BUDGET_INFRA_PATTERNS = (
    "Task failed at the profiling server.",
    "within the memory budget",
    "after clearing cached baselines",
    "Timed out after 240s waiting for evaluate"
)


def is_infra_failure_output(text: str) -> bool:
    if "INFRA_TIMEOUT" in text:
        return True
    return any(pattern in text for pattern in MEMORY_BUDGET_INFRA_PATTERNS)


def is_api_failure_output(text: str) -> bool:
    return any(pattern in text for pattern in API_FAILURE_PATTERNS)


def is_context_window_failure_output(text: str) -> bool:
    return any(pattern in text for pattern in CONTEXT_WINDOW_FAILURE_PATTERNS)


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text() if path.exists() else ""
    except OSError:
        return ""


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    extra = message.get("extra") or {}
    raw_output = extra.get("raw_output") or ""
    return f"{content}\n{raw_output}"


def backup_path(path: Path, label: str) -> Path:
    candidate = path.with_name(f"{path.name}.{label}_{int(time.time())}.bak")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.{label}_{int(time.time())}_{suffix}.bak")
        suffix += 1
    return candidate


def extract_cpp_code(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    match = re.search(r"```\s*cpp\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def kernel_hash(source: str) -> str:
    return hashlib.sha256(source.replace("\r\n", "\n").strip().encode()).hexdigest()


def kept_kernel_turns(messages: list[dict[str, Any]], starting_turn: int) -> dict[str, list[int]]:
    kept: dict[str, list[int]] = {}
    for turn in range(starting_turn):
        assistant_idx = 2 + 2 * turn
        if assistant_idx >= len(messages):
            break
        source = extract_cpp_code(messages[assistant_idx].get("content"))
        if source:
            kept.setdefault(kernel_hash(source), []).append(turn)
    return kept


def rebuild_success_dir(
    output_root: Path,
    exp_name: str,
    messages: list[dict[str, Any]],
    starting_turn: int,
) -> dict[str, Any] | None:
    success_dir = output_root / "success" / exp_name
    if not success_dir.exists():
        return None

    backup = backup_path(success_dir, "resume_success")
    success_dir.rename(backup)
    success_dir.mkdir(parents=True, exist_ok=True)

    kept_hashes = kept_kernel_turns(messages, starting_turn)
    record_path = backup / "record.json"
    try:
        old_records = json.loads(record_path.read_text()) if record_path.exists() else []
    except (json.JSONDecodeError, OSError):
        old_records = []
    if not isinstance(old_records, list):
        old_records = []

    new_records: list[dict[str, Any]] = []
    for record in old_records:
        if not isinstance(record, dict):
            continue
        version = record.get("version")
        if not isinstance(version, int):
            continue
        old_kernel = backup / f"kernel_v{version}.cu"
        if not old_kernel.exists():
            continue
        try:
            source = old_kernel.read_text()
        except OSError:
            continue
        turns = kept_hashes.get(kernel_hash(source))
        if not turns:
            continue
        record_turn = record.get("turn")
        turn = record_turn if record_turn in turns else turns[0]
        new_version = len(new_records)
        shutil.copy2(old_kernel, success_dir / f"kernel_v{new_version}.cu")
        new_record = dict(record)
        new_record["version"] = new_version
        new_record["turn"] = turn
        new_records.append(new_record)

    if new_records:
        (success_dir / "record.json").write_text(json.dumps(new_records, indent=2) + "\n")
    else:
        shutil.rmtree(success_dir)

    return {
        "success_backup": str(backup),
        "success_records_preserved": len(new_records),
        "success_removed": not new_records,
    }


def first_infra_failure_turn(messages: list[dict[str, Any]]) -> int | None:
    turn = 0
    while True:
        observation_idx = 3 + 2 * turn
        if observation_idx >= len(messages):
            return None
        observation = messages[observation_idx]
        if observation.get("role") != "user":
            return None
        if is_infra_failure_output(_message_text(observation)):
            return turn
        turn += 1


def trim_infra_trajectory(trajectory: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(trajectory.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None

    failed_turn = first_infra_failure_turn(messages)
    if failed_turn is None:
        return None

    starting_turn = failed_turn
    if starting_turn <= 0:
        backup = trajectory.with_suffix(trajectory.suffix + f".infra_failed_{int(time.time())}.bak")
        trajectory.rename(backup)
        return {
            "starting_turn": 0,
            "resume_trajectory": None,
            "action": "archived",
            "backup": str(backup),
            "failed_turn": failed_turn,
        }

    keep = 2 + 2 * starting_turn
    backup = trajectory.with_suffix(trajectory.suffix + f".infra_failed_{int(time.time())}.bak")
    trajectory.rename(backup)
    data["messages"] = messages[:keep]
    trajectory.write_text(json.dumps(data, indent=2) + "\n")
    return {
        "starting_turn": starting_turn,
        "resume_trajectory": str(trajectory),
        "action": "trimmed",
        "backup": str(backup),
        "failed_turn": failed_turn,
    }


def trajectory_infra_failure_turn(trajectory: Path) -> int | None:
    try:
        data = json.loads(trajectory.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    return first_infra_failure_turn(messages)


def completed_trajectory_turns(messages: list[dict[str, Any]]) -> int:
    completed = 0
    while True:
        assistant_idx = 2 + 2 * completed
        observation_idx = assistant_idx + 1
        if observation_idx >= len(messages):
            return completed
        if messages[assistant_idx].get("role") != "assistant":
            return completed
        if messages[observation_idx].get("role") != "user":
            return completed
        completed += 1


def completed_turns_for_trajectory(trajectory: Path) -> int | None:
    try:
        data = json.loads(trajectory.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    return completed_trajectory_turns(messages)


def trim_api_failure_trajectory(trajectory: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(trajectory.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None

    starting_turn = completed_trajectory_turns(messages)
    backup = trajectory.with_suffix(trajectory.suffix + f".api_failed_{int(time.time())}.bak")
    trajectory.rename(backup)
    if starting_turn <= 0:
        return {
            "starting_turn": 0,
            "resume_trajectory": None,
            "action": "api_archived",
            "backup": str(backup),
        }

    keep = 2 + 2 * starting_turn
    data["messages"] = messages[:keep]
    trajectory.write_text(json.dumps(data, indent=2) + "\n")
    return {
        "starting_turn": starting_turn,
        "resume_trajectory": str(trajectory),
        "action": "api_trimmed",
        "backup": str(backup),
    }


def load_previous_statuses(output_root: Path) -> dict[str, str]:
    summary_path = output_root / "summary.json"
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    results = summary.get("results")
    if not isinstance(results, list):
        return {}
    statuses: dict[str, str] = {}
    for result in results:
        exp_name = result.get("exp_name")
        status = result.get("status")
        if isinstance(exp_name, str) and isinstance(status, str):
            statuses[exp_name] = status
    return statuses


def trajectory_submitted_successfully(trajectory: Path) -> bool:
    try:
        data = json.loads(trajectory.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    last = messages[-1]
    if not isinstance(last, dict):
        return False
    extra = last.get("extra") or {}
    return last.get("role") == "exit" and extra.get("exit_status") == "Submitted"


def prepare_extend_turns(experiments: list[tuple[int, dict[str, Any]]], output_root: Path) -> dict[int, dict[str, Any]]:
    extend_info: dict[int, dict[str, Any]] = {}
    for exp_index, item in experiments:
        exp_name = f"exp_{exp_index:03d}"
        trajectory = output_root / "trajectories" / f"{exp_name}.json"
        planned_turns = int(item["num_turns"])
        if trajectory_submitted_successfully(trajectory):
            continue
        completed_turns = completed_turns_for_trajectory(trajectory)
        if completed_turns is None:
            continue
        if completed_turns >= planned_turns:
            continue
        extend_info[exp_index] = {
            "starting_turn": completed_turns,
            "resume_trajectory": str(trajectory) if completed_turns > 0 else None,
            "action": "extend",
            "planned_turns": planned_turns,
            "completed_turns": completed_turns,
        }
        if completed_turns > 0:
            item["_resume_turn"] = completed_turns
            item["_resume_trajectory"] = str(trajectory)
    return extend_info


def prepare_resume(experiments: list[tuple[int, dict[str, Any]]], output_root: Path) -> dict[int, dict[str, Any]]:
    resume_info: dict[int, dict[str, Any]] = {}
    previous_statuses = load_previous_statuses(output_root)
    for exp_index, item in experiments:
        exp_name = f"exp_{exp_index:03d}"
        trajectory = output_root / "trajectories" / f"{exp_name}.json"
        if not trajectory.exists():
            info = {
                "starting_turn": 0,
                "resume_trajectory": None,
                "action": "missing",
            }
            success_info = rebuild_success_dir(output_root, exp_name, [], 0)
            if success_info:
                info.update(success_info)
            resume_info[exp_index] = info
            continue
        if trajectory_submitted_successfully(trajectory):
            continue

        trim = trim_infra_trajectory(trajectory)
        if trim is not None:
            if trim["resume_trajectory"]:
                try:
                    trimmed_messages = json.loads(Path(trim["resume_trajectory"]).read_text()).get("messages", [])
                except (json.JSONDecodeError, OSError):
                    trimmed_messages = []
            else:
                trimmed_messages = []
            success_info = rebuild_success_dir(output_root, exp_name, trimmed_messages, trim["starting_turn"])
            if success_info:
                trim.update(success_info)
            resume_info[exp_index] = trim
            item["_resume_turn"] = trim["starting_turn"]
            item["_resume_trajectory"] = trim["resume_trajectory"]
            continue

        previous_status = previous_statuses.get(exp_name)
        log_file = output_root / "logs" / f"{exp_name}.log"
        api_failure_text = f"{_read_text_if_exists(log_file)}\n{_read_text_if_exists(trajectory)}"
        if is_context_window_failure_output(api_failure_text):
            continue
        if previous_status != "success" and is_api_failure_output(api_failure_text):
            trim = trim_api_failure_trajectory(trajectory)
            if trim is None:
                continue
            if trim["resume_trajectory"]:
                try:
                    trimmed_messages = json.loads(Path(trim["resume_trajectory"]).read_text()).get("messages", [])
                except (json.JSONDecodeError, OSError):
                    trimmed_messages = []
            else:
                trimmed_messages = []
            success_info = rebuild_success_dir(output_root, exp_name, trimmed_messages, trim["starting_turn"])
            if success_info:
                trim.update(success_info)
            resume_info[exp_index] = trim
            item["_resume_turn"] = trim["starting_turn"]
            item["_resume_trajectory"] = trim["resume_trajectory"]
            continue

        planned_turns = int(item["num_turns"])
        completed_turns = completed_turns_for_trajectory(trajectory)
        if completed_turns is None or completed_turns >= planned_turns:
            continue
        resume_info[exp_index] = {
            "starting_turn": completed_turns,
            "resume_trajectory": str(trajectory) if completed_turns > 0 else None,
            "action": "incomplete",
            "planned_turns": planned_turns,
            "completed_turns": completed_turns,
        }
        if completed_turns > 0:
            item["_resume_turn"] = completed_turns
            item["_resume_trajectory"] = str(trajectory)
    return resume_info
