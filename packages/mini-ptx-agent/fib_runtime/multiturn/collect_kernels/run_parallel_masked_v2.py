#!/usr/bin/env python3
"""Parallel launcher for masked-kernel fill-in runs.

This is a narrow variant of ../run_parallel_v2.py for prompt configs whose
items include `masked_kernel_path`, for example:

    {
        "num_trajectories": 4,
        "num_turns": 1,
        "target_speedup": 0.62,
        "prompt_tag": "hopper-no-hint",
        "masked_kernel_path": "/path/to/masked_kernel.cu",
        "definition": "mha_with_lse_d128",
        "test_path": "/path/to/test.py"
    }

Each expanded experiment starts run_v2.py with a per-experiment user template
whose first user prompt is:

    Please fill in the missing part marked as ?

    ```cpp
    <masked kernel source>
    ```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
MULTITURN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MULTITURN_DIR))

from build_doc_v2 import build_doc  # noqa: E402
from common import GPU_ARCH_NVCC  # noqa: E402


TINKER_MODEL_NAMES = {"Qwen3.5-35B-A3B-tinker"}
INFRA_FAILED = "infra_failed"
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


class InfraAbort(RuntimeError):
    pass


def _kill_container(workspace: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel launcher for run_v2.py using masked-kernel first prompts",
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to config JSON. Each item must include "
            "num_trajectories/num_turns/target_speedup/prompt_tag/"
            "masked_kernel_path/definition/test_path."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max concurrent workers (default: total experiments)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="mini-swe-eval:latest",
        help="Docker image passed to run_v2.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Parent output directory (default: ./eval_runs/masked_v2_TIMESTAMP)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing --output-root after profiling-service infra failure. "
            "Polluted trajectory tails are trimmed before relaunch."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First expanded experiment index to launch (default: 0)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last expanded experiment index to launch, inclusive (default: run through the end)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="Per-experiment subprocess timeout in seconds (default: 14400)",
    )
    parser.add_argument(
        "--turn-timeout",
        type=int,
        default=360,
        help="Per-turn timeout in seconds, passed to run_v2.py (default: 360)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output passed to run_v2.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name passed to run_v2.py",
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default="http://localhost:10000",
        help="Profiling service URL passed to run_v2.py",
    )
    parser.add_argument(
        "--gpu-arch",
        type=str,
        choices=list(GPU_ARCH_NVCC.keys()),
        default="hopper",
        help="GPU architecture passed to run_v2.py (default: hopper)",
    )
    parser.add_argument(
        "--tinker-checkpoints-jsonl",
        type=str,
        default=None,
        help=(
            "Path to Tinker checkpoints.jsonl. Used only when --model is a Tinker model, "
            "unless TINKER_MODEL_PATH or TINKER_CHECKPOINTS_JSONL is already set."
        ),
    )
    parser.add_argument(
        "--tinker-checkpoint-name",
        type=str,
        default=None,
        help="Checkpoint name inside --tinker-checkpoints-jsonl (default: final)",
    )
    return parser.parse_args()


def expand_config(config_items: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    experiments: list[tuple[int, dict[str, Any]]] = []
    exp_index = 0
    for item_index, item in enumerate(config_items):
        missing = [
            key
            for key in (
                "num_trajectories",
                "num_turns",
                "target_speedup",
                "prompt_tag",
                "masked_kernel_path",
                "definition",
                "test_path",
            )
            if key not in item
        ]
        if missing:
            raise ValueError(f"config item {item_index} missing required key(s): {', '.join(missing)}")
        if not str(item["definition"]).strip():
            raise ValueError(f"config item {item_index} has empty definition")
        if not str(item["test_path"]).strip():
            raise ValueError(f"config item {item_index} has empty test_path")
        for _ in range(int(item["num_trajectories"])):
            experiments.append((exp_index, dict(item)))
            exp_index += 1
    return experiments


def build_child_env(args: argparse.Namespace) -> dict[str, str] | None:
    if args.model not in TINKER_MODEL_NAMES:
        return None
    if not args.tinker_checkpoints_jsonl and not args.tinker_checkpoint_name:
        return None
    env = dict(os.environ)
    if args.tinker_checkpoints_jsonl:
        env["TINKER_CHECKPOINTS_JSONL"] = str(Path(args.tinker_checkpoints_jsonl).resolve())
    if args.tinker_checkpoint_name:
        env["TINKER_CHECKPOINT_NAME"] = args.tinker_checkpoint_name
    return env


def build_masked_prompt(masked_kernel_path: Path) -> str:
    masked_source = masked_kernel_path.read_text()
    return (
        "Your task is to implement a CUDA kernel with a host `run` function that "
        "implements the given PyTorch reference.\n\n"
        "Task:\n"
        "{task_content}\n\n"
        "Please fill in the missing part marked as ?\n\n"
        "```cpp\n"
        f"{masked_source.rstrip()}\n"
        "```\n"
    )


def is_infra_failure_output(text: str) -> bool:
    return "INFRA_TIMEOUT" in text


def is_api_failure_output(text: str) -> bool:
    return any(pattern in text for pattern in API_FAILURE_PATTERNS)


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text() if path.exists() else ""
    except OSError:
        return ""


def _combined_output(stdout: str | None, stderr: str | None) -> str:
    return f"{stdout or ''}\n{stderr or ''}"


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


def rebuild_success_dir(output_root: Path, exp_name: str, messages: list[dict[str, Any]], starting_turn: int) -> dict[str, Any] | None:
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
    if starting_turn <= 1:
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
    return resume_info


def run_single_experiment(
    exp_index: int,
    item: dict[str, Any],
    args: argparse.Namespace,
    gpu_queue: queue.Queue,
    output_root: Path,
) -> dict[str, Any]:
    gpu_id = gpu_queue.get()
    try:
        exp_name = f"exp_{exp_index:03d}"
        workspace = output_root / exp_name
        log_file = output_root / "logs" / f"{exp_name}.log"
        trajectory = output_root / "trajectories" / f"{exp_name}.json"
        success_dir = output_root / "success" / exp_name
        prompt_file = output_root / "prompts" / f"{exp_name}.txt"
        masked_kernel_path = Path(item["masked_kernel_path"]).expanduser().resolve()
        test_path = Path(item["test_path"]).expanduser().resolve()
        definition = str(item["definition"])

        workspace.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(build_masked_prompt(masked_kernel_path))

        launch_script = MULTITURN_DIR / "run_v2.py"
        container_timeout = f"{args.timeout + 7200}s"
        cmd = [
            sys.executable,
            str(launch_script),
            "--definition",
            definition,
            "--model",
            args.model,
            "--test-path",
            str(test_path),
            "--log-path",
            str(trajectory),
            "--output-dir",
            str(workspace),
            "--success-dir",
            str(success_dir),
            "--image",
            args.image,
            "--max-turns",
            str(item["num_turns"]),
            "--target-speedup",
            str(item["target_speedup"]),
            "--prompt-tag",
            item["prompt_tag"],
            "--user-template",
            str(prompt_file),
            "--service-url",
            args.service_url,
            "--gpu-arch",
            args.gpu_arch,
            "--container-timeout",
            container_timeout,
            "--turn-timeout",
            str(args.turn_timeout),
        ]
        cmd.append("--without-local-gpu")
        if args.verbose:
            cmd.append("-v")

        resume_turn = int(item.get("_resume_turn") or 0)
        resume_trajectory = item.get("_resume_trajectory")
        if resume_turn > 0 and resume_trajectory:
            cmd.extend([
                "--resume-trajectory",
                str(resume_trajectory),
                "--resume-turn",
                str(resume_turn),
            ])

        result: dict[str, Any] = {
            "exp_index": exp_index,
            "exp_name": exp_name,
            "gpu_id": gpu_id,
            "prompt_tag": item["prompt_tag"],
            "definition": definition,
            "test_path": str(test_path),
            "masked_kernel_path": str(masked_kernel_path),
            "prompt_file": str(prompt_file),
            "num_turns": item["num_turns"],
            "target_speedup": item["target_speedup"],
            "status": "unknown",
            "duration": 0.0,
            "returncode": None,
            "error": None,
            "trajectory": str(trajectory),
        }

        child_env = build_child_env(args)
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=child_env,
            )
            result["duration"] = time.time() - start
            result["returncode"] = proc.returncode
            infra_turn = trajectory_infra_failure_turn(trajectory)
            if infra_turn is not None:
                result["status"] = INFRA_FAILED
                result["error"] = f"Profiling service infra failure in trajectory turn {infra_turn}"
                result["infra_failure_turn"] = infra_turn
            elif is_infra_failure_output(_combined_output(proc.stdout, proc.stderr)):
                result["status"] = INFRA_FAILED
                result["error"] = "Profiling service infra failure"
            elif proc.returncode == 0:
                result["status"] = "success"
            else:
                result["status"] = "failed"
            if resume_turn > 0:
                result["resume_turn"] = resume_turn
                result["resume_trajectory"] = str(resume_trajectory)

            with log_file.open("w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== MASKED KERNEL ===\n{masked_kernel_path}\n\n")
                f.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
                f.write(f"=== STDERR ===\n{proc.stderr}\n")

        except subprocess.TimeoutExpired as exc:
            result["duration"] = time.time() - start
            result["status"] = "timeout"
            result["error"] = f"Timed out after {args.timeout}s"
            _kill_container(workspace)
            with log_file.open("w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== MASKED KERNEL ===\n{masked_kernel_path}\n\n")
                f.write(f"=== TIMEOUT after {args.timeout}s ===\n")
                if exc.stdout:
                    f.write(f"=== STDOUT ===\n{exc.stdout}\n\n")
                if exc.stderr:
                    f.write(f"=== STDERR ===\n{exc.stderr}\n")

        except Exception as exc:
            result["duration"] = time.time() - start
            result["status"] = "error"
            result["error"] = str(exc)
            _kill_container(workspace)
            with log_file.open("w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== MASKED KERNEL ===\n{masked_kernel_path}\n\n")
                f.write(f"=== ERROR ===\n{exc}\n")

        return result
    finally:
        gpu_queue.put(gpu_id)


def summarize_results(results: list[dict[str, Any]], output_root: Path) -> None:
    results.sort(key=lambda r: r["exp_index"])

    total = len(results)
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]
    infra_failures = [r for r in results if r["status"] == INFRA_FAILED]
    timeouts = [r for r in results if r["status"] == "timeout"]
    errors = [r for r in results if r["status"] == "error"]

    durations = [r["duration"] for r in results]
    avg_duration = sum(durations) / len(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0
    min_duration = min(durations) if durations else 0.0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:    {total}")
    print(f"Success:  {len(successes)}")
    print(f"Failed:   {len(failures)}")
    print(f"Infra:    {len(infra_failures)}")
    print(f"Timeout:  {len(timeouts)}")
    print(f"Error:    {len(errors)}")
    print(f"Duration: avg={avg_duration:.1f}s  min={min_duration:.1f}s  max={max_duration:.1f}s")

    if failures:
        print(f"\nFailed experiments: {[r['exp_name'] for r in failures]}")
    if infra_failures:
        print(f"\nInfra-failed experiments: {[r['exp_name'] for r in infra_failures]}")
    if timeouts:
        print(f"Timed out experiments: {[r['exp_name'] for r in timeouts]}")
    if errors:
        print(f"Errored experiments: {[r['exp_name'] for r in errors]}")

    by_tag: dict[str, dict[str, int]] = {}
    for result in results:
        tag = result.get("prompt_tag", "?")
        bucket = by_tag.setdefault(
            tag,
            {"total": 0, "success": 0, "failed": 0, INFRA_FAILED: 0, "timeout": 0, "error": 0},
        )
        bucket["total"] += 1
        bucket[result["status"]] = bucket.get(result["status"], 0) + 1
    if by_tag:
        print("\nPer prompt_tag:")
        for tag, bucket in sorted(by_tag.items()):
            print(
                f"  {tag}: total={bucket['total']} success={bucket.get('success', 0)} "
                f"failed={bucket.get('failed', 0)} infra={bucket.get(INFRA_FAILED, 0)} "
                f"timeout={bucket.get('timeout', 0)} "
                f"error={bucket.get('error', 0)}"
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "success": len(successes),
        "failed": len(failures),
        "infra_failed": len(infra_failures),
        "timeout": len(timeouts),
        "error": len(errors),
        "avg_duration": round(avg_duration, 2),
        "min_duration": round(min_duration, 2),
        "max_duration": round(max_duration, 2),
        "by_prompt_tag": by_tag,
        "results": results,
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary written to: {summary_path}")


def run_experiments(
    experiments: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    gpu_queue: queue.Queue,
    output_root: Path,
    max_parallel: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_pos = 0
    futures: dict[Any, int] = {}
    stop_scheduling = False

    def submit_next(pool: ThreadPoolExecutor) -> bool:
        nonlocal next_pos
        if next_pos >= len(experiments):
            return False
        exp_index, item = experiments[next_pos]
        ok, error = check_service_health(args.service_url, str(item["definition"]))
        if not ok:
            raise InfraAbort(
                f"INFRA_FAILED: profiling service unhealthy before exp_{exp_index:03d}: {error}"
            )
        futures[pool.submit(run_single_experiment, exp_index, item, args, gpu_queue, output_root)] = exp_index
        next_pos += 1
        return True

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        for _ in range(max_parallel):
            if not submit_next(pool):
                break

        completed = 0
        while futures:
            for future in as_completed(list(futures)):
                exp_index = futures.pop(future)
                result = future.result()
                results.append(result)
                completed += 1
                status_str = result["status"].upper()
                duration = result["duration"]
                print(
                    f"[{completed}/{len(experiments)}] exp_{exp_index:03d} ({result['prompt_tag']}) "
                    f"on no GPU: {status_str} ({duration:.1f}s)"
                )
                if result["status"] == INFRA_FAILED:
                    print(
                        f"INFRA_FAILED: {result['exp_name']} saw profiling-service failure; "
                        "stopping new experiment scheduling.",
                        file=sys.stderr,
                    )
                    stop_scheduling = True
                if not stop_scheduling:
                    submit_next(pool)
                break

    return results


def validate_tinker_args(args: argparse.Namespace) -> None:
    if args.model not in TINKER_MODEL_NAMES:
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError("--model is a Tinker model but TINKER_API_KEY is not set")
    if (
        not args.tinker_checkpoints_jsonl
        and not os.environ.get("TINKER_MODEL_PATH")
        and not os.environ.get("TINKER_CHECKPOINTS_JSONL")
    ):
        raise ValueError(
            "--model is a Tinker model but no checkpoint is configured. "
            "Pass --tinker-checkpoints-jsonl, or set TINKER_MODEL_PATH / TINKER_CHECKPOINTS_JSONL."
        )
    if args.tinker_checkpoints_jsonl and not Path(args.tinker_checkpoints_jsonl).exists():
        raise ValueError(f"--tinker-checkpoints-jsonl {args.tinker_checkpoints_jsonl!r} does not exist")


def load_config(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        config_items = json.load(f)
    if not isinstance(config_items, list) or not config_items:
        raise ValueError(f"{path} must be a non-empty JSON list")
    return config_items


def main() -> None:
    args = parse_args()

    try:
        validate_tinker_args(args)
        config_path = Path(args.config).resolve()
        config_items = load_config(config_path)
        experiments = expand_config(config_items)
        for _, item in experiments:
            masked_kernel_path = Path(item["masked_kernel_path"]).expanduser()
            if not masked_kernel_path.exists():
                raise ValueError(f"masked_kernel_path does not exist: {masked_kernel_path}")
            test_path = Path(item["test_path"]).expanduser()
            if not test_path.is_file():
                raise ValueError(f"test_path does not exist: {test_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    with (MULTITURN_DIR / "prompt_configs" / "hub.json").open() as f:
        hub = json.load(f)
    unique_tags = sorted({item["prompt_tag"] for item in config_items})
    print(f"Pre-building prompt docs for {len(unique_tags)} tags: {unique_tags}")
    for tag in unique_tags:
        build_doc(tag, hub)

    planned_total = len(experiments)
    if args.start_index < 0:
        print(f"Error: --start-index must be >= 0, got {args.start_index}.", file=sys.stderr)
        raise SystemExit(1)
    if args.end_index is not None and args.end_index < args.start_index:
        print(
            f"Error: --end-index ({args.end_index}) must be >= --start-index ({args.start_index}).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    experiments = [
        (index, item)
        for index, item in experiments
        if index >= args.start_index and (args.end_index is None or index <= args.end_index)
    ]
    if not experiments:
        end_label = args.end_index if args.end_index is not None else planned_total - 1
        print(
            f"Error: no experiments selected from {planned_total} planned experiments "
            f"for inclusive range [{args.start_index}, {end_label}].",
            file=sys.stderr,
        )
        raise SystemExit(1)

    total = len(experiments)
    max_parallel = args.max_parallel or total

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path(f"./eval_runs/masked_v2_{timestamp}")
    if args.resume and not args.output_root:
        print("Error: --resume requires --output-root.", file=sys.stderr)
        raise SystemExit(1)
    if args.resume and not output_root.exists():
        print(f"Error: --resume output root {output_root} does not exist.", file=sys.stderr)
        raise SystemExit(1)
    if not args.resume and output_root.exists():
        print(f"Error: Output root {output_root} already exists.", file=sys.stderr)
        raise SystemExit(1)
    output_root = output_root.resolve()

    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "trajectories").mkdir(parents=True, exist_ok=True)
    (output_root / "success").mkdir(parents=True, exist_ok=True)
    (output_root / "prompts").mkdir(parents=True, exist_ok=True)

    resume_info: dict[int, dict[str, Any]] = {}
    if args.resume:
        resume_info = prepare_resume(experiments, output_root)
        experiments = [(index, item) for index, item in experiments if index in resume_info]
        if not experiments:
            print("No missing, infra-failed, or API-failed trajectories found to resume.")
            return
        total = len(experiments)
        max_parallel = args.max_parallel or total

    plan = [{"exp_index": index, **item} for index, item in experiments]
    if args.resume:
        (output_root / "resume_plan.json").write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": str(config_path),
                    "prompt_kind": "masked_fill",
                    "resume_info": resume_info,
                    "plan": plan,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        (output_root / "plan.json").write_text(
            json.dumps({"config": str(config_path), "prompt_kind": "masked_fill", "plan": plan}, indent=2) + "\n"
        )

    print(f"Launching {total} masked fill-in experiments from {config_path}")
    if total != planned_total:
        end_label = args.end_index if args.end_index is not None else planned_total - 1
        print(f"Selected experiment range: [{args.start_index}, {end_label}] of {planned_total}")
    print(f"GPUs: none (--without-local-gpu)  |  Max parallel: {max_parallel}")
    definitions = sorted({str(item["definition"]) for _, item in experiments})
    test_paths = sorted({str(Path(item["test_path"]).expanduser().resolve()) for _, item in experiments})
    print(
        f"Model: {args.model}  |  Definitions: {definitions}  |  "
        f"Test paths: {len(test_paths)}  |  GPU arch: {args.gpu_arch}"
    )
    print(f"Service URL: {args.service_url}")
    print(f"Output: {output_root}")
    print(f"Timeout: {args.timeout}s per experiment")
    if args.resume:
        print(f"Resume mode: wrote {output_root / 'resume_plan.json'}")
    print()

    gpu_queue: queue.Queue = queue.Queue()
    for _ in range(max_parallel):
        gpu_queue.put("none")

    wall_start = time.time()

    try:
        results = run_experiments(experiments, args, gpu_queue, output_root, max_parallel)
    except InfraAbort as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    wall_time = time.time() - wall_start
    print(f"\nAll experiments completed in {wall_time:.1f}s wall time")
    summarize_results(results, output_root)
    if any(result["status"] == INFRA_FAILED for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
