#!/usr/bin/env python3
"""Parallel launcher for failed-kernel fixup runs.

This is a narrow variant of ../run_parallel_v2.py for prompt configs whose
items include `error_kernel_path` and `error_log_path`, for example:

    {
        "num_trajectories": 4,
        "num_turns": 1,
        "target_speedup": 0.62,
        "prompt_tag": "hopper-no-hint",
        "error_kernel_path": "/path/to/kernel.cu",
        "error_log_path": "/path/to/error.log",
        "definition": "mha_with_lse_d128",
        "test_path": "/path/to/test.py"
    }

Each expanded experiment starts run_v2.py with a per-experiment user template
whose first user prompt asks the model to fix the provided kernel using the
provided error log.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MULTITURN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MULTITURN_DIR))

from build_doc_v2 import build_doc  # noqa: E402
from common import GPU_ARCH_NVCC  # noqa: E402
from launcher_utils import (  # noqa: E402
    InfraAbort,
    build_child_env,
    check_service_health,
    combined_output,
    experiments_from_plan,
    kill_container,
    load_config,
    load_current_plan_data,
    materialize_run_fields,
    plan_from_experiments,
    run_targets_from_args,
    validate_tinker_args,
)
from resume_utils import (  # noqa: E402
    API_FAILED,
    INFRA_FAILED,
    is_api_failure_output,
    is_infra_failure_output,
    prepare_resume,
    trajectory_infra_failure_turn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel launcher for run_v2.py using failed-kernel fixup prompts",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to config JSON, required for fresh runs and rejected for --resume. "
            "Each item must include "
            "num_trajectories/num_turns/target_speedup/prompt_tag/"
            "error_kernel_path/error_log_path, plus definition/test_path either "
            "per item or from CLI target arguments."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max concurrent workers (default: total experiments)",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=None,
        help=(
            "Max concurrent trajectories allowed to run the compile+profile test.py stage. "
            "The fix launcher always runs children with --without-local-gpu; set --max-parallel "
            "higher than this to keep LLM generation oversubscribed while profiling stays bounded."
        ),
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
        help="Output directory for a single-target run (default: ./eval_runs/fix_v2_TIMESTAMP)",
    )
    parser.add_argument(
        "--output-roots",
        nargs="+",
        default=None,
        help=(
            "Output directories for a multi-target run, paired positionally with "
            "--definitions/--test-paths/--configs. Existing roots are resumed automatically; "
            "missing roots are launched fresh."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume existing --output-root/--output-roots after profiling-service infra failure. "
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
        "--llm-context-policy",
        choices=("full", "latest-pair", "single-user"),
        default="full",
        help="LLM-visible context policy passed to run_v2.py (default: full)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output passed to run_v2.py",
    )
    parser.add_argument(
        "--definition",
        type=str,
        default=None,
        help="Definition name for a single-target run; config items may also provide definition",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to the test file for a single-target run; config items may also provide test_path",
    )
    parser.add_argument(
        "--definitions",
        nargs="+",
        default=None,
        help=(
            "Definition names for a multi-target run. Must be paired positionally with "
            "--test-paths and --configs."
        ),
    )
    parser.add_argument(
        "--test-paths",
        nargs="+",
        default=None,
        help="Test file paths for a multi-target run, paired positionally with --definitions.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "Fix prompt config JSON paths for a multi-target run, paired positionally with "
            "--definitions/--test-paths/--output-roots. Required for multi-target runs."
        ),
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
                "error_kernel_path",
                "error_log_path",
            )
            if key not in item
        ]
        if missing:
            raise ValueError(f"config item {item_index} missing required key(s): {', '.join(missing)}")
        for _ in range(int(item["num_trajectories"])):
            experiments.append((exp_index, dict(item)))
            exp_index += 1
    return experiments


def expand_target_configs(
    targets: list[dict[str, str]],
    *,
    resume_existing: bool = False,
) -> tuple[str, list[tuple[int, dict[str, Any]]]]:
    experiments: list[tuple[int, dict[str, Any]]] = []
    config_labels: list[str] = []
    for target in targets:
        config_path = Path(target["config"]).resolve()
        output_root = Path(target["output_root"]).expanduser().resolve()

        if resume_existing and output_root.exists():
            current_plan_data = load_current_plan_data(output_root)
            config_labels.append(str(current_plan_data.get("config", output_root / "plan.json")))
            target_experiments = experiments_from_plan(current_plan_data["plan"])
            for exp_index, item in target_experiments:
                item = dict(item)
                item["_output_root"] = str(output_root)
                item["_resume_existing_root"] = "1"
                experiments.append((exp_index, item))
            continue

        config_labels.append(str(config_path))
        target_without_config = {
            key: value
            for key, value in target.items()
            if key not in ("config", "output_root")
        }
        target_experiments = expand_config(load_config(config_path))
        target_experiments = materialize_run_fields(target_experiments, [target_without_config])
        for exp_index, item in target_experiments:
            item = dict(item)
            item["config"] = str(config_path)
            item["_output_root"] = str(output_root)
            experiments.append((exp_index, item))
    return json.dumps(config_labels), experiments


def experiment_output_root(item: dict[str, Any], default_output_root: Path) -> Path:
    return Path(str(item.get("_output_root") or default_output_root)).expanduser().resolve()


def group_experiments_by_root(
    experiments: list[tuple[int, dict[str, Any]]],
    default_output_root: Path,
) -> dict[Path, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[Path, list[tuple[int, dict[str, Any]]]] = {}
    for exp_index, item in experiments:
        grouped.setdefault(experiment_output_root(item, default_output_root), []).append((exp_index, item))
    return grouped


def group_results_by_root(results: list[dict[str, Any]]) -> dict[Path, list[dict[str, Any]]]:
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(Path(str(result["output_root"])).resolve(), []).append(result)
    return grouped


def build_fixit_prompt(error_kernel_path: Path, error_log_path: Path) -> str:
    error_kernel_source = error_kernel_path.read_text()
    error_log = error_log_path.read_text()
    return (
        "Task:\n"
        "{task_content}\n\n"
        "Kernel\n\n"
        "```cpp\n"
        f"{error_kernel_source.rstrip()}\n"
        "```\n"
        "Error Message\n\n"
        f"{error_log.rstrip()}\n"
        "Your task is to fix the CUDA kernel with the minimal change. The kernel implements the given PyTorch reference.\n\n"
    )


def run_single_experiment(
    exp_index: int,
    item: dict[str, Any],
    args: argparse.Namespace,
    gpu_queue: queue.Queue,
    output_root: Path,
) -> dict[str, Any]:
    gpu_id = gpu_queue.get()
    try:
        output_root = experiment_output_root(item, output_root)
        exp_name = f"exp_{exp_index:03d}"
        workspace = output_root / exp_name
        log_file = output_root / "logs" / f"{exp_name}.log"
        trajectory = output_root / "trajectories" / f"{exp_name}.json"
        success_dir = output_root / "success" / exp_name
        prompt_file = output_root / "prompts" / f"{exp_name}.txt"
        error_kernel_path = Path(item["error_kernel_path"]).expanduser().resolve()
        error_log_path = Path(item["error_log_path"]).expanduser().resolve()
        test_path = Path(item["test_path"]).expanduser().resolve()
        definition = str(item["definition"])

        workspace.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(build_fixit_prompt(error_kernel_path, error_log_path))

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
            "--llm-context-policy",
            args.llm_context_policy,
        ]
        cmd.append("--without-local-gpu")
        if args.verbose:
            cmd.append("-v")

        result: dict[str, Any] = {
            "exp_index": exp_index,
            "exp_name": exp_name,
            "gpu_id": gpu_id,
            "prompt_tag": item["prompt_tag"],
            "definition": definition,
            "test_path": str(test_path),
            "output_root": str(output_root),
            "error_kernel_path": str(error_kernel_path),
            "error_log_path": str(error_log_path),
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
        if args.max_profiles:
            child_env = dict(os.environ) if child_env is None else dict(child_env)
            child_env["ACCRL_PROFILE_STAGE_TICKET_LOCK"] = "1"
            child_env["ACCRL_PROFILE_SLOT_DIR"] = str(
                getattr(args, "_profile_slot_dir", output_root / ".profile_slots")
            )
            child_env["ACCRL_MAX_PROFILES"] = str(args.max_profiles)
        resume_turn = int(item.get("_resume_turn") or 0)
        resume_trajectory = item.get("_resume_trajectory")
        if resume_turn > 0 and resume_trajectory:
            cmd.extend([
                "--resume-trajectory",
                str(resume_trajectory),
                "--resume-turn",
                str(resume_turn),
            ])

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
            output_text = combined_output(proc.stdout, proc.stderr)
            try:
                output_text = f"{output_text}\n{trajectory.read_text()}"
            except OSError:
                pass
            infra_turn = trajectory_infra_failure_turn(trajectory)
            if infra_turn is not None:
                result["status"] = INFRA_FAILED
                result["error"] = f"Profiling service infra failure in trajectory turn {infra_turn}"
                result["infra_failure_turn"] = infra_turn
            elif is_infra_failure_output(output_text):
                result["status"] = INFRA_FAILED
                result["error"] = "Profiling service infra failure"
            elif proc.returncode == 0:
                result["status"] = "success"
            elif is_api_failure_output(output_text):
                result["status"] = API_FAILED
                result["error"] = "Model/provider API failure"
            else:
                result["status"] = "failed"
            if resume_turn > 0:
                result["resume_turn"] = resume_turn
                result["resume_trajectory"] = str(resume_trajectory)

            with log_file.open("w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== ERROR KERNEL ===\n{error_kernel_path}\n\n")
                f.write(f"=== ERROR LOG ===\n{error_log_path}\n\n")
                f.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
                f.write(f"=== STDERR ===\n{proc.stderr}\n")

        except subprocess.TimeoutExpired as exc:
            result["duration"] = time.time() - start
            timeout_output = combined_output(exc.stdout, exc.stderr)
            if is_api_failure_output(timeout_output):
                result["status"] = API_FAILED
                result["error"] = "Model/provider API failure before timeout"
            else:
                result["status"] = "timeout"
                result["error"] = f"Timed out after {args.timeout}s"
            kill_container(workspace)
            with log_file.open("w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== ERROR KERNEL ===\n{error_kernel_path}\n\n")
                f.write(f"=== ERROR LOG ===\n{error_log_path}\n\n")
                f.write(f"=== TIMEOUT after {args.timeout}s ===\n")
                if exc.stdout:
                    f.write(f"=== STDOUT ===\n{exc.stdout}\n\n")
                if exc.stderr:
                    f.write(f"=== STDERR ===\n{exc.stderr}\n")

        except Exception as exc:
            result["duration"] = time.time() - start
            if is_api_failure_output(str(exc)):
                result["status"] = API_FAILED
                result["error"] = "Model/provider API failure"
            else:
                result["status"] = "error"
                result["error"] = str(exc)
            kill_container(workspace)
            with log_file.open("w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== ERROR KERNEL ===\n{error_kernel_path}\n\n")
                f.write(f"=== ERROR LOG ===\n{error_log_path}\n\n")
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
    api_failures = [r for r in results if r["status"] == API_FAILED]
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
    print(f"API:      {len(api_failures)}")
    print(f"Timeout:  {len(timeouts)}")
    print(f"Error:    {len(errors)}")
    print(f"Duration: avg={avg_duration:.1f}s  min={min_duration:.1f}s  max={max_duration:.1f}s")

    if failures:
        print(f"\nFailed experiments: {[r['exp_name'] for r in failures]}")
    if infra_failures:
        print(f"\nInfra-failed experiments: {[r['exp_name'] for r in infra_failures]}")
    if api_failures:
        print(f"\nAPI-failed experiments: {[r['exp_name'] for r in api_failures]}")
    if timeouts:
        print(f"Timed out experiments: {[r['exp_name'] for r in timeouts]}")
    if errors:
        print(f"Errored experiments: {[r['exp_name'] for r in errors]}")

    by_tag: dict[str, dict[str, int]] = {}
    for result in results:
        tag = result.get("prompt_tag", "?")
        bucket = by_tag.setdefault(
            tag,
            {
                "total": 0,
                "success": 0,
                "failed": 0,
                INFRA_FAILED: 0,
                API_FAILED: 0,
                "timeout": 0,
                "error": 0,
            },
        )
        bucket["total"] += 1
        bucket[result["status"]] = bucket.get(result["status"], 0) + 1
    if by_tag:
        print("\nPer prompt_tag:")
        for tag, bucket in sorted(by_tag.items()):
            print(
                f"  {tag}: total={bucket['total']} success={bucket.get('success', 0)} "
                f"failed={bucket.get('failed', 0)} infra={bucket.get(INFRA_FAILED, 0)} "
                f"api={bucket.get(API_FAILED, 0)} "
                f"timeout={bucket.get('timeout', 0)} "
                f"error={bucket.get('error', 0)}"
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "success": len(successes),
        "failed": len(failures),
        "infra_failed": len(infra_failures),
        "api_failed": len(api_failures),
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
        futures[pool.submit(run_single_experiment, exp_index, item, args, gpu_queue, output_root)] = (
            exp_index,
            experiment_output_root(item, output_root),
        )
        next_pos += 1
        return True

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        for _ in range(max_parallel):
            if not submit_next(pool):
                break

        completed = 0
        while futures:
            for future in as_completed(list(futures)):
                exp_index, item_output_root = futures.pop(future)
                result = future.result()
                results.append(result)
                completed += 1
                status_str = result["status"].upper()
                duration = result["duration"]
                print(
                    f"[{completed}/{len(experiments)}] {item_output_root.name}/exp_{exp_index:03d} "
                    f"({result['prompt_tag']}) on no GPU: {status_str} ({duration:.1f}s)"
                )
                if result["status"] == INFRA_FAILED:
                    print(
                        f"INFRA_FAILED: {result['exp_name']} saw profiling-service failure; "
                        "stopping new experiment scheduling.",
                        file=sys.stderr,
                    )
                    stop_scheduling = True
                elif result["status"] == API_FAILED:
                    print(
                        f"API_FAILED: {result['exp_name']} saw model/provider API failure; "
                        "stopping new experiment scheduling.",
                        file=sys.stderr,
                    )
                    stop_scheduling = True
                if not stop_scheduling:
                    submit_next(pool)
                break

    return results


def main() -> None:
    args = parse_args()
    multi_tuple_args = bool(args.output_roots and args.configs)
    if args.resume and (args.config or args.configs) and not multi_tuple_args:
        print("Error: --config/--configs is not allowed with --resume.", file=sys.stderr)
        raise SystemExit(1)
    if not args.resume and not args.config and not args.configs:
        print("Error: --config or --configs is required for a fresh run.", file=sys.stderr)
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path(f"./eval_runs/fix_v2_{timestamp}")
    output_roots = [Path(path).expanduser().resolve() for path in args.output_roots or []]
    multi_root_mode = bool(output_roots)
    if args.output_root and output_roots:
        print("Error: --output-root cannot be used together with --output-roots.", file=sys.stderr)
        raise SystemExit(1)
    if args.resume and not args.output_root and not output_roots:
        print("Error: --resume requires --output-root or --output-roots.", file=sys.stderr)
        raise SystemExit(1)
    output_root = output_root.resolve()
    roots_to_check = output_roots if output_roots else [output_root]
    if args.resume and not multi_tuple_args:
        missing_roots = [root for root in roots_to_check if not root.exists()]
        if missing_roots:
            print(f"Error: --resume output root does not exist: {missing_roots[0]}", file=sys.stderr)
            raise SystemExit(1)
    else:
        existing_roots = [root for root in roots_to_check if root.exists()]
        if existing_roots and not multi_root_mode:
            print(f"Error: Output root {existing_roots[0]} already exists.", file=sys.stderr)
            raise SystemExit(1)

    try:
        run_targets = None if args.resume and not multi_tuple_args else run_targets_from_args(args)
        validate_tinker_args(args)
        if args.max_profiles is not None and args.max_profiles <= 0:
            raise ValueError("--max-profiles must be positive when set")
        if args.resume and not multi_tuple_args:
            loaded_experiments: list[tuple[int, dict[str, Any]]] = []
            config_labels = []
            for root in roots_to_check:
                current_plan_data = load_current_plan_data(root)
                config_labels.append(str(current_plan_data.get("config", root / "plan.json")))
                root_experiments = experiments_from_plan(current_plan_data["plan"])
                for exp_index, item in root_experiments:
                    item = dict(item)
                    item["_output_root"] = str(root)
                    loaded_experiments.append((exp_index, item))
            config_label = json.dumps(config_labels) if len(config_labels) > 1 else config_labels[0]
            experiments = loaded_experiments
        else:
            if args.configs:
                if args.config:
                    raise ValueError("--config cannot be used together with --configs")
                if not run_targets or not all(target.get("config") for target in run_targets):
                    raise ValueError("--configs must be paired with --definitions and --test-paths")
                config_label, experiments = expand_target_configs(
                    run_targets,
                    resume_existing=multi_root_mode,
                )
            else:
                config_path = Path(args.config).resolve()
                config_label = str(config_path)
                config_items = load_config(config_path)
                experiments = expand_config(config_items)
                experiments = materialize_run_fields(experiments, run_targets)
        for _, item in experiments:
            error_kernel_path = Path(item["error_kernel_path"]).expanduser()
            if not error_kernel_path.is_file():
                raise ValueError(f"error_kernel_path does not exist: {error_kernel_path}")
            error_log_path = Path(item["error_log_path"]).expanduser()
            if not error_log_path.is_file():
                raise ValueError(f"error_log_path does not exist: {error_log_path}")
            test_path = Path(item["test_path"]).expanduser()
            if not test_path.is_file():
                raise ValueError(f"test_path does not exist: {test_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    with (MULTITURN_DIR / "prompt_configs" / "hub.json").open() as f:
        hub = json.load(f)
    unique_tags = sorted({item["prompt_tag"] for _, item in experiments})
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

    roots_for_selected = group_experiments_by_root(experiments, output_root)
    for root in roots_for_selected:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "trajectories").mkdir(parents=True, exist_ok=True)
        (root / "success").mkdir(parents=True, exist_ok=True)
        (root / "prompts").mkdir(parents=True, exist_ok=True)

    resume_roots = {
        root
        for root, root_experiments in roots_for_selected.items()
        if args.resume or any(item.get("_resume_existing_root") for _, item in root_experiments)
    }
    resume_info_by_root: dict[str, dict[int, dict[str, Any]]] = {}

    if resume_roots:
        resume_definitions = {
            str(item["definition"])
            for root, root_experiments in roots_for_selected.items()
            if root in resume_roots
            for _, item in root_experiments
        }
        for definition in sorted(resume_definitions):
            ok, error = check_service_health(args.service_url, definition)
            if not ok:
                print(
                    f"Error: profiling service unhealthy before resume for {definition}: {error}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
        for root, root_experiments in roots_for_selected.items():
            if root not in resume_roots:
                continue
            resume_info_by_root[str(root)] = prepare_resume(root_experiments, root)
        experiments = [
            (index, item)
            for index, item in experiments
            if (
                experiment_output_root(item, output_root) not in resume_roots
                or index in resume_info_by_root.get(str(experiment_output_root(item, output_root)), {})
            )
        ]
        if not experiments:
            print("No missing, infra-failed, or API-failed trajectories found to resume.")
            return

    total = len(experiments)
    max_parallel = args.max_parallel or total

    roots_for_selected = group_experiments_by_root(experiments, output_root)
    plan_by_root = {
        root: plan_from_experiments(root_experiments)
        for root, root_experiments in roots_for_selected.items()
    }
    active_resume_roots = set(plan_by_root) & resume_roots
    if active_resume_roots:
        for root, root_plan in plan_by_root.items():
            if root not in active_resume_roots:
                continue
            (root / "resume_plan.json").write_text(
                json.dumps(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "config": config_label,
                        "prompt_kind": "fixit",
                        "resume_info": resume_info_by_root.get(str(root), {}),
                        "plan": root_plan,
                    },
                    indent=2,
                )
                + "\n"
            )
    for root, root_plan in plan_by_root.items():
        if root in resume_roots:
            continue
        (root / "plan.json").write_text(
            json.dumps({"config": config_label, "prompt_kind": "fixit", "plan": root_plan}, indent=2) + "\n"
        )

    print(f"Launching {total} fix-kernel experiments from {config_label}")
    if total != planned_total:
        end_label = args.end_index if args.end_index is not None else planned_total - 1
        print(f"Selected experiment range: [{args.start_index}, {end_label}] of {planned_total}")
    max_profiles_label = args.max_profiles if args.max_profiles is not None else "disabled"
    print(
        f"GPUs: none (--without-local-gpu)  |  Max parallel: {max_parallel}  |  "
        f"Max profiles: {max_profiles_label}"
    )
    definitions = sorted({str(item["definition"]) for _, item in experiments})
    test_paths = sorted({str(Path(item["test_path"]).expanduser().resolve()) for _, item in experiments})
    print(
        f"Model: {args.model}  |  Definitions: {definitions}  |  "
        f"Test paths: {len(test_paths)}  |  GPU arch: {args.gpu_arch}"
    )
    print(f"Service URL: {args.service_url}")
    if multi_root_mode:
        print(f"Outputs: {sorted(str(root) for root in roots_for_selected)}")
    else:
        print(f"Output: {output_root}")
    print(f"Timeout: {args.timeout}s per experiment")
    if active_resume_roots:
        print("Resume mode: wrote resume_plan.json under selected output root(s)")
    print()

    gpu_queue: queue.Queue = queue.Queue()
    for _ in range(max_parallel):
        gpu_queue.put("none")

    wall_start = time.time()
    if args.max_profiles:
        profile_slot_parent = sorted(roots_for_selected, key=str)[0]
        args._profile_slot_dir = profile_slot_parent / ".profile_slots"
    try:
        results = run_experiments(experiments, args, gpu_queue, output_root, max_parallel)
    except InfraAbort as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    wall_time = time.time() - wall_start
    print(f"\nAll experiments completed in {wall_time:.1f}s wall time")
    for root, root_results in sorted(group_results_by_root(results).items()):
        summarize_results(root_results, root)
    if any(result["status"] in (INFRA_FAILED, API_FAILED) for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
