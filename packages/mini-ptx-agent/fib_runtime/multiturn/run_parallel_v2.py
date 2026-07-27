#!/usr/bin/env python3
"""Parallel launcher for run_v2.py driven by a prompt_configs JSON file.

Each item in the config JSON looks like:
    {"num_trajectories": N, "num_turns": M, "target_speedup": S, "prompt_tag": TAG}

Experiments are numbered sequentially in config-file order: item i contributes
N_i experiments whose indices are [sum_{j<i} N_j, sum_{j<i} N_j + N_i).

Example:
    python run_parallel_v2.py \
        --config prompt_configs/2026-0421-2352.json \
        --definition gemm_n6144_k4096 \
        --model gemini-3.1-pro-preview \
        --test-path ../mini_swe_agent_docker/envs/test_profile_cuda_gemm_n6144_k4096.py

To use a Tinker-hosted SFT checkpoint via Tinker's OpenAI-compatible API:
    TINKER_API_KEY=... python run_parallel_v2.py \
        --config ... --definition ... --test-path ... \
        --model Qwen3.5-35B-A3B-tinker \
        --tinker-checkpoints-jsonl /path/to/run/checkpoints.jsonl \
        [--tinker-checkpoint-name final]
"""

import argparse
import json
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
import os

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import GPU_ARCH_NVCC  # noqa: E402
from build_doc_v2 import build_doc  # noqa: E402
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
    prepare_extend_turns,
    prepare_resume,
    trajectory_infra_failure_turn,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel launcher for run_v2.py driven by a config JSON",
    )
    parser.add_argument(
        "--config", default=None,
        help=(
            "Path to config JSON (required for fresh runs and --resume --extend-turns; "
            "rejected for plain --resume)."
        ),
    )
    parser.add_argument(
        "--gpus", type=str, default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=None,
        help="Max concurrent workers (default: number of GPUs, or total experiments with --without-local-gpu)",
    )
    parser.add_argument(
        "--max-profiles", type=int, default=None,
        help=(
            "Max concurrent trajectories allowed to run the compile+profile test.py stage. "
            "Only valid with --without-local-gpu; set --max-parallel higher than this to keep "
            "LLM generation oversubscribed while profiling stays bounded."
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.environ.get("PTXBENCH_EVAL_IMAGE", "ptxbench-eval:dev"),
        help="Docker image (passthrough to run_v2.py)",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Output directory for a single-target run (default: ./eval_runs/multiturn_v2_TIMESTAMP)",
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
        "--resume", action="store_true",
        help=(
            "Resume existing --output-root/--output-roots after profiling-service infra failure. "
            "Polluted trajectory tails are trimmed before relaunch."
        ),
    )
    parser.add_argument(
        "--extend-turns", action="store_true",
        help=(
            "Extend existing --output-root/--output-roots by replacing plan.json only when the "
            "new expanded plan differs from the current plan only in num_turns."
        ),
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="First expanded experiment index to launch (default: 0).",
    )
    parser.add_argument(
        "--end-index", type=int, default=None,
        help="Last expanded experiment index to launch, inclusive (default: run through the end).",
    )
    parser.add_argument(
        "--timeout", type=int, default=14400,
        help="Per-experiment timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--turn-timeout", type=int, default=360,
        help="Per-turn timeout in seconds, passed to run_v2.py (default: 180)",
    )
    parser.add_argument(
        "--llm-context-policy",
        choices=("full", "latest-pair", "single-user"),
        default="full",
        help="LLM-visible context policy passed to run_v2.py (default: full)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output (passthrough to run_v2.py)",
    )
    parser.add_argument(
        "--definition", type=str, default=None,
        help="Definition name for a single-target run; config items may also provide definition",
    )
    parser.add_argument(
        "--test-path", type=str, default=None,
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
            "Prompt config JSON paths for a multi-target run, paired positionally with "
            "--definitions/--test-paths/--output-roots. Required for multi-target runs."
        ),
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (passthrough to run_v2.py)",
    )
    parser.add_argument(
        "--service-url", type=str, default="http://localhost:10000",
        help="Profiling service URL (passthrough to run_v2.py)",
    )
    parser.add_argument(
        "--gpu-arch", type=str, choices=list(GPU_ARCH_NVCC.keys()), default="hopper",
        help="GPU architecture (passthrough to run_v2.py, default: hopper)",
    )
    parser.add_argument(
        "--without-local-gpu", action="store_true",
        help="Run without local GPU access (passthrough to run_v2.py)",
    )
    parser.add_argument(
        "--tinker-checkpoints-jsonl", type=str, default=None,
        help="Path to a Tinker checkpoints.jsonl. Required when --model is a Tinker model "
             "checkpoint (e.g. Qwen3.5-35B-A3B-tinker), unless TINKER_MODEL_PATH or TINKER_CHECKPOINTS_JSONL "
             "is already set in the environment.",
    )
    parser.add_argument(
        "--tinker-checkpoint-name", type=str, default=None,
        help="Name of the checkpoint inside --tinker-checkpoints-jsonl (default: 'final').",
    )
    return parser.parse_args()


def expand_config(config_items: list[dict]) -> list[tuple[int, dict]]:
    """Return [(exp_index, item), ...] with indices in config-file order."""
    experiments = []
    exp_index = 0
    for item in config_items:
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
            validate_plan_run_fields(target_experiments)
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


def validate_plan_run_fields(experiments: list[tuple[int, dict[str, Any]]]) -> None:
    missing = [
        index
        for index, item in experiments
        if not item.get("definition") or not item.get("test_path")
    ]
    if missing:
        raise ValueError(
            "plan.json is missing definition/test_path for experiments "
            f"{missing[:5]}. Backfill plan.json from the original trajectories/logs before resuming."
        )


def _plan_without_turns(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for entry in plan:
        stripped.append({key: value for key, value in entry.items() if key != "num_turns"})
    return stripped


def _plan_turns(plan: list[dict[str, Any]]) -> list[Any]:
    return [entry.get("num_turns") for entry in plan]


def load_current_plan(output_root: Path) -> list[dict[str, Any]]:
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
    return plan


def validate_extend_plan(current_plan: list[dict[str, Any]], new_plan: list[dict[str, Any]]) -> None:
    if len(current_plan) != len(new_plan):
        raise ValueError(
            f"plan length changed: current has {len(current_plan)} entries, "
            f"new has {len(new_plan)} entries"
        )
    if _plan_without_turns(current_plan) != _plan_without_turns(new_plan):
        for idx, (current_entry, new_entry) in enumerate(zip(current_plan, new_plan)):
            if {k: v for k, v in current_entry.items() if k != "num_turns"} != {
                k: v for k, v in new_entry.items() if k != "num_turns"
            }:
                raise ValueError(
                    "new plan differs from current plan in fields other than num_turns "
                    f"at entry {idx}: current={current_entry}, new={new_entry}"
                )
        raise ValueError("new plan differs from current plan in fields other than num_turns")
    if _plan_turns(current_plan) == _plan_turns(new_plan):
        raise ValueError("new plan has the same num_turns values as the current plan")
    for idx, (current_turns, new_turns) in enumerate(zip(_plan_turns(current_plan), _plan_turns(new_plan))):
        if int(new_turns) < int(current_turns):
            raise ValueError(
                "new plan reduces num_turns instead of extending them "
                f"at entry {idx}: current={current_turns}, new={new_turns}"
            )


def run_single_experiment(
    exp_index: int,
    item: dict,
    args,
    gpu_queue: queue.Queue,
    output_root: Path,
) -> dict:
    """Run a single experiment as a subprocess, reserving a GPU from the queue."""
    gpu_id = gpu_queue.get()
    try:
        output_root = experiment_output_root(item, output_root)
        exp_name = f"exp_{exp_index:03d}"
        workspace = output_root / exp_name
        log_file = output_root / "logs" / f"{exp_name}.log"
        trajectory = output_root / "trajectories" / f"{exp_name}.json"
        success_dir = output_root / "success" / exp_name

        definition = str(item["definition"])
        test_path = str(item["test_path"])

        workspace.mkdir(parents=True, exist_ok=True)
        launch_script = SCRIPT_DIR / "run_v2.py"
        container_timeout = f"{args.timeout + 7200}s"
        cmd = [
            sys.executable, str(launch_script),
            "--definition", definition,
            "--model", args.model,
            "--test-path", test_path,
            "--log-path", str(trajectory),
            "--output-dir", str(workspace),
            "--success-dir", str(success_dir),
            "--image", args.image,
            "--max-turns", str(item["num_turns"]),
            "--target-speedup", str(item["target_speedup"]),
            "--prompt-tag", item["prompt_tag"],
            "--service-url", args.service_url,
            "--gpu-arch", args.gpu_arch,
            "--container-timeout", container_timeout,
            "--turn-timeout", str(args.turn_timeout),
            "--llm-context-policy", args.llm_context_policy,
        ]
        if args.without_local_gpu:
            cmd.append("--without-local-gpu")
        else:
            cmd.extend(["--gpus", f'"device={gpu_id}"'])
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

        result = {
            "exp_index": exp_index,
            "exp_name": exp_name,
            "gpu_id": gpu_id,
            "prompt_tag": item["prompt_tag"],
            "definition": definition,
            "test_path": test_path,
            "output_root": str(output_root),
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

            with open(log_file, "w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
                f.write(f"=== STDERR ===\n{proc.stderr}\n")

        except subprocess.TimeoutExpired as e:
            result["duration"] = time.time() - start
            timeout_output = combined_output(e.stdout, e.stderr)
            if is_api_failure_output(timeout_output):
                result["status"] = API_FAILED
                result["error"] = "Model/provider API failure before timeout"
            else:
                result["status"] = "timeout"
                result["error"] = f"Timed out after {args.timeout}s"
            kill_container(workspace)
            with open(log_file, "w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== TIMEOUT after {args.timeout}s ===\n")
                if e.stdout:
                    f.write(f"=== STDOUT ===\n{e.stdout}\n\n")
                if e.stderr:
                    f.write(f"=== STDERR ===\n{e.stderr}\n")

        except Exception as e:
            result["duration"] = time.time() - start
            if is_api_failure_output(str(e)):
                result["status"] = API_FAILED
                result["error"] = "Model/provider API failure"
            else:
                result["status"] = "error"
                result["error"] = str(e)
            kill_container(workspace)
            with open(log_file, "w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== ERROR ===\n{e}\n")

        return result
    finally:
        gpu_queue.put(gpu_id)


def summarize_results(results: list[dict], output_root: Path) -> None:
    """Print summary and write summary.json."""
    results.sort(key=lambda r: r["exp_index"])

    total = len(results)
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]
    infra_failures = [r for r in results if r["status"] == INFRA_FAILED]
    api_failures = [r for r in results if r["status"] == API_FAILED]
    timeouts = [r for r in results if r["status"] == "timeout"]
    errors = [r for r in results if r["status"] == "error"]

    durations = [r["duration"] for r in results]
    avg_duration = sum(durations) / len(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

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

    # Per-prompt-tag breakdown
    by_tag: dict[str, dict] = {}
    for r in results:
        tag = r.get("prompt_tag", "?")
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
        bucket[r["status"]] = bucket.get(r["status"], 0) + 1
    if by_tag:
        print("\nPer prompt_tag:")
        for tag, b in sorted(by_tag.items()):
            print(f"  {tag}: total={b['total']} success={b.get('success', 0)} "
                  f"failed={b.get('failed', 0)} infra={b.get(INFRA_FAILED, 0)} "
                  f"api={b.get(API_FAILED, 0)} "
                  f"timeout={b.get('timeout', 0)} error={b.get('error', 0)}")

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
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")


def run_experiments(
    experiments: list[tuple[int, dict]],
    args,
    gpu_queue: queue.Queue,
    output_root: Path,
    max_parallel: int,
) -> list[dict]:
    results: list[dict] = []
    next_pos = 0
    futures: dict[Any, int] = {}
    stop_scheduling = False

    def submit_next(pool: ThreadPoolExecutor) -> bool:
        nonlocal next_pos
        if next_pos >= len(experiments):
            return False
        exp_index, item = experiments[next_pos]
        definition = str(item["definition"])
        ok, error = check_service_health(args.service_url, definition)
        if not ok:
            raise InfraAbort(
                f"INFRA_FAILED: profiling service unhealthy before exp_{exp_index:03d} "
                f"({definition}): {error}"
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
                gpu_id = result["gpu_id"]
                duration = result["duration"]
                gpu_label = "no GPU" if args.without_local_gpu else f"GPU {gpu_id}"
                print(
                    f"[{completed}/{len(experiments)}] {item_output_root.name}/exp_{exp_index:03d} "
                    f"({result['prompt_tag']}) on {gpu_label}: {status_str} ({duration:.1f}s)"
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


def main():
    args = parse_args()
    multi_tuple_args = bool(args.output_roots and args.configs)
    if args.extend_turns and not args.resume:
        print("Error: --extend-turns must be used together with --resume.", file=sys.stderr)
        sys.exit(1)
    if args.resume and (args.config or args.configs) and not args.extend_turns and not multi_tuple_args:
        print(
            "Warning: --config/--configs is ignored with plain --resume; using output-root plan.json.",
            file=sys.stderr,
        )
    if args.extend_turns and not args.config and not args.configs:
        print("Error: --resume --extend-turns requires --config or --configs.", file=sys.stderr)
        sys.exit(1)
    if not args.resume and not args.config and not args.configs:
        print("Error: --config or --configs is required for a fresh run.", file=sys.stderr)
        sys.exit(1)
    profile_limit_explicit = args.max_profiles is not None
    args._profile_limit_explicit = profile_limit_explicit
    if args.max_profiles is not None and args.max_profiles <= 0:
        print("Error: --max-profiles must be positive when set.", file=sys.stderr)
        sys.exit(1)
    if args.max_profiles is not None and not args.without_local_gpu:
        print("Error: --max-profiles requires --without-local-gpu.", file=sys.stderr)
        sys.exit(1)

    try:
        run_targets = (
            None
            if args.resume and not args.extend_turns and not multi_tuple_args
            else run_targets_from_args(args)
        )
        validate_tinker_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path(f"./eval_runs/multiturn_v2_{timestamp}")
    output_roots = [Path(path).expanduser().resolve() for path in args.output_roots or []]
    multi_root_mode = bool(output_roots)
    if args.output_root and output_roots:
        print("Error: --output-root cannot be used together with --output-roots.", file=sys.stderr)
        sys.exit(1)
    if args.resume and not args.output_root and not output_roots:
        print("Error: --resume requires --output-root or --output-roots.", file=sys.stderr)
        sys.exit(1)
    output_root = output_root.resolve()
    roots_to_check = output_roots if output_roots else [output_root]
    if args.resume and not multi_tuple_args:
        missing_roots = [root for root in roots_to_check if not root.exists()]
        if missing_roots:
            print(f"Error: --resume output root does not exist: {missing_roots[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        existing_roots = [root for root in roots_to_check if root.exists()]
        if existing_roots and not multi_root_mode:
            print(f"Error: Output root {existing_roots[0]} already exists.")
            sys.exit(1)

    try:
        if args.resume and not args.extend_turns and not multi_tuple_args:
            loaded_experiments: list[tuple[int, dict[str, Any]]] = []
            config_labels = []
            for root in roots_to_check:
                current_plan_data = load_current_plan_data(root)
                config_labels.append(str(current_plan_data.get("config", root / "plan.json")))
                root_experiments = experiments_from_plan(current_plan_data["plan"])
                validate_plan_run_fields(root_experiments)
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
                    resume_existing=multi_root_mode and not args.extend_turns,
                )
            else:
                config_path = Path(args.config).resolve()
                config_label = str(config_path)
                config_items = load_config(config_path)
                experiments = expand_config(config_items)
                experiments = materialize_run_fields(experiments, run_targets)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Pre-build prompt docs up-front to avoid concurrent-write races between workers.
    # Callers are expected to run `build_doc_v2.py <config>` before launching, but
    # we also run it here as a safety net.
    with open(SCRIPT_DIR / "prompt_configs" / "hub.json") as f:
        hub = json.load(f)
    unique_tags = sorted({item["prompt_tag"] for _, item in experiments})
    print(f"Pre-building prompt docs for {len(unique_tags)} tags: {unique_tags}")
    for tag in unique_tags:
        build_doc(tag, hub)

    planned_total = len(experiments)
    if args.start_index < 0:
        print(f"Error: --start-index must be >= 0, got {args.start_index}.")
        sys.exit(1)
    if args.end_index is not None and args.end_index < args.start_index:
        print(
            f"Error: --end-index ({args.end_index}) must be >= "
            f"--start-index ({args.start_index})."
        )
        sys.exit(1)
    experiments = [
        (i, item)
        for i, item in experiments
        if i >= args.start_index and (args.end_index is None or i <= args.end_index)
    ]
    if not experiments:
        end_label = args.end_index if args.end_index is not None else planned_total - 1
        print(
            f"Error: no experiments selected from {planned_total} planned "
            f"experiments for inclusive range [{args.start_index}, {end_label}]."
        )
        sys.exit(1)
    total = len(experiments)

    if args.without_local_gpu:
        gpu_ids: list[str] = []
        max_parallel = args.max_parallel or total
    else:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if not gpu_ids:
            print("Error: --gpus must list at least one GPU when local GPU mode is enabled.", file=sys.stderr)
            sys.exit(1)
        max_parallel = args.max_parallel or len(gpu_ids)

    roots_for_selected = group_experiments_by_root(experiments, output_root)
    for root in roots_for_selected:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "trajectories").mkdir(parents=True, exist_ok=True)
        (root / "success").mkdir(parents=True, exist_ok=True)

    # Persist the expanded plan so failures/restarts can be reasoned about.
    full_plan_by_root = {
        root: plan_from_experiments(root_experiments)
        for root, root_experiments in roots_for_selected.items()
    }
    extend_info: dict[int, dict[str, Any]] = {}
    if args.extend_turns:
        try:
            for root, root_plan in full_plan_by_root.items():
                current_plan = load_current_plan(root)
                validate_plan_run_fields(experiments_from_plan(current_plan))
                validate_extend_plan(current_plan, root_plan)
        except ValueError as exc:
            print(f"Error: --extend-turns rejected: {exc}", file=sys.stderr)
            sys.exit(1)

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
                sys.exit(1)
        if args.extend_turns:
            for root, root_plan in full_plan_by_root.items():
                with open(root / "plan.json", "w") as f:
                    json.dump({"config": config_label, "plan": root_plan}, f, indent=2)
        for root, root_experiments in roots_for_selected.items():
            if root not in resume_roots:
                continue
            resume_info_by_root[str(root)] = prepare_resume(root_experiments, root)
        if args.extend_turns:
            for root, root_experiments in roots_for_selected.items():
                if root not in resume_roots:
                    continue
                root_resume_info = resume_info_by_root[str(root)]
                root_extend_info = prepare_extend_turns(
                    [(index, item) for index, item in root_experiments if index not in root_resume_info],
                    root,
                )
                root_resume_info.update(root_extend_info)
                extend_info.update(root_extend_info)
        experiments = [
            (index, item)
            for index, item in experiments
            if (
                experiment_output_root(item, output_root) not in resume_roots
                or index in resume_info_by_root.get(str(experiment_output_root(item, output_root)), {})
            )
        ]
        if not experiments:
            if args.extend_turns:
                print("No missing, infra-failed, API-failed, or turn-extendable trajectories found to resume.")
            else:
                print("No missing, infra-failed, or API-failed trajectories found to resume.")
            return
        total = len(experiments)

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
            with open(root / "resume_plan.json", "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "config": config_label,
                        "extend_turns": args.extend_turns,
                        "extend_info": extend_info,
                        "resume_info": resume_info_by_root.get(str(root), {}),
                        "plan": root_plan,
                    },
                    f,
                    indent=2,
                )
    if not args.extend_turns:
        for root, root_plan in full_plan_by_root.items():
            if root in resume_roots:
                continue
            with open(root / "plan.json", "w") as f:
                json.dump({"config": config_label, "plan": root_plan}, f, indent=2)

    print(f"Launching {total} experiments from {config_label}")
    if total != planned_total:
        end_label = args.end_index if args.end_index is not None else planned_total - 1
        print(f"Selected experiment range: [{args.start_index}, {end_label}] of {planned_total}")
    max_profiles_label = args.max_profiles if args.max_profiles is not None else "disabled"
    if args.without_local_gpu:
        print(f"GPUs: none (--without-local-gpu)  |  Max parallel: {max_parallel}  |  Max profiles: {max_profiles_label}")
    else:
        print(f"GPUs: {gpu_ids}  |  Max parallel: {max_parallel}  |  Max profiles: {max_profiles_label}")
    definitions = sorted({str(item["definition"]) for _, item in experiments})
    print(f"Model: {args.model}  |  Definitions: {definitions}  |  GPU arch: {args.gpu_arch}")
    print(f"Service URL: {args.service_url}")
    if multi_root_mode:
        print(f"Outputs: {sorted(str(root) for root in roots_for_selected)}")
    else:
        print(f"Output: {output_root}")
    print(f"Timeout: {args.timeout}s per experiment")
    if active_resume_roots:
        print("Resume mode: wrote resume_plan.json under selected output root(s)")
    if args.extend_turns:
        print("Extend-turns mode: updated plan.json under selected output root(s)")
    print()

    gpu_queue: queue.Queue = queue.Queue()
    if args.without_local_gpu:
        for _ in range(max_parallel):
            gpu_queue.put("none")
    else:
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)

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
