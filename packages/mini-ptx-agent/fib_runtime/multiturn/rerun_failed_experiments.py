#!/usr/bin/env python3
"""Parallel launcher for failed, incomplete, or never-started multiturn runs.

Consumes the JSON list produced by `scan_container_kills.py` or
`scan_incomplete_turns.py`. Each entry has
{base_path, exp_id, starting_turn, num_turns, target_speedup, prompt_tag}.

For entries with `starting_turn > 0`, this script spawns `run_v2.py` with
`--resume-trajectory <base_path>/trajectories/<exp_id>.json` and
`--resume-turn <starting_turn>` so the agent continues from the first dead turn,
preserving all prior assistant responses and evaluations in its message history.
Entries with `starting_turn == 0` are launched fresh, which also covers planned
experiments that never wrote a trajectory.

Layout under --output-root (required, no auto-timestamp):
    <output_root>/
      plan.json                   <- copy of the resolved rerun list
      summary.json                <- written at the end
      logs/<exp_id>.log
      trajectories/<exp_id>.json
      success/<exp_id>/...        <- seeded with prior success records so
                                     new passing kernels append rather than
                                     overwrite version numbers
      <exp_id>/                   <- per-run workspace (container .container_id,
                                     kernel.cu, etc.)

Example:
    python rerun_failed_experiments.py \\
        --rerun-list  container_kills_0422.json \\
        --output-root /home/ubuntu/AccRL-exps/eval_runs/rerun_0422_2352 \\
        --definition  gemm_n6144_k4096 \\
        --model       gemini-3.1-pro-preview \\
        --test-path   ../mini_swe_agent_docker/envs/test_profile_cuda_gemm_n6144_k4096.py \\
        --gpus        0,1,2,3,4,5,6,7
"""

import argparse
import json
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import GPU_ARCH_NVCC  # noqa: E402
from build_doc_v2 import build_doc  # noqa: E402
from run_parallel_v2 import _kill_container, summarize_results  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel launcher for failed, incomplete, or never-started experiments.",
    )
    parser.add_argument("--rerun-list", required=True,
                        help="Path to the JSON list emitted by scan_container_kills.py "
                             "or scan_incomplete_turns.py")
    parser.add_argument("--output-root", required=True,
                        help="Parent output directory (must not exist — refuses to overwrite).")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max concurrent workers (default: number of GPUs).")
    parser.add_argument("--image", type=str, default="mini-swe-eval:latest",
                        help="Docker image (passthrough to run_v2.py)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-experiment timeout in seconds (default: 3600)")
    parser.add_argument("--turn-timeout", type=int, default=130,
                        help="Per-turn timeout in seconds, passed to run_v2.py (default: 130)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output (passthrough to run_v2.py)")
    parser.add_argument("--definition", type=str, required=True,
                        help="Definition name (passthrough to run_v2.py)")
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to the test file (passthrough to run_v2.py)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (passthrough to run_v2.py)")
    parser.add_argument("--service-url", type=str, default="http://localhost:10000",
                        help="Profiling service URL (passthrough to run_v2.py)")
    parser.add_argument("--gpu-arch", type=str, choices=list(GPU_ARCH_NVCC.keys()), default="hopper",
                        help="GPU architecture (passthrough to run_v2.py, default: hopper)")
    parser.add_argument("--without-local-gpu", action="store_true",
                        help="Run without local GPU access (passthrough to run_v2.py)")
    return parser.parse_args()


def _seed_success_dir(prior_success_dir: Path, new_success_dir: Path) -> None:
    """Copy existing kernel_v*.cu and record.json from a prior run so the new
    resume continues versioning (kernel_v{N}.cu, kernel_v{N+1}.cu, ...) instead
    of starting over at v0 and overwriting the history.
    """
    if not prior_success_dir.is_dir():
        return
    prior_files = [item for item in prior_success_dir.iterdir() if item.is_file()]
    if not any(item.match("kernel_v*.cu") for item in prior_files):
        return
    new_success_dir.mkdir(parents=True, exist_ok=True)
    for item in prior_files:
        shutil.copy2(item, new_success_dir / item.name)


def _should_resume(entry: dict) -> bool:
    return int(entry["starting_turn"]) > 0


def run_single_rerun(
    entry: dict,
    args,
    gpu_queue: queue.Queue,
    output_root: Path,
) -> dict:
    """Spawn run_v2.py for a single rerun entry, reserving a GPU from the queue."""
    gpu_id = gpu_queue.get()
    try:
        exp_id = entry["exp_id"]
        base_path = Path(entry["base_path"])
        starting_turn = int(entry["starting_turn"])
        num_turns = int(entry["num_turns"])
        target_speedup = float(entry["target_speedup"])
        prompt_tag = entry["prompt_tag"]

        workspace = output_root / exp_id
        log_file = output_root / "logs" / f"{exp_id}.log"
        trajectory = output_root / "trajectories" / f"{exp_id}.json"
        success_dir = output_root / "success" / exp_id

        workspace.mkdir(parents=True, exist_ok=True)
        _seed_success_dir(base_path / "success" / exp_id, success_dir)

        resume_trajectory_src = base_path / "trajectories" / f"{exp_id}.json"
        launch_script = SCRIPT_DIR / "run_v2.py"
        container_timeout = f"{args.timeout + 7200}s"
        cmd = [
            sys.executable, str(launch_script),
            "--definition", args.definition,
            "--model", args.model,
            "--test-path", args.test_path,
            "--log-path", str(trajectory),
            "--output-dir", str(workspace),
            "--success-dir", str(success_dir),
            "--image", args.image,
            "--max-turns", str(num_turns),
            "--target-speedup", str(target_speedup),
            "--prompt-tag", prompt_tag,
            "--service-url", args.service_url,
            "--gpu-arch", args.gpu_arch,
            "--container-timeout", container_timeout,
            "--turn-timeout", str(args.turn_timeout),
        ]
        if _should_resume(entry):
            cmd.extend([
                "--resume-trajectory", str(resume_trajectory_src),
                "--resume-turn", str(starting_turn),
            ])
        if args.without_local_gpu:
            cmd.append("--without-local-gpu")
        else:
            cmd.extend(["--gpus", f'"device={gpu_id}"'])
        if args.verbose:
            cmd.append("-v")

        # Extract integer index for downstream sorting/summary reuse.
        try:
            exp_index = int(exp_id.split("_")[1])
        except (IndexError, ValueError):
            exp_index = -1

        result = {
            "exp_index": exp_index,
            "exp_name": exp_id,
            "gpu_id": gpu_id,
            "prompt_tag": prompt_tag,
            "num_turns": num_turns,
            "target_speedup": target_speedup,
            "starting_turn": starting_turn,
            "status": "unknown",
            "duration": 0.0,
            "returncode": None,
            "error": None,
            "trajectory": str(trajectory),
        }
        if _should_resume(entry):
            result["resume_trajectory"] = str(resume_trajectory_src)

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            result["duration"] = time.time() - start
            result["returncode"] = proc.returncode
            result["status"] = "success" if proc.returncode == 0 else "failed"

            with open(log_file, "w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
                f.write(f"=== STDERR ===\n{proc.stderr}\n")

        except subprocess.TimeoutExpired as e:
            result["duration"] = time.time() - start
            result["status"] = "timeout"
            result["error"] = f"Timed out after {args.timeout}s"
            _kill_container(workspace)
            with open(log_file, "w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== TIMEOUT after {args.timeout}s ===\n")
                if e.stdout:
                    f.write(f"=== STDOUT ===\n{e.stdout}\n\n")
                if e.stderr:
                    f.write(f"=== STDERR ===\n{e.stderr}\n")

        except Exception as e:
            result["duration"] = time.time() - start
            result["status"] = "error"
            result["error"] = str(e)
            _kill_container(workspace)
            with open(log_file, "w") as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== ERROR ===\n{e}\n")

        return result
    finally:
        gpu_queue.put(gpu_id)


def main():
    args = parse_args()

    rerun_path = Path(args.rerun_list).resolve()
    with open(rerun_path) as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        print(f"Error: {rerun_path} must be a non-empty JSON list.")
        sys.exit(1)

    # Validate each entry has required fields.
    required = {"base_path", "exp_id", "starting_turn", "num_turns", "target_speedup", "prompt_tag"}
    for i, entry in enumerate(entries):
        missing = required - set(entry.keys())
        if missing:
            print(f"Error: entry[{i}] missing fields {missing}")
            sys.exit(1)
        src_traj = Path(entry["base_path"]) / "trajectories" / f"{entry['exp_id']}.json"
        if _should_resume(entry) and not src_traj.exists():
            print(f"Error: source trajectory {src_traj} not found for entry[{i}]")
            sys.exit(1)

    # Pre-build prompt docs up-front to avoid concurrent-write races.
    with open(SCRIPT_DIR / "prompt_configs" / "hub.json") as f:
        hub = json.load(f)
    unique_tags = sorted({e["prompt_tag"] for e in entries})
    print(f"Pre-building prompt docs for {len(unique_tags)} tags: {unique_tags}")
    for tag in unique_tags:
        build_doc(tag, hub)

    total = len(entries)

    if args.without_local_gpu:
        gpu_ids: list[str] = []
        max_parallel = args.max_parallel or total
    else:
        gpu_ids = [g.strip() for g in args.gpus.split(",")]
        max_parallel = args.max_parallel or len(gpu_ids)

    output_root = Path(args.output_root)
    if output_root.exists():
        print(f"Error: Output root {output_root} already exists.")
        sys.exit(1)
    output_root = output_root.resolve()

    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "trajectories").mkdir(parents=True, exist_ok=True)
    (output_root / "success").mkdir(parents=True, exist_ok=True)

    # Persist the resolved rerun list so failures/restarts can be reasoned about.
    with open(output_root / "plan.json", "w") as f:
        json.dump({"rerun_list": str(rerun_path), "entries": entries}, f, indent=2)

    print(f"Launching {total} rerun experiments from {rerun_path}")
    if args.without_local_gpu:
        print(f"GPUs: none (--without-local-gpu)  |  Max parallel: {max_parallel}")
    else:
        print(f"GPUs: {gpu_ids}  |  Max parallel: {max_parallel}")
    print(f"Model: {args.model}  |  Definition: {args.definition}  |  GPU arch: {args.gpu_arch}")
    print(f"Output: {output_root}")
    print(f"Timeout: {args.timeout}s per experiment")
    print()

    gpu_queue: queue.Queue = queue.Queue()
    if args.without_local_gpu:
        for _ in range(max_parallel):
            gpu_queue.put("none")
    else:
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)

    results: list[dict] = []
    completed = 0
    wall_start = time.time()

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_single_rerun, entry, args, gpu_queue, output_root): entry["exp_id"]
            for entry in entries
        }
        for future in as_completed(futures):
            exp_id = futures[future]
            result = future.result()
            results.append(result)
            completed += 1
            status_str = result["status"].upper()
            gpu_id = result["gpu_id"]
            duration = result["duration"]
            gpu_label = "no GPU" if args.without_local_gpu else f"GPU {gpu_id}"
            start_label = (
                f"resume_turn={result['starting_turn']}"
                if result["starting_turn"] > 0 else "fresh_start"
            )
            print(f"[{completed}/{total}] {exp_id} "
                  f"({start_label}, {result['prompt_tag']}) "
                  f"on {gpu_label}: {status_str} ({duration:.1f}s)")

    wall_time = time.time() - wall_start
    print(f"\nAll reruns completed in {wall_time:.1f}s wall time")
    summarize_results(results, output_root)
    print(f"Run dir: {output_root}  |  Started: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
