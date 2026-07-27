#!/usr/bin/env python3
"""Parallel launcher for multiturn kernel generation experiments.

Spawns N concurrent subprocess calls to run.py with reactive GPU assignment —
when a task finishes, its GPU is immediately reused by the next queued experiment.

Mirrors launch_eval_v8_parallel.py from mini_swe_agent_docker/.
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
from common import GPU_ARCH_NVCC

SCRIPT_DIR = Path(__file__).resolve().parent

def _kill_container(workspace: Path) -> None:
    """Kill the Docker container associated with a workspace, if still running."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple multiturn kernel generation experiments in parallel",
    )
    parser.add_argument(
        "-n", "--num-experiments", type=int, default=64,
        help="Total number of experiments to run (default: 64)",
    )
    parser.add_argument(
        "--gpus", type=str, default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=None,
        help="Max concurrent workers (default: number of GPUs)",
    )
    parser.add_argument(
        "--image", type=str, default="mini-swe-eval:latest",
        help="Docker image (passthrough to run.py)",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Parent output directory (default: ./eval_runs/multiturn_TIMESTAMP)",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Per-experiment timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output (passthrough to run.py)",
    )
    parser.add_argument(
        "--definition", type=str, required=True,
        help="Definition name (passthrough to run.py)",
    )
    parser.add_argument(
        "--test-path", type=str, required=True,
        help="Path to the test file (passthrough to run.py)",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (passthrough to run.py)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=5,
        help="Max turns per experiment (passthrough to run.py)",
    )
    parser.add_argument(
        "--target-speedup", type=float, default=1.0,
        help="Target speedup (passthrough to run.py)",
    )
    parser.add_argument(
        "--service-url", type=str, default="http://localhost:10000",
        help="Profiling service URL (passthrough to run.py)",
    )
    parser.add_argument(
        "--gpu-arch", type=str, choices=list(GPU_ARCH_NVCC.keys()), default="hopper",
        help="GPU architecture (passthrough to run.py, default: hopper)",
    )
    parser.add_argument(
        "--without-local-gpu", action="store_true",
        help="Run without local GPU access (passthrough to run.py)",
    )
    parser.add_argument(
        "--script", type=str, default="run.py",
        help="Path to the launch script (default: run.py)",
    )
    return parser.parse_args()


def run_single_experiment(exp_index: int, args, gpu_queue: queue.Queue, output_root: Path) -> dict:
    """Run a single experiment as a subprocess.

    Acquires a GPU from gpu_queue before starting and returns it when done,
    so the next queued experiment can immediately reuse the freed GPU.
    """
    gpu_id = gpu_queue.get()
    try:
        exp_name = f"exp_{exp_index:03d}"
        workspace = output_root / exp_name
        log_file = output_root / "logs" / f"{exp_name}.log"
        trajectory = output_root / "trajectories" / f"{exp_name}.json"
        success_dir = output_root / "success" / exp_name

        workspace.mkdir(parents=True, exist_ok=True)
        LAUNCH_SCRIPT = SCRIPT_DIR / args.script
        cmd = [
            sys.executable, str(LAUNCH_SCRIPT),
            "--definition", args.definition,
            "--model", args.model,
            "--test-path", args.test_path,
            "--log-path", str(trajectory),
            "--output-dir", str(workspace),
            "--success-dir", str(success_dir),
            "--image", args.image,
            "--max-turns", str(args.max_turns),
            "--target-speedup", str(args.target_speedup),
            "--service-url", args.service_url,
            "--gpu-arch", args.gpu_arch,
        ]
        if args.without_local_gpu:
            cmd.append("--without-local-gpu")
        else:
            cmd.extend(["--gpus", f'"device={gpu_id}"'])
        if args.verbose:
            cmd.append("-v")

        result = {
            "exp_index": exp_index,
            "exp_name": exp_name,
            "gpu_id": gpu_id,
            "status": "unknown",
            "duration": 0.0,
            "returncode": None,
            "error": None,
            "trajectory": str(trajectory),
        }

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


def summarize_results(results: list[dict], output_root: Path) -> None:
    """Print summary and write summary.json."""
    results.sort(key=lambda r: r["exp_index"])

    total = len(results)
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]
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
    print(f"Timeout:  {len(timeouts)}")
    print(f"Error:    {len(errors)}")
    print(f"Duration: avg={avg_duration:.1f}s  min={min_duration:.1f}s  max={max_duration:.1f}s")

    if failures:
        print(f"\nFailed experiments: {[r['exp_name'] for r in failures]}")
    if timeouts:
        print(f"Timed out experiments: {[r['exp_name'] for r in timeouts]}")
    if errors:
        print(f"Errored experiments: {[r['exp_name'] for r in errors]}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "success": len(successes),
        "failed": len(failures),
        "timeout": len(timeouts),
        "error": len(errors),
        "avg_duration": round(avg_duration, 2),
        "min_duration": round(min_duration, 2),
        "max_duration": round(max_duration, 2),
        "results": results,
    }

    summary_path = output_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")


def main():
    args = parse_args()

    if args.without_local_gpu:
        gpu_ids = []
        max_parallel = args.max_parallel or args.num_experiments
    else:
        gpu_ids = [g.strip() for g in args.gpus.split(",")]
        max_parallel = args.max_parallel or len(gpu_ids)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path(f"./eval_runs/multiturn_{timestamp}")
    if output_root.exists():
        print(f"Error: Output root {output_root} already exists.")
        sys.exit(1)
    output_root = output_root.resolve()

    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "trajectories").mkdir(parents=True, exist_ok=True)
    (output_root / "success").mkdir(parents=True, exist_ok=True)

    print(f"Launching {args.num_experiments} experiments")
    if args.without_local_gpu:
        print(f"GPUs: none (--without-local-gpu)  |  Max parallel: {max_parallel}")
    else:
        print(f"GPUs: {gpu_ids}  |  Max parallel: {max_parallel}")
    print(f"Model: {args.model}  |  Definition: {args.definition}  |  GPU arch: {args.gpu_arch}")
    print(f"Max turns: {args.max_turns}  |  Target speedup: {args.target_speedup}x")
    print(f"Output: {output_root}")
    print(f"Timeout: {args.timeout}s per experiment")
    print()

    gpu_queue = queue.Queue()
    if args.without_local_gpu:
        for i in range(max_parallel):
            gpu_queue.put("none")
    else:
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)

    results = []
    completed = 0
    wall_start = time.time()

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {}
        for i in range(args.num_experiments):
            future = pool.submit(run_single_experiment, i, args, gpu_queue, output_root)
            futures[future] = i

        for future in as_completed(futures):
            exp_index = futures[future]
            result = future.result()
            results.append(result)
            completed += 1
            status_str = result["status"].upper()
            gpu_id = result["gpu_id"]
            duration = result["duration"]
            gpu_label = "no GPU" if args.without_local_gpu else f"GPU {gpu_id}"
            print(f"[{completed}/{args.num_experiments}] exp_{exp_index:03d} on {gpu_label}: {status_str} ({duration:.1f}s)")

    wall_time = time.time() - wall_start
    print(f"\nAll experiments completed in {wall_time:.1f}s wall time")
    summarize_results(results, output_root)


if __name__ == "__main__":
    main()
