#!/usr/bin/env python3
"""Measure /evaluate throughput with baseline-cache reuse versus bypass.

Run one mode after each clean one-GPU profiling-service restart:

    python measure_baseline_cache.py run --mode reuse
    python measure_baseline_cache.py run --mode bypass
    python measure_baseline_cache.py plot

Each run sends requests sequentially and records wall time from immediately
before POST /evaluate until the blocking task result has been received.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = os.environ.get("PROFILE_BASE_URL", "http://127.0.0.1:10000")
DEFAULT_KERNEL = Path(
    "/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/measure_baseline_cache/cuda_gemm_n7168_k5120_kernel.cu"
)
DEFINITION = "gemm_n7168_k5120"
WORKLOAD_UUID = "94920358-01a8-4c5b-9209-3103fd490e94"
DEFAULT_REQUESTS = 128
DEFAULT_ROLLING_WINDOW = 10
# Default evaluator: 3 trials * (1 correctness + 10 warmup + 50 timed calls).
CANDIDATE_CALLS_PER_REQUEST = 3 * (1 + 10 + 50)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def get_json(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple[float, float] = (5.0, 30.0),
) -> dict[str, Any]:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(value).__name__}")
    return value


def assert_one_gpu_service(health: dict[str, Any]) -> None:
    workers = health.get("backends")
    if workers is None:
        workers = health.get("workers")
    if not isinstance(workers, list) or len(workers) != 1:
        raise RuntimeError(f"Expected exactly one profiling backend/worker, got: {workers!r}")
    if health.get("status") != "ok" or not workers[0].get("healthy", False):
        raise RuntimeError(f"Profiling service is not healthy: {health}")
    if health.get("queue_size") != 0:
        raise RuntimeError(f"Profiling service queue is not empty: {health}")


def build_solution(kernel_source: str) -> dict[str, Any]:
    return {
        "name": "baseline_cache_measurement",
        "definition": DEFINITION,
        "spec": {
            "language": "cuda",
            "binding": "tvm-ffi",
            "target_hardware": ["H100"],
            "entry_point": "kernel.cu::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "author": "baseline-cache-measurement",
        "sources": [{"path": "kernel.cu", "content": kernel_source}],
    }


def extract_evaluation(task: dict[str, Any]) -> dict[str, Any]:
    traces = task.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise RuntimeError(f"Expected exactly one evaluation trace, got: {traces!r}")
    evaluation = traces[0].get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError(f"Evaluation is missing from task result: {task}")
    return evaluation


def run_mode(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.mode}_raw.jsonl"
    metadata_path = output_dir / f"{args.mode}_metadata.json"
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing raw data: {raw_path}")

    kernel_source = args.kernel.read_text()
    kernel_sha256 = hashlib.sha256(kernel_source.encode()).hexdigest()
    solution = build_solution(kernel_source)
    payload = {
        "solution": solution,
        "workload_uuids": [WORKLOAD_UUID],
        "baseline_cache_mode": args.mode,
        "timeout": args.evaluation_timeout,
    }

    session = requests.Session()
    base_url = args.base_url.rstrip("/")
    health = get_json(session, f"{base_url}/health")
    assert_one_gpu_service(health)
    server_info = get_json(session, f"{base_url}/")
    definition = get_json(session, f"{base_url}/definitions/{DEFINITION}")
    workload = get_json(session, f"{base_url}/workloads/{WORKLOAD_UUID}")

    metadata = {
        "mode": args.mode,
        "request_count": args.requests,
        "base_url": base_url,
        "started_at": utc_now(),
        "kernel_path": str(args.kernel.resolve()),
        "kernel_sha256": kernel_sha256,
        "definition": DEFINITION,
        "workload_uuid": WORKLOAD_UUID,
        "rolling_window": args.rolling_window,
        "evaluation_timeout_seconds": args.evaluation_timeout,
        "health_before": health,
        "server_info": server_info,
        "definition_payload": definition,
        "workload_payload": workload,
    }
    atomic_json(metadata_path, metadata)

    durations: list[float] = []
    run_started = time.perf_counter()
    with raw_path.open("w") as raw:
        for request_index in range(1, args.requests + 1):
            started_at = utc_now()
            started = time.perf_counter()
            post = session.post(
                f"{base_url}/evaluate",
                json=payload,
                timeout=(args.connect_timeout, args.submit_timeout),
            )
            post.raise_for_status()
            submission = post.json()
            task_id = submission["task_id"]

            task_response = session.get(
                f"{base_url}/tasks/{task_id}",
                params={"timeout": args.task_wait_timeout},
                timeout=(args.connect_timeout, args.task_wait_timeout + 30),
            )
            task_response.raise_for_status()
            task = task_response.json()
            duration_s = time.perf_counter() - started
            durations.append(duration_s)

            evaluation = extract_evaluation(task)
            performance = evaluation.get("performance") or {}
            record = {
                "mode": args.mode,
                "request_index": request_index,
                "started_at": started_at,
                "completed_at": utc_now(),
                "duration_s": duration_s,
                "instantaneous_requests_per_s": 1.0 / duration_s,
                "cumulative_requests_per_s": request_index / sum(durations),
                "rolling_requests_per_s": min(args.rolling_window, len(durations))
                / sum(durations[-args.rolling_window :]),
                "task_id": task_id,
                "normalized_solution_name": submission.get("normalized_solution_name"),
                "task_status": task.get("status"),
                "evaluation_status": evaluation.get("status"),
                "solution_latency_ms": performance.get("latency_ms"),
                "reference_latency_ms": performance.get("reference_latency_ms"),
                "task_response": task,
            }
            raw.write(json.dumps(record, sort_keys=True) + "\n")
            raw.flush()
            os.fsync(raw.fileno())
            print(
                f"{args.mode} {request_index:03d}/{args.requests}: "
                f"{duration_s:.6f}s, rolling={record['rolling_requests_per_s']:.4f} req/s, "
                f"status={record['evaluation_status']}",
                flush=True,
            )

            if task.get("status") != "completed" or evaluation.get("status") != "PASSED":
                raise RuntimeError(
                    f"Request {request_index} failed: task={task.get('status')} "
                    f"evaluation={evaluation.get('status')}; raw response saved to {raw_path}"
                )

    metadata["completed_at"] = utc_now()
    metadata["total_elapsed_s"] = time.perf_counter() - run_started
    metadata["aggregate_requests_per_s"] = args.requests / sum(durations)
    if len(durations) > 1:
        steady = durations[1:]
        metadata["steady_state_excluding_first"] = {
            "request_count": len(steady),
            "aggregate_requests_per_s": len(steady) / sum(steady),
            "mean_duration_s": statistics.fmean(steady),
            "median_duration_s": statistics.median(steady),
        }
    metadata["health_after"] = get_json(session, f"{base_url}/health")
    atomic_json(metadata_path, metadata)
    print(f"Wrote {raw_path}")
    print(f"Wrote {metadata_path}")

    if all((output_dir / f"{mode}_raw.jsonl").exists() for mode in ("reuse", "bypass")):
        plot_results(output_dir, args.rolling_window)


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def derive_rows(records: list[dict[str, Any]], rolling_window: int) -> list[dict[str, Any]]:
    durations: list[float] = []
    rows = []
    for record in records:
        duration = float(record["duration_s"])
        durations.append(duration)
        request_index = int(record["request_index"])
        window = min(rolling_window, len(durations))
        rows.append(
            {
                "mode": record["mode"],
                "request_index": request_index,
                "duration_s": duration,
                "instantaneous_requests_per_s": 1.0 / duration,
                "cumulative_runtime_s": sum(durations),
                "cumulative_requests_per_s": request_index / sum(durations),
                "rolling_requests_per_s": window / sum(durations[-window:]),
                "solution_latency_ms": record.get("solution_latency_ms"),
                "reference_latency_ms": record.get("reference_latency_ms"),
                "task_id": record.get("task_id"),
                "evaluation_status": record.get("evaluation_status"),
            }
        )
    return rows


def plot_results(output_dir: Path, rolling_window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ("reuse", "bypass")
    colors = {"reuse": "#1874CD", "bypass": "#D95F02"}
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"rolling_window": rolling_window, "modes": {}}

    for mode in modes:
        raw_path = output_dir / f"{mode}_raw.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw data: {raw_path}")
        records = load_records(raw_path)
        rows = derive_rows(records, rolling_window)
        if not rows:
            raise RuntimeError(f"No measurement records in {raw_path}")
        all_rows.extend(rows)
        durations = [float(row["duration_s"]) for row in rows]
        steady = durations[1:] if len(durations) > 1 else durations
        summary["modes"][mode] = {
            "request_count": len(durations),
            "total_measured_s": sum(durations),
            "aggregate_requests_per_s": len(durations) / sum(durations),
            "steady_state_excluding_first": {
                "request_count": len(steady),
                "aggregate_requests_per_s": len(steady) / sum(steady),
                "mean_duration_s": statistics.fmean(steady),
                "median_duration_s": statistics.median(steady),
            },
        }

    reuse_rps = summary["modes"]["reuse"]["steady_state_excluding_first"][
        "aggregate_requests_per_s"
    ]
    bypass_rps = summary["modes"]["bypass"]["steady_state_excluding_first"][
        "aggregate_requests_per_s"
    ]
    summary["steady_state_reuse_over_bypass_speedup"] = reuse_rps / bypass_rps
    total_runtime_speedup = (
        summary["modes"]["bypass"]["total_measured_s"]
        / summary["modes"]["reuse"]["total_measured_s"]
    )
    candidate_kernel_total_by_mode_s = {
        mode: sum(
            float(row["solution_latency_ms"])
            for row in all_rows
            if row["mode"] == mode
        )
        * CANDIDATE_CALLS_PER_REQUEST
        / 1000.0
        for mode in modes
    }
    candidate_latency_by_mode = {
        mode: {
            int(row["request_index"]): float(row["solution_latency_ms"])
            for row in all_rows
            if row["mode"] == mode
        }
        for mode in modes
    }
    candidate_request_numbers = sorted(
        set.intersection(
            *(set(candidate_latency_by_mode[mode]) for mode in modes)
        )
    )
    candidate_kernel_cumulative_s: list[float] = []
    cumulative_kernel_s = 0.0
    for request_index in candidate_request_numbers:
        mean_latency_ms = statistics.fmean(
            candidate_latency_by_mode[mode][request_index] for mode in modes
        )
        cumulative_kernel_s += mean_latency_ms * CANDIDATE_CALLS_PER_REQUEST / 1000.0
        candidate_kernel_cumulative_s.append(cumulative_kernel_s)
    candidate_kernel_total_s = candidate_kernel_cumulative_s[-1]
    summary["total_runtime_bypass_over_reuse_ratio"] = total_runtime_speedup
    summary["candidate_calls_per_request"] = CANDIDATE_CALLS_PER_REQUEST
    summary["candidate_kernel_total_by_mode_s"] = candidate_kernel_total_by_mode_s
    summary["plotted_candidate_kernel_total_s"] = candidate_kernel_total_s
    summary["reuse_runtime_over_kernel_execution_ratio"] = (
        summary["modes"]["reuse"]["total_measured_s"] / candidate_kernel_total_s
    )
    atomic_json(output_dir / "summary.json", summary)

    csv_path = output_dir / "timings.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    fig, (ax_time, ax_rps) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for mode in modes:
        rows = [row for row in all_rows if row["mode"] == mode]
        x = [row["request_index"] for row in rows]
        ax_time.plot(
            x,
            [row["duration_s"] for row in rows],
            color=colors[mode],
            linewidth=1.3,
            label=mode,
        )
        ax_rps.plot(
            x,
            [row["rolling_requests_per_s"] for row in rows],
            color=colors[mode],
            linewidth=1.8,
            label=f"{mode} ({rolling_window}-request rolling)",
        )

    ax_time.set_ylabel("Request wall time (s)")
    ax_time.set_title("Baseline cache reuse vs bypass: 128 sequential /evaluate requests")
    ax_time.grid(alpha=0.25)
    ax_time.legend()
    ax_rps.set_xlabel("Completed request number")
    ax_rps.set_ylabel("Requests / s")
    ax_rps.grid(alpha=0.25)
    ax_rps.legend()
    fig.tight_layout()

    fig.savefig(output_dir / "baseline_cache_measurement.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    last_request = 0
    for mode in modes:
        rows = [row for row in all_rows if row["mode"] == mode]
        x = [row["request_index"] for row in rows]
        cumulative_runtime = [row["cumulative_runtime_s"] for row in rows]
        last_request = max(last_request, x[-1])
        ax.plot(
            x,
            cumulative_runtime,
            color=colors[mode],
            linewidth=2,
            label=mode,
        )

    ax.plot(
        candidate_request_numbers,
        candidate_kernel_cumulative_s,
        color="#555555",
        linestyle="--",
        linewidth=1.7,
        label="kernel execution",
    )

    reuse_total_s = summary["modes"]["reuse"]["total_measured_s"]
    bypass_total_s = summary["modes"]["bypass"]["total_measured_s"]
    speedup_x = last_request + 5
    ax.annotate(
        "",
        xy=(speedup_x, bypass_total_s),
        xytext=(speedup_x, reuse_total_s),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "linewidth": 1.8},
        annotation_clip=False,
    )
    ax.text(
        speedup_x + 1.2,
        (reuse_total_s + bypass_total_s) / 2,
        f"{total_runtime_speedup:.2f}×",
        color="#333333",
        fontweight="bold",
        ha="left",
        va="center",
    )

    kernel_speedup_x = speedup_x
    kernel_speedup = reuse_total_s / candidate_kernel_total_s
    ax.annotate(
        "",
        xy=(kernel_speedup_x, reuse_total_s),
        xytext=(kernel_speedup_x, candidate_kernel_total_s),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "linewidth": 1.8},
        annotation_clip=False,
    )
    ax.text(
        kernel_speedup_x + 1.2,
        (candidate_kernel_total_s + reuse_total_s) / 2,
        f"{kernel_speedup:.2f}×",
        color="#333333",
        fontweight="bold",
        ha="left",
        va="center",
    )

    ax.set_xlabel("Completed request number")
    ax.set_ylabel("Cumulative request wall time (s)")
    ax.set_title("Total runtime through each sequential /evaluate request")
    ax.set_xlim(0, last_request + 13)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_cache_cumulative_runtime.png", dpi=200)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'baseline_cache_measurement.png'}")
    print(f"Wrote {output_dir / 'baseline_cache_cumulative_runtime.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one measurement mode")
    run.add_argument("--mode", choices=("reuse", "bypass"), required=True)
    run.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    run.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    run.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    run.add_argument("--evaluation-timeout", type=int, default=300)
    run.add_argument("--task-wait-timeout", type=float, default=360)
    run.add_argument("--connect-timeout", type=float, default=5)
    run.add_argument("--submit-timeout", type=float, default=30)
    run.add_argument("--overwrite", action="store_true")

    plot = subparsers.add_parser("plot", help="Regenerate CSV, summary, and plots")
    plot.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    plot.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)

    args = parser.parse_args()
    if getattr(args, "requests", 1) < 1:
        parser.error("--requests must be positive")
    if getattr(args, "rolling_window", 1) < 1:
        parser.error("--rolling-window must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "run":
        run_mode(args)
    else:
        plot_results(args.output_dir.resolve(), args.rolling_window)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
