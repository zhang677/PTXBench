#!/usr/bin/env python3
"""Measure FlashInfer reference-vs-solution parity through the profiling service.

This expects definitions in the service to have already been rewritten so their
`reference` field contains the FlashInfer wrapper source. The submitted solution
is the copied FlashInfer baseline solution JSON. A passing migration should have
correctness PASSED and speedup_factor close to 1.0 for every measured workload.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_MANIFEST_MD = Path(
    "/home/ubuntu/AccRL-exps/tasks/clean_data/artifacts/"
    "flashinfer_hopper_baseline_definitions.md"
)
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/accrl-training")
DEFAULT_BASE_URL = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")


@dataclass(frozen=True)
class Entry:
    family: str
    definition: str
    workload_rows: int
    solution_name: str


def parse_manifest(path: Path) -> list[Entry]:
    family: str | None = None
    entries: list[Entry] = []
    row_re = re.compile(
        r"^\| `(?P<definition>[^`]+)` \|\s*(?P<rows>\d+)\s*\|"
        r"\s*\d+\s*\| `(?P<solution>[^`]+)` \|$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^## (.+)$", line)
        if heading:
            family = heading.group(1)
            continue
        match = row_re.match(line)
        if match:
            if family is None:
                raise ValueError(f"manifest row before family heading: {line}")
            entries.append(
                Entry(
                    family=family,
                    definition=match.group("definition"),
                    workload_rows=int(match.group("rows")),
                    solution_name=match.group("solution"),
                )
            )
    if not entries:
        raise ValueError(f"no manifest entries parsed from {path}")
    return entries


def selected_entries(
    entries: list[Entry],
    families: set[str] | None,
    definitions: set[str] | None,
) -> list[Entry]:
    selected = []
    for entry in entries:
        if entry.workload_rows == 0:
            continue
        if families and entry.family not in families:
            continue
        if definitions and entry.definition not in definitions:
            continue
        selected.append(entry)
    return selected


def get_json(base_url: str, path: str, timeout: int = 30) -> Any:
    resp = requests.get(f"{base_url}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def submit_and_poll(base_url: str, payload: dict[str, Any], deadline_s: int) -> dict[str, Any]:
    resp = requests.post(f"{base_url}/evaluate", json=payload, timeout=30)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"  task_id={task_id}", flush=True)

    deadline = time.time() + deadline_s
    while time.time() < deadline:
        data = get_json(base_url, f"/tasks/{task_id}?timeout=30", timeout=60)
        if data["status"] in ("completed", "failed"):
            return data
        print(f"  status={data['status']}, polling...", flush=True)
        time.sleep(2)
    raise TimeoutError(f"evaluate task {task_id} timed out after {deadline_s}s")


def load_solution(dataset_root: Path, entry: Entry) -> dict[str, Any]:
    path = (
        dataset_root
        / "solutions"
        / entry.family
        / entry.definition
        / f"{entry.solution_name}.json"
    )
    with path.open(encoding="utf-8") as f:
        solution = json.load(f)
    if solution.get("definition") != entry.definition:
        raise ValueError(
            f"{path} has definition={solution.get('definition')!r}, expected {entry.definition!r}"
        )
    if solution.get("name") != entry.solution_name:
        raise ValueError(
            f"{path} has name={solution.get('name')!r}, expected {entry.solution_name!r}"
        )
    return solution


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-md", type=Path, default=DEFAULT_MANIFEST_MD)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--family", action="append", help="limit to one family; repeatable")
    parser.add_argument("--definition", action="append", help="limit to one definition; repeatable")
    parser.add_argument("--start-at-definition", help="skip entries before this definition in manifest order")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--parallel-tasks", type=int, default=1)
    parser.add_argument("--max-workloads-per-definition", type=int)
    parser.add_argument("--deadline-s", type=int, default=1200)
    parser.add_argument("--ratio-threshold", type=float, default=0.05)
    parser.add_argument("--csv", type=Path, default=Path("flashinfer_reference_parity.csv"))
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.parallel_tasks <= 0:
        raise ValueError("--parallel-tasks must be positive")
    if args.ratio_threshold < 0:
        raise ValueError("--ratio-threshold must be non-negative")

    entries = selected_entries(
        parse_manifest(args.manifest_md),
        set(args.family) if args.family else None,
        set(args.definition) if args.definition else None,
    )
    if not entries:
        raise ValueError("selection is empty")
    if args.start_at_definition:
        for index, entry in enumerate(entries):
            if entry.definition == args.start_at_definition:
                entries = entries[index:]
                break
        else:
            raise ValueError(f"--start-at-definition {args.start_at_definition!r} is not in selection")

    print(f"profiling service: {args.base_url}")
    health = get_json(args.base_url, "/health")
    print(f"service health: {health}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "definition",
                "solution",
                "workload_uuid",
                "status",
                "latency_ms",
                "reference_latency_ms",
                "speedup_factor",
                "ratio_delta",
                "max_absolute_error",
                "max_relative_error",
                "error",
            ],
        )
        writer.writeheader()
        f.flush()

        failures = 0
        outliers = 0
        measured = 0
        for entry in entries:
            print(f"definition: {entry.family}/{entry.definition}")
            service_definition = get_json(args.base_url, f"/definitions/{entry.definition}")
            if service_definition.get("name") != entry.definition:
                raise ValueError(f"service returned wrong definition for {entry.definition}")
            if "flashinfer" not in str(service_definition.get("reference", "")).lower():
                print("  warning: active reference does not mention flashinfer", flush=True)

            solution = load_solution(args.dataset_root, entry)
            workloads = get_json(args.base_url, f"/definitions/{entry.definition}/workloads")
            uuids = [w["uuid"] for w in workloads]
            if len(uuids) != entry.workload_rows:
                print(
                    f"  warning: service has {len(uuids)} workloads, manifest has {entry.workload_rows}",
                    flush=True,
                )
            if args.max_workloads_per_definition is not None:
                uuids = uuids[: args.max_workloads_per_definition]
            uuid_chunks = chunks(uuids, args.chunk_size)
            print(
                f"  measuring {len(uuids)} workloads in chunks of {args.chunk_size} "
                f"with {args.parallel_tasks} parallel task(s)"
            )

            def evaluate_uuid_chunk(uuid_chunk: list[str]) -> tuple[list[str], dict[str, Any] | None, str | None]:
                payload = {
                    "solution": solution,
                    "workload_uuids": uuid_chunk,
                    "run_baseline": True,
                    "profile_baseline": True,
                }
                try:
                    return uuid_chunk, submit_and_poll(args.base_url, payload, args.deadline_s), None
                except Exception as exc:
                    return uuid_chunk, None, f"{type(exc).__name__}: {exc}"

            def handle_result(
                uuid_chunk: list[str],
                data: dict[str, Any] | None,
                task_error: str | None,
            ) -> None:
                nonlocal failures, measured, outliers
                if task_error is not None:
                    error = task_error
                    print(f"  TASK ERROR: {error}", flush=True)
                    failures += len(uuid_chunk)
                    for workload_uuid in uuid_chunk:
                        writer.writerow(
                            {
                                "family": entry.family,
                                "definition": entry.definition,
                                "solution": entry.solution_name,
                                "workload_uuid": workload_uuid,
                                "status": "TASK_ERROR",
                                "latency_ms": None,
                                "reference_latency_ms": None,
                                "speedup_factor": None,
                                "ratio_delta": None,
                                "max_absolute_error": None,
                                "max_relative_error": None,
                                "error": error,
                            }
                        )
                    f.flush()
                    measured += len(uuid_chunk)
                    return

                assert data is not None
                if data["status"] == "failed":
                    error = data.get("error", "unknown")
                    print(f"  TASK FAILED: {error}", flush=True)
                    failures += len(uuid_chunk)
                    for workload_uuid in uuid_chunk:
                        writer.writerow(
                            {
                                "family": entry.family,
                                "definition": entry.definition,
                                "solution": entry.solution_name,
                                "workload_uuid": workload_uuid,
                                "status": "TASK_FAILED",
                                "latency_ms": None,
                                "reference_latency_ms": None,
                                "speedup_factor": None,
                                "ratio_delta": None,
                                "max_absolute_error": None,
                                "max_relative_error": None,
                                "error": error,
                            }
                        )
                    f.flush()
                    measured += len(uuid_chunk)
                    return

                seen_workloads: set[str] = set()
                for trace in data.get("traces") or []:
                    ev = trace.get("evaluation") or {}
                    corr = ev.get("correctness") or {}
                    perf = ev.get("performance") or {}
                    status = ev.get("status", "MISSING")
                    latency_ms = perf.get("latency_ms")
                    reference_latency_ms = perf.get("reference_latency_ms")
                    speedup = perf.get("speedup_factor")
                    ratio_delta = None
                    if isinstance(speedup, (int, float)):
                        ratio_delta = abs(float(speedup) - 1.0)
                    workload = trace.get("workload") or {}
                    workload_uuid = workload.get("uuid", "")
                    seen_workloads.add(workload_uuid)
                    measured += 1
                    if status != "PASSED":
                        failures += 1
                    if ratio_delta is not None and ratio_delta > args.ratio_threshold:
                        outliers += 1
                    error = ev.get("log") or data.get("error")

                    writer.writerow(
                        {
                            "family": entry.family,
                            "definition": entry.definition,
                            "solution": entry.solution_name,
                            "workload_uuid": workload_uuid,
                            "status": status,
                            "latency_ms": latency_ms,
                            "reference_latency_ms": reference_latency_ms,
                            "speedup_factor": speedup,
                            "ratio_delta": ratio_delta,
                            "max_absolute_error": corr.get("max_absolute_error"),
                            "max_relative_error": corr.get("max_relative_error"),
                            "error": error,
                        }
                    )
                    f.flush()
                    marker = "PASS" if status == "PASSED" else f"FAIL[{status}]"
                    ratio_text = "?" if ratio_delta is None else f"{ratio_delta:.4f}"
                    print(
                        f"    {workload_uuid} {marker} "
                        f"lat={latency_ms} ref={reference_latency_ms} "
                        f"speedup={speedup} ratio_delta={ratio_text}",
                        flush=True,
                    )
                missing_workloads = [uuid for uuid in uuid_chunk if uuid not in seen_workloads]
                if missing_workloads:
                    failures += len(missing_workloads)
                    measured += len(missing_workloads)
                    error = "completed task omitted workload trace"
                    for workload_uuid in missing_workloads:
                        writer.writerow(
                            {
                                "family": entry.family,
                                "definition": entry.definition,
                                "solution": entry.solution_name,
                                "workload_uuid": workload_uuid,
                                "status": "MISSING_TRACE",
                                "latency_ms": None,
                                "reference_latency_ms": None,
                                "speedup_factor": None,
                                "ratio_delta": None,
                                "max_absolute_error": None,
                                "max_relative_error": None,
                                "error": error,
                            }
                        )
                        print(f"    {workload_uuid} FAIL[MISSING_TRACE] {error}", flush=True)
                    f.flush()

            if args.parallel_tasks == 1:
                for uuid_chunk in uuid_chunks:
                    handle_result(*evaluate_uuid_chunk(uuid_chunk))
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_tasks) as executor:
                    futures = [executor.submit(evaluate_uuid_chunk, uuid_chunk) for uuid_chunk in uuid_chunks]
                    for future in as_completed(futures):
                        handle_result(*future.result())

        print(
            f"measured={measured} failures={failures} "
            f"ratio_outliers>{args.ratio_threshold}={outliers}"
        )
        return 1 if failures or outliers else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text, file=sys.stderr)
        raise
