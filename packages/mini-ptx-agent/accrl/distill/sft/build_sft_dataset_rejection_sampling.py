#!/usr/bin/env python3
"""Build an SFT parquet from successful mini-swe-agent trajectories.

The input CSV is expected to contain a ``trajectory_path`` column, plus optional
run/experiment metadata columns such as ``run_name`` and ``exp_name``. Each
trajectory JSON is treated as one rejection-sampling winner: the output row keeps
the conversation from the beginning through the last assistant message and drops
post-assistant evaluation/exit records.

Output rows use the repo's standard chat schema:

    id: str
    messages: list[{role: str, content: str}]
    metadata: dict

The script also writes ``<output>.manifest.json`` and
``<output>.length_stats.json`` next to the parquet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_CSV = Path(
    "/home/ubuntu/AccRL-exps/sft_experiments/"
    "2026-0609-1535_wgmma_desc_masked/data/"
    "2026-0608-1500-2026-0609-0100-success-trajectories.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/ubuntu/AccRL-exps/sft_experiments/"
    "2026-0609-1535_wgmma_desc_masked/data/"
    "2026-0608-1500-2026-0609-0100-success.parquet"
)

ALLOWED_CHAT_ROLES = {"system", "user", "assistant"}
TARGET_FORMAT_WITH_REASONING = "qwen_think_plus_assistant_response"
TARGET_FORMAT_VISIBLE_ONLY = "assistant_completion_from_successful_trajectory"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: missing CSV header")
        if "trajectory_path" not in reader.fieldnames:
            raise ValueError(f"{path}: missing required trajectory_path column")
        return [dict(row) for row in reader]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return data


def clean_message(message: dict[str, Any], *, path: Path, index: int) -> dict[str, str]:
    role = message.get("role")
    if role not in ALLOWED_CHAT_ROLES:
        raise ValueError(f"{path}: message {index} has unsupported chat role {role!r}")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{path}: message {index} content must be a string")
    return {"role": role, "content": content}


def extract_reasoning_content(message: dict[str, Any]) -> str:
    """Return hidden reasoning from common LiteLLM/OpenAI-compatible locations."""
    candidates: list[Any] = [
        message.get("reasoning_content"),
        (message.get("provider_specific_fields") or {}).get("reasoning_content")
        if isinstance(message.get("provider_specific_fields"), dict)
        else None,
    ]

    extra = message.get("extra") or {}
    response = extra.get("response") if isinstance(extra, dict) else None
    if isinstance(response, dict):
        for choice in response.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            choice_message = choice.get("message")
            if isinstance(choice_message, dict):
                candidates.extend(
                    [
                        choice_message.get("reasoning_content"),
                        (
                            choice_message.get("provider_specific_fields") or {}
                        ).get("reasoning_content")
                        if isinstance(
                            choice_message.get("provider_specific_fields"), dict
                        )
                        else None,
                    ]
                )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def content_contains_reasoning(content: str, reasoning_content: str) -> bool:
    reasoning = reasoning_content.strip()
    if not reasoning:
        return False
    if reasoning in content:
        return True
    normalized_reasoning = " ".join(reasoning.split())
    normalized_content = " ".join(content.split())
    return bool(normalized_reasoning and normalized_reasoning in normalized_content)


def build_assistant_target(
    message: dict[str, Any], *, path: Path, index: int
) -> tuple[dict[str, str], dict[str, Any]]:
    cleaned = clean_message(message, path=path, index=index)
    content = cleaned["content"]
    reasoning_content = extract_reasoning_content(message)
    reasoning_in_content = content_contains_reasoning(content, reasoning_content)
    has_reasoning = bool(reasoning_content.strip())

    if has_reasoning and not reasoning_in_content:
        cleaned["content"] = f"<think>\n{reasoning_content.strip()}\n</think>\n{content}"

    return cleaned, {
        "reasoning_content_available": has_reasoning,
        "reasoning_content_chars": len(reasoning_content),
        "reasoning_content_in_content": reasoning_in_content,
        "assistant_content_already_has_think": (
            "<think>" in content and "</think>" in content
        ),
        "assistant_raw_content_chars": len(content),
        "assistant_target_wrapped_reasoning": has_reasoning
        and not reasoning_in_content,
    }


def last_assistant_index(messages: list[dict[str, Any]], path: Path) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError(f"{path}: no assistant message found")


def first_evaluation_after(
    messages: list[dict[str, Any]], assistant_index: int
) -> dict[str, Any] | None:
    for message in messages[assistant_index + 1 :]:
        extra = message.get("extra") or {}
        if isinstance(extra, dict) and extra.get("event") == "evaluation":
            return extra
    return None


def split_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(";") if part]


def row_id(csv_row: dict[str, str], trajectory_path: Path, seen: Counter[str]) -> str:
    run_name = csv_row.get("run_name") or csv_row.get("run_dir") or "run"
    exp_name = csv_row.get("exp_name") or trajectory_path.stem
    base = f"{Path(run_name).name}_{exp_name}"
    seen[base] += 1
    if seen[base] == 1:
        return base
    return f"{base}_{seen[base]}"


def evaluation_metadata(evaluation: dict[str, Any] | None) -> dict[str, Any]:
    if not evaluation:
        return {}

    traces = evaluation.get("traces") or []
    trace_definitions: list[str] = []
    trace_solution_ids: list[str] = []
    trace_speedups: list[float] = []
    trace_latencies_ms: list[float] = []
    trace_reference_latencies_ms: list[float] = []
    trace_statuses: list[str] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        if trace.get("definition") is not None:
            trace_definitions.append(str(trace["definition"]))
        if trace.get("solution") is not None:
            trace_solution_ids.append(str(trace["solution"]))
        trace_eval = trace.get("evaluation") or {}
        if isinstance(trace_eval, dict):
            if trace_eval.get("status") is not None:
                trace_statuses.append(str(trace_eval["status"]))
            perf = trace_eval.get("performance") or {}
            if isinstance(perf, dict):
                if isinstance(perf.get("speedup_factor"), (int, float)):
                    trace_speedups.append(float(perf["speedup_factor"]))
                if isinstance(perf.get("latency_ms"), (int, float)):
                    trace_latencies_ms.append(float(perf["latency_ms"]))
                if isinstance(perf.get("reference_latency_ms"), (int, float)):
                    trace_reference_latencies_ms.append(
                        float(perf["reference_latency_ms"])
                    )

    return {
        "evaluation_returncode": evaluation.get("returncode"),
        "evaluation_event": evaluation.get("event"),
        "evaluation_all_passed": evaluation.get("all_passed"),
        "evaluation_min_speedup": evaluation.get("min_speedup"),
        "evaluation_target_met": evaluation.get("target_met"),
        "evaluation_trace_count": len(traces),
        "evaluation_trace_definitions": trace_definitions,
        "evaluation_trace_solution_ids": trace_solution_ids,
        "evaluation_trace_statuses": trace_statuses,
        "evaluation_trace_speedups": trace_speedups,
        "evaluation_trace_latencies_ms": trace_latencies_ms,
        "evaluation_trace_reference_latencies_ms": trace_reference_latencies_ms,
    }


def make_row(
    csv_row: dict[str, str],
    *,
    seen_ids: Counter[str],
) -> dict[str, Any]:
    raw_path = csv_row.get("trajectory_path") or ""
    trajectory_path = Path(raw_path)
    if not trajectory_path.is_file():
        raise ValueError(f"missing trajectory JSON: {trajectory_path}")

    data = load_json(trajectory_path)
    messages_raw = data.get("messages")
    if not isinstance(messages_raw, list):
        raise ValueError(f"{trajectory_path}: missing list field 'messages'")

    assistant_index = last_assistant_index(messages_raw, trajectory_path)
    sft_messages = []
    assistant_target_metadata: dict[str, Any] = {}
    for index, message in enumerate(messages_raw[: assistant_index + 1]):
        if index == assistant_index:
            cleaned, assistant_target_metadata = build_assistant_target(
                message, path=trajectory_path, index=index
            )
        else:
            cleaned = clean_message(message, path=trajectory_path, index=index)
        sft_messages.append(cleaned)
    evaluation = first_evaluation_after(messages_raw, assistant_index)
    target_format = (
        TARGET_FORMAT_WITH_REASONING
        if assistant_target_metadata.get("reasoning_content_available")
        else TARGET_FORMAT_VISIBLE_ONLY
    )

    metadata: dict[str, Any] = {
        "source": "rejection_sampling_trajectory",
        "target_format": target_format,
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "trajectory_format": data.get("trajectory_format"),
        "source_message_count": len(messages_raw),
        "sft_message_count": len(sft_messages),
        "assistant_message_index": assistant_index,
        "assistant_chars": len(sft_messages[-1]["content"]),
        "prompt_chars": sum(len(message["content"]) for message in sft_messages[:-1]),
        "run_dir": csv_row.get("run_dir"),
        "run_name": csv_row.get("run_name"),
        "exp_name": csv_row.get("exp_name"),
        "success_dir": csv_row.get("success_dir"),
        "success_kernel_count": int(csv_row.get("success_kernel_count") or 0),
        "success_kernel_paths": split_paths(csv_row.get("success_kernel_paths")),
    }
    metadata.update(assistant_target_metadata)
    metadata.update(evaluation_metadata(evaluation))

    return {
        "id": row_id(csv_row, trajectory_path, seen_ids),
        "messages": sft_messages,
        "metadata": metadata,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow is required to write parquet; run in the Miles image") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def percentile(sorted_values: list[int], pct: float) -> int:
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def summarize_ints(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p50": percentile(sorted_values, 0.50),
        "p90": percentile(sorted_values, 0.90),
        "p95": percentile(sorted_values, 0.95),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": statistics.fmean(sorted_values),
    }


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    message_counts = [len(row["messages"]) for row in rows]
    prompt_chars = [
        sum(len(message["content"]) for message in row["messages"][:-1])
        for row in rows
    ]
    assistant_chars = [len(row["messages"][-1]["content"]) for row in rows]
    total_chars = [
        prompt + assistant
        for prompt, assistant in zip(prompt_chars, assistant_chars, strict=True)
    ]
    definitions = Counter()
    run_names = Counter()
    all_passed = Counter()
    target_met = Counter()
    target_formats = Counter()
    reasoning_content_available = Counter()
    reasoning_content_in_content = Counter()
    wrapped_reasoning = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        run_names[str(metadata.get("run_name"))] += 1
        all_passed[str(metadata.get("evaluation_all_passed"))] += 1
        target_met[str(metadata.get("evaluation_target_met"))] += 1
        target_formats[str(metadata.get("target_format"))] += 1
        reasoning_content_available[
            str(metadata.get("reasoning_content_available"))
        ] += 1
        reasoning_content_in_content[
            str(metadata.get("reasoning_content_in_content"))
        ] += 1
        wrapped_reasoning[
            str(metadata.get("assistant_target_wrapped_reasoning"))
        ] += 1
        for definition in metadata.get("evaluation_trace_definitions") or []:
            definitions[str(definition)] += 1

    return {
        "num_rows": len(rows),
        "messages": summarize_ints(message_counts),
        "prompt_chars": summarize_ints(prompt_chars),
        "assistant_chars": summarize_ints(assistant_chars),
        "total_chars": summarize_ints(total_chars),
        "run_names": dict(run_names),
        "target_formats": dict(target_formats),
        "reasoning_content_available": dict(reasoning_content_available),
        "reasoning_content_in_content": dict(reasoning_content_in_content),
        "assistant_target_wrapped_reasoning": dict(wrapped_reasoning),
        "evaluation_all_passed": dict(all_passed),
        "evaluation_target_met": dict(target_met),
        "evaluation_trace_definitions": dict(definitions),
        "top_10_longest": sorted(
            (
                {
                    "id": row["id"],
                    "total_chars": total,
                    "assistant_chars": assistant,
                }
                for row, total, assistant in zip(
                    rows, total_chars, assistant_chars, strict=True
                )
            ),
            key=lambda item: item["total_chars"],
            reverse=True,
        )[:10],
    }


def companion_path(output: Path, suffix: str) -> Path:
    return output.with_suffix(f"{output.suffix}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSONL instead of parquet, useful when pyarrow is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"missing input CSV: {args.input_csv}")
    if not args.jsonl and args.output.suffix != ".parquet":
        raise SystemExit("--output must end with .parquet unless --jsonl is set")
    if args.jsonl and args.output.suffix != ".jsonl":
        raise SystemExit("--output must end with .jsonl when --jsonl is set")

    csv_rows = load_csv_rows(args.input_csv)
    seen_ids: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for index, csv_row in enumerate(csv_rows, 1):
        try:
            rows.append(make_row(csv_row, seen_ids=seen_ids))
        except Exception as e:
            dropped.append(
                {
                    "csv_row": str(index),
                    "trajectory_path": csv_row.get("trajectory_path") or "",
                    "reason": str(e),
                }
            )

    if not rows:
        raise SystemExit(f"no rows exported; dropped={len(dropped)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        write_jsonl(args.output, rows)
    else:
        write_parquet(args.output, rows)

    stats_path = companion_path(args.output, ".length_stats.json")
    manifest_path = companion_path(args.output, ".manifest.json")
    write_json(stats_path, dataset_stats(rows))
    manifest = {
        "builder": "accrl.distill.sft.build_sft_dataset_rejection_sampling",
        "input_csv": str(args.input_csv),
        "input_csv_sha256": file_sha256(args.input_csv),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "num_csv_rows": len(csv_rows),
        "num_rows": len(rows),
        "num_dropped": len(dropped),
        "dropped": dropped,
        "trajectory_paths": [row["metadata"]["trajectory_path"] for row in rows],
    }
    write_json(manifest_path, manifest)

    print(f"wrote rows: {len(rows)} -> {args.output} (dropped {len(dropped)})")
    print(f"manifest:   {manifest_path}")
    print(f"stats:      {stats_path}")


if __name__ == "__main__":
    main()
