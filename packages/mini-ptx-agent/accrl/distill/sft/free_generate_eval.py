#!/usr/bin/env python3
"""Run standalone free-generation eval against an SGLang /generate server."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(sorted_values: list[int | float], q: float) -> float:
    if not sorted_values:
        return math.nan
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def summarize(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "min": ordered[0],
    }


def find_subsequence(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    for idx in range(len(values) - len(needle) + 1):
        if values[idx : idx + len(needle)] == needle:
            return idx
    return None


def render_prompt(row: dict[str, Any], tokenizer: Any, prompt_key: str) -> str:
    prompt = row.get(prompt_key)
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise ValueError(f"prompt key {prompt_key!r} must contain a string or messages list")


def finish_reason_type(finish_reason: Any) -> str:
    if isinstance(finish_reason, dict):
        return str(finish_reason.get("type") or finish_reason.get("reason") or "")
    if finish_reason is None:
        return ""
    return str(finish_reason)


def optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def output_token_ids_from_meta(meta_info: dict[str, Any]) -> list[int] | None:
    """Extract generated token ids when SGLang returns token-level metadata."""
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if not isinstance(output_token_logprobs, list):
        return None
    token_ids: list[int] = []
    for item in output_token_logprobs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        token_id = optional_int(item[1])
        if token_id is None:
            return None
        token_ids.append(token_id)
    return token_ids


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_one(
    base_url: str,
    prompt: str,
    sampling_params: dict[str, Any],
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={"text": prompt, "sampling_params": sampling_params},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("text", "")), dict(payload.get("meta_info") or {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    rows = read_jsonl(args.prompts)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no prompts found in {args.prompts}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("")

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    outputs: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        prompt = render_prompt(row, tokenizer, args.prompt_key)
        start = time.time()
        response, meta_info = generate_one(args.base_url, prompt, sampling_params, args.timeout)
        latency = time.time() - start

        prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
        target = str(row.get("target", ""))
        target_tokens = tokenizer(target, add_special_tokens=False).input_ids
        decoded_response_tokens = tokenizer(response, add_special_tokens=False).input_ids
        server_output_token_ids = output_token_ids_from_meta(meta_info)
        server_completion_tokens = optional_int(meta_info.get("completion_tokens"))
        if server_output_token_ids is not None:
            response_token_count = len(server_output_token_ids)
            count_source = "meta_info.output_token_logprobs"
            thinking_scan_tokens = server_output_token_ids
        elif server_completion_tokens is not None:
            response_token_count = server_completion_tokens
            count_source = "meta_info.completion_tokens"
            thinking_scan_tokens = decoded_response_tokens
        else:
            response_token_count = len(decoded_response_tokens)
            count_source = "decoded_text_retokenized"
            thinking_scan_tokens = decoded_response_tokens

        end_pos = find_subsequence(thinking_scan_tokens, end_think_ids)
        found_end = end_pos is not None
        if end_pos is None:
            thinking_tokens = response_token_count
            visible_tokens = 0
            thinking_source = count_source
        else:
            thinking_tokens = end_pos
            visible_tokens = max(0, len(thinking_scan_tokens) - end_pos - len(end_think_ids))
            thinking_source = (
                "meta_info.output_token_logprobs"
                if server_output_token_ids is not None
                else "decoded_text_retokenized"
            )
        finish_reason = meta_info.get("finish_reason")
        finish_type = finish_reason_type(finish_reason)

        output_row = {
            "sample_index": sample_index,
            "metadata": row.get("metadata") or {},
            "prompt": prompt,
            "target": target,
            "response": response,
            "prompt_tokens": len(prompt_tokens),
            "target_tokens": len(target_tokens),
            "response_tokens": response_token_count,
            "decoded_response_tokens": len(decoded_response_tokens),
            "response_token_count_source": count_source,
            "generated_thinking_tokens": thinking_tokens,
            "generated_visible_after_think_tokens": visible_tokens,
            "thinking_token_count_source": thinking_source,
            "found_end_think": found_end,
            "finish_reason": finish_reason,
            "meta_info": meta_info,
            "latency_sec": latency,
        }
        outputs.append(output_row)
        append_jsonl(args.output_jsonl, output_row)
        print(
            json.dumps(
                {
                    "sample_index": sample_index,
                    "response_tokens": response_token_count,
                    "decoded_response_tokens": len(decoded_response_tokens),
                    "token_count_source": count_source,
                    "thinking_tokens": thinking_tokens,
                    "visible_tokens": visible_tokens,
                    "found_end_think": found_end,
                    "finish_reason": finish_type,
                    "latency_sec": round(latency, 3),
                }
            ),
            flush=True,
        )

    response_lengths = [int(row["response_tokens"]) for row in outputs]
    decoded_response_lengths = [int(row["decoded_response_tokens"]) for row in outputs]
    thinking_lengths = [int(row["generated_thinking_tokens"]) for row in outputs]
    visible_lengths = [int(row["generated_visible_after_think_tokens"]) for row in outputs]
    prompt_lengths = [int(row["prompt_tokens"]) for row in outputs]
    target_lengths = [int(row["target_tokens"]) for row in outputs]
    found_end = [bool(row["found_end_think"]) for row in outputs]
    finish_types = [finish_reason_type(row.get("finish_reason")) for row in outputs]
    truncated = [
        ("length" in finish_type.lower()) or int(row["response_tokens"]) >= args.max_new_tokens
        for row, finish_type in zip(outputs, finish_types, strict=True)
    ]

    summary = {
        "model": args.model,
        "prompts": str(args.prompts),
        "output_jsonl": str(args.output_jsonl),
        "tokenizer": str(args.tokenizer),
        "base_url": args.base_url,
        "num_samples": len(outputs),
        "sampling_params": sampling_params,
        "prompt_tokens": summarize(prompt_lengths),
        "target_tokens": summarize(target_lengths),
        "response_tokens": summarize(response_lengths),
        "decoded_response_tokens": summarize(decoded_response_lengths),
        "generated_thinking": summarize(thinking_lengths),
        "generated_visible_after_think": summarize(visible_lengths),
        "thinking_end_ratio": statistics.fmean(1.0 if value else 0.0 for value in found_end),
        "truncated_ratio": statistics.fmean(1.0 if value else 0.0 for value in truncated),
        "finish_reason_counts": {value: finish_types.count(value) for value in sorted(set(finish_types))},
        "latency_sec": summarize([float(row["latency_sec"]) for row in outputs]),
        "sample_row_ids": [
            (row.get("metadata") or {}).get("sft_eval_row_id")
            or (row.get("metadata") or {}).get("id")
            or row["sample_index"]
            for row in outputs
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
