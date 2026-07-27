#!/usr/bin/env python3
"""Build Miles eval prompts from SFT parquet/jsonl rows.

By default this removes the assistant target from each SFT row and evaluates on
the remaining SFT prompt. When --turns is provided, it instead reconstructs the
original Gemini policy prompt from the extracted per-turn records and keeps the
SFT row's assistant content only as the comparison target.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit("pyarrow is required for parquet input") from e
        return pq.read_table(path).to_pylist()

    raise SystemExit(f"unsupported input format: {path}")


def canonical_key_from_metadata(metadata: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(metadata["exp_id"]),
        str(metadata["definition_name"]),
        int(metadata["turn"]),
    )


def canonical_key_from_turn(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["exp_id"]),
        str(row["definition_name"]),
        int(row["turn"]),
    )


def index_turns(
    path: Path,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    turns = load_rows(path)
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for turn in turns:
        key = canonical_key_from_turn(turn)
        if key in by_key:
            raise SystemExit(f"duplicate turn key in {path}: {key}")
        by_key[key] = turn
        by_run.setdefault((key[0], key[1]), []).append(turn)

    for run_turns in by_run.values():
        run_turns.sort(key=lambda row: int(row["turn"]))
    return by_key, by_run


def build_gemini_policy_prompt(
    key: tuple[str, str, int],
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    by_run: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, str]]:
    exp_id, definition_name, turn_idx = key
    if key not in by_key:
        raise SystemExit(f"SFT row key not found in Gemini turns: {key}")

    current = by_key[key]
    messages = [
        {"role": "system", "content": current.get("system_prompt", "")},
        {"role": "user", "content": current.get("task_prompt", "")},
    ]
    for previous in by_run.get((exp_id, definition_name), []):
        if int(previous["turn"]) >= turn_idx:
            break
        assistant_content = previous.get("raw_assistant_content") or ""
        feedback_content = previous.get("raw_feedback_content") or ""
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        if feedback_content:
            messages.append({"role": "user", "content": feedback_content})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--turns",
        type=Path,
        default=None,
        help=(
            "Optional extracted Gemini turns JSONL. When set, eval prompts are "
            "rebuilt from the original Gemini policy conversation instead of "
            "the SFT row's reverse-CoT prompt."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=None,
        help="If set, take a contiguous slice starting at this row. Use 0 for the first rows.",
    )
    parser.add_argument("--tokenizer", type=Path, default=None, help="If set, render prompt text with chat template")
    parser.add_argument("--output-key", default=None, help="Prompt key to write. Default: prompt when rendered, messages otherwise")
    args = parser.parse_args()

    rows = load_rows(args.data)
    if not rows:
        raise SystemExit(f"empty dataset: {args.data}")

    if args.sample_offset is not None:
        if args.sample_offset < 0:
            raise SystemExit(f"--sample-offset must be non-negative, got {args.sample_offset}")
        indices = list(range(args.sample_offset, min(args.sample_offset + args.num_samples, len(rows))))
    elif len(rows) <= args.num_samples:
        indices = list(range(len(rows)))
    else:
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.num_samples))

    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    output_key = args.output_key or ("prompt" if tokenizer is not None else "messages")

    turns_by_key = None
    turns_by_run = None
    if args.turns is not None:
        turns_by_key, turns_by_run = index_turns(args.turns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row_idx in indices:
            row = rows[row_idx]
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise SystemExit(f"row {row_idx} has invalid messages")
            if messages[-1].get("role") == "assistant":
                prompt_messages = messages[:-1]
                target_text = messages[-1].get("content", "")
            else:
                prompt_messages = messages
                target_text = ""
            metadata = dict(row.get("metadata") or {})
            if turns_by_key is not None and turns_by_run is not None:
                key = canonical_key_from_metadata(metadata)
                prompt_messages = build_gemini_policy_prompt(key, turns_by_key, turns_by_run)
                metadata["sft_eval_prompt_source"] = "gemini_turns"
                metadata["sft_eval_prompt_key"] = "/".join(map(str, key))
            else:
                metadata["sft_eval_prompt_source"] = "sft_messages"
            metadata["sft_eval_row_idx"] = row_idx
            metadata["sft_eval_row_id"] = row.get("id") or metadata.get("id") or str(row_idx)
            prompt_value = prompt_messages
            if tokenizer is not None:
                prompt_value = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            out = {
                output_key: prompt_value,
                "target": target_text,
                "metadata": metadata,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(args.data),
        "output": str(args.output),
        "turns": str(args.turns) if args.turns is not None else None,
        "num_rows": len(indices),
        "indices": indices,
        "sample_offset": args.sample_offset,
        "seed": args.seed,
        "format": (
            "miles_eval_rendered_gemini_policy_prompt"
            if tokenizer is not None and args.turns is not None
            else "miles_eval_gemini_policy_messages"
            if args.turns is not None
            else "miles_eval_rendered_prompt"
            if tokenizer is not None
            else "miles_eval_messages_without_assistant_target_in_label"
        ),
        "input_key": output_key,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
