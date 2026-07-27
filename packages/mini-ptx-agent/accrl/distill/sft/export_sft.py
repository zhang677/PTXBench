#!/usr/bin/env python3
"""Export distill reasoning pairs to Miles SFT data.

Input records are produced by accrl.distill.run_experiment and contain the
distill prompt, visible reasoning, optional hidden thinking, and metadata.

The default `reverse_cot` style uses the distill system prompt and full distill
prompt, then trains assistant=<think>reasoning</think>. The reasoning field is
the visible <my_reasoning> block extracted from the teacher output; the
teacher's hidden scratchpad is not used for SFT.

The optional `kernel_sft` style removes the current/expert kernel from the user
prompt and trains assistant=<think>reasoning</think> plus the code block.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SFT_SYSTEM_PROMPT = (
    "You are an expert CUDA kernel developer. Given a kernel optimization task "
    "and any previous evaluation feedback, think step-by-step about the "
    "optimization strategy, then provide your implementation in a single "
    "```cpp code block."
)

NEXT_KERNEL_INSTRUCTION = (
    "Write the next CUDA kernel for this task. First provide your reasoning in "
    "<think>...</think>, then provide the complete implementation in a single "
    "```cpp code block."
)

CURRENT_KERNEL_RE = re.compile(
    r"(?:### Turn \d+ [^\n]*Kernel \(CURRENT\)|## Expert's Kernel)\s*"
    r"```(?:cpp|cuda|c\+\+)?\n(.*?)\n```",
    re.DOTALL,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_kernel_and_user_prompt(prompt: str) -> tuple[str, str]:
    match = CURRENT_KERNEL_RE.search(prompt)
    if not match:
        raise ValueError("could not find current/expert kernel block in prompt")
    kernel = match.group(1).strip()
    user_prompt = prompt[: match.start()].rstrip()
    user_prompt = f"{user_prompt}\n\n{NEXT_KERNEL_INSTRUCTION}"
    return kernel, user_prompt


def build_assistant_content(reasoning: str, kernel_code: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n```cpp\n{kernel_code.strip()}\n```"


def build_think_only_content(reasoning: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>"


def convert_record(
    record: dict[str, Any],
    *,
    reasoning_field: str,
    style: str,
    include_distill_system: bool,
) -> dict[str, Any]:
    if style == "reverse_cot":
        reasoning = record.get(reasoning_field) or ""
        if not reasoning.strip():
            raise ValueError(f"empty reasoning field: {reasoning_field}")
        metadata = record.get("metadata") or {}
        exp_id = metadata.get("exp_id", "unknown")
        turn = metadata.get("turn", "unknown")
        metadata = dict(metadata)
        metadata.update(
            {
                "reasoning_field": reasoning_field,
                "target_source_field": reasoning_field,
                "target_format": "qwen_think_wrapped",
                "teacher_hidden_thinking_available": bool((record.get("thinking") or "").strip()),
                "reasoning_chars": len(reasoning.strip()),
            }
        )
        return {
            "id": f"{exp_id}_t{turn}",
            "messages": [
                {"role": "system", "content": record.get("system_prompt", SFT_SYSTEM_PROMPT)},
                {"role": "user", "content": record["input"]},
                {"role": "assistant", "content": build_think_only_content(reasoning)},
            ],
            "metadata": metadata,
        }

    if style != "kernel_sft":
        raise ValueError(f"unknown style: {style}")

    kernel_code, user_prompt = extract_kernel_and_user_prompt(record["input"])
    reasoning = record.get(reasoning_field) or ""
    if not reasoning.strip():
        raise ValueError(f"empty reasoning field: {reasoning_field}")

    messages = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": build_assistant_content(reasoning, kernel_code)},
    ]

    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "reasoning_field": reasoning_field,
            "target_source_field": reasoning_field,
            "target_format": "qwen_think_wrapped_plus_code",
            "teacher_hidden_thinking_available": bool((record.get("thinking") or "").strip()),
            "reasoning_chars": len(reasoning),
            "kernel_chars": len(kernel_code),
        }
    )
    if include_distill_system:
        metadata["distill_system_prompt"] = record.get("system_prompt", "")

    return {"messages": messages, "metadata": metadata}


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow is required to write parquet; use a .jsonl output or run in the Miles image") from e

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="Input reasoning_pairs.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output .parquet or .jsonl")
    parser.add_argument(
        "--reasoning-field",
        choices=["reasoning", "thinking"],
        default="reasoning",
        help="Which distill field to place inside <think> tags; main experiments should use reasoning.",
    )
    parser.add_argument(
        "--style",
        choices=["reverse_cot", "kernel_sft"],
        default="reverse_cot",
        help=(
            "reverse_cot: assistant is <think>reasoning</think>; "
            "kernel_sft: assistant is <think>reasoning</think> plus code"
        ),
    )
    parser.add_argument(
        "--include-distill-system",
        action="store_true",
        help="Copy the distill generation system prompt into metadata for provenance",
    )
    args = parser.parse_args()

    rows = []
    dropped = 0
    for i, record in enumerate(load_jsonl(args.pairs), 1):
        try:
            rows.append(
                convert_record(
                    record,
                    reasoning_field=args.reasoning_field,
                    style=args.style,
                    include_distill_system=args.include_distill_system,
                )
            )
        except Exception as e:
            dropped += 1
            print(f"drop line {i}: {e}")

    if not rows:
        raise SystemExit("no rows exported")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        write_parquet(rows, args.output)
    elif args.output.suffix == ".jsonl":
        write_jsonl(rows, args.output)
    else:
        raise SystemExit("output must end with .parquet or .jsonl")

    if args.style == "reverse_cot":
        reasoning_lens = [len(row["messages"][-1]["content"]) for row in rows]
    else:
        reasoning_lens = [row["metadata"]["reasoning_chars"] for row in rows]
    print(f"wrote {len(rows)} rows to {args.output} (dropped {dropped})")
    print(
        "reasoning chars: "
        f"min={min(reasoning_lens)} mean={sum(reasoning_lens) // len(reasoning_lens)} "
        f"max={max(reasoning_lens)}"
    )


if __name__ == "__main__":
    main()
