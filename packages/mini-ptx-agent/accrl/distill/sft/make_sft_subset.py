#!/usr/bin/env python3
"""Create deterministic small SFT parquet/jsonl data for smoke runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit("pyarrow is required for parquet; run in the Miles image") from e
        return pq.read_table(path).to_pylist()

    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    raise SystemExit("input must end with .parquet or .jsonl")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit("pyarrow is required for parquet; run in the Miles image") from e
        pq.write_table(pa.Table.from_pylist(rows), path)
        return

    if path.suffix == ".jsonl":
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return

    raise SystemExit("output must end with .parquet or .jsonl")


def row_size_score(row: dict[str, Any]) -> tuple[int, int]:
    messages = row.get("messages") or []
    target_chars = sum(len(message.get("content", "")) for message in messages if message.get("role") == "assistant")
    prompt_chars = sum(len(message.get("content", "")) for message in messages if message.get("role") != "assistant")
    return prompt_chars + target_chars, target_chars


def make_synthetic_rows(num_rows: int) -> list[dict[str, Any]]:
    rows = []
    for idx in range(num_rows):
        nonce = f"ACCRL_OVERFIT_{idx:04d}_K7Q9_Z{(idx * 17 + 31) % 97:02d}"
        target = (
            "<think>\n"
            f"This is controlled SFT overfit trace {idx}.\n"
            f"The checksum phrase is {nonce}.\n"
            "I will repeat the checksum once so the tiny run has a clear memorization signal: "
            f"{nonce}.\n"
            "The correct behavior is to preserve this exact private reasoning trace and then stop.\n"
            "</think>"
        )
        rows.append(
            {
                "id": f"synthetic_overfit_{idx:04d}",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are generating private reasoning traces for an SFT sanity check. "
                            "Return only the requested reasoning inside think tags."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Produce the exact controlled reasoning trace for item {idx}. "
                            f"The unique checksum is {nonce}."
                        ),
                    },
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "definition_name": "synthetic_overfit",
                    "exp_id": f"synthetic_overfit_{idx:04d}",
                    "source_label": "synthetic",
                    "source_reasoning_model": "synthetic",
                    "target_format": "think_wrapped_reasoning",
                    "target_source_field": "synthetic_reasoning",
                    "teacher_hidden_thinking_available": False,
                    "turn": idx,
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mode",
        choices=["head", "sample", "shortest", "longest", "synthetic"],
        default="head",
        help=(
            "head keeps source order; sample chooses a deterministic random subset; "
            "shortest/longest sort by prompt+assistant character count; "
            "synthetic writes tiny controlled rows for overfit sanity checks"
        ),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.rows <= 0:
        raise SystemExit("--rows must be positive")

    source_rows = 0
    if args.mode == "synthetic":
        subset = make_synthetic_rows(args.rows)
        idxs = []
    else:
        rows = load_rows(args.input)
        source_rows = len(rows)
        if not rows:
            raise SystemExit("input contains no rows")

    if args.mode == "synthetic":
        pass
    elif args.mode == "head":
        subset = rows[: args.rows]
        idxs = list(range(min(args.rows, len(rows))))
    elif args.mode == "sample":
        rng = random.Random(args.seed)
        idxs = sorted(rng.sample(range(len(rows)), min(args.rows, len(rows))))
        subset = [rows[i] for i in idxs]
    else:
        reverse = args.mode == "longest"
        scored = sorted(((row_size_score(row), i, row) for i, row in enumerate(rows)), reverse=reverse)
        selected = scored[: args.rows]
        idxs = [i for _, i, _ in selected]
        subset = [row for _, _, row in selected]

    write_rows(args.output, subset)

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "input_rows": source_rows,
        "output_rows": len(subset),
        "mode": args.mode,
        "seed": args.seed,
        "source_indices": idxs,
    }
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"wrote {len(subset)} rows to {args.output} from {source_rows} input rows")


if __name__ == "__main__":
    main()
