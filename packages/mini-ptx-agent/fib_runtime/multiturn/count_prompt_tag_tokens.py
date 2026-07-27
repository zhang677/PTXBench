#!/usr/bin/env python3
"""Sum structural-doc token counts for every prompt tag in hub.json.

The output schema is:
    prompt_tag,num_tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIB_RUNTIME_ROOT = HERE.parent
PROMPT_CONFIG_DIR = HERE / "prompt_configs"
DEFAULT_TOKEN_COUNTS = (
    FIB_RUNTIME_ROOT / "structural_doc" / "file_token_counts__Qwen_Qwen3.6-35B-A3B.csv"
)


def default_output_path(token_counts_path: Path) -> Path:
    name = token_counts_path.name
    if name.startswith("file_token_counts"):
        name = name.replace("file_token_counts", "prompt_tag_token_counts", 1)
    else:
        name = "prompt_tag_token_counts.csv"
    return PROMPT_CONFIG_DIR / name


def load_token_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        expected = {"path", "num_tokens"}
        if not reader.fieldnames or set(reader.fieldnames) != expected:
            raise ValueError(f"{path} must have columns: path,num_tokens")
        for row in reader:
            doc_path = row["path"]
            if doc_path in counts:
                raise ValueError(f"duplicate token-count path: {doc_path}")
            counts[doc_path] = int(row["num_tokens"])
    return counts


def load_hub(path: Path) -> dict[str, list[str]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    hub: dict[str, list[str]] = {}
    for tag, items in data.items():
        if not isinstance(tag, str) or not isinstance(items, list):
            raise ValueError(f"invalid hub entry for {tag!r}; expected list of strings")
        if not all(isinstance(item, str) for item in items):
            raise ValueError(f"invalid hub entry for {tag!r}; expected list of strings")
        hub[tag] = items
    return hub


def resolve_tag(
    tag: str,
    hub: dict[str, list[str]],
    token_counts: dict[str, int],
    memo: dict[str, int],
    stack: tuple[str, ...] = (),
) -> int:
    if tag in memo:
        return memo[tag]
    if tag in stack:
        cycle = " -> ".join((*stack, tag))
        raise ValueError(f"cycle in hub tag references: {cycle}")
    if tag not in hub:
        raise KeyError(f"unknown prompt tag: {tag}")

    total = 0
    for item in hub[tag]:
        if item in hub:
            total += resolve_tag(item, hub, token_counts, memo, (*stack, tag))
        elif item in token_counts:
            total += token_counts[item]
        else:
            raise KeyError(f"{tag} references unknown tag/path: {item}")

    memo[tag] = total
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count total structural-doc tokens for every prompt tag in hub.json."
    )
    parser.add_argument(
        "--hub",
        type=Path,
        default=PROMPT_CONFIG_DIR / "hub.json",
        help="Path to hub.json (default: ./prompt_configs/hub.json)",
    )
    parser.add_argument(
        "--token-counts",
        type=Path,
        default=DEFAULT_TOKEN_COUNTS,
        help=f"CSV with path,num_tokens rows (default: {DEFAULT_TOKEN_COUNTS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: prompt_tag_token_counts*.csv under prompt_configs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or default_output_path(args.token_counts)

    token_counts = load_token_counts(args.token_counts)
    hub = load_hub(args.hub)

    memo: dict[str, int] = {}
    rows = [
        {"prompt_tag": tag, "num_tokens": resolve_tag(tag, hub, token_counts, memo)}
        for tag in hub
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_tag", "num_tokens"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} row(s) to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
