#!/usr/bin/env python3
"""Render and verify the shared prompt registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
HUB = json.loads((PROMPTS / "hub.json").read_text())


def document(tag: str, stack: tuple[str, ...] = ()) -> str:
    if tag in stack:
        raise ValueError(f"Prompt tag cycle: {' -> '.join((*stack, tag))}")
    if tag not in HUB:
        raise KeyError(f"Unknown prompt tag: {tag}")
    parts = []
    for item in HUB[tag]:
        if "/" in item:
            relative = Path(item)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe prompt fragment: {item}")
            value = (PROMPTS / "fragments" / relative).read_text()
        else:
            value = document(item, (*stack, tag))
        parts.append(value + "\n\n")
    return "".join(parts)


def check() -> None:
    assembled = PROMPTS / "assembled"
    for tag in HUB:
        path = assembled / f"{tag}.md"
        if path.read_text() != document(tag):
            raise ValueError(f"Prompt document differs from registry: {path}")
    extra = {path.stem for path in assembled.glob("*.md")} - set(HUB)
    if extra:
        raise ValueError(f"Unregistered prompt documents: {sorted(extra)}")
    print(f"Checked {len(HUB)} prompt tags")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", help="Tag to render")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if not args.tag or not args.output:
        parser.error("a tag and --output are required to render a document")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document(args.tag))


if __name__ == "__main__":
    main()
