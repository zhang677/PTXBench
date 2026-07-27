#!/usr/bin/env python3
"""Pre-build base prompt `.md` files from a run_parallel_v2 config JSON.

The config JSON is a list of items with a `prompt_tag` field, e.g.
`prompt_configs/2026-0421-2352.json`. This script extracts the unique
prompt_tags and assembles each into `prompt_configs/<prompt_tag>.md` by
concatenating the doc fragments listed under that tag in
`prompt_configs/hub.json`.

run_v2.py / run_parallel_v2.py assume these `.md` files already exist — run
this script once beforehand.

Usage:
    python build_doc_v2.py prompt_configs/2026-0421-2352.json
    python build_doc_v2.py prompt_configs/2026-0421-2352.json --force
    python build_doc_v2.py                  # build every tag in hub.json
    python build_doc_v2.py --force          # rebuild every tag in hub.json
"""

import argparse
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HUB_PATH = SCRIPT_DIR / "prompt_configs" / "hub.json"


def build_doc(prompt_tag: str, hub: dict, *, force: bool = False) -> Path:
    """Recursively assemble `prompt_configs/<prompt_tag>.md` and return its path.

    Leaf entries in hub.json are paths relative to `fib_runtime/` (including
    the `structural_doc/...` prefix and `.md` suffix). Entries without a `/`
    are treated as nested prompt tags and resolved recursively.

    Writes atomically (tmp file + os.replace) so parallel callers don't race.
    """
    base_prompt_path = SCRIPT_DIR / "prompt_configs" / f"{prompt_tag}.md"
    if base_prompt_path.exists() and not force:
        return base_prompt_path

    if prompt_tag not in hub:
        raise KeyError(f"prompt_tag {prompt_tag!r} not found in {HUB_PATH}")

    output_content = ""
    for partial_doc_path in hub[prompt_tag]:
        if "/" not in partial_doc_path:
            doc_path = build_doc(partial_doc_path, hub, force=force)
        else:
            doc_path = SCRIPT_DIR.parent / partial_doc_path
            if not doc_path.exists():
                raise FileNotFoundError(
                    f"Doc fragment {doc_path} (referenced from hub[{prompt_tag!r}]) does not exist"
                )
        output_content += doc_path.read_text() + "\n\n"

    tmp_path = base_prompt_path.with_suffix(base_prompt_path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(output_content)
    os.replace(tmp_path, base_prompt_path)
    return base_prompt_path


def tags_from_config(config_path: Path) -> list[str]:
    """Extract unique prompt_tags from a run_parallel_v2 config JSON, preserving order."""
    with open(config_path) as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{config_path} must be a JSON list of items")
    seen: list[str] = []
    for item in items:
        tag = item.get("prompt_tag")
        if tag is None:
            raise ValueError(f"Config item missing 'prompt_tag': {item}")
        if tag not in seen:
            seen.append(tag)
    return seen


def tags_from_hub(hub: dict) -> list[str]:
    """Return all top-level prompt_tags declared in hub.json, in declared order."""
    return list(hub.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-build base prompt .md files for a run_parallel_v2 config JSON",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help=(
            "Path to config JSON (e.g. prompt_configs/2026-0421-2352.json). "
            "If omitted, every top-level tag in hub.json is built."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the cached .md already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(HUB_PATH) as f:
        hub = json.load(f)

    if args.config is None:
        tags = tags_from_hub(hub)
        print(f"Building all {len(tags)} prompt doc(s) from {HUB_PATH}: {tags}")
    else:
        config_path = Path(args.config).resolve()
        tags = tags_from_config(config_path)
        print(f"Building {len(tags)} prompt doc(s) from {config_path}: {tags}")

    for tag in tags:
        path = build_doc(tag, hub, force=args.force)
        size = path.stat().st_size
        print(f"  {tag} -> {path}  ({size} bytes)")


if __name__ == "__main__":
    main()
