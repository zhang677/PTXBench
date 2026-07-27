#!/usr/bin/env python3
"""Extract reference function names from a Hopper prompt-config markdown file
and write them to a YAML cache consumed by `analyze_patterns_batch.py`.

The heavy lifting lives in `analyze_pattern.extract_reference_fn_names`;
this script is a thin CLI wrapper that persists the result so the batch
analysis doesn't re-parse the markdown on every run.

Usage:
    python extract_reference_fns.py [--reference PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pattern import (  # noqa: E402
    DEFAULT_REFERENCE,
    extract_reference_fn_names,
)


def default_output_for(reference: Path) -> Path:
    return reference.parent / f"{reference.stem}.fns.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                    help=f"Markdown prompt config to scan "
                         f"(default: {DEFAULT_REFERENCE})")
    ap.add_argument("--output", type=Path, default=None,
                    help="YAML output path (default: "
                         "<reference-dir>/<reference-stem>.fns.yaml)")
    args = ap.parse_args()

    reference: Path = args.reference.resolve()
    if not reference.is_file():
        sys.exit(f"reference not found: {reference}")

    output: Path = (args.output or default_output_for(reference)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    names = sorted(extract_reference_fn_names(reference))
    payload = {
        "source": reference.name,
        "count": len(names),
        "functions": names,
    }
    output.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"wrote {output} ({len(names)} functions from {reference})")


if __name__ == "__main__":
    main()
