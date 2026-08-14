#!/usr/bin/env python3
"""Render one Triton multiturn test.py from the dedicated Triton template."""

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "template_compile_measure_triton.txt"


def render(definition: str, workload_uuid: str) -> str:
    """Fill definition and workload placeholders in the Triton transport."""
    template = TEMPLATE_PATH.read_text()
    template = template.replace("<definition_name>", definition)
    template = template.replace("<workload_uuid>", workload_uuid)
    assert "cutedsl" not in template.lower()
    assert "CUTE_DSL_ARCH" not in template
    assert '"language": "triton"' in template
    assert '"dependencies": []' in template
    return template


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--workload-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.definition, args.workload_uuid))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
