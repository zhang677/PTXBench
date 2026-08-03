"""PTX artifact inspection helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

ARCHITECTURE_INSTRUCTION_FAMILIES = {
    "ampere": (
        "cp.async.ca.shared.global",
        "cp.async.cg.shared.global",
        "ldmatrix.",
        "mma.sync.",
    ),
    "hopper": (
        "barrier.cluster.",
        "cp.async.bulk.",
        "elect.sync",
        "mapa.",
        "mbarrier.",
        "setmaxnreg.",
        "stmatrix.",
        "wgmma.",
    ),
    "blackwell": (
        "red.global.v4",
        "tcgen05.",
    ),
}

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _instruction_opcodes(ptx_text: str) -> list[str]:
    """Extract opcodes from compiled PTX, ignoring directives and comments."""
    text = _BLOCK_COMMENT.sub("", ptx_text)
    opcodes: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith(".") or line in {"{", "}"}:
            continue
        if line.endswith(":"):
            continue
        tokens = line.rstrip(";").split()
        if tokens and tokens[0].startswith("@"):
            tokens = tokens[1:]
        if tokens:
            opcodes.append(tokens[0])
    return opcodes


def extract_architecture_usage(ptx_text: str, virtual_arch: str) -> dict[str, Any]:
    """Summarize architecture-specific instructions from compiled PTX only."""
    family_counts: Counter[str] = Counter()
    architecture_counts: Counter[str] = Counter()
    for opcode in _instruction_opcodes(ptx_text):
        for architecture, families in ARCHITECTURE_INSTRUCTION_FAMILIES.items():
            for family in families:
                if opcode.startswith(family):
                    family_counts[family] += 1
                    architecture_counts[architecture] += 1
                    break

    return {
        "source": "compiled_ptx_artifact",
        "virtual_arch": virtual_arch,
        "instruction_counts": dict(sorted(family_counts.items())),
        "architecture_counts": dict(sorted(architecture_counts.items())),
    }
