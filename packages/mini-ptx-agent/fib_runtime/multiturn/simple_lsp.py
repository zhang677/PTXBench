#!/usr/bin/env python3
"""Lint ``` ```cpp ``` ``` blocks in structural_doc/patterns/*.md against the
hopper-no-hint function allowlist.

Any callee not in the yaml allowlist (and not a C/C++ keyword or a known
CUDA/stdlib builtin) is reported as a violation. Exits non-zero when
anything is flagged so the tool can be wired into a lint check.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import yaml

from analyze_pattern import C_KEYWORDS_NOT_A_FN, Scanner, strip_comments

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PATTERNS_DIR = SCRIPT_DIR.parent / "structural_doc" / "patterns"
DEFAULT_ALLOWLIST = SCRIPT_DIR / "prompt_configs" / "hopper-no-hint.fns.yaml"

BUILTIN_SKIP: set[str] = {
    # CUDA / PTX intrinsics
    "__syncthreads", "__syncwarp", "__threadfence", "__threadfence_block",
    "__threadfence_system",
    "__ballot_sync", "__any_sync", "__all_sync", "__activemask",
    "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync", "__shfl_xor_sync",
    "__ldg",
    "atomicAdd", "atomicSub", "atomicMin", "atomicMax", "atomicExch",
    "atomicCAS", "atomicAnd", "atomicOr", "atomicXor",
    "__popc", "__clz", "__ffs", "__brev",
    # stdlib / common
    "min", "max", "abs", "printf", "memcpy", "memset",
    # primitive-type ctor-looking names
    "uint32_t", "int32_t", "uint64_t", "int64_t", "size_t",
}

SKIP = C_KEYWORDS_NOT_A_FN | BUILTIN_SKIP

CPP_FENCE_RE = re.compile(r"^```cpp\s*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def load_allowlist(path: Path) -> set[str]:
    doc = yaml.safe_load(path.read_text())
    fns = doc.get("functions") or []
    return set(fns)


def iter_cpp_blocks(md_text: str) -> Iterator[tuple[int, str]]:
    """Yield (start_line_1_based, block_text) for each ```cpp fenced block."""
    for m in CPP_FENCE_RE.finditer(md_text):
        start_line = md_text.count("\n", 0, m.start(1)) + 1
        yield start_line, m.group(1)


def extract_calls(block: str) -> list[tuple[int, str]]:
    """Return (line_1_based_within_block, callee_name) for every call.

    A 'call' is an identifier followed by optional `<...>` template args and
    then `(`. Identifiers in `SKIP` (keywords + builtins) are filtered out.
    Strings and char literals are skipped so identifiers inside inline PTX
    don't leak through.
    """
    stripped = strip_comments(block)
    calls: list[tuple[int, str]] = []
    n = len(stripped)
    i = 0
    while i < n:
        c = stripped[i]
        if c == '"':
            sc = Scanner(stripped, i)
            sc.advance_over_string()
            i = sc.pos
            continue
        if c == "'":
            sc = Scanner(stripped, i)
            sc.advance_over_char()
            i = sc.pos
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (stripped[j].isalnum() or stripped[j] == "_"):
                j += 1
            ident = stripped[i:j]
            k = j
            while k < n and stripped[k].isspace():
                k += 1
            if k < n and stripped[k] == "<":
                k = _skip_template(stripped, k, n)
                if k < 0:
                    i = j
                    continue
                while k < n and stripped[k].isspace():
                    k += 1
            if k < n and stripped[k] == "(" and ident not in SKIP:
                line = stripped.count("\n", 0, i) + 1
                calls.append((line, ident))
            i = j
            continue
        i += 1
    return calls


def _skip_template(text: str, start: int, n: int) -> int:
    """Balance `<...>` starting at `text[start] == '<'`. Return index just past
    the closing `>`, or -1 if this was actually a comparison (hit `(` before
    balance, or walked off the end).
    """
    depth = 0
    p = start
    while p < n:
        c = text[p]
        if c == '"':
            sc = Scanner(text, p)
            sc.advance_over_string()
            p = sc.pos
            continue
        if c == "'":
            sc = Scanner(text, p)
            sc.advance_over_char()
            p = sc.pos
            continue
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return p + 1
        elif c == "(":
            return -1
        p += 1
    return -1


def lint_file(md_path: Path, allowlist: set[str]) -> list[tuple[Path, int, str]]:
    text = md_path.read_text()
    violations: list[tuple[Path, int, str]] = []
    for block_start, block in iter_cpp_blocks(text):
        for line_in_block, name in extract_calls(block):
            if name in allowlist:
                continue
            abs_line = block_start + line_in_block - 1
            violations.append((md_path, abs_line, name))
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patterns-dir", type=Path, default=DEFAULT_PATTERNS_DIR)
    ap.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not args.patterns_dir.is_dir():
        print(f"error: patterns dir not found: {args.patterns_dir}", file=sys.stderr)
        return 2
    if not args.allowlist.is_file():
        print(f"error: allowlist not found: {args.allowlist}", file=sys.stderr)
        return 2

    allowlist = load_allowlist(args.allowlist)
    md_files = sorted(args.patterns_dir.glob("*.md"))

    all_violations: list[tuple[Path, int, str]] = []
    for md in md_files:
        all_violations.extend(lint_file(md, allowlist))

    all_violations.sort(key=lambda v: (str(v[0]), v[1], v[2]))
    for path, line, name in all_violations:
        print(f"{path}:{line}  {name}")

    files_with_violations = len({v[0] for v in all_violations})
    if args.verbose or all_violations:
        print(
            f"\n{len(all_violations)} violation(s) across {files_with_violations} file(s) "
            f"(scanned {len(md_files)} file(s), allowlist={len(allowlist)})",
            file=sys.stderr,
        )
    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
