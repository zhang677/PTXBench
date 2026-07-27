#!/usr/bin/env python3
"""Extract a structural skeleton from a generated CUDA kernel.

The output keeps:
  * The `__global__` kernel's signature, plus a filtered body that preserves
    control flow (if/for/while/do/switch/return/break/continue) and any
    function-call or inline-PTX asm statement.
  * The host `run(...)` function and any other helper function whose name is
    NOT already defined in the Hopper reference prompt.
  * Namespace wrappers, structs/unions, macros, extern shared, and FFI export
    macros pass through verbatim.

Everything else inside the __global__ body (variable declarations,
assignments, bare arithmetic, pointer casts) is dropped so the remaining
text captures the structure of the kernel.

Usage:
    python analyze_pattern.py <kernel.cu> [--reference PATH] [--output PATH]

--------------------------------------------------------------------------
CFG construction (the same statements as a tree, not as text)
--------------------------------------------------------------------------
The second half of this file re-builds the filtered skeleton as a tree of
`CFGNode` (frozen dataclass, see `build_cfg`), so two kernels can be
compared structurally instead of as strings.

  * Interior kinds: SEQ, IF, FOR, WHILE, DO, SWITCH, CASE, DEFAULT.
  * Leaf kinds:
      - CALL  — label is the callee's bare name (namespace qualifiers
                stripped by `_extract_call_name`).
      - JMP   — label is one of `return|break|continue|goto`.
      - ASM   — inline PTX / `__asm__` block.

`build_cfg(body)` mirrors `filter_block_text`'s statement-level dispatch:
`_cfg_statement` routes to `_cfg_if`, `_cfg_for_or_while`, `_cfg_do_while`,
`_cfg_switch`, `_cfg_jump`, `_cfg_asm`; bare `{}` blocks recurse via
`_cfg_seq_from_block`; bare expression statements that look like function
calls become CALL leaves.

Condition expressions are dropped, *except* for any function calls embedded
in them: `_cfg_calls_in_cond` extracts those in order and prepends them to
the body (or appends them, for `do/while`), so tree edit distance still sees
work that was shuffled between a condition and its body.

--------------------------------------------------------------------------
TED (tree edit distance over CFGs)
--------------------------------------------------------------------------
`cfg_to_zss(node)` converts a `CFGNode` tree into a `zss.Node` tree and
delegates to the `zss` library's `simple_distance` (Zhang-Shasha, unit-cost
insert / delete / relabel). `zss` is imported lazily inside `cfg_to_zss` so
this module stays importable without it.

Each zss node's label is `f"{kind}:{label}"`. Collapsing the node kind and
the call name into a single string means one Zhang-Shasha edit covers
either dimension in the same currency:
  * swapping a callee  (`CALL:memcpy`  -> `CALL:cudaMemcpy`) costs 1
  * swapping a control structure (`IF:` -> `WHILE:`)         costs 1
So algorithmic changes and control-flow changes are directly comparable.

Normalization is done by callers (e.g. `analyze_patterns_batch.py`):
`ted / max(node_count_a, node_count_b)` in [0, 1], where 0 means identical
structure and 1 means no shared structure.

--------------------------------------------------------------------------
Companion utilities
--------------------------------------------------------------------------
  * `cfg_hash`                    — 128-bit Blake2b Weisfeiler-Lehman-style
                                    rolling hash; cheap O(N) prefilter for
                                    exact-CFG dedup before O(N^2) TED.
  * `cfg_sexpr`                   — compact S-expression for inspection.
  * `cfg_stats`                   — node count, depth, #branches, #loops,
                                    #calls.
  * `extract_filtered_kernel_text` — text view that maps 1:1 to the CFG
                                     tree (same source drives both).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE = SCRIPT_DIR / "prompt_configs" / "hopper-no-hint.md"

FN_MODIFIERS = {
    "__device__", "__global__", "__host__", "__forceinline__",
    "__noinline__", "static", "inline", "extern", "constexpr",
}

C_KEYWORDS_NOT_A_FN = {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "return", "break", "continue", "goto",
    "sizeof", "alignof", "alignas", "typeof", "decltype",
    "struct", "class", "union", "enum", "typedef", "namespace", "using",
    "template", "typename", "this", "new", "delete", "operator",
    "public", "private", "protected", "virtual", "override", "final",
    "static", "extern", "inline", "const", "constexpr", "volatile",
    "mutable", "register", "thread_local", "auto",
    "__device__", "__global__", "__host__", "__forceinline__", "__noinline__",
    "__launch_bounds__", "__shared__", "__grid_constant__", "__restrict__",
    "asm", "__asm__", "__asm", "try", "catch", "throw",
}


# ---------------------------------------------------------------------------
# Comment stripping (string/char-literal aware, preserves line count).

def strip_comments(src: str) -> str:
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            j = i + 1
            while j < n:
                if src[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(src[i:j])
            i = j
            continue
        if c == "'":
            j = i + 1
            while j < n:
                if src[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if src[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(src[i:j])
            i = j
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            j = i + 2
            while j < n and src[j] != '\n':
                j += 1
            i = j
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '*':
            j = i + 2
            while j + 1 < n and not (src[j] == '*' and src[j + 1] == '/'):
                if src[j] == '\n':
                    out.append('\n')
                j += 1
            i = j + 2 if j + 1 < n else n
            continue
        out.append(c)
        i += 1
    return ''.join(out)


# ---------------------------------------------------------------------------
# Cursor-style scanner.

class Scanner:
    def __init__(self, text: str, pos: int = 0, end: int | None = None):
        self.text = text
        self.pos = pos
        self.end = end if end is not None else len(text)

    def eof(self) -> bool:
        return self.pos >= self.end

    def peek(self, offset: int = 0) -> str:
        p = self.pos + offset
        return self.text[p] if p < self.end else ''

    def skip_ws(self) -> None:
        while self.pos < self.end and self.text[self.pos].isspace():
            self.pos += 1

    def peek_identifier(self) -> str:
        p = self.pos
        if p >= self.end:
            return ''
        c = self.text[p]
        if not (c.isalpha() or c == '_'):
            return ''
        q = p
        while q < self.end and (self.text[q].isalnum() or self.text[q] == '_'):
            q += 1
        return self.text[p:q]

    def advance_identifier(self) -> str:
        ident = self.peek_identifier()
        self.pos += len(ident)
        return ident

    def advance_over_string(self) -> None:
        assert self.peek() == '"'
        self.pos += 1
        while self.pos < self.end:
            c = self.text[self.pos]
            if c == '\\' and self.pos + 1 < self.end:
                self.pos += 2
                continue
            if c == '"':
                self.pos += 1
                return
            self.pos += 1

    def advance_over_char(self) -> None:
        assert self.peek() == "'"
        self.pos += 1
        while self.pos < self.end:
            c = self.text[self.pos]
            if c == '\\' and self.pos + 1 < self.end:
                self.pos += 2
                continue
            if c == "'":
                self.pos += 1
                return
            self.pos += 1

    def match_balanced(self, open_ch: str, close_ch: str) -> tuple[int, int]:
        """Positioned at open_ch. Advance past matching close_ch. Return (start, end)."""
        assert self.peek() == open_ch, f"expected {open_ch} at {self.pos}"
        start = self.pos
        depth = 0
        while self.pos < self.end:
            c = self.text[self.pos]
            if c == '"':
                self.advance_over_string()
                continue
            if c == "'":
                self.advance_over_char()
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    return (start, self.pos)
            self.pos += 1
        raise ValueError(f"Unbalanced {open_ch}{close_ch} starting at {start}")


def _consume_preproc(sc: Scanner) -> None:
    """Consume a preprocessor directive, respecting `\\<newline>` continuation."""
    while not sc.eof():
        c = sc.peek()
        if c == '\\' and sc.peek(1) == '\n':
            sc.pos += 2
            continue
        if c == '\n':
            sc.pos += 1
            return
        sc.pos += 1


# ---------------------------------------------------------------------------
# Reference function-name extraction from the hopper-no-hint.md file.

def extract_reference_fn_names(md_path: Path) -> set[str]:
    """Scan the markdown for every IDENT followed by a balanced `(...)` and then
    a `{` body, return the set of names.

    The md file mixes prose with embedded C++/CUDA code blocks (both inside
    ```cpp fences and inline). This heuristic catches any real function
    definition regardless of fence, while C keywords (`if`, `for`, ...) and
    declarations (ending in `;` before `{`) are rejected.
    """
    text = md_path.read_text()
    stripped = strip_comments(text)
    names: set[str] = set()
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
        if c.isalpha() or c == '_':
            j = i
            while j < n and (stripped[j].isalnum() or stripped[j] == '_'):
                j += 1
            ident = stripped[i:j]
            k = j
            while k < n and stripped[k] in ' \t':
                k += 1
            if k < n and stripped[k] == '(' and ident not in C_KEYWORDS_NOT_A_FN:
                sc = Scanner(stripped, k)
                try:
                    _, close_pos = sc.match_balanced('(', ')')
                except ValueError:
                    i = j
                    continue
                p = close_pos
                while p < n:
                    cc = stripped[p]
                    if cc.isspace():
                        p += 1
                        continue
                    if cc == '{' or cc == ';':
                        break
                    if cc == '(':
                        try:
                            _, pp = Scanner(stripped, p).match_balanced('(', ')')
                            p = pp
                            continue
                        except ValueError:
                            break
                    if cc.isalpha() or cc == '_':
                        q = p
                        while q < n and (stripped[q].isalnum() or stripped[q] == '_'):
                            q += 1
                        p = q
                        continue
                    break
                if p < n and stripped[p] == '{':
                    names.add(ident)
                i = close_pos
                continue
            i = j
            continue
        i += 1
    return names


# ---------------------------------------------------------------------------
# Top-level parser for a .cu file.

@dataclass
class TopItem:
    kind: str  # 'ns_open' | 'ns_close' | 'func' | 'passthrough'
    text: str = ''
    name: str = ''
    signature: str = ''
    body: str = ''
    qualifiers: set[str] = field(default_factory=set)


def parse_top_level(src: str) -> list[TopItem]:
    items: list[TopItem] = []
    sc = Scanner(src)
    _parse_scope(sc, items)
    return items


def _parse_scope(sc: Scanner, items: list[TopItem]) -> None:
    while not sc.eof():
        ws_start = sc.pos
        sc.skip_ws()
        if ws_start != sc.pos:
            items.append(TopItem(kind='passthrough', text=sc.text[ws_start:sc.pos]))
        if sc.eof():
            return
        c = sc.peek()
        if c == '}':
            return
        if c == '#':
            start = sc.pos
            _consume_preproc(sc)
            items.append(TopItem(kind='passthrough', text=sc.text[start:sc.pos]))
            continue
        if sc.peek_identifier() == 'namespace':
            _parse_namespace(sc, items)
            continue
        _parse_top_level_item(sc, items)


def _parse_namespace(sc: Scanner, items: list[TopItem]) -> None:
    start = sc.pos
    sc.advance_identifier()  # 'namespace'
    sc.skip_ws()
    name = sc.peek_identifier()
    if name:
        sc.advance_identifier()
        sc.skip_ws()
    if sc.peek() != '{':
        # not a real namespace — fall back to declaration passthrough
        end_pos = _scan_to_semi_top(sc)
        items.append(TopItem(kind='passthrough', text=sc.text[start:end_pos]))
        return
    sc.pos += 1  # consume '{'
    items.append(TopItem(kind='ns_open', text=sc.text[start:sc.pos], name=name))
    _parse_scope(sc, items)
    if sc.peek() == '}':
        sc.pos += 1
        items.append(TopItem(kind='ns_close', text='}'))
    saved = sc.pos
    sc.skip_ws()
    if sc.peek() == ';':
        # stray ';' after namespace close — keep as passthrough
        items.append(TopItem(kind='passthrough', text=sc.text[saved:sc.pos + 1]))
        sc.pos += 1


def _scan_to_semi_top(sc: Scanner) -> int:
    depth = 0
    while not sc.eof():
        c = sc.peek()
        if c == '"':
            sc.advance_over_string()
            continue
        if c == "'":
            sc.advance_over_char()
            continue
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ';' and depth == 0:
            sc.pos += 1
            return sc.pos
        sc.pos += 1
    return sc.pos


def _parse_top_level_item(sc: Scanner, items: list[TopItem]) -> None:
    """Parse one top-level item: a function definition, or a declaration/struct/union/enum/macro expansion passthrough."""
    start = sc.pos
    depth_paren = 0
    depth_angle = 0
    depth_brack = 0
    saw_call_paren = False
    name: str | None = None

    p = sc.pos
    n = sc.end
    while p < n:
        c = sc.text[p]
        if c == '"':
            sub = Scanner(sc.text, p, n)
            sub.advance_over_string()
            p = sub.pos
            continue
        if c == "'":
            sub = Scanner(sc.text, p, n)
            sub.advance_over_char()
            p = sub.pos
            continue
        if c == '(':
            if depth_paren == 0 and depth_brack == 0 and depth_angle == 0:
                # Find IDENT immediately before this '(' — it's the candidate function name.
                q = p - 1
                while q >= sc.pos and sc.text[q].isspace():
                    q -= 1
                if q >= sc.pos and (sc.text[q].isalnum() or sc.text[q] == '_'):
                    r = q
                    while r > sc.pos and (sc.text[r - 1].isalnum() or sc.text[r - 1] == '_'):
                        r -= 1
                    name = sc.text[r:q + 1]
                    saw_call_paren = True
            depth_paren += 1
            p += 1
            continue
        if c == ')':
            depth_paren -= 1
            p += 1
            continue
        if c == '<' and depth_paren == 0 and not saw_call_paren:
            depth_angle += 1
            p += 1
            continue
        if c == '>' and depth_angle > 0:
            depth_angle -= 1
            p += 1
            continue
        if c == '[':
            depth_brack += 1
            p += 1
            continue
        if c == ']':
            depth_brack -= 1
            p += 1
            continue
        if c == ';' and depth_paren == 0 and depth_angle == 0 and depth_brack == 0:
            sc.pos = p + 1
            items.append(TopItem(kind='passthrough', text=sc.text[start:sc.pos]))
            return
        if c == '}' and depth_paren == 0 and depth_angle == 0 and depth_brack == 0:
            sc.pos = p
            items.append(TopItem(kind='passthrough', text=sc.text[start:p]))
            return
        if c == '{' and depth_paren == 0 and depth_angle == 0 and depth_brack == 0:
            if saw_call_paren and name:
                sc.pos = p
                s_body, e_body = sc.match_balanced('{', '}')
                signature = sc.text[start:s_body].rstrip()
                body = sc.text[s_body + 1:e_body - 1]
                qual = {
                    tok for tok in re.findall(r'\b([A-Za-z_]\w*)\b', signature)
                    if tok in FN_MODIFIERS
                }
                items.append(TopItem(
                    kind='func', name=name, signature=signature, body=body,
                    qualifiers=qual, text=sc.text[start:sc.pos],
                ))
                return
            # Struct / union / enum / extern "C" { ... }; — passthrough to trailing ';'.
            sc.pos = p
            try:
                sc.match_balanced('{', '}')
            except ValueError:
                items.append(TopItem(kind='passthrough', text=sc.text[start:sc.pos]))
                return
            depth2 = 0
            while sc.pos < n:
                cc = sc.peek()
                if cc == '"':
                    sc.advance_over_string()
                    continue
                if cc == "'":
                    sc.advance_over_char()
                    continue
                if cc in '([{':
                    depth2 += 1
                elif cc in ')]}':
                    depth2 -= 1
                elif cc == ';' and depth2 == 0:
                    sc.pos += 1
                    break
                elif cc == '\n' and depth2 == 0:
                    # passthrough likely ends here if no ';' seen; be lenient.
                    pass
                sc.pos += 1
            items.append(TopItem(kind='passthrough', text=sc.text[start:sc.pos]))
            return
        p += 1

    sc.pos = p
    items.append(TopItem(kind='passthrough', text=sc.text[start:p]))


# ---------------------------------------------------------------------------
# Kernel body filter.

def filter_block_text(body: str, indent_spaces: int) -> str:
    sc = Scanner(body)
    lines: list[str] = []
    while True:
        sc.skip_ws()
        if sc.eof():
            break
        stmt = _filter_statement(sc, indent_spaces)
        if stmt is not None and stmt.strip():
            lines.append(stmt)
    return '\n'.join(lines)


def _filter_statement(sc: Scanner, indent_spaces: int) -> str | None:
    sc.skip_ws()
    if sc.eof():
        return None
    start = sc.pos
    c = sc.peek()

    if c == '#':
        _consume_preproc(sc)
        return None

    if c == '{':
        s, e = sc.match_balanced('{', '}')
        inner = sc.text[s + 1:e - 1]
        inner_filtered = filter_block_text(inner, indent_spaces + 4)
        if not inner_filtered.strip():
            return None
        return (' ' * indent_spaces + '{\n'
                + inner_filtered + '\n'
                + ' ' * indent_spaces + '}')

    if c == ';':
        sc.pos += 1
        return None

    ident = sc.peek_identifier()
    if ident == 'if':
        return _scan_if(sc, indent_spaces)
    if ident in ('for', 'while'):
        return _scan_for_or_while(sc, ident, indent_spaces)
    if ident == 'do':
        return _scan_do_while(sc, indent_spaces)
    if ident == 'switch':
        return _scan_switch(sc, indent_spaces)
    if ident in ('return', 'break', 'continue', 'goto'):
        return _scan_keep_to_semi(sc, indent_spaces)
    if ident in ('case', 'default'):
        return _scan_case_label(sc, indent_spaces)
    if ident in ('asm', '__asm__', '__asm'):
        return _scan_asm(sc, indent_spaces)

    end = _scan_to_semi_end(sc)
    stmt_text = sc.text[start:end].strip()
    if _is_call_statement(stmt_text):
        return ' ' * indent_spaces + _collapse_ws(stmt_text)
    return None


def _scan_if(sc: Scanner, indent_spaces: int) -> str:
    sc.advance_identifier()  # 'if'
    sc.skip_ws()
    if sc.peek() != '(':
        return ' ' * indent_spaces + 'if /* malformed */'
    cs, ce = sc.match_balanced('(', ')')
    cond = sc.text[cs:ce]
    sc.skip_ws()
    body = _emit_body(sc, indent_spaces)
    out = ' ' * indent_spaces + f"if {_collapse_ws(cond)} {body}"
    saved = sc.pos
    sc.skip_ws()
    if sc.peek_identifier() == 'else':
        sc.advance_identifier()
        sc.skip_ws()
        if sc.peek_identifier() == 'if':
            nested = _scan_if(sc, indent_spaces)
            nested_stripped = nested[indent_spaces:] if nested.startswith(' ' * indent_spaces) else nested.lstrip()
            out += " else " + nested_stripped
        else:
            else_body = _emit_body(sc, indent_spaces)
            out += f" else {else_body}"
    else:
        sc.pos = saved
    return out


def _scan_for_or_while(sc: Scanner, kw: str, indent_spaces: int) -> str:
    sc.advance_identifier()
    sc.skip_ws()
    if sc.peek() != '(':
        return ' ' * indent_spaces + f'{kw} /* malformed */'
    cs, ce = sc.match_balanced('(', ')')
    cond = sc.text[cs:ce]
    sc.skip_ws()
    body = _emit_body(sc, indent_spaces)
    return ' ' * indent_spaces + f"{kw} {_collapse_ws(cond)} {body}"


def _scan_do_while(sc: Scanner, indent_spaces: int) -> str:
    sc.advance_identifier()  # 'do'
    sc.skip_ws()
    body = _emit_body(sc, indent_spaces)
    sc.skip_ws()
    tail = ''
    if sc.peek_identifier() == 'while':
        sc.advance_identifier()
        sc.skip_ws()
        if sc.peek() == '(':
            cs, ce = sc.match_balanced('(', ')')
            cond = sc.text[cs:ce]
            sc.skip_ws()
            if sc.peek() == ';':
                sc.pos += 1
            tail = f" while {_collapse_ws(cond)};"
    return ' ' * indent_spaces + f"do {body}{tail}"


def _scan_switch(sc: Scanner, indent_spaces: int) -> str:
    sc.advance_identifier()
    sc.skip_ws()
    if sc.peek() != '(':
        return ' ' * indent_spaces + 'switch /* malformed */'
    cs, ce = sc.match_balanced('(', ')')
    cond = sc.text[cs:ce]
    sc.skip_ws()
    body = _emit_body(sc, indent_spaces)
    return ' ' * indent_spaces + f"switch {_collapse_ws(cond)} {body}"


def _scan_case_label(sc: Scanner, indent_spaces: int) -> str:
    start = sc.pos
    kw = sc.advance_identifier()
    if kw == 'case':
        depth = 0
        while not sc.eof():
            c = sc.peek()
            if c == '"':
                sc.advance_over_string()
                continue
            if c == "'":
                sc.advance_over_char()
                continue
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            elif c == ':' and depth == 0:
                sc.pos += 1
                break
            sc.pos += 1
    else:
        sc.skip_ws()
        if sc.peek() == ':':
            sc.pos += 1
    text = sc.text[start:sc.pos].strip()
    return ' ' * indent_spaces + _collapse_ws(text)


def _scan_keep_to_semi(sc: Scanner, indent_spaces: int) -> str:
    start = sc.pos
    end = _scan_to_semi_end(sc)
    return ' ' * indent_spaces + _collapse_ws(sc.text[start:end].strip())


def _scan_asm(sc: Scanner, indent_spaces: int) -> str:
    start = sc.pos
    sc.advance_identifier()  # 'asm' / '__asm__' / '__asm'
    sc.skip_ws()
    if sc.peek_identifier() in ('volatile', '__volatile__'):
        sc.advance_identifier()
        sc.skip_ws()
    if sc.peek() == '(':
        sc.match_balanced('(', ')')
    sc.skip_ws()
    if sc.peek() == ';':
        sc.pos += 1
    text = sc.text[start:sc.pos].strip()
    # Preserve asm body verbatim, but reflow the wrapper onto a single-ish line:
    # collapse pure whitespace runs outside of strings so the output reads clean.
    return ' ' * indent_spaces + text


def _scan_to_semi_end(sc: Scanner) -> int:
    depth = 0
    while not sc.eof():
        c = sc.peek()
        if c == '"':
            sc.advance_over_string()
            continue
        if c == "'":
            sc.advance_over_char()
            continue
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ';' and depth == 0:
            sc.pos += 1
            return sc.pos
        sc.pos += 1
    return sc.pos


def _emit_body(sc: Scanner, indent_spaces: int) -> str:
    sc.skip_ws()
    if sc.peek() == '{':
        s, e = sc.match_balanced('{', '}')
        inner = sc.text[s + 1:e - 1]
        filtered = filter_block_text(inner, indent_spaces + 4)
        if filtered.strip():
            return '{\n' + filtered + '\n' + ' ' * indent_spaces + '}'
        return '{ }'
    stmt = _filter_statement(sc, indent_spaces + 4)
    if stmt is None or not stmt.strip():
        return '{ }'
    return '{\n' + stmt + '\n' + ' ' * indent_spaces + '}'


def _is_call_statement(stmt: str) -> bool:
    s = stmt.strip()
    if s.endswith(';'):
        s = s[:-1].rstrip()
    if not s:
        return False
    m = re.match(r'^[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*', s)
    if not m:
        return False
    first = s[:m.end()].strip()
    first_bare = first.split('::')[-1].strip()
    if first_bare in C_KEYWORDS_NOT_A_FN:
        return False
    i = m.end()
    while i < len(s) and s[i].isspace():
        i += 1
    if i < len(s) and s[i] == '<':
        depth = 0
        j = i
        while j < len(s):
            c = s[j]
            if c == '"':
                sc = Scanner(s, j)
                sc.advance_over_string()
                j = sc.pos
                continue
            if c == "'":
                sc = Scanner(s, j)
                sc.advance_over_char()
                j = sc.pos
                continue
            if c == '<':
                depth += 1
            elif c == '>':
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            elif c == '(':
                # unmatched '<'; it was actually a comparison — not a call.
                return False
            j += 1
        else:
            return False
        i = j
        while i < len(s) and s[i].isspace():
            i += 1
    if i >= len(s) or s[i] != '(':
        return False
    sc = Scanner(s, i)
    try:
        _, end_paren = sc.match_balanced('(', ')')
    except ValueError:
        return False
    tail = s[end_paren:].strip()
    return tail == ''


def _collapse_ws(text: str) -> str:
    """Collapse whitespace runs outside of string/char literals into single spaces."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            sc = Scanner(text, i)
            sc.advance_over_string()
            out.append(text[i:sc.pos])
            i = sc.pos
            continue
        if c == "'":
            sc = Scanner(text, i)
            sc.advance_over_char()
            out.append(text[i:sc.pos])
            i = sc.pos
            continue
        if c.isspace():
            out.append(' ')
            while i < n and text[i].isspace():
                i += 1
            continue
        out.append(c)
        i += 1
    return ''.join(out).strip()


# ---------------------------------------------------------------------------
# Top-level assembly.

def process_source(src: str, reference_names: set[str]) -> str:
    stripped = strip_comments(src)
    items = parse_top_level(stripped)
    out_parts: list[str] = []
    prev_omitted = False
    for item in items:
        if item.kind == 'passthrough':
            if prev_omitted and item.text.strip() == '':
                # Absorb trailing whitespace of an omitted helper so we don't leave
                # a big blank gap in the output.
                continue
            out_parts.append(item.text)
            prev_omitted = False
        elif item.kind == 'ns_open':
            out_parts.append(item.text)
            prev_omitted = False
        elif item.kind == 'ns_close':
            out_parts.append(item.text)
            prev_omitted = False
        elif item.kind == 'func':
            if '__global__' in item.qualifiers:
                filtered = filter_block_text(item.body, indent_spaces=4)
                sig = item.signature.rstrip()
                out_parts.append(f"{sig} {{\n{filtered}\n}}")
                prev_omitted = False
            elif item.name in reference_names:
                prev_omitted = True  # omit — already in the reference
            else:
                out_parts.append(item.text)
                prev_omitted = False
    return ''.join(out_parts)


# ---------------------------------------------------------------------------
# CFG extraction — a structured control-flow tree over the same statements
# that filter_block_text surfaces, but as data instead of text. Node labels
# are name-aware (CALL:callee) so tree edit distance between two CFGs
# penalizes both call-name and control-flow differences in the same currency.

import hashlib as _hashlib

CFG_INTERIOR_KINDS = {'SEQ', 'IF', 'FOR', 'WHILE', 'DO', 'SWITCH', 'CASE', 'DEFAULT'}


@dataclass(frozen=True)
class CFGNode:
    kind: str          # SEQ | IF | FOR | WHILE | DO | SWITCH | CASE | DEFAULT | CALL | JMP | ASM
    label: str = ''    # CALL: callee; JMP: return|break|continue|goto
    children: tuple = ()

    def __repr__(self) -> str:  # pragma: no cover — debug only
        lab = f":{self.label}" if self.label else ""
        return f"CFGNode({self.kind}{lab}, n={len(self.children)})"


def build_cfg(body: str) -> CFGNode:
    """Parse a kernel/function body into a structured CFG tree.

    Mirrors filter_block_text's statement-level dispatch, but returns a tree of
    CFGNodes instead of an indented string. Condition expressions are dropped
    except for any embedded call statements, which are prepended to the body's
    SEQ so TED doesn't miss work moved between condition and body.
    """
    sc = Scanner(body)
    stmts: list[CFGNode] = []
    while True:
        sc.skip_ws()
        if sc.eof():
            break
        node = _cfg_statement(sc)
        if node is not None:
            stmts.append(node)
    return CFGNode('SEQ', '', tuple(stmts))


def _cfg_seq_from_block(block: str) -> CFGNode:
    sc = Scanner(block)
    stmts: list[CFGNode] = []
    while True:
        sc.skip_ws()
        if sc.eof():
            break
        node = _cfg_statement(sc)
        if node is not None:
            stmts.append(node)
    return CFGNode('SEQ', '', tuple(stmts))


def _cfg_calls_in_cond(cond_text: str) -> list[CFGNode]:
    """Extract callee names from a (cond) expression as CALL leaves, in order.

    Walks the cond string skipping string/char literals so identifiers inside
    quoted asm-like cond fragments don't leak through. Mirrors the filter-side
    call tokenization.
    """
    out: list[CFGNode] = []
    i, n = 0, len(cond_text)
    while i < n:
        c = cond_text[i]
        if c == '"':
            sub = Scanner(cond_text, i)
            sub.advance_over_string()
            i = sub.pos
            continue
        if c == "'":
            sub = Scanner(cond_text, i)
            sub.advance_over_char()
            i = sub.pos
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < n and (cond_text[j].isalnum() or cond_text[j] == '_'):
                j += 1
            ident = cond_text[i:j]
            k = j
            while k < n and cond_text[k] in ' \t':
                k += 1
            if (k < n and cond_text[k] == '('
                    and ident not in C_KEYWORDS_NOT_A_FN
                    and ident not in FN_MODIFIERS):
                out.append(CFGNode('CALL', ident, ()))
            i = j
            continue
        i += 1
    return out


def _cfg_statement(sc: Scanner):
    sc.skip_ws()
    if sc.eof():
        return None
    start = sc.pos
    c = sc.peek()

    if c == '#':
        _consume_preproc(sc)
        return None

    if c == '{':
        s, e = sc.match_balanced('{', '}')
        inner = sc.text[s + 1:e - 1]
        return _cfg_seq_from_block(inner)

    if c == ';':
        sc.pos += 1
        return None

    ident = sc.peek_identifier()
    if ident == 'if':
        return _cfg_if(sc)
    if ident in ('for', 'while'):
        return _cfg_for_or_while(sc, ident)
    if ident == 'do':
        return _cfg_do_while(sc)
    if ident == 'switch':
        return _cfg_switch(sc)
    if ident in ('return', 'break', 'continue', 'goto'):
        return _cfg_jump(sc, ident)
    if ident in ('case', 'default'):
        return _cfg_case_label(sc)
    if ident in ('asm', '__asm__', '__asm'):
        return _cfg_asm(sc)

    end = _scan_to_semi_end(sc)
    stmt_text = sc.text[start:end].strip()
    if _is_call_statement(stmt_text):
        name = _extract_call_name(stmt_text)
        if name:
            return CFGNode('CALL', name, ())
    return None


def _extract_call_name(stmt: str) -> str:
    s = stmt.strip()
    if s.endswith(';'):
        s = s[:-1].rstrip()
    m = re.match(r'^[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*', s)
    if not m:
        return ''
    name = s[:m.end()].strip()
    return name.split('::')[-1].strip()


def _cfg_if(sc: Scanner) -> CFGNode:
    sc.advance_identifier()  # 'if'
    sc.skip_ws()
    cond_calls: list[CFGNode] = []
    if sc.peek() == '(':
        cs, ce = sc.match_balanced('(', ')')
        cond_calls = _cfg_calls_in_cond(sc.text[cs + 1:ce - 1])
        sc.skip_ws()
    then_body = _cfg_emit_body(sc)
    else_body: CFGNode = CFGNode('SEQ', '', ())
    saved = sc.pos
    sc.skip_ws()
    if sc.peek_identifier() == 'else':
        sc.advance_identifier()
        sc.skip_ws()
        if sc.peek_identifier() == 'if':
            nested = _cfg_if(sc)
            else_body = CFGNode('SEQ', '', (nested,))
        else:
            else_body = _cfg_emit_body(sc)
    else:
        sc.pos = saved
    if cond_calls:
        # Prepend condition-side calls into the THEN branch so they appear before
        # the branch body (closest analog to "work done before the branch").
        then_body = CFGNode('SEQ', '',
                            tuple(cond_calls) + tuple(then_body.children))
    return CFGNode('IF', '', (then_body, else_body))


def _cfg_for_or_while(sc: Scanner, kw: str) -> CFGNode:
    sc.advance_identifier()
    sc.skip_ws()
    cond_calls: list[CFGNode] = []
    if sc.peek() == '(':
        cs, ce = sc.match_balanced('(', ')')
        cond_calls = _cfg_calls_in_cond(sc.text[cs + 1:ce - 1])
        sc.skip_ws()
    body = _cfg_emit_body(sc)
    if cond_calls:
        body = CFGNode('SEQ', '', tuple(cond_calls) + tuple(body.children))
    kind = 'FOR' if kw == 'for' else 'WHILE'
    return CFGNode(kind, '', (body,))


def _cfg_do_while(sc: Scanner) -> CFGNode:
    sc.advance_identifier()  # 'do'
    sc.skip_ws()
    body = _cfg_emit_body(sc)
    sc.skip_ws()
    trailing_cond_calls: list[CFGNode] = []
    if sc.peek_identifier() == 'while':
        sc.advance_identifier()
        sc.skip_ws()
        if sc.peek() == '(':
            cs, ce = sc.match_balanced('(', ')')
            trailing_cond_calls = _cfg_calls_in_cond(sc.text[cs + 1:ce - 1])
            sc.skip_ws()
            if sc.peek() == ';':
                sc.pos += 1
    if trailing_cond_calls:
        body = CFGNode('SEQ', '', tuple(body.children) + tuple(trailing_cond_calls))
    return CFGNode('DO', '', (body,))


def _cfg_switch(sc: Scanner) -> CFGNode:
    sc.advance_identifier()  # 'switch'
    sc.skip_ws()
    cond_calls: list[CFGNode] = []
    if sc.peek() == '(':
        cs, ce = sc.match_balanced('(', ')')
        cond_calls = _cfg_calls_in_cond(sc.text[cs + 1:ce - 1])
        sc.skip_ws()
    body = _cfg_emit_body(sc)
    # `body` is a SEQ whose children are a mix of CASE/DEFAULT headers and
    # their following statements. Reshape: each CASE/DEFAULT captures the
    # statements until the next CASE/DEFAULT.
    cases: list[CFGNode] = []
    pending_kind: str | None = None
    pending_stmts: list[CFGNode] = []
    for child in body.children:
        if child.kind in ('CASE', 'DEFAULT'):
            if pending_kind is not None:
                cases.append(CFGNode(pending_kind, '',
                                     (CFGNode('SEQ', '', tuple(pending_stmts)),)))
            pending_kind = child.kind
            pending_stmts = list(child.children)
        else:
            if pending_kind is None:
                pending_kind = 'CASE'
                pending_stmts = []
            pending_stmts.append(child)
    if pending_kind is not None:
        cases.append(CFGNode(pending_kind, '',
                             (CFGNode('SEQ', '', tuple(pending_stmts)),)))
    if cond_calls:
        cases = [CFGNode('SEQ', '', tuple(cond_calls)), *cases]
    return CFGNode('SWITCH', '', tuple(cases))


def _cfg_case_label(sc: Scanner) -> CFGNode:
    """Emit a transient CASE/DEFAULT marker with no body. _cfg_switch reshapes these."""
    kw = sc.advance_identifier()
    if kw == 'case':
        depth = 0
        while not sc.eof():
            c = sc.peek()
            if c == '"':
                sc.advance_over_string()
                continue
            if c == "'":
                sc.advance_over_char()
                continue
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            elif c == ':' and depth == 0:
                sc.pos += 1
                break
            sc.pos += 1
        return CFGNode('CASE', '', ())
    sc.skip_ws()
    if sc.peek() == ':':
        sc.pos += 1
    return CFGNode('DEFAULT', '', ())


def _cfg_jump(sc: Scanner, kw: str) -> CFGNode:
    _scan_to_semi_end(sc)
    return CFGNode('JMP', kw, ())


def _cfg_asm(sc: Scanner) -> CFGNode:
    sc.advance_identifier()  # 'asm' / '__asm__' / '__asm'
    sc.skip_ws()
    if sc.peek_identifier() in ('volatile', '__volatile__'):
        sc.advance_identifier()
        sc.skip_ws()
    if sc.peek() == '(':
        sc.match_balanced('(', ')')
    sc.skip_ws()
    if sc.peek() == ';':
        sc.pos += 1
    return CFGNode('ASM', '', ())


def _cfg_emit_body(sc: Scanner) -> CFGNode:
    sc.skip_ws()
    if sc.peek() == '{':
        s, e = sc.match_balanced('{', '}')
        inner = sc.text[s + 1:e - 1]
        return _cfg_seq_from_block(inner)
    stmt = _cfg_statement(sc)
    if stmt is None:
        return CFGNode('SEQ', '', ())
    return CFGNode('SEQ', '', (stmt,))


# ---------------------------------------------------------------------------
# CFG utilities: hashing, serialization, stats, and zss conversion.

def cfg_hash(node: CFGNode) -> str:
    """128-bit Weisfeiler–Lehman-style rolling hash over the ordered tree."""
    h = _hashlib.blake2b(digest_size=16)
    h.update(node.kind.encode())
    h.update(b'\0')
    h.update(node.label.encode())
    for ch in node.children:
        h.update(b'\0')
        h.update(bytes.fromhex(cfg_hash(ch)))
    return h.hexdigest()


_SEXPR_CODE = {
    'SEQ': 'S', 'IF': 'I', 'FOR': 'F', 'WHILE': 'W', 'DO': 'D',
    'SWITCH': 'SW', 'CASE': 'CS', 'DEFAULT': 'DF',
    'CALL': 'C', 'JMP': 'J', 'ASM': 'A',
}


def cfg_sexpr(node: CFGNode) -> str:
    code = _SEXPR_CODE.get(node.kind, node.kind)
    if node.kind in ('CALL', 'JMP'):
        return f"{code}:{node.label}"
    if node.kind == 'ASM':
        return code
    inner = ','.join(cfg_sexpr(c) for c in node.children)
    return f"{code}[{inner}]"


def cfg_stats(node: CFGNode) -> dict:
    count = 0
    branches = 0
    loops = 0
    calls = 0

    def walk(n: CFGNode, depth: int) -> int:
        nonlocal count, branches, loops, calls
        count += 1
        if n.kind == 'IF':
            branches += 1
        elif n.kind in ('FOR', 'WHILE', 'DO'):
            loops += 1
        elif n.kind == 'CALL':
            calls += 1
        max_d = depth
        for ch in n.children:
            max_d = max(max_d, walk(ch, depth + 1))
        return max_d

    max_depth = walk(node, 0) if node.children or node.kind != 'SEQ' else 0
    return {
        'node_count': count,
        'depth': max_depth,
        'branches': branches,
        'loops': loops,
        'calls': calls,
    }


def extract_filtered_kernel_text(src: str) -> str:
    """Return the filtered `__global__` kernels of a .cu source as one string.

    Each `__global__` kernel is emitted as `{signature} {\\n  {filtered_body}\\n}`,
    matching what process_source emits for globals — but *only* globals are
    emitted; namespaces, #includes, helpers, and other passthroughs are dropped.
    This is the text view that maps 1:1 to the CFG tree drawn by
    draw_pattern_tree (same source for build_cfg).
    """
    stripped = strip_comments(src)
    items = parse_top_level(stripped)
    parts: list[str] = []
    for item in items:
        if item.kind == 'func' and '__global__' in item.qualifiers:
            filtered = filter_block_text(item.body, indent_spaces=4)
            sig = item.signature.rstrip()
            parts.append(f"{sig} {{\n{filtered}\n}}")
    if not parts:
        return "// no __global__ kernel in source\n"
    return "\n\n".join(parts) + "\n"


def cfg_to_zss(node: CFGNode):
    """Convert a CFGNode tree into a zss.Node tree for tree edit distance.

    Label is `kind:label` so zss's default unit-cost relabel/insert/delete
    matches the semantics we want: one edit per call-name change, one edit
    per control-flow kind change.
    """
    import zss  # local import so analyze_pattern stays importable without zss
    label = f"{node.kind}:{node.label}" if node.label else node.kind
    z = zss.Node(label)
    for ch in node.children:
        z.addkid(cfg_to_zss(ch))
    return z


# ---------------------------------------------------------------------------
# CLI.

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else '')
    ap.add_argument('kernel', type=Path, help='Path to the CUDA kernel .cu file')
    ap.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE,
                    help=f'Path to the reference markdown (default: {DEFAULT_REFERENCE})')
    ap.add_argument('--output', type=Path, default=None,
                    help='Write output here (default: stdout)')
    args = ap.parse_args()

    reference_names = extract_reference_fn_names(args.reference)
    src = args.kernel.read_text()
    out = process_source(src, reference_names)

    if args.output is not None:
        args.output.write_text(out)
    else:
        sys.stdout.write(out)


if __name__ == '__main__':
    main()
