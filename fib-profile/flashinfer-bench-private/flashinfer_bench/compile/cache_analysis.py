"""Standalone CUDA/C++ source check for persistent-state-dependent work skipping.

This opt-in check is separate from compilation and evaluation. It is a lexical
control-flow heuristic, not a C++ verifier. It follows direct host helper calls
and simple assignments, and reports persistent state guarding
returns or compute launches. Allocation/descriptor/attribute setup alone is OK.
Macros are not expanded; indirect calls, aliasing and device-side cache protocols
can evade this screen. A clean result is not proof of absence of result reuse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class AnalysisError(ValueError):
    """The source could not be screened; never treat this as a clean result."""


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    state: tuple[str, ...]

    def __str__(self):
        return f"{self.path}:{self.line}: {self.rule} (state: {', '.join(self.state)})"


# Strip comments, quoted strings (including PTX) and directives without changing
# line offsets. Keywords in diagnostics and comments must not drive findings.
_NO_CODE = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)"'
    r'|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r"|^[ \t]*\#(?:[^\n]*\\\n)*[^\n]*",
    re.S | re.M,
)
_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d*)?|<<<|>>>|::|->|==|!=|<=|>=|&&|\|\||\+\+|--|[^\s]")
_ID = re.compile(r"^[A-Za-z_]\w*$")
_CONTROL = {"if", "for", "while", "switch", "catch"}
_COMPUTE_API = re.compile(r"^(?:cu(?:da)?Launch\w*|cublas\w*|cudnn\w*)$")


def _names(tokens):
    return {t for t in tokens if _ID.match(t)}


def _declared(tokens):
    """Names of simple scalar/pointer/object declarators; ignore constants."""
    if not tokens or "constexpr" in tokens or "__shared__" in tokens:
        return set()
    if tokens[0] in {"using", "typedef", "return", "enum"}:
        return set()
    # const T* p is mutable; T* const p and const T n are not.
    head = tokens[: tokens.index("=")] if "=" in tokens else tokens
    if "const" in head and ("*" not in head or head.index("const") > head.index("*")):
        return set()
    result = set()
    for part in " ".join(tokens).split(","):
        lhs = part.split("=")[0].split("[")[0].strip().split()
        ids = [t for t in lhs if _ID.match(t)]
        if ids:
            result.add(ids[-1])
    return result


def analyze_sources(sources: dict[str, str], entry_point: str = "run") -> list[Finding]:
    """Screen supplied source files from a direct host entry symbol.

    Raises AnalysisError for unmatched delimiters or an unresolved entry point.
    Findings describe suspicious control flow, not a misconduct classification.
    No submitted code is compiled or executed.
    """
    tokens, locations = [], []
    for path, source in sources.items():
        clean = _NO_CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), source)
        line, previous = 1, 0
        for match in _TOKEN.finditer(clean):
            line += clean.count("\n", previous, match.start())
            previous = match.start()
            tokens.append(match.group())
            locations.append((path, line))
    pair, stack = {}, []
    for i, t in enumerate(tokens):
        if t in {"(", "[", "{"}:
            stack.append(i)
        elif t in {")", "]", "}"}:
            if not stack or tokens[stack[-1]] != {")": "(", "]": "[", "}": "{"}[t]:
                raise AnalysisError(f"{locations[i]}: unbalanced delimiter {t}")
            j = stack.pop()
            pair[i], pair[j] = j, i
    if stack:
        raise AnalysisError(f"{locations[stack[-1]]}: unclosed delimiter")

    functions, bodies = {}, []
    for i, t in enumerate(tokens):
        if t != "{" or i == 0 or tokens[i - 1] != ")":
            continue
        paren = pair[i - 1]
        name = tokens[paren - 1] if paren else ""
        if not _ID.match(name) or name in _CONTROL:
            continue
        start = paren - 1
        while start and tokens[start - 1] not in {";", "{", "}"}:
            start -= 1
        device = bool({"__global__", "__device__"} & set(tokens[start:paren]))
        bodies.append((start, pair[i]))
        if not device:
            functions.setdefault(name, []).append((i + 1, pair[i]))
    if entry_point not in functions:
        raise AnalysisError(f"Cannot resolve host entry point {entry_point!r}")

    # Find namespace/file mutable variables, excluding function and struct bodies.
    hidden = set()
    for lo, hi in bodies:
        hidden.update(range(lo, hi + 1))
    for i, t in enumerate(tokens):
        if t in {"struct", "class", "enum"}:
            j = i + 1
            while j < len(tokens) and tokens[j] not in {"{", ";"}:
                j += 1
            if j < len(tokens) and tokens[j] == "{":
                hidden.update(range(i, pair[j] + 1))
    globals_ = set()
    start = 0
    for i, t in enumerate(tokens):
        if i not in hidden and t == ";":
            decl = tokens[start:i]
            if "(" not in decl and len(decl) >= 2:
                globals_.update(_declared(decl))
        if t in {";", "{", "}"}:
            start = i + 1

    def statement_end(i):
        if i >= len(tokens):
            raise AnalysisError("Incomplete statement")
        if tokens[i] == "{":
            return pair[i] + 1
        if tokens[i] == "if" and tokens[i + 1] == "(":
            end = statement_end(pair[i + 1] + 1)
            return statement_end(end + 1) if tokens[end : end + 1] == ["else"] else end
        while i < len(tokens) and tokens[i] != ";":
            i = pair[i] + 1 if tokens[i] in {"(", "[", "{"} else i + 1
        return i + 1

    def active_body(lo, hi):
        # Ignore explicitly dead if(false)/if(0) branches, including false && X.
        active = set(range(lo, hi))
        for i in range(lo, hi - 2):
            if tokens[i : i + 2] == ["if", "("]:
                end = pair[i + 1]
                cond = tokens[i + 2 : end]
                if cond in (["false"], ["0"]) or cond[:2] in (["false", "&&"], ["0", "&&"]):
                    active.difference_update(range(end + 1, statement_end(end + 1)))
        return active

    active = {
        name: set().union(*(active_body(lo, hi) for lo, hi in ranges))
        for name, ranges in functions.items()
    }
    calls = {
        name: {tokens[i] for i in inds if tokens[i] in functions and tokens[i + 1 : i + 2] == ["("]}
        for name, inds in active.items()
    }
    reachable, pending = set(), [entry_point]
    while pending:
        name = pending.pop()
        if name not in reachable:
            reachable.add(name)
            pending.extend(calls[name])
    # A direct helper that launches a kernel is a compute effect too.
    compute = {
        name
        for name, inds in active.items()
        if any(tokens[i] == "<<<" or _COMPUTE_API.match(tokens[i]) for i in inds)
    }
    while True:
        expanded = compute | {name for name in functions if calls[name] & compute}
        if expanded == compute:
            break
        compute = expanded

    findings = []
    for name in sorted(reachable):
        inds = active[name]
        persistent = set(globals_)
        for i in sorted(inds):
            if tokens[i] in {"static", "thread_local"}:
                end = statement_end(i)
                decl = tokens[i + 1 : end - 1]
                if "(" not in decl:  # not a static function declaration
                    persistent.update(_declared(decl))
        # Simple identifier/member assignments propagate dependence into guards
        # such as `bool hit = object.valid && object.ptr == input`.
        deps = {p: {p} for p in persistent}
        assignments = []
        for i in sorted(inds):
            if tokens[i] != "=":
                continue
            j = i - 1
            if not _ID.match(tokens[j]):
                continue
            while j >= 2 and tokens[j - 1] in {".", "->"}:
                j -= 2
            end = i + 1
            while end in inds and tokens[end] not in {";", ","}:
                end += 1
            assignments.append((tokens[j], _names(tokens[i + 1 : end])))
        while True:
            changed = False
            for lhs, rhs in assignments:
                roots = set().union(*(deps.get(n, set()) for n in rhs))
                old = deps.setdefault(lhs, set())
                if roots - old:
                    old.update(roots)
                    changed = True
            if not changed:
                break
        for i in sorted(inds):
            if tokens[i] not in {"if", "switch", "while", "for"} or tokens[i + 1] != "(":
                continue
            end = pair[i + 1]
            roots = set().union(*(deps.get(n, set()) for n in _names(tokens[i + 2 : end])))
            if not roots:
                continue
            stop = statement_end(end + 1)
            if tokens[stop : stop + 1] == ["else"]:
                stop = statement_end(stop + 1)
            branch = [tokens[j] for j in range(end + 1, stop) if j in inds]
            effects = set(branch)
            launch = (
                "<<<" in effects
                or bool(effects & compute)
                or any(_COMPUTE_API.match(t) for t in effects)
            )
            if "return" in effects or launch:
                path, line = locations[i]
                rule = (
                    "persistent_state_guards_return"
                    if "return" in effects
                    else "persistent_state_guards_compute"
                )
                findings.append(Finding(path, line, rule, tuple(sorted(roots))))
    return findings
