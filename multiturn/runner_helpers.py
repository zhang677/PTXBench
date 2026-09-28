"""Small source helpers needed by the bundled multi-turn runner."""

from source_utils import extract_code_block


def strip_comments(src: str) -> str:
    """Strip C/C++ comments while preserving strings and line breaks."""
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
