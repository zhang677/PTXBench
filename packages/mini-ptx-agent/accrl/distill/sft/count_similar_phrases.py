#!/usr/bin/env python3
"""Count typo-tolerant phrase matches in SFT parquet message text."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+")
DEFAULT_METADATA_KEYS = ["run_id", "exp_id", "definition_name", "turn", "source_label"]


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Match:
    row_idx: int
    message_idx: int
    role: str
    phrase: str
    matched_text: str
    token_distances: tuple[int, ...]
    start: int
    end: int
    snippet: str


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("pyarrow is required for parquet inspection; run in the Miles image") from e
    return pq.read_table(path).to_pylist()


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0).lower(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def edit_distance(a: str, b: str, *, max_distance: int) -> int:
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def phrase_tokens(phrase: str) -> list[str]:
    tokens = [token.text for token in tokenize(phrase)]
    if not tokens:
        raise SystemExit(f"phrase has no word tokens: {phrase!r}")
    return tokens


def make_snippet(text: str, start: int, end: int, *, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    snippet = text[left:right].replace("\n", "\\n")
    if left:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def find_phrase_matches(
    text: str,
    *,
    phrase: str,
    phrase_words: list[str],
    max_token_distance: int,
    context_chars: int,
) -> list[tuple[str, tuple[int, ...], int, int, str]]:
    tokens = tokenize(text)
    width = len(phrase_words)
    matches = []
    distance_cache: dict[tuple[str, str], int] = {}
    for i in range(0, len(tokens) - width + 1):
        window = tokens[i : i + width]
        distances_list = []
        matched = True
        for expected, actual in zip(phrase_words, window, strict=True):
            cache_key = (expected, actual.text)
            distance = distance_cache.get(cache_key)
            if distance is None:
                distance = edit_distance(expected, actual.text, max_distance=max_token_distance)
                distance_cache[cache_key] = distance
            if distance > max_token_distance:
                matched = False
                break
            distances_list.append(distance)
        if not matched:
            continue
        distances = tuple(distances_list)
        start = window[0].start
        end = window[-1].end
        matched_text = text[start:end]
        matches.append(
            (
                matched_text,
                distances,
                start,
                end,
                make_snippet(text, start, end, context_chars=context_chars),
            )
        )
    return matches


def selected_messages(row: dict[str, Any], roles: set[str] | None) -> list[tuple[int, dict[str, Any]]]:
    messages = row.get("messages") or []
    out = []
    for idx, message in enumerate(messages):
        role = str(message.get("role") or "")
        if roles is None or role in roles:
            out.append((idx, message))
    return out


def collect_matches(
    rows: list[dict[str, Any]],
    *,
    phrases: list[str],
    roles: set[str] | None,
    max_token_distance: int,
    context_chars: int,
) -> list[Match]:
    phrase_lookup = [(phrase, phrase_tokens(phrase)) for phrase in phrases]
    matches: list[Match] = []
    for row_idx, row in enumerate(rows):
        for message_idx, message in selected_messages(row, roles):
            role = str(message.get("role") or "")
            text = str(message.get("content") or "")
            for phrase, words in phrase_lookup:
                for matched_text, distances, start, end, snippet in find_phrase_matches(
                    text,
                    phrase=phrase,
                    phrase_words=words,
                    max_token_distance=max_token_distance,
                    context_chars=context_chars,
                ):
                    matches.append(
                        Match(
                            row_idx=row_idx,
                            message_idx=message_idx,
                            role=role,
                            phrase=phrase,
                            matched_text=matched_text,
                            token_distances=distances,
                            start=start,
                            end=end,
                            snippet=snippet,
                        )
                    )
    return matches


def row_metadata(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {key: metadata.get(key) for key in keys if key in metadata}


def print_text_report(
    rows: list[dict[str, Any]],
    matches: list[Match],
    *,
    metadata_keys: list[str],
    max_examples: int,
) -> None:
    row_hit_indices = {m.row_idx for m in matches}
    message_hits = {(m.row_idx, m.message_idx) for m in matches}
    print(f"Rows: {len(rows):,}")
    print(f"Total matches: {len(matches):,}")
    print(f"Rows with matches: {len(row_hit_indices):,}")
    print(f"Messages with matches: {len(message_hits):,}")

    by_phrase: dict[str, int] = {}
    by_variant: dict[str, int] = {}
    for match in matches:
        by_phrase[match.phrase] = by_phrase.get(match.phrase, 0) + 1
        by_variant[match.matched_text.lower()] = by_variant.get(match.matched_text.lower(), 0) + 1
    if by_phrase:
        print("\nBy phrase")
        for phrase, count in sorted(by_phrase.items(), key=lambda item: (-item[1], item[0])):
            print(f"{phrase!r}: {count}")
    if by_variant:
        print("\nMatched variants")
        for variant, count in sorted(by_variant.items(), key=lambda item: (-item[1], item[0])):
            print(f"{variant!r}: {count}")

    if not matches:
        return

    per_row: dict[int, int] = {}
    for match in matches:
        per_row[match.row_idx] = per_row.get(match.row_idx, 0) + 1
    print("\nTop rows")
    for row_idx, count in sorted(per_row.items(), key=lambda item: (-item[1], item[0]))[:max_examples]:
        row = rows[row_idx]
        print(f"row={row_idx} matches={count} id={row.get('id')}")
        metadata = row_metadata(row, metadata_keys)
        if metadata:
            print("  metadata=" + json.dumps(metadata, sort_keys=True))

    print("\nExamples")
    for match in matches[:max_examples]:
        row = rows[match.row_idx]
        print(
            f"row={match.row_idx} msg={match.message_idx} role={match.role} "
            f"phrase={match.phrase!r} matched={match.matched_text!r} "
            f"distances={list(match.token_distances)} id={row.get('id')}"
        )
        print(f"  {match.snippet}")


def print_json_report(
    rows: list[dict[str, Any]],
    matches: list[Match],
    *,
    metadata_keys: list[str],
) -> None:
    payload = {
        "num_rows": len(rows),
        "total_matches": len(matches),
        "rows_with_matches": len({m.row_idx for m in matches}),
        "messages_with_matches": len({(m.row_idx, m.message_idx) for m in matches}),
        "matches": [
            {
                "row_idx": match.row_idx,
                "message_idx": match.message_idx,
                "role": match.role,
                "id": rows[match.row_idx].get("id"),
                "metadata": row_metadata(rows[match.row_idx], metadata_keys),
                "phrase": match.phrase,
                "matched_text": match.matched_text,
                "token_distances": list(match.token_distances),
                "start": match.start,
                "end": match.end,
                "snippet": match.snippet,
            }
            for match in matches
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_roles(raw_roles: str) -> set[str] | None:
    if raw_roles == "all":
        return None
    roles = {role.strip() for role in raw_roles.split(",") if role.strip()}
    if not roles:
        raise SystemExit("--roles must be 'all' or a comma-separated role list")
    return roles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Input .parquet file")
    parser.add_argument(
        "--phrase",
        action="append",
        default=None,
        help="Phrase to count fuzzily; repeat to count multiple phrases",
    )
    parser.add_argument(
        "--roles",
        default="assistant",
        help="Comma-separated message roles to search, or 'all' (default: assistant)",
    )
    parser.add_argument(
        "--max-token-distance",
        type=int,
        default=2,
        help=(
            "Maximum edit distance allowed per token "
            "(default catches tranposes/transposes/transpose/hardware-transpose; use 0 for exact tokens)"
        ),
    )
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument(
        "--metadata-key",
        action="append",
        default=None,
        help="Metadata key to include in reports; repeat to customize",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    if not args.parquet.is_file():
        raise SystemExit(f"missing parquet file: {args.parquet}")
    if args.max_token_distance < 0:
        raise SystemExit("--max-token-distance must be non-negative")
    if args.context_chars < 0:
        raise SystemExit("--context-chars must be non-negative")
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")

    phrases = args.phrase or ["hardware tranposes"]
    rows = load_rows(args.parquet)
    matches = collect_matches(
        rows,
        phrases=phrases,
        roles=parse_roles(args.roles),
        max_token_distance=args.max_token_distance,
        context_chars=args.context_chars,
    )
    metadata_keys = args.metadata_key or DEFAULT_METADATA_KEYS
    if args.json:
        print_json_report(rows, matches, metadata_keys=metadata_keys)
    else:
        print_text_report(rows, matches, metadata_keys=metadata_keys, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
