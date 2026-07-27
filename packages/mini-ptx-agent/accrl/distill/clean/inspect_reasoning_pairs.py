"""Inspect a reasoning_pairs.jsonl file and report data-quality anomalies.

Usage:
    python -m accrl.distill.clean.inspect_reasoning_pairs <path/to/reasoning_pairs.jsonl>

Reports:
- Top-level field shape and per-field length stats
- Metadata distributions (turns, definitions, passed flag, speedup=None count)
- Hidden-thinking leak (records whose `reasoning` still contains `</think>` or
  `<my_reasoning>` - these survive Kimi K2.6 generation when the post-processor
  fails to split visible answer from hidden CoT)
- Suspicious special tokens (chat template, think tags, EOS-like tokens)
- Non-ASCII / Chinese language leakage
- Naive 50-char-substring repetition detection (catches generation loops)
- Ending punctuation check (catches abrupt truncation)
- Estimated token-count buckets (chars / 3.5) vs common SFT context lengths
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import unicodedata
from collections import Counter
from pathlib import Path


CHINESE_RE = re.compile(r"[一-鿿]")
SUSPICIOUS_TOKENS = [
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<think>",
    "</think>",
    "<thinking>",
    "</thinking>",
    "<my_reasoning>",
    "</my_reasoning>",
    "<unk>",
]
ENDING_PUNCT = {".", "!", "?", '"', "'", ")", "]", "`"}
CHARS_PER_TOKEN_ESTIMATE = 3.5
CONTEXT_BUCKETS = [4096, 8192, 16384, 32768, 65536, 128000]


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def report_shape(records: list[dict]) -> None:
    print("=== SHAPE ===")
    print(f"total records: {len(records)}")
    if not records:
        return
    fields = set()
    for r in records:
        fields.update(r.keys())
    print(f"top-level fields: {sorted(fields)}")
    sample = records[0]
    for k, v in sample.items():
        kind = type(v).__name__
        size = len(v) if hasattr(v, "__len__") else "n/a"
        print(f"  {k}: {kind} (len={size})")


def report_lengths(records: list[dict], fields: list[str]) -> None:
    print("\n=== LENGTH STATS (chars) ===")
    for field in fields:
        if field not in records[0]:
            continue
        lens = [len(r.get(field) or "") for r in records]
        print(
            f"{field}: min={min(lens)}, max={max(lens)}, "
            f"mean={int(st.mean(lens))}, median={int(st.median(lens))}, "
            f"p95={sorted(lens)[int(0.95 * (len(lens) - 1))]}"
        )

    print("\n=== EMPTY / WHITESPACE-ONLY ===")
    for field in fields:
        if field not in records[0]:
            continue
        empties = sum(1 for r in records if not (r.get(field) or "").strip())
        print(f"{field}: {empties} empty/whitespace-only")


def report_metadata(records: list[dict]) -> None:
    print("\n=== METADATA DISTRIBUTIONS ===")
    if not records or "metadata" not in records[0]:
        print("no metadata")
        return
    turn_dist = Counter(r["metadata"].get("turn") for r in records)
    def_dist = Counter(r["metadata"].get("definition_name") for r in records)
    passed_dist = Counter(r["metadata"].get("passed") for r in records)
    speedup_none = sum(1 for r in records if r["metadata"].get("speedup") is None)
    print(f"turn: {sorted(turn_dist.items())}")
    print(f"definition_name (top 10): {def_dist.most_common(10)}")
    print(f"passed: {dict(passed_dist)}")
    print(f"speedup is None: {speedup_none}/{len(records)}")


def report_format_leaks(records: list[dict]) -> None:
    print("\n=== FORMAT LEAK CHECKS (reasoning field) ===")
    starts_with_tag = sum(
        1 for r in records if (r.get("reasoning") or "").lstrip().startswith("<my_reasoning>")
    )
    ends_with_tag = sum(
        1 for r in records if (r.get("reasoning") or "").rstrip().endswith("</my_reasoning>")
    )
    contains_open = sum(1 for r in records if "<my_reasoning>" in (r.get("reasoning") or ""))
    contains_close = sum(1 for r in records if "</my_reasoning>" in (r.get("reasoning") or ""))
    contains_close_think = sum(1 for r in records if "</think>" in (r.get("reasoning") or ""))
    contains_open_think = sum(1 for r in records if "<think>" in (r.get("reasoning") or ""))
    print(f"reasoning starts with <my_reasoning>: {starts_with_tag}/{len(records)}")
    print(f"reasoning ends with </my_reasoning>:  {ends_with_tag}/{len(records)}")
    print(f"reasoning contains <my_reasoning>:    {contains_open}/{len(records)}")
    print(f"reasoning contains </my_reasoning>:   {contains_close}/{len(records)}")
    print(f"reasoning contains <think>:           {contains_open_think}/{len(records)}")
    print(f"reasoning contains </think>:          {contains_close_think}/{len(records)} (HIDDEN-THINKING LEAK)")

    leaked_indices = [
        i for i, r in enumerate(records)
        if "</think>" in (r.get("reasoning") or "")
        or "<my_reasoning>" in (r.get("reasoning") or "")
    ]
    if leaked_indices:
        print(f"\nleaked record indices: {leaked_indices}")
        for i in leaked_indices:
            r = records[i]
            text = r.get("reasoning") or ""
            close_idx = text.find("</think>")
            my_idx = text.find("<my_reasoning>")
            md = r.get("metadata") or {}
            print(
                f"  idx {i}: total_chars={len(text)}, </think>@{close_idx}, "
                f"<my_reasoning>@{my_idx}, metadata={md}"
            )


def report_suspicious_tokens(records: list[dict]) -> None:
    print("\n=== SUSPICIOUS SPECIAL TOKENS ===")
    for tok in SUSPICIOUS_TOKENS:
        in_reasoning = sum(1 for r in records if tok in (r.get("reasoning") or ""))
        in_input = sum(1 for r in records if tok in (r.get("input") or ""))
        if in_reasoning or in_input:
            print(f"  {tok!r}: reasoning={in_reasoning}, input={in_input}")


def report_chinese(records: list[dict]) -> None:
    print("\n=== CHINESE CHARACTER LEAKAGE ===")
    rows = []
    for i, r in enumerate(records):
        cnt = len(CHINESE_RE.findall(r.get("reasoning") or ""))
        if cnt:
            rows.append((i, cnt))
    print(f"records with Chinese in reasoning: {len(rows)}/{len(records)}")
    rows.sort(key=lambda x: -x[1])
    for i, c in rows[:10]:
        text = records[i]["reasoning"]
        m = CHINESE_RE.search(text)
        snippet = text[max(0, m.start() - 60) : m.end() + 80] if m else ""
        print(f"  idx {i} ({c} CJK chars): ...{snippet.strip()}...")


def report_unicode_health(records: list[dict]) -> None:
    print("\n=== UNICODE HEALTH (control / surrogate / private-use) ===")
    bad = []
    bad_cats = {"Cc", "Cf", "Co", "Cs", "Cn"}
    for i, r in enumerate(records):
        text = r.get("reasoning") or ""
        for c in text:
            if c in ("\n", "\t", "\r"):
                continue
            if unicodedata.category(c) in bad_cats:
                bad.append((i, c, ord(c)))
                break
    print(f"records with control/format/private-use chars: {len(bad)}")
    for i, c, o in bad[:5]:
        print(f"  idx {i}: U+{o:04X} ({unicodedata.category(c)})")


def report_repetition(records: list[dict], window: int = 50) -> None:
    print(f"\n=== REPETITION (max count of any {window}-char substring) ===")
    rows = []
    for i, r in enumerate(records):
        text = r.get("reasoning") or ""
        if len(text) < window:
            rows.append((i, 0))
            continue
        counts: dict[str, int] = {}
        for j in range(0, len(text) - window, 5):
            sub = text[j : j + window]
            counts[sub] = counts.get(sub, 0) + 1
        rows.append((i, max(counts.values()) if counts else 0))
    rows.sort(key=lambda x: -x[1])
    print("top 5 most-repetitive records:")
    for i, c in rows[:5]:
        print(f"  idx {i}: max repetition={c}, len={len(records[i].get('reasoning') or '')}")


def report_truncation(records: list[dict]) -> None:
    print("\n=== TRUNCATION HINTS ===")
    no_punct = []
    unbalanced_fences = []
    for i, r in enumerate(records):
        text = (r.get("reasoning") or "").rstrip()
        if not text:
            continue
        if text[-1] not in ENDING_PUNCT:
            no_punct.append((i, text[-60:]))
        if text.count("```") % 2 == 1:
            unbalanced_fences.append((i, text[-80:]))
    print(f"records not ending with punctuation: {len(no_punct)}")
    for i, end in no_punct[:5]:
        print(f"  idx {i}: ...{end!r}")
    print(f"records with unbalanced ``` fences: {len(unbalanced_fences)}")
    for i, end in unbalanced_fences[:5]:
        print(f"  idx {i}: ...{end!r}")


def report_token_estimates(records: list[dict]) -> None:
    print(f"\n=== ESTIMATED TOKENS (chars / {CHARS_PER_TOKEN_ESTIMATE}) ===")
    totals = []
    for r in records:
        total_chars = (
            len(r.get("system_prompt") or "")
            + len(r.get("input") or "")
            + len(r.get("reasoning") or "")
        )
        totals.append(int(total_chars / CHARS_PER_TOKEN_ESTIMATE))
    totals.sort()
    print(
        f"total tokens (sys+input+reasoning): "
        f"min={totals[0]}, max={totals[-1]}, "
        f"mean={int(st.mean(totals))}, "
        f"p95={totals[int(0.95 * (len(totals) - 1))]}"
    )
    for ctx in CONTEXT_BUCKETS:
        over = sum(1 for t in totals if t > ctx)
        print(f"  > {ctx}: {over}/{len(records)} ({100 * over / len(records):.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="reasoning_pairs.jsonl path")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 1

    records = load_records(args.path)
    if not records:
        print("no records loaded", file=sys.stderr)
        return 1

    report_shape(records)
    report_lengths(records, ["system_prompt", "input", "reasoning", "thinking"])
    report_metadata(records)
    report_format_leaks(records)
    report_suspicious_tokens(records)
    report_chinese(records)
    report_unicode_health(records)
    report_repetition(records)
    report_truncation(records)
    report_token_estimates(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
