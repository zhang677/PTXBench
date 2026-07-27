"""Filter a reasoning_pairs.jsonl, dropping records likely to destabilize SFT.

Default filters (each can be turned off with --keep-*):
  1. drop hidden-thinking leak: record's `reasoning` contains `</think>`,
     `<think>`, `<my_reasoning>`, or `</my_reasoning>` - the Kimi K2.6
     post-processor sometimes fails to split visible reasoning from hidden CoT,
     leaving stray tags that become a malformed `<think>...</think>` target
     after Qwen wrapping (causes loss spikes around the premature `</think>`).
  2. drop chat-template / EOS-like special tokens that would break tokenization.
  3. drop records with `passed=False` (kernel failed compile/correctness, the
     reasoning argues confidently for a broken kernel).
  4. drop records where reasoning is shorter than --min-reasoning-chars.

Optional filters (off by default):
  --drop-run-ids: drop records whose metadata.run_id is in the provided list.
  --drop-chinese: drop records whose reasoning contains CJK characters.
  --drop-no-punct-ending: drop records whose reasoning doesn't end in
                          standard punctuation (truncation indicator).

Writes the cleaned JSONL to --output and a JSON drop-report to --report.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CHINESE_RE = re.compile(r"[一-鿿]")
LEAK_TOKENS = ("</think>", "<think>", "<my_reasoning>", "</my_reasoning>")
SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<unk>",
)
ENDING_PUNCT = {".", "!", "?", '"', "'", ")", "]", "`"}


def has_any(text: str, needles: tuple[str, ...]) -> str | None:
    for n in needles:
        if n in text:
            return n
    return None


def classify(record: dict, args: argparse.Namespace) -> tuple[bool, str | None]:
    """Return (keep, drop_reason). drop_reason is None when keep is True."""
    reasoning = record.get("reasoning") or ""
    metadata = record.get("metadata") or {}

    if not reasoning.strip():
        return False, "empty_reasoning"

    if args.drop_run_ids:
        run_id = metadata.get("run_id") or record.get("run_id")
        if run_id is not None and str(run_id) in args.drop_run_ids:
            return False, f"run_id:{run_id}"

    if not args.keep_leak:
        hit = has_any(reasoning, LEAK_TOKENS)
        if hit is not None:
            return False, f"hidden_thinking_leak:{hit}"

    if not args.keep_special_tokens:
        hit = has_any(reasoning, SPECIAL_TOKENS)
        if hit is not None:
            return False, f"special_token:{hit}"

    if not args.keep_failed:
        passed = metadata.get("passed")
        speedup = metadata.get("speedup")
        if passed is False or speedup is None:
            return False, "failed_kernel"

    if len(reasoning) < args.min_reasoning_chars:
        return False, f"too_short<{args.min_reasoning_chars}"

    if args.drop_chinese and CHINESE_RE.search(reasoning):
        return False, "chinese_leak"

    if args.drop_no_punct_ending:
        stripped = reasoning.rstrip()
        if stripped and stripped[-1] not in ENDING_PUNCT:
            return False, "no_punct_ending"

    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input reasoning_pairs.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output reasoning_pairs.jsonl (cleaned)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write a JSON drop report (defaults to <output>.report.json)",
    )
    parser.add_argument(
        "--min-reasoning-chars",
        type=int,
        default=200,
        help="drop records with reasoning shorter than this (default: 200)",
    )
    parser.add_argument("--keep-leak", action="store_true", help="keep hidden-thinking-leak records")
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="keep records whose reasoning contains chat-template / EOS tokens",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="keep records where the corresponding kernel failed (passed=False or speedup=None)",
    )
    parser.add_argument(
        "--drop-chinese",
        action="store_true",
        help="also drop records whose reasoning contains CJK characters (Kimi language leakage)",
    )
    parser.add_argument(
        "--drop-run-ids",
        nargs="+",
        default=(),
        help="drop records whose metadata.run_id matches any of these run IDs",
    )
    parser.add_argument(
        "--drop-no-punct-ending",
        action="store_true",
        help="also drop records whose reasoning doesn't end with standard punctuation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report only, don't write outputs",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input not found: {args.input}")

    args.drop_run_ids = set(args.drop_run_ids)
    report = args.report or args.output.with_suffix(args.output.suffix + ".report.json")

    kept_rows: list[dict] = []
    drop_rows: list[dict] = []
    drop_reasons: dict[str, int] = {}

    with args.input.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            keep, reason = classify(record, args)
            if keep:
                kept_rows.append(record)
            else:
                assert reason is not None
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                meta = record.get("metadata") or {}
                drop_rows.append(
                    {
                        "line": line_no,
                        "reason": reason,
                        "run_id": meta.get("run_id") or record.get("run_id"),
                        "exp_id": meta.get("exp_id"),
                        "turn": meta.get("turn"),
                        "definition_name": meta.get("definition_name"),
                        "passed": meta.get("passed"),
                        "speedup": meta.get("speedup"),
                        "reasoning_chars": len(record.get("reasoning") or ""),
                    }
                )

    total = len(kept_rows) + len(drop_rows)
    print(f"input rows : {total}")
    print(f"kept rows  : {len(kept_rows)}")
    print(f"dropped    : {len(drop_rows)}")
    if drop_reasons:
        print("drop breakdown:")
        for reason, count in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")

    if args.dry_run:
        print("dry-run: not writing outputs")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for row in kept_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w") as out:
        json.dump(
            {
                "input_path": str(args.input),
                "output_path": str(args.output),
                "filters": {
                    "min_reasoning_chars": args.min_reasoning_chars,
                    "keep_leak": args.keep_leak,
                    "keep_special_tokens": args.keep_special_tokens,
                    "keep_failed": args.keep_failed,
                    "drop_chinese": args.drop_chinese,
                    "drop_run_ids": sorted(args.drop_run_ids),
                    "drop_no_punct_ending": args.drop_no_punct_ending,
                },
                "summary": {
                    "input_rows": total,
                    "kept_rows": len(kept_rows),
                    "dropped_rows": len(drop_rows),
                    "drop_reason_counts": drop_reasons,
                },
                "dropped_records": drop_rows,
            },
            out,
            indent=2,
        )
        out.write("\n")
    print(f"wrote cleaned jsonl: {args.output}")
    print(f"wrote drop report : {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
