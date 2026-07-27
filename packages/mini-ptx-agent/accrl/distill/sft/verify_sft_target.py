#!/usr/bin/env python3
"""Verify Qwen thinking-channel SFT targets before launching training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


THINK_ONLY_FORMAT = "qwen_think_wrapped"
THINK_PLUS_VISIBLE_FORMAT = "qwen_think_plus_gemini_response"
ASSISTANT_THINK_PLUS_VISIBLE_FORMAT = "qwen_think_plus_assistant_response"
THINK_PLUS_VISIBLE_FORMATS = {
    THINK_PLUS_VISIBLE_FORMAT,
    ASSISTANT_THINK_PLUS_VISIBLE_FORMAT,
}
ALLOWED_TARGET_FORMATS = {None, THINK_ONLY_FORMAT, *THINK_PLUS_VISIBLE_FORMATS}


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit("pyarrow is required for parquet input; run this in the Miles image") from e
        return pq.read_table(path).to_pylist()

    raise SystemExit(f"unsupported input format: {path}")


def token_label(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    decoded = tokenizer.decode([token_id])
    return f"id={token_id} token={token!r} text={decoded!r}"


def token_preview(
    tokenizer: Any,
    token_ids: list[int],
    loss_mask: list[int],
    *,
    center: int,
    radius: int,
) -> list[dict[str, Any]]:
    start = max(0, center - radius)
    end = min(len(token_ids), center + radius + 1)
    special_ids = set(tokenizer.all_special_ids)
    return [
        {
            "idx": idx,
            "id": token_ids[idx],
            "token": tokenizer.convert_ids_to_tokens([token_ids[idx]])[0],
            "text": tokenizer.decode([token_ids[idx]]),
            "loss_mask": loss_mask[idx],
            "special": token_ids[idx] in special_ids,
        }
        for idx in range(start, end)
    ]


def assert_true(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def find_thinking_span(
    token_ids: list[int],
    loss_mask: list[int],
    response_length: int,
    tokenizer: Any,
) -> tuple[int, int, bool]:
    """Count supervised thinking and visible tokens in the response segment."""
    response_start = max(0, len(token_ids) - response_length)
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")

    supervised_indices = [
        idx for idx in range(response_start, len(token_ids)) if loss_mask[idx] == 1
    ]
    if not supervised_indices:
        return 0, 0, False

    end_positions = [idx for idx in supervised_indices if token_ids[idx] == end_think_id]
    if not end_positions:
        return 0, sum(1 for idx in supervised_indices if idx >= response_start), False

    end_pos = end_positions[0]
    thinking_tokens = sum(1 for idx in supervised_indices if idx <= end_pos)
    visible_tokens = sum(1 for idx in supervised_indices if idx > end_pos)
    return thinking_tokens, visible_tokens, True


def verify_row(
    row: dict[str, Any],
    *,
    row_idx: int,
    tokenizer: Any,
    masker: Any,
    preview_radius: int,
    require_visible_code_fence: bool,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    messages = row.get("messages")
    metadata = row.get("metadata") or {}

    assert_true(isinstance(messages, list) and len(messages) >= 3, "messages must have system/user/assistant", errors)
    if errors:
        return errors, {}

    assistant = messages[-1]
    assistant_content = assistant.get("content", "")
    target_format = metadata.get("target_format")
    if target_format is None:
        target_format = THINK_ONLY_FORMAT
    expect_visible_after_think = target_format in THINK_PLUS_VISIBLE_FORMATS

    assert_true(assistant.get("role") == "assistant", "last message must be assistant", errors)
    assert_true(assistant_content.startswith("<think>\n"), "assistant content must start with '<think>\\n'", errors)
    assert_true(assistant.get("step_loss_mask", 1) == 1, "final assistant must be supervised", errors)
    for msg_idx, message in enumerate(messages[:-1]):
        if message.get("role") == "assistant":
            assert_true(
                message.get("step_loss_mask") == 0,
                f"prior assistant message {msg_idx} must have step_loss_mask=0",
                errors,
            )
    assert_true("<my_reasoning>" not in assistant_content, "assistant content still contains <my_reasoning>", errors)
    assert_true("</my_reasoning>" not in assistant_content, "assistant content still contains </my_reasoning>", errors)
    assert_true(metadata.get("target_source_field") in (None, "reasoning"), "target_source_field should be reasoning", errors)
    assert_true(
        metadata.get("target_format") in ALLOWED_TARGET_FORMATS,
        f"target_format should be one of {sorted(str(x) for x in ALLOWED_TARGET_FORMATS)}",
        errors,
    )
    assert_true(assistant_content.count("<think>") == 1, "assistant content should contain exactly one <think>", errors)
    assert_true(assistant_content.count("</think>") == 1, "assistant content should contain exactly one </think>", errors)
    if "</think>" in assistant_content:
        visible_text = assistant_content.split("</think>", 1)[1]
        if expect_visible_after_think:
            assert_true(visible_text.strip() != "", "visible target after </think> must be non-empty", errors)
            if require_visible_code_fence:
                assert_true("```" in visible_text, "visible target after </think> should contain a code fence", errors)
        else:
            assert_true(
                assistant_content.endswith("\n</think>"),
                "think-only target must end with '\\n</think>'",
                errors,
            )

    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    assert_true(prompt_text.endswith("<think>\n"), "generation prompt should end with '<think>\\n'", errors)

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)
    assert_true(rendered.count("<think>") >= 1, "rendered chat should contain <think>", errors)
    assert_true(rendered.count("</think>") >= 1, "rendered chat should contain </think>", errors)

    tools = metadata.get("tools")
    token_ids, loss_mask = masker.get_loss_mask(messages, tools=tools)
    assert_true(len(token_ids) == len(loss_mask), "token_ids and loss_mask length mismatch", errors)
    assert_true(1 in loss_mask, "loss_mask has no supervised tokens", errors)
    if errors:
        return errors, {}

    first_loss = loss_mask.index(1)
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    think_positions = [idx for idx, token_id in enumerate(token_ids) if token_id == think_id]
    end_positions = [idx for idx, token_id in enumerate(token_ids) if token_id == end_think_id]
    masked_end_positions = [idx for idx in end_positions if loss_mask[idx] == 1]
    assert_true(think_positions, "expected at least one <think> token", errors)
    assert_true(end_positions, "expected at least one </think> token", errors)
    assert_true(
        len(masked_end_positions) == 1,
        f"expected one supervised </think> token, found {len(masked_end_positions)}",
        errors,
    )
    if errors:
        return errors, {}

    end_pos = masked_end_positions[0]
    think_pos = max(idx for idx in think_positions if idx < end_pos)
    assert_true(loss_mask[think_pos] == 0, "<think> token should not be supervised", errors)
    assert_true(think_pos < first_loss < end_pos, "first supervised token should be between <think> and </think>", errors)
    assert_true(loss_mask[end_pos] == 1, "</think> token should be supervised", errors)
    assert_true(all(mask == 1 for mask in loss_mask[first_loss : end_pos + 1]), "target span should be fully supervised", errors)

    response_length = masker.get_response_lengths([loss_mask])[0]
    thinking_tokens, visible_tokens, found_end = find_thinking_span(
        token_ids, loss_mask, response_length, tokenizer
    )
    assert_true(found_end, "_find_thinking_span did not find </think>", errors)
    assert_true(thinking_tokens > 0, "thinking token count must be positive", errors)
    if expect_visible_after_think:
        assert_true(visible_tokens > 0, "think+visible target should have visible_after_think tokens", errors)
    else:
        assert_true(visible_tokens == 0, "think-only target should have zero visible_after_think tokens", errors)

    preview = {
        "row_idx": row_idx,
        "id": row.get("id"),
        "target_source_field": metadata.get("target_source_field"),
        "target_format": target_format,
        "prompt_tail": prompt_text[-120:],
        "first_loss": token_label(tokenizer, token_ids[first_loss]),
        "think_token": token_label(tokenizer, token_ids[think_pos]),
        "end_think_token": token_label(tokenizer, token_ids[end_pos]),
        "total_tokens": len(token_ids),
        "response_length": response_length,
        "thinking_tokens": thinking_tokens,
        "visible_after_think_tokens": visible_tokens,
        "visible_after_think_preview": assistant_content.split("</think>", 1)[1].strip()[:240]
        if "</think>" in assistant_content
        else "",
        "boundary_preview": token_preview(
            tokenizer,
            token_ids,
            loss_mask,
            center=first_loss,
            radius=preview_radius,
        ),
        "end_preview": token_preview(
            tokenizer,
            token_ids,
            loss_mask,
            center=end_pos,
            radius=preview_radius,
        ),
    }
    return errors, preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="SFT parquet/jsonl")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--miles-src", type=Path, default=Path(__file__).resolve().parent / "miles")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--check-all", action="store_true", help="Verify every row instead of sampling")
    parser.add_argument("--max-previews", type=int, default=8, help="Maximum row previews to print")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-radius", type=int, default=8)
    parser.add_argument(
        "--no-require-visible-code-fence",
        action="store_true",
        help="Do not require v2 visible targets to contain a markdown code fence.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.miles_src))
    from miles.utils.mask_utils import MultiTurnLossMaskGenerator
    from miles.utils.processing_utils import load_tokenizer

    rows = load_rows(args.data)
    if not rows:
        raise SystemExit(f"empty dataset: {args.data}")

    if args.check_all:
        sample_indices = list(range(len(rows)))
    else:
        rng = random.Random(args.seed)
        sample_indices = list(range(min(args.num_samples, len(rows))))
        if len(rows) > args.num_samples:
            sample_indices = sorted(rng.sample(range(len(rows)), args.num_samples))

    tokenizer = load_tokenizer(str(args.tokenizer), trust_remote_code=True)
    masker = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")

    previews = []
    all_errors = []
    for idx in sample_indices:
        errors, preview = verify_row(
            rows[idx],
            row_idx=idx,
            tokenizer=tokenizer,
            masker=masker,
            preview_radius=args.preview_radius,
            require_visible_code_fence=not args.no_require_visible_code_fence,
        )
        if errors:
            all_errors.append({"row_idx": idx, "id": rows[idx].get("id"), "errors": errors})
        if preview and len(previews) < args.max_previews:
            previews.append(preview)

    print(json.dumps({"data": str(args.data), "checked_rows": sample_indices, "previews": previews}, indent=2))
    if all_errors:
        print(json.dumps({"errors": all_errors}, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: verified {len(sample_indices)} rows from {args.data}")


if __name__ == "__main__":
    main()
