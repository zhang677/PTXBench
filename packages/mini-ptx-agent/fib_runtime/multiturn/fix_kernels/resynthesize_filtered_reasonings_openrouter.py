#!/usr/bin/env python3
"""Repair fix-it reasoning records rejected by the SFT parquet builder.

The input and output use the JSONL record shape emitted by
``synthesize_pair_reasoning_openrouter.py``. Only records whose composed SFT
chat contains a forbidden reasoning delimiter or exceeds the configured token
limit are sent to the LLM. The output preserves every input row in source order,
replacing only rows that were successfully resynthesized.

Each filtered row receives one API call. Successful repairs are appended to a
repairs-only checkpoint JSONL so an interrupted run can resume without repeating
completed calls. Replacement records store the actual resynthesis system and
user prompts, rewritten reasoning, hidden thinking, and provenance metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
import yaml


ACCRL_ROOT = Path(__file__).resolve().parents[3]
if str(ACCRL_ROOT) not in sys.path:
    sys.path.insert(0, str(ACCRL_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from accrl.distill.sft.build_sft_dataset_fixit import (  # noqa: E402
    FORBIDDEN_PARQUET_TOKENS,
    build_row,
    export_messages_from_chat_template,
    load_kernel_pairs_csv,
    load_tokenizer,
    path_key,
    record_correct_kernel_path,
    token_count,
)
from synthesize_pair_reasoning_openrouter import (  # noqa: E402
    extract_reasoning,
    model_extra_body,
)


LOGGER = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """\
You are a precise editor of CUDA-kernel repair reasoning used as supervised training data.
Rewrite the supplied reasoning while preserving its concrete technical substance, causal analysis, calculations, and implementation plan.
The rewritten text must stand alone as the expert's first-person reasoning before writing the corrected kernel.
Never output the literal strings <my_reasoning> or </my_reasoning>.
Return only the rewritten reasoning text, with no wrapper, preface, summary, or markdown fence around the whole response."""


@dataclass(frozen=True)
class ValidationResult:
    row_id: str
    issues: tuple[str, ...]
    total_tokens: int
    fixed_tokens: int
    reasoning_tokens: int

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class RepairResult:
    correct_kernel_path: str
    record: dict[str, Any] | None
    error: str | None = None


@dataclass(frozen=True)
class TargetReasoningLimits:
    min_tokens: int
    max_tokens: int
    min_chars: int
    max_chars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_yaml",
        type=Path,
        help=(
            "Synthesis YAML containing pairs_csv, output_jsonl, reasoning_model, and request "
            "settings. An optional resynthesis mapping can override repair-specific settings."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override resynthesis.overwrite and replace an existing valid output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and report failing records without making API calls or writing output.",
    )
    cli = parser.parse_args()
    args = load_config(cli.config_yaml)
    args.dry_run = bool(cli.dry_run or args.dry_run)
    args.overwrite = bool(cli.overwrite or args.overwrite)

    positive = {
        "--max-sequence-tokens": args.max_sequence_tokens,
        "--request-max-tokens": args.request_max_tokens,
        "--max-concurrent": args.max_concurrent,
        "--timeout": args.timeout,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.safety_tokens < 0:
        parser.error("--safety-tokens must be non-negative")
    if not 0 < args.min_target_ratio <= 1:
        parser.error("resynthesis.min_target_ratio must be in (0, 1]")
    if args.input_jsonl.resolve() == args.output_jsonl.resolve():
        parser.error("--input-jsonl and --output-jsonl must be different paths")
    if args.output_jsonl.exists() and not (args.overwrite or args.dry_run):
        parser.error(f"output exists; pass --overwrite to replace it: {args.output_jsonl}")
    return args


def resolve_config_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else base_dir / path


def derived_output_path(input_jsonl: Path, max_sequence_tokens: int) -> Path:
    return input_jsonl.with_name(
        f"{input_jsonl.stem}-full-{max_sequence_tokens}{input_jsonl.suffix}"
    )


def load_config(path: Path) -> argparse.Namespace:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config YAML not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    repair = raw.get("resynthesis") or {}
    if not isinstance(repair, dict):
        raise ValueError(f"{path}: resynthesis must be a mapping")

    missing = {"pairs_csv", "output_jsonl"}.difference(raw)
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(sorted(missing))}")

    known_repair_keys = {
        "input_jsonl",
        "kernel_pairs_csv",
        "output_jsonl",
        "checkpoint_jsonl",
        "provenance_json",
        "tokenizer",
        "source_label",
        "model",
        "max_sequence_tokens",
        "request_max_tokens",
        "safety_tokens",
        "min_reasoning_chars",
        "min_target_ratio",
        "max_concurrent",
        "temperature",
        "top_p",
        "timeout",
        "overwrite",
        "dry_run",
    }
    unknown = set(repair).difference(known_repair_keys)
    if unknown:
        raise ValueError(
            f"{path}: unknown resynthesis keys: {', '.join(sorted(unknown))}"
        )

    base_dir = path.parent
    input_jsonl = resolve_config_path(
        repair.get("input_jsonl", raw["output_jsonl"]), base_dir=base_dir
    )
    kernel_pairs_csv = resolve_config_path(
        repair.get("kernel_pairs_csv", raw["pairs_csv"]), base_dir=base_dir
    )
    max_sequence_tokens = int(repair.get("max_sequence_tokens", 65536))
    output_jsonl = resolve_config_path(
        repair.get(
            "output_jsonl", derived_output_path(input_jsonl, max_sequence_tokens)
        ),
        base_dir=base_dir,
    )
    checkpoint_jsonl = resolve_config_path(
        repair.get(
            "checkpoint_jsonl",
            output_jsonl.with_name(f"{output_jsonl.stem}.repairs.jsonl"),
        ),
        base_dir=base_dir,
    )
    provenance_json = resolve_config_path(
        repair.get(
            "provenance_json",
            output_jsonl.with_name(f"{output_jsonl.stem}.provenance.json"),
        ),
        base_dir=base_dir,
    )

    return argparse.Namespace(
        config_yaml=path,
        input_jsonl=input_jsonl,
        kernel_pairs_csv=kernel_pairs_csv,
        output_jsonl=output_jsonl,
        checkpoint_jsonl=checkpoint_jsonl,
        provenance_json=provenance_json,
        tokenizer=str(repair.get("tokenizer", "Qwen/Qwen3.6-27B")),
        source_label=str(repair.get("source_label", raw.get("name", "fixit-v5-glm52"))),
        model=str(repair.get("model", raw.get("reasoning_model", "openrouter/z-ai/glm-5.2"))),
        max_sequence_tokens=max_sequence_tokens,
        request_max_tokens=int(
            repair.get("request_max_tokens", raw.get("max_tokens", 131072))
        ),
        safety_tokens=int(repair.get("safety_tokens", 1024)),
        min_reasoning_chars=int(
            repair.get("min_reasoning_chars", raw.get("min_reasoning_chars", 2000))
        ),
        min_target_ratio=float(repair.get("min_target_ratio", 0.95)),
        max_concurrent=int(
            repair.get("max_concurrent", raw.get("max_concurrent", 8))
        ),
        temperature=float(repair.get("temperature", raw.get("temperature", 0.4))),
        top_p=float(repair.get("top_p", raw.get("top_p", 0.95))),
        timeout=float(repair.get("timeout", raw.get("timeout", 600.0))),
        overwrite=bool(repair.get("overwrite", False)),
        dry_run=bool(repair.get("dry_run", False)),
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(record)
    return records


def record_path_key(record: dict[str, Any], *, base_dir: Path) -> str:
    return path_key(record_correct_kernel_path(record), base_dir=base_dir)


def index_records(
    records: list[dict[str, Any]], *, base_dir: Path, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_path_key(record, base_dir=base_dir)
        if key in indexed:
            raise ValueError(f"{label}: duplicate correct_kernel_path: {key}")
        indexed[key] = record
    return indexed


def index_checkpoint_records(
    records: list[dict[str, Any]], *, base_dir: Path, label: str
) -> dict[str, dict[str, Any]]:
    """Index append-only checkpoints, keeping the newest record per pair."""
    indexed: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for record in records:
        key = record_path_key(record, base_dir=base_dir)
        if key in indexed:
            duplicates += 1
        indexed[key] = record
    if duplicates:
        LOGGER.warning("%s: kept the newest value for %d duplicate checkpoint rows", label, duplicates)
    return indexed


def sanitized_reasoning(reasoning: str) -> str:
    cleaned = reasoning
    for token in FORBIDDEN_PARQUET_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def record_with_reasoning(record: dict[str, Any], reasoning: str) -> dict[str, Any]:
    updated = copy.deepcopy(record)
    updated["reasoning"] = reasoning
    return updated


def normalized_row_and_tokens(
    record: dict[str, Any],
    csv_row: dict[str, str],
    *,
    csv_dir: Path,
    source_label: str,
    tokenizer: Any,
) -> tuple[dict[str, Any], int]:
    row = build_row(
        record=record,
        csv_row=csv_row,
        csv_dir=csv_dir,
        source_label=source_label,
    )
    messages, _ = export_messages_from_chat_template(tokenizer, row["messages"])
    tokens = token_count(tokenizer, messages)
    if tokens is None:
        raise ValueError(f"failed to count chat-template tokens for row {row['id']}")
    normalized = dict(row)
    normalized["messages"] = messages
    return normalized, tokens


def validate_record(
    record: dict[str, Any],
    csv_row: dict[str, str],
    *,
    csv_dir: Path,
    source_label: str,
    tokenizer: Any,
    max_sequence_tokens: int,
) -> ValidationResult:
    reasoning = record.get("reasoning") or ""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("empty reasoning field")

    issues: list[str] = []
    present_forbidden = [token for token in FORBIDDEN_PARQUET_TOKENS if token in reasoning]
    if present_forbidden:
        issues.append("forbidden_token")

    measurable_reasoning = sanitized_reasoning(reasoning)
    if not measurable_reasoning:
        measurable_reasoning = "I will repair the kernel."
    measurable_record = record_with_reasoning(record, measurable_reasoning)
    row, total_tokens = normalized_row_and_tokens(
        measurable_record,
        csv_row,
        csv_dir=csv_dir,
        source_label=source_label,
        tokenizer=tokenizer,
    )
    if total_tokens > max_sequence_tokens:
        issues.append("over_max_tokens")

    placeholder_record = record_with_reasoning(record, "x")
    _, fixed_tokens = normalized_row_and_tokens(
        placeholder_record,
        csv_row,
        csv_dir=csv_dir,
        source_label=source_label,
        tokenizer=tokenizer,
    )
    reasoning_tokens = max(1, total_tokens - fixed_tokens)
    return ValidationResult(
        row_id=row["id"],
        issues=tuple(issues),
        total_tokens=total_tokens,
        fixed_tokens=fixed_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def target_reasoning_limits(
    reasoning: str,
    validation: ValidationResult,
    *,
    max_sequence_tokens: int,
    safety_tokens: int,
    min_target_ratio: float,
) -> TargetReasoningLimits:
    if "over_max_tokens" not in validation.issues:
        max_tokens = validation.reasoning_tokens
        max_chars = len(reasoning)
        return TargetReasoningLimits(
            min_tokens=max(1, math.ceil(max_tokens * min_target_ratio)),
            max_tokens=max_tokens,
            min_chars=max(1, math.ceil(max_chars * min_target_ratio)),
            max_chars=max_chars,
        )

    available_tokens = max_sequence_tokens - validation.fixed_tokens - safety_tokens
    if available_tokens <= 0:
        raise ValueError(
            f"fixed prompt/kernel uses {validation.fixed_tokens} tokens, leaving no room "
            f"under max_sequence_tokens={max_sequence_tokens} with safety_tokens={safety_tokens}"
        )
    chars_per_token = len(reasoning) / max(1, validation.reasoning_tokens)
    max_tokens = min(validation.reasoning_tokens, available_tokens)
    estimated_chars = int(max_tokens * chars_per_token)
    max_chars = max(1, min(len(reasoning), estimated_chars))
    return TargetReasoningLimits(
        min_tokens=max(1, math.ceil(max_tokens * min_target_ratio)),
        max_tokens=max_tokens,
        min_chars=max(1, math.ceil(max_chars * min_target_ratio)),
        max_chars=max_chars,
    )


def build_rewrite_prompt(
    reasoning: str,
    validation: ValidationResult,
    *,
    target_limits: TargetReasoningLimits,
) -> str:
    issue_lines: list[str] = []
    if "forbidden_token" in validation.issues:
        issue_lines.append(
            "- Remove every literal <my_reasoning> and </my_reasoning> string from the rewritten text."
        )
    if "over_max_tokens" in validation.issues:
        issue_lines.append(
            "- Shorten the reasoning only enough to reach the target reasoning range below."
        )
    elif "forbidden_token" in validation.issues:
        issue_lines.append(
            "- Make only the minimal edits needed to remove the delimiters; do not condense or summarize the reasoning."
        )
    issue_text = "\n".join(issue_lines)
    return f"""\
Rewrite the existing CUDA repair reasoning below.

Required corrections:
{issue_text}

Reasoning length:
- Original reasoning: {len(reasoning)} characters.
- Requested reasoning: approximately {target_limits.min_chars}-{target_limits.max_chars} characters.
- Aim near the upper end of that range and do not shorten more than necessary.

Editing requirements:
- Preserve the important diagnosis, numerical calculations, CUDA constraints, rejected alternatives, and final implementation strategy.
- Keep the reasoning in first person and forward-flowing, as if written before the corrected kernel.
- Do not mention this editing request, length limits, forbidden delimiters, or the fact that the text was shortened.
- Preserve detail and target approximately {target_limits.min_chars}-{target_limits.max_chars} characters.
- Return only the revised reasoning text.
- Do not return <my_reasoning> or </my_reasoning>; those delimiters below only mark the input.

<my_reasoning>
{reasoning}
</my_reasoning>"""


async def request_rewrite(
    prompt: str,
    *,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, str | None, dict[str, Any] | None]:
    extra_body = model_extra_body(args.model)
    kwargs: dict[str, Any] = {}
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    async with semaphore:
        response = await litellm.acompletion(
            model=args.model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            max_tokens=args.request_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            drop_params=True,
            **kwargs,
        )
    message = response.choices[0].message
    content = message.content or ""
    thinking = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )
    return content, thinking, getattr(response, "model", None), extra_body


def make_replacement_record(
    original: dict[str, Any],
    *,
    rewrite_prompt: str,
    reasoning: str,
    thinking: str,
    original_validation: ValidationResult,
    final_validation: ValidationResult,
    response_model: str | None,
    request_extra_body: dict[str, Any] | None,
    target_limits: TargetReasoningLimits,
    args: argparse.Namespace,
) -> dict[str, Any]:
    original_reasoning = str(original.get("reasoning") or "")
    metadata = dict(copy.deepcopy(original.get("metadata") or {}))
    metadata.update(
        {
            "reasoning_len": len(reasoning),
            "reasoning_chars": len(reasoning),
            "resynthesized": True,
            "resynthesis_model": args.model,
            "resynthesis_response_model": response_model,
            "resynthesis_original_issues": list(original_validation.issues),
            "resynthesis_original_total_tokens": original_validation.total_tokens,
            "resynthesis_final_total_tokens": final_validation.total_tokens,
            "resynthesis_max_sequence_tokens": args.max_sequence_tokens,
            "resynthesis_target_reasoning_min_tokens": target_limits.min_tokens,
            "resynthesis_target_reasoning_max_tokens": target_limits.max_tokens,
            "resynthesis_target_reasoning_min_chars": target_limits.min_chars,
            "resynthesis_target_reasoning_max_chars": target_limits.max_chars,
            "resynthesis_original_reasoning_sha256": sha256_text(original_reasoning),
            "resynthesis_reasoning_sha256": sha256_text(reasoning),
            "resynthesis_request_extra_body": request_extra_body,
        }
    )
    return {
        "system_prompt": REWRITE_SYSTEM_PROMPT,
        "input": rewrite_prompt,
        "reasoning": reasoning,
        "thinking": thinking,
        "metadata": metadata,
    }


async def repair_one(
    original: dict[str, Any],
    csv_row: dict[str, str],
    original_validation: ValidationResult,
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    semaphore: asyncio.Semaphore,
) -> RepairResult:
    correct_key = record_path_key(original, base_dir=args.input_jsonl.parent)
    reasoning = str(original.get("reasoning") or "")
    validation = original_validation
    try:
        target_limits = target_reasoning_limits(
            reasoning,
            validation,
            max_sequence_tokens=args.max_sequence_tokens,
            safety_tokens=args.safety_tokens,
            min_target_ratio=args.min_target_ratio,
        )
        prompt = build_rewrite_prompt(
            reasoning,
            validation,
            target_limits=target_limits,
        )
        content, thinking, response_model, extra_body = await request_rewrite(
            prompt, args=args, semaphore=semaphore
        )
    except Exception as exc:
        LOGGER.warning("%s API failure: %s", original_validation.row_id, exc)
        return RepairResult(correct_key, None, f"API failure: {exc}")

    candidate = extract_reasoning(content).strip()
    candidate_record = record_with_reasoning(original, candidate)
    try:
        candidate_validation = validate_record(
            candidate_record,
            csv_row,
            csv_dir=args.kernel_pairs_csv.parent,
            source_label=args.source_label,
            tokenizer=tokenizer,
            max_sequence_tokens=args.max_sequence_tokens,
        )
    except Exception as exc:
        LOGGER.warning("%s could not be validated: %s", original_validation.row_id, exc)
        return RepairResult(correct_key, None, f"validation error: {exc}")

    if not candidate_validation.ok:
        LOGGER.warning(
            "%s still fails issues=%s total_tokens=%d chars=%d",
            original_validation.row_id,
            list(candidate_validation.issues),
            candidate_validation.total_tokens,
            len(candidate),
        )
        return RepairResult(
            correct_key,
            None,
            f"rewrite still fails: issues={list(candidate_validation.issues)} "
            f"total_tokens={candidate_validation.total_tokens}",
        )

    replacement = make_replacement_record(
        original,
        rewrite_prompt=prompt,
        reasoning=candidate,
        thinking=thinking,
        original_validation=original_validation,
        final_validation=candidate_validation,
        response_model=response_model,
        request_extra_body=extra_body,
        target_limits=target_limits,
        args=args,
    )
    return RepairResult(correct_key, replacement)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def merged_records_in_source_order(
    source_records: list[dict[str, Any]],
    repaired_by_key: dict[str, dict[str, Any]],
    *,
    source_dir: Path,
) -> list[dict[str, Any]]:
    """Preserve source rows and replace repaired rows without changing order."""
    output: list[dict[str, Any]] = []
    for source_record in source_records:
        key = record_path_key(source_record, base_dir=source_dir)
        output.append(repaired_by_key.get(key, source_record))
    return output


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def provenance_payload(args: argparse.Namespace, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "config": jsonable_args(args),
        "environment": {"openrouter_api_key_set": bool(os.environ.get("OPENROUTER_API_KEY"))},
        "model_extra_body": model_extra_body(args.model),
        "stats": stats,
    }


async def run(args: argparse.Namespace) -> None:
    records = load_jsonl(args.input_jsonl)
    source_by_key = index_records(
        records, base_dir=args.input_jsonl.parent, label=str(args.input_jsonl)
    )
    csv_by_key = load_kernel_pairs_csv(args.kernel_pairs_csv)
    missing_csv = sorted(set(source_by_key).difference(csv_by_key))
    extra_csv = sorted(set(csv_by_key).difference(source_by_key))
    if missing_csv or extra_csv:
        raise SystemExit(
            f"input/CSV coverage mismatch: missing_csv={len(missing_csv)} extra_csv={len(extra_csv)}; "
            f"first_missing={missing_csv[:3]} first_extra={extra_csv[:3]}"
        )

    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer is None:
        raise SystemExit(f"failed to load tokenizer: {args.tokenizer}")

    validations: dict[str, ValidationResult] = {}
    for index, record in enumerate(records, 1):
        key = record_path_key(record, base_dir=args.input_jsonl.parent)
        try:
            validations[key] = validate_record(
                record,
                csv_by_key[key],
                csv_dir=args.kernel_pairs_csv.parent,
                source_label=args.source_label,
                tokenizer=tokenizer,
                max_sequence_tokens=args.max_sequence_tokens,
            )
        except Exception as exc:
            raise ValueError(f"failed to validate input record {index} ({key}): {exc}") from exc

    failing_keys = [key for key in source_by_key if not validations[key].ok]
    failing_key_set = set(failing_keys)
    issue_counts = Counter(issue for key in failing_keys for issue in validations[key].issues)
    LOGGER.info(
        "audited records=%d valid=%d failing=%d issues=%s",
        len(records),
        len(records) - len(failing_keys),
        len(failing_keys),
        dict(sorted(issue_counts.items())),
    )
    for key in failing_keys:
        validation = validations[key]
        LOGGER.info(
            "needs repair id=%s issues=%s total_tokens=%d fixed_tokens=%d",
            validation.row_id,
            list(validation.issues),
            validation.total_tokens,
            validation.fixed_tokens,
        )

    base_stats: dict[str, Any] = {
        "input_records": len(records),
        "valid_before": len(records) - len(failing_keys),
        "repair_candidates": len(failing_keys),
        "initial_issue_counts": dict(sorted(issue_counts.items())),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        write_json(args.provenance_json, provenance_payload(args, base_stats))
        LOGGER.info("dry run complete; wrote provenance=%s", args.provenance_json)
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")

    checkpoint_records = (
        load_jsonl(args.checkpoint_jsonl) if args.checkpoint_jsonl.is_file() else []
    )
    checkpoint_by_key = index_checkpoint_records(
        checkpoint_records,
        base_dir=args.checkpoint_jsonl.parent,
        label=str(args.checkpoint_jsonl),
    )
    accepted_checkpoint: dict[str, dict[str, Any]] = {}
    for key, record in checkpoint_by_key.items():
        if key not in source_by_key:
            LOGGER.warning("ignoring checkpoint record absent from input: %s", key)
            continue
        if key not in failing_key_set:
            LOGGER.warning(
                "ignoring checkpoint record for source row that needs no repair: %s", key
            )
            continue
        if record.get("system_prompt") != REWRITE_SYSTEM_PROMPT or not str(
            record.get("input") or ""
        ).startswith("Rewrite the existing CUDA repair reasoning below."):
            LOGGER.warning("ignoring legacy checkpoint record without rewrite prompts: %s", key)
            continue
        validation = validate_record(
            record,
            csv_by_key[key],
            csv_dir=args.kernel_pairs_csv.parent,
            source_label=args.source_label,
            tokenizer=tokenizer,
            max_sequence_tokens=args.max_sequence_tokens,
        )
        if validation.ok:
            accepted_checkpoint[key] = record
        else:
            LOGGER.warning(
                "ignoring invalid checkpoint record %s issues=%s",
                validation.row_id,
                list(validation.issues),
            )

    pending_keys = [key for key in failing_keys if key not in accepted_checkpoint]
    LOGGER.info(
        "checkpoint_valid=%d pending_repairs=%d", len(accepted_checkpoint), len(pending_keys)
    )
    semaphore = asyncio.Semaphore(args.max_concurrent)
    checkpoint_lock = asyncio.Lock()
    repaired: dict[str, dict[str, Any]] = dict(accepted_checkpoint)

    async def process(key: str) -> RepairResult:
        result = await repair_one(
            source_by_key[key],
            csv_by_key[key],
            validations[key],
            args=args,
            tokenizer=tokenizer,
            semaphore=semaphore,
        )
        if result.record is not None:
            async with checkpoint_lock:
                append_jsonl(args.checkpoint_jsonl, result.record)
                repaired[key] = result.record
            final = validate_record(
                result.record,
                csv_by_key[key],
                csv_dir=args.kernel_pairs_csv.parent,
                source_label=args.source_label,
                tokenizer=tokenizer,
                max_sequence_tokens=args.max_sequence_tokens,
            )
            LOGGER.info(
                "repaired id=%s total_tokens=%d",
                final.row_id,
                final.total_tokens,
            )
        return result

    results = await asyncio.gather(*(process(key) for key in pending_keys))
    failures = [result for result in results if result.record is None]
    output_records = merged_records_in_source_order(
        records, repaired, source_dir=args.input_jsonl.parent
    )
    final_issue_counts: Counter[str] = Counter()
    final_invalid_records = 0
    for record in output_records:
        key = record_path_key(record, base_dir=args.input_jsonl.parent)
        validation = validate_record(
            record,
            csv_by_key[key],
            csv_dir=args.kernel_pairs_csv.parent,
            source_label=args.source_label,
            tokenizer=tokenizer,
            max_sequence_tokens=args.max_sequence_tokens,
        )
        final_issue_counts.update(validation.issues)
        if not validation.ok:
            final_invalid_records += 1
            LOGGER.warning(
                "output retains unresolved source id=%s issues=%s",
                validation.row_id,
                list(validation.issues),
            )

    write_jsonl_atomic(args.output_jsonl, output_records)
    stats = {
        **base_stats,
        "checkpoint_reused": len(accepted_checkpoint),
        "repaired_this_run": len(results) - len(failures),
        "failed": len(failures),
        "failures": [asdict(result) for result in failures],
        "output_records": len(output_records),
        "resynthesized_output_records": len(repaired),
        "unchanged_output_records": len(output_records) - len(repaired),
        "final_invalid_records": final_invalid_records,
        "final_issue_counts": dict(sorted(final_issue_counts.items())),
        "output_written": True,
        "output_jsonl": str(args.output_jsonl),
        "checkpoint_jsonl": str(args.checkpoint_jsonl),
    }
    write_json(args.provenance_json, provenance_payload(args, stats))
    LOGGER.info(
        "complete: output_records=%d failed=%d output=%s provenance=%s",
        len(output_records),
        len(failures),
        args.output_jsonl,
        args.provenance_json,
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} of {len(failing_keys)} one-shot resyntheses failed validation; "
            f"wrote all {len(output_records)} source-ordered records with successful replacements "
            f"to {args.output_jsonl}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
