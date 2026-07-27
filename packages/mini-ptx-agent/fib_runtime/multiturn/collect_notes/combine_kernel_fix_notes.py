#!/usr/bin/env python3
"""Use an LLM to summarize kernel-fix notes into instruction cookbooks."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
import yaml

MULTITURN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MULTITURN_DIR))
from common import make_model  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "GLM-5.2"

SYSTEM_PROMPT = """\
You are summarizing CUDA kernel repair notes into an operator-agnostic CUDA
instruction cookbook. Merge only examples for the same primitive or instruction
family, keep evidence-backed variants, and preserve exact CUDA/PTX/API
contracts."""

REDUCE_PROMPT_TEMPLATE = """\
You are running round {round_number} of an iterative LLM summarization pipeline.
Below are {input_count} input notes or prior summaries from CUDA kernel repair examples.

{inputs_block}

Return only valid JSON with this schema:
{{
  "summary": "concise synthesis of these inputs",
  "instruction_items": [
    {{
      "instruction": "specific CUDA/PTX/API primitive or instruction family",
      "source_count": 1,
      "tags": ["short-topic-tag"],
      "variants": [
        {{
          "shape_context": "dtype/rank/tile/layout/alignment context where this variant is valid",
          "correct_pattern": "small exact valid snippet or pattern",
          "wrong_patterns": ["small exact invalid snippets or patterns"],
          "operand_contract": ["required operand ordering, address spaces, immediates, or descriptor fields"],
          "required_sequence": ["ordered instructions or API calls that must appear together"],
          "diagnostics": ["compiler/runtime/profiler messages this variant explains"],
          "do_not_do": ["specific invalid shortcut or cargo-cult pattern to avoid"],
          "example_completeness": "complete or partial",
          "missing_details": ["details omitted from correct_pattern, empty when complete"],
          "source_count": 1
        }}
      ]
    }}
  ],
  "dropped_themes": ["themes intentionally omitted because they were too vague or unsupported"]
}}

Rules:
- Summarize these inputs into one new note for the next reduction round.
- Group by primitive or instruction family, not by operator or task.
- Preserve exact operand order, address-space qualifiers, immediates, descriptor assumptions, synchronization order, and shape/layout constraints.
- Do not merge examples that require different tensor shapes, layouts, swizzles, address spaces, or synchronization sequences; keep them as separate variants.
- Extract instruction-use contracts, not optimization advice. Do not say "use TMA/WGMMA"; say how to use a specific primitive correctly.
- Do not replace operands, template parameters, descriptor arguments, address-space qualifiers, barrier sequences, or launch arguments with "...".
- Prefer one complete 3-8 line snippet over a one-line sketch with ellipses.
- If two examples differ only in irrelevant variable names, normalize names. If they differ in actual operands, ranks, address spaces, swizzles, or synchronization sequence, keep separate variants.
- If the input examples are incomplete, set example_completeness to "partial" and list missing_details instead of inventing missing operands.
- Preserve the strongest evidence and source counts.
- Do not introduce claims that are unsupported by the notes."""

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ExperimentConfig:
    notes_jsonl: Path
    output_json: Path
    output_markdown: Path | None = None
    config_yaml: Path | None = None
    provenance_json: Path | None = None
    name: str = "kernel_fix_note_combiner"
    description: str = ""
    model: str = DEFAULT_MODEL
    timeout: float = 600.0
    limit_notes: int | None = None
    inputs_per_summary: int = 20
    max_note_chars: int = 120000
    max_rounds: int = 20
    overwrite: bool = False


def resolve_config_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(path: Path) -> ExperimentConfig:
    path = path.expanduser()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    fields = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw).difference(fields)
    if unknown:
        raise ValueError(f"{path}: unknown keys: {', '.join(sorted(unknown))}")
    missing = {"notes_jsonl", "output_json"}.difference(raw)
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(sorted(missing))}")
    base_dir = path.resolve().parent
    raw["notes_jsonl"] = resolve_config_path(raw["notes_jsonl"], base_dir=base_dir)
    raw["output_json"] = resolve_config_path(raw["output_json"], base_dir=base_dir)
    raw["output_markdown"] = resolve_config_path(raw.get("output_markdown"), base_dir=base_dir)
    raw["provenance_json"] = resolve_config_path(raw.get("provenance_json"), base_dir=base_dir)
    raw["config_yaml"] = path.resolve()
    return ExperimentConfig(**raw)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_yaml", type=Path)
    return load_config(parser.parse_args().config_yaml)


def load_notes(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            notes.append(json.loads(line))
            if limit is not None and len(notes) >= limit:
                break
    return notes


def compact_raw_note(note: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = note.get("metadata", {})
    return {
        "note_index": metadata.get("raw_note_index", index),
        "kind": "raw_note",
        "summary": note.get("summary", ""),
        "tags": note.get("tags", []),
        "definition": metadata.get("definition", ""),
        "wrong_turn": metadata.get("wrong_turn", ""),
        "wrong_kernel_path": metadata.get("wrong_kernel_path", ""),
        "correct_kernel_path": metadata.get("correct_kernel_path", ""),
        "instruction_notes": note.get("instruction_notes", note.get("insights", [])),
    }


def compact_summary_note(note: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = note.get("metadata", {})
    return {
        "note_index": index,
        "kind": "summary_note",
        "summary": note.get("summary", ""),
        "instruction_items": note.get("instruction_items", note.get("actionable_items", [])),
        "dropped_themes": note.get("dropped_themes", []),
        "source_count": metadata.get("source_count", note.get("source_count", 1)),
        "source_note_indices": metadata.get("source_note_indices", []),
        "round": metadata.get("round"),
        "group_index": metadata.get("group_index"),
    }


def compact_input(item: dict[str, Any], index: int) -> dict[str, Any]:
    if item.get("_summary_note"):
        return compact_summary_note(item, index)
    return compact_raw_note(item, index)


def inputs_block(items: list[dict[str, Any]], *, max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for index, item in enumerate(items, 1):
        text = json.dumps(compact_input(item, index), ensure_ascii=False)
        if lines and used + len(text) + 2 > max_chars:
            raise ValueError(
                f"group with {len(items)} inputs exceeds max_note_chars={max_chars}; "
                "lower inputs_per_summary or raise max_note_chars"
            )
        if not lines and len(text) > max_chars:
            LOGGER.warning("single input exceeds max_note_chars; including it alone")
        lines.append(text)
        used += len(text) + 2
    return "\n".join(lines)


def source_indices_for_item(item: dict[str, Any], fallback_index: int) -> list[int]:
    if item.get("_summary_note"):
        values = item.get("metadata", {}).get("source_note_indices", [])
        return [int(value) for value in values] if isinstance(values, list) else []
    raw_index = item.get("metadata", {}).get("raw_note_index")
    if raw_index is not None:
        return [int(raw_index)]
    return [fallback_index]


def batched(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size < 2:
        raise ValueError("inputs_per_summary must be at least 2")
    return [items[i: i + size] for i in range(0, len(items), size)]


def parse_json_object(content: str, *, required_list: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(content)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("model response JSON is not an object")
    if not isinstance(data.get(required_list), list):
        raise ValueError(f"model response missing list field: {required_list}")
    return data


async def call_llm(
    config: ExperimentConfig,
    prompt: str,
    *,
    required_list: str,
) -> tuple[dict[str, Any], Any]:
    model = make_model(config.model)
    model_kwargs: dict[str, Any] = dict(model.config.model_kwargs)
    model_kwargs.setdefault("timeout", config.timeout)
    response = await litellm.acompletion(
        model=model.config.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        **model_kwargs,
    )
    content = response.choices[0].message.content or ""
    return parse_json_object(content, required_list=required_list), response


def markdown_from_payload(payload: dict[str, Any]) -> str:
    lines = ["# CUDA Instruction Cookbook", ""]
    if payload.get("summary"):
        lines.extend([str(payload["summary"]).strip(), ""])
    for index, item in enumerate(payload.get("instruction_items", []), 1):
        lines.append(f"## {index}. {item.get('instruction', '').strip()}")
        tags = item.get("tags") or []
        if tags:
            lines.extend(["", f"Tags: {', '.join(str(tag) for tag in tags)}"])
        if item.get("source_count") is not None:
            lines.extend(["", f"Source count: {item['source_count']}"])
        for variant_index, variant in enumerate(item.get("variants") or [], 1):
            if not isinstance(variant, dict):
                continue
            lines.extend(["", f"### Variant {variant_index}"])
            shape_context = str(variant.get("shape_context") or "").strip()
            if shape_context:
                lines.extend(["", f"Shape/context: {shape_context}"])
            completeness = str(variant.get("example_completeness") or "").strip()
            missing_details = variant.get("missing_details") or []
            if completeness:
                lines.extend(["", f"Example completeness: {completeness}"])
            if missing_details:
                lines.extend(["", "Missing details:"])
                for detail in missing_details:
                    text = str(detail).strip()
                    if text:
                        lines.append(f"- {text}")
            correct_pattern = str(variant.get("correct_pattern") or "").strip()
            if correct_pattern:
                label = "Partial pattern" if completeness == "partial" else "Correct pattern"
                lines.extend(["", f"{label}:", "", "```cpp", correct_pattern, "```"])
            wrong_patterns = variant.get("wrong_patterns") or []
            if wrong_patterns:
                lines.extend(["", "Wrong patterns:"])
                for pattern in wrong_patterns:
                    text = str(pattern).strip()
                    if text:
                        lines.extend(["", "```cpp", text, "```"])
            for field, label in (
                ("operand_contract", "Operand contract"),
                ("required_sequence", "Required sequence"),
                ("diagnostics", "Diagnostics"),
                ("do_not_do", "Do not do"),
            ):
                values = variant.get(field) or []
                if values:
                    lines.extend(["", f"{label}:"])
                    for value in values:
                        lines.append(f"- {value}")
        lines.append("")
    dropped = payload.get("dropped_themes") or []
    if dropped:
        lines.extend(["## Dropped Themes", ""])
        for theme in dropped:
            lines.append(f"- {theme}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def lint_payload_examples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for item_index, item in enumerate(payload.get("instruction_items", []), 1):
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or f"item_{item_index}")
        for variant_index, variant in enumerate(item.get("variants") or [], 1):
            if not isinstance(variant, dict):
                continue
            correct = str(variant.get("correct_pattern") or "")
            required_sequence = variant.get("required_sequence") or []
            operand_contract = variant.get("operand_contract") or []
            diagnostics = variant.get("diagnostics") or []
            issues: list[str] = []
            if "..." in correct or "<...>" in correct:
                issues.append("correct_pattern contains ellipsis or placeholder")
            if "asm volatile" in correct and "::" not in correct:
                issues.append("PTX asm pattern omits operand constraint section")
            if "cuTensorMapEncodeTiled" in correct:
                for term in ("globalDim", "globalStrides", "boxDim"):
                    if term not in correct and not any(term in str(value) for value in operand_contract):
                        issues.append(f"cuTensorMapEncodeTiled pattern does not mention {term}")
                if "SWIZZLE" not in correct and not any("swizzle" in str(value).lower() for value in operand_contract):
                    issues.append("cuTensorMapEncodeTiled pattern does not mention swizzle")
            if "mbarrier" in instruction.lower():
                sequence_text = " ".join(str(value) for value in required_sequence).lower()
                for term in ("init", "expect_tx", "wait"):
                    if term not in sequence_text:
                        issues.append(f"mbarrier variant required_sequence omits {term}")
            if issues:
                warnings.append(
                    {
                        "instruction": instruction,
                        "variant_index": variant_index,
                        "issues": sorted(set(issues)),
                    }
                )
    return warnings


async def run(config: ExperimentConfig) -> None:
    if config.output_json.exists() and not config.overwrite:
        raise FileExistsError(f"output exists; set overwrite: true to replace: {config.output_json}")

    raw_notes = load_notes(config.notes_jsonl, limit=config.limit_notes)
    for index, note in enumerate(raw_notes, 1):
        note.setdefault("metadata", {})
        note["metadata"]["raw_note_index"] = index
    current: list[dict[str, Any]] = raw_notes
    rounds: list[dict[str, Any]] = []
    response_models: list[dict[str, Any]] = []
    LOGGER.info("loaded notes=%d inputs_per_summary=%d", len(raw_notes), config.inputs_per_summary)

    round_number = 0
    while len(current) > 1:
        round_number += 1
        if round_number > config.max_rounds:
            raise RuntimeError(f"exceeded max_rounds={config.max_rounds} with {len(current)} notes left")
        groups = batched(current, config.inputs_per_summary)
        LOGGER.info("round %d: inputs=%d groups=%d", round_number, len(current), len(groups))

        next_round: list[dict[str, Any]] = []
        round_records: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups, 1):
            if len(group) == 1:
                next_round.append(group[0])
                round_records.append(
                    {
                        "group_index": group_index,
                        "input_count": 1,
                        "carried_forward": True,
                    }
                )
                continue
            block = inputs_block(group, max_chars=config.max_note_chars)
            prompt = REDUCE_PROMPT_TEMPLATE.format(
                round_number=round_number,
                input_count=len(group),
                inputs_block=block,
            )
            payload, response = await call_llm(config, prompt, required_list="instruction_items")
            source_indices: list[int] = []
            for offset, item in enumerate(group):
                source_indices.extend(source_indices_for_item(item, offset + 1))
            source_indices = sorted(set(source_indices))
            payload["_summary_note"] = True
            payload.setdefault("metadata", {})
            payload["metadata"].update(
                {
                    "round": round_number,
                    "group_index": group_index,
                    "input_count": len(group),
                    "source_count": len(source_indices),
                    "source_note_indices": source_indices,
                    "response_model": getattr(response, "model", None),
                }
            )
            next_round.append(payload)
            round_records.append(
                {
                    "group_index": group_index,
                    "input_count": len(group),
                    "source_count": len(source_indices),
                    "instruction_items": len(payload.get("instruction_items", [])),
                }
            )
            response_models.append(
                {
                    "round": round_number,
                    "group_index": group_index,
                    "response_model": getattr(response, "model", None),
                }
            )
            LOGGER.info(
                "round %d group %d/%d summarized inputs=%d",
                round_number,
                group_index,
                len(groups),
                len(group),
            )
        rounds.append(
            {
                "round": round_number,
                "input_count": len(current),
                "output_count": len(next_round),
                "groups": round_records,
            }
        )
        current = next_round

    if not current:
        raise ValueError(f"no notes loaded from {config.notes_jsonl}")
    payload = current[0]
    payload.pop("_summary_note", None)
    if "instruction_items" not in payload and "instruction_notes" in payload:
        payload["instruction_items"] = [
            {
                "instruction": note.get("instruction", ""),
                "source_count": 1,
                "tags": note.get("tags", payload.get("tags", [])),
                "variants": [
                    {
                        "shape_context": note.get("shape_context", ""),
                        "correct_pattern": note.get("correct_example", ""),
                        "wrong_patterns": [note.get("wrong_example", "")] if note.get("wrong_example") else [],
                        "operand_contract": note.get("operand_contract", []),
                        "required_sequence": note.get("required_sequence", []),
                        "diagnostics": note.get("diagnostics", []),
                        "do_not_do": note.get("do_not_do", []),
                        "example_completeness": note.get("example_completeness", "partial"),
                        "missing_details": note.get("missing_details", []),
                        "source_count": 1,
                    }
                ],
            }
            for note in payload.get("instruction_notes", [])
            if isinstance(note, dict)
        ]
    lint_warnings = lint_payload_examples(payload)
    for warning in lint_warnings:
        LOGGER.warning(
            "example lint: %s variant=%s issues=%s",
            warning["instruction"],
            warning["variant_index"],
            "; ".join(warning["issues"]),
        )
    payload.setdefault("metadata", {})
    payload["metadata"].update(
        {
            "notes_jsonl": str(config.notes_jsonl),
            "notes_loaded": len(raw_notes),
            "inputs_per_summary": config.inputs_per_summary,
            "rounds": rounds,
            "model": config.model,
            "response_models": response_models,
            "example_lint_warnings": lint_warnings,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if config.output_markdown is not None:
        config.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        config.output_markdown.write_text(markdown_from_payload(payload))
    write_provenance(
        config,
        {
            "notes_loaded": len(raw_notes),
            "inputs_per_summary": config.inputs_per_summary,
            "rounds": len(rounds),
            "example_lint_warnings": len(lint_warnings),
            "output_json": str(config.output_json),
        },
    )
    LOGGER.info("wrote %s items=%d", config.output_json, len(payload.get("instruction_items", [])))


def jsonable_config(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def default_provenance_path(config: ExperimentConfig) -> Path:
    return config.provenance_json or config.output_json.with_suffix(config.output_json.suffix + ".provenance.json")


def write_provenance(config: ExperimentConfig, stats: dict[str, Any]) -> None:
    path = default_provenance_path(config)
    model = make_model(config.model)
    payload = {
        "run_name": config.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "config": jsonable_config(config),
        "environment": {
            "openrouter_api_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
            "llm_api_timeout_set": bool(os.environ.get("LLM_API_TIMEOUT")),
        },
        "resolved_model": model.config.model_name,
        "resolved_model_kwargs": {
            key: value
            for key, value in model.config.model_kwargs.items()
            if key != "api_key"
        },
        "stats": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
