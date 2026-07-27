#!/usr/bin/env python3
"""Convert combined kernel-fix instruction items into a structural-doc prompt.

The input is the JSON emitted by combine_kernel_fix_notes.py. The output is an
operator-agnostic CUDA instruction cookbook intended to be included from
fib_runtime/structural_doc/notes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/AccRL/fib_runtime/structural_doc/notes")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "kernel-fix-action-items"


def load_payload(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    items = payload.get("instruction_items", payload.get("actionable_items"))
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected list field 'instruction_items'")
    return payload


def output_path_for(input_path: Path, output_dir: Path, *, name: str | None) -> Path:
    stem = slugify(name or input_path.stem)
    return output_dir / f"{stem}.prompt.md"


def comma_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(str(value) for value in values if str(value).strip())


def render_prompt(payload: dict[str, Any], *, source_path: Path, title: str | None) -> str:
    items = payload.get("instruction_items") or payload.get("actionable_items") or []
    dropped = payload.get("dropped_themes") or []
    prompt_title = title or "CUDA Instruction Usage Notes"

    lines: list[str] = [
        f"# {prompt_title}",
        "",
        "Use these notes as an operator-agnostic cookbook for CUDA/PTX/API instruction usage.",
        "Do not treat the presence of an instruction as optimization advice. Use a variant only when its shape, layout, address-space, and synchronization contract match the current kernel.",
        "Task names are source metadata only; the instruction contract is the lesson.",
        "",
    ]

    summary = str(payload.get("summary") or "").strip()
    if summary:
        lines.extend(["## Summary", "", summary, ""])

    lines.extend(
        [
            "## Instruction Cookbook",
            "",
            "Each section documents how to use one primitive legally. Preserve operand order, address-space qualifiers, immediates, descriptor assumptions, synchronization order, and shape/layout constraints.",
            "",
        ]
    )

    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or item.get("item") or "").strip()
        if not instruction:
            continue
        lines.append(f"### {index}. {instruction}")

        tags = comma_list(item.get("tags"))
        metadata: list[str] = []
        if item.get("source_count") is not None:
            metadata.append(f"source_count={item['source_count']}")
        if tags:
            metadata.append(f"tags={tags}")
        if metadata:
            lines.extend(["", f"Metadata: {'; '.join(metadata)}"])

        variants = item.get("variants")
        if not isinstance(variants, list):
            variants = [
                {
                    "shape_context": item.get("shape_context", ""),
                    "correct_pattern": item.get("correct_example", ""),
                    "wrong_patterns": [item.get("wrong_example", "")] if item.get("wrong_example") else [],
                    "operand_contract": item.get("operand_contract", []),
                    "required_sequence": item.get("required_sequence", []),
                    "diagnostics": item.get("diagnostics", item.get("signals", [])),
                    "do_not_do": item.get("do_not_do", item.get("avoid", [])),
                }
            ]
        for variant_index, variant in enumerate(variants, 1):
            if not isinstance(variant, dict):
                continue
            lines.extend(["", f"#### Variant {variant_index}"])
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
                    detail_text = str(detail).strip()
                    if detail_text:
                        lines.append(f"- {detail_text}")

            correct_pattern = str(variant.get("correct_pattern") or "").strip()
            if correct_pattern:
                label = "Partial pattern" if completeness == "partial" else "Minimal correct pattern"
                lines.extend(["", f"{label}:", "", "```cpp", correct_pattern, "```"])

            wrong_patterns = variant.get("wrong_patterns") or []
            if wrong_patterns:
                lines.extend(["", "Wrong patterns:"])
                for pattern in wrong_patterns:
                    pattern_text = str(pattern).strip()
                    if pattern_text:
                        lines.extend(["", "```cpp", pattern_text, "```"])

            for field, label in (
                ("operand_contract", "Operand contract"),
                ("required_sequence", "Required sequence"),
                ("diagnostics", "Diagnostics"),
                ("do_not_do", "Do not do"),
            ):
                values = variant.get(field) or []
                if isinstance(values, str):
                    values = [values]
                if isinstance(values, list) and values:
                    lines.extend(["", f"{label}:"])
                    for value in values:
                        value_text = str(value).strip()
                        if value_text:
                            lines.append(f"- {value_text}")
        lines.append("")

    if dropped:
        lines.extend(["## Do Not Overgeneralize", ""])
        for theme in dropped:
            theme_text = str(theme).strip()
            if theme_text:
                lines.append(f"- {theme_text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="Combined action_items JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated prompts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--output", type=Path, help="Exact output Markdown path")
    parser.add_argument("--name", help="Output filename stem when --output is not set")
    parser.add_argument("--title", help="Prompt title")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_json.expanduser().resolve()
    payload = load_payload(input_path)

    output_path = args.output.expanduser() if args.output else output_path_for(input_path, args.output_dir, name=args.name)
    if not output_path.is_absolute():
        output_path = output_path.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output_path}")

    prompt = render_prompt(payload, source_path=input_path, title=args.title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt)
    print(output_path)


if __name__ == "__main__":
    main()
