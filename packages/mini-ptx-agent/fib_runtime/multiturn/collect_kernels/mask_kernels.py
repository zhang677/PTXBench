#!/usr/bin/env python3
"""Mask selected parts of collected CUDA kernels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


MASKED_CSV_NAME = "masked_kernels.csv"
SUPPORTED_LEVELS = ("l1",)
L1_CALL_NAME_SUBSTRING = "desc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mask CUDA kernels. Level l1 replaces the arguments of calls to "
            'functions whose names contain "desc" with question marks.'
        )
    )
    parser.add_argument(
        "--level",
        choices=SUPPORTED_LEVELS,
        default="l1",
        help="Masking level to apply.",
    )
    parser.add_argument(
        "--input-kernel",
        type=Path,
        help="Path to a single input kernel.",
    )
    parser.add_argument(
        "--output-kernel",
        type=Path,
        help="Path where the masked single kernel should be written.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="CSV containing kernel_path rows to mask in batch mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for masked kernels and the output CSV in batch mode.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help=f"Optional batch output CSV path. Defaults to OUTPUT_DIR/{MASKED_CSV_NAME}.",
    )
    args = parser.parse_args()

    single_mode = args.input_kernel is not None or args.output_kernel is not None
    batch_mode = args.input_csv is not None or args.output_dir is not None or args.output_csv is not None
    if single_mode and batch_mode:
        parser.error("single-kernel mode and batch CSV mode cannot be mixed")
    if single_mode:
        if args.input_kernel is None or args.output_kernel is None:
            parser.error("single-kernel mode requires --input-kernel and --output-kernel")
    elif batch_mode:
        if args.input_csv is None or args.output_dir is None:
            parser.error("batch mode requires --input-csv and --output-dir")
    else:
        parser.error("provide either --input-kernel/--output-kernel or --input-csv/--output-dir")

    return args


def split_top_level_args(args_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    angle_depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    i = 0

    while i < len(args_text):
        char = args_text[i]
        next_char = args_text[i + 1] if i + 1 < len(args_text) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            i += 1
            continue

        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue

        if char == "/" and next_char == "/":
            line_comment = True
            i += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            i += 2
            continue
        if char in ("'", '"'):
            quote = char
            i += 1
            continue

        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth > 0:
            angle_depth -= 1
        elif (
            char == ","
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
            and angle_depth == 0
        ):
            args.append(args_text[start:i].strip())
            start = i + 1

        i += 1

    tail = args_text[start:].strip()
    if tail or args_text.strip():
        args.append(tail)
    return args


def find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    i = open_index

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            i += 1
            continue

        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue

        if char == "/" and next_char == "/":
            line_comment = True
            i += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            i += 2
            continue
        if char in ("'", '"'):
            quote = char
            i += 1
            continue

        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    raise ValueError(f"unclosed parenthesis at offset {open_index}")


def next_nonspace(text: str, start: int) -> str:
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    return text[i] if i < len(text) else ""


def is_identifier_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def is_identifier_start_char(char: str) -> bool:
    return char.isalpha() or char == "_"


def find_next_l1_call(source: str, cursor: int) -> tuple[int, int, int] | None:
    i = cursor
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False

    while i < len(source):
        char = source[i]
        next_char = source[i + 1] if i + 1 < len(source) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            i += 1
            continue

        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue

        if char == "/" and next_char == "/":
            line_comment = True
            i += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            i += 2
            continue
        if char in ("'", '"'):
            quote = char
            i += 1
            continue

        if not is_identifier_start_char(char):
            i += 1
            continue

        name_index = i
        i += 1
        while i < len(source) and is_identifier_char(source[i]):
            i += 1
        name_end_index = i
        name = source[name_index:name_end_index]

        if L1_CALL_NAME_SUBSTRING not in name:
            continue

        open_index = name_end_index
        while open_index < len(source) and source[open_index].isspace():
            open_index += 1
        if open_index < len(source) and source[open_index] == "(":
            return name_index, name_end_index, open_index

    return None


def mask_l1(source: str) -> tuple[str, int]:
    out: list[str] = []
    cursor = 0
    masked = 0

    while True:
        match = find_next_l1_call(source, cursor)
        if match is None:
            out.append(source[cursor:])
            break
        name_index, name_end_index, open_index = match

        close_index = find_matching_paren(source, open_index)
        if next_nonspace(source, close_index + 1) == "{":
            out.append(source[cursor : close_index + 1])
            cursor = close_index + 1
            continue

        args_text = source[open_index + 1 : close_index]
        args = split_top_level_args(args_text)
        replacement = ", ".join("?" for _ in args)
        out.append(source[cursor : open_index + 1])
        out.append(replacement)
        cursor = close_index
        masked += 1

    return "".join(out), masked


def mask_source(source: str, level: str) -> tuple[str, int]:
    if level == "l1":
        return mask_l1(source)
    raise ValueError(f"unsupported mask level: {level}")


def mask_kernel(input_kernel: Path, output_kernel: Path, level: str) -> int:
    source = input_kernel.read_text()
    masked_source, count = mask_source(source, level)
    output_kernel.parent.mkdir(parents=True, exist_ok=True)
    output_kernel.write_text(masked_source)
    return count


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def output_name(index: int, kernel_path: Path, row: dict[str, str]) -> str:
    digest_source = "\n".join(
        [
            str(kernel_path),
            row.get("model", ""),
            row.get("arch", ""),
            row.get("definition", ""),
            row.get("workload", ""),
        ]
    )
    digest = hashlib.sha1(digest_source.encode()).hexdigest()[:10]
    return f"{index:06d}_{kernel_path.stem}_{digest}{kernel_path.suffix or '.cu'}"


def mask_csv(
    input_csv: Path,
    output_dir: Path,
    output_csv: Path | None,
    level: str,
) -> tuple[int, int, int, list[Path], Path]:
    fieldnames, rows = read_csv_rows(input_csv)
    if "kernel_path" not in fieldnames:
        raise ValueError(f"{input_csv} must contain a kernel_path column")

    kernels_dir = output_dir / "kernels"
    csv_path = output_csv or output_dir / MASKED_CSV_NAME
    output_rows: list[dict[str, str]] = []
    kernels_with_masks = 0
    total_masked_calls = 0
    unchanged_kernel_paths: list[Path] = []

    for index, row in enumerate(rows):
        kernel_path = Path(row["kernel_path"]).expanduser()
        masked_path = kernels_dir / output_name(index, kernel_path, row)
        count = mask_kernel(kernel_path, masked_path, level)
        if count > 0:
            kernels_with_masks += 1
        else:
            unchanged_kernel_paths.append(kernel_path)
        total_masked_calls += count
        output_row = dict(row)
        output_row["masked_kernel_path"] = str(masked_path)
        output_row["mask_level"] = level
        output_row["masked_descriptor_calls"] = str(count)
        output_rows.append(output_row)

    output_fieldnames = list(fieldnames)
    for extra_field in ("masked_kernel_path", "mask_level", "masked_descriptor_calls"):
        if extra_field not in output_fieldnames:
            output_fieldnames.append(extra_field)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows), kernels_with_masks, total_masked_calls, unchanged_kernel_paths, csv_path


def main() -> None:
    args = parse_args()
    try:
        if args.input_kernel is not None:
            count = mask_kernel(args.input_kernel, args.output_kernel, args.level)
            print(f"wrote {args.output_kernel}; masked {count} desc-named calls")
        else:
            total, kernels_with_masks, total_calls, unchanged_kernel_paths, csv_path = mask_csv(
                args.input_csv,
                args.output_dir,
                args.output_csv,
                args.level,
            )
            unchanged = total - kernels_with_masks
            print(
                f"wrote {total} masked-kernel files into {args.output_dir} "
                f"({kernels_with_masks} contained '?' masks, {unchanged} unchanged; "
                f"{total_calls} desc-named calls masked); wrote {csv_path}"
            )
            if unchanged_kernel_paths:
                print("unchanged kernel paths:")
                for kernel_path in unchanged_kernel_paths:
                    print(f"  {kernel_path}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
