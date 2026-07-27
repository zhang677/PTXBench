#!/usr/bin/env python3
"""Count tokenizer tokens for files in structural_doc content directories.

By default this scans ../document, ../headers, and ../patterns, then writes:
    ../file_token_counts__Qwen_Qwen3.6-35B-A3B.csv

The output schema is:
    path,num_tokens
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = ("document", "headers", "patterns", "headers_wo_doc")
DEFAULT_TOKENIZER = "Qwen/Qwen3.6-35B-A3B"


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to count PDF text") from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8")


def collect_files(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            raise FileNotFoundError(f"input directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"input path is not a directory: {directory}")
        files.extend(p for p in directory.rglob("*") if p.is_file())
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def tokenizer_slug(tokenizer_name: str) -> str:
    safe_chars = []
    for char in tokenizer_name:
        if char.isalnum() or char in ".-_":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "tokenizer"


def default_output_path(tokenizer_name: str) -> Path:
    return ROOT / f"file_token_counts__{tokenizer_slug(tokenizer_name)}.csv"


def hub_style_path(path: Path) -> str:
    return f"structural_doc/{path.relative_to(ROOT).as_posix()}"


def build_row(path: Path, tokenizer: Any) -> dict[str, str | int]:
    text = extract_text(path)
    return {
        "path": hub_style_path(path),
        "num_tokens": count_tokens(tokenizer, text),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count tokenizer tokens for each file in document, headers, and patterns."
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Hugging Face tokenizer name or local path (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: file_token_counts__<tokenizer>.csv under structural_doc)",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=list(DEFAULT_DIRS),
        help="Directories under structural_doc to scan (default: document headers patterns)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download tokenizer files if they are not cached locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = [(ROOT / d) if not Path(d).is_absolute() else Path(d) for d in args.dirs]
    output = args.output or default_output_path(args.tokenizer)

    print(f"loading tokenizer: {args.tokenizer}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
    )

    files = collect_files(input_dirs)
    print(f"counting {len(files)} file(s)", file=sys.stderr)

    rows = [build_row(path, tokenizer) for path in files]
    fieldnames = ["path", "num_tokens"]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} row(s) to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
