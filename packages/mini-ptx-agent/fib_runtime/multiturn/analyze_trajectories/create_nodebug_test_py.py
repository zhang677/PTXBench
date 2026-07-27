#!/usr/bin/env python3
"""Generate nodebug test.py files from template_compile_measure_cuda_nodebug.txt."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


DEFAULT_CSV = Path(__file__).with_name("nodebug_20260624_0939_problems.csv")
DEFINITION_RE = re.compile(r'^DEFINITION_NAME\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
WORKLOAD_RE = re.compile(r'^WORKLOAD_UUID\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Problem manifest CSV. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output test.py files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print intended writes without writing files.",
    )
    return parser.parse_args()


def require_columns(row: dict[str, str], row_number: int) -> None:
    required = {
        "definition",
        "workload_uuid",
        "source_test_path",
        "template_path",
        "output_test_path",
    }
    missing = sorted(key for key in required if not row.get(key))
    if missing:
        raise ValueError(f"row {row_number}: missing required columns: {', '.join(missing)}")


def constants_from_test(path: Path) -> tuple[str, str]:
    text = path.read_text()
    definition_match = DEFINITION_RE.search(text)
    workload_match = WORKLOAD_RE.search(text)
    if not definition_match or not workload_match:
        raise ValueError(f"{path}: could not find DEFINITION_NAME and WORKLOAD_UUID")
    return definition_match.group(1), workload_match.group(1)


def render_template(template: str, definition: str, workload_uuid: str) -> str:
    rendered = template.replace("<definition_name>", definition)
    rendered = rendered.replace("<workload_uuid>", workload_uuid)
    if "<definition_name>" in rendered or "<workload_uuid>" in rendered:
        raise ValueError("template placeholders were not fully replaced")
    return rendered


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"missing CSV: {args.csv}", file=sys.stderr)
        return 2

    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"empty CSV: {args.csv}", file=sys.stderr)
        return 2

    wrote = 0
    for row_number, row in enumerate(rows, start=2):
        require_columns(row, row_number)

        definition = row["definition"]
        workload_uuid = row["workload_uuid"]
        source_test_path = Path(row["source_test_path"])
        template_path = Path(row["template_path"])
        output_test_path = Path(row["output_test_path"])

        if not source_test_path.exists():
            raise FileNotFoundError(f"row {row_number}: missing source test: {source_test_path}")
        if not template_path.exists():
            raise FileNotFoundError(f"row {row_number}: missing template: {template_path}")

        source_definition, source_workload_uuid = constants_from_test(source_test_path)
        if (source_definition, source_workload_uuid) != (definition, workload_uuid):
            raise ValueError(
                f"row {row_number}: CSV constants do not match {source_test_path}: "
                f"{definition}/{workload_uuid} != {source_definition}/{source_workload_uuid}"
            )

        rendered = render_template(template_path.read_text(), definition, workload_uuid)

        if args.dry_run:
            print(f"would write {output_test_path}")
            continue

        if output_test_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_test_path} exists; pass --overwrite to replace it")

        output_test_path.parent.mkdir(parents=True, exist_ok=True)
        output_test_path.write_text(rendered)
        wrote += 1
        print(f"wrote {output_test_path}")

    if args.dry_run:
        print(f"validated {len(rows)} rows")
    else:
        print(f"wrote {wrote} nodebug test.py files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
