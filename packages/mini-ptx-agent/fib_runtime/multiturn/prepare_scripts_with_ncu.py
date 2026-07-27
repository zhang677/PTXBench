"""Render the per-problem compile/measure scripts from the shared CUDA templates.

Reads `prepare_config.csv` (columns: subdir, definition_id, workload_uuid) and
writes two scripts into each subdir for every row:

  <subdir>/<definition_id>_<workload_uuid>.py
      - from template_compile_measure_cuda.txt (compile + memcheck + /evaluate)
  <subdir>/ncu_<definition_id>_<workload_uuid>.py
      - from template_compile_measure_profile_cuda.txt (adds /profile via NCU)

Templates contain literal placeholders `<definition_name>` and `<workload_uuid>`
that are substituted with the CSV row's values.

Usage:
    python prepare_scripts.py
    python prepare_scripts.py --config prepare_config.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "prepare_config.csv"
PLAIN_TEMPLATE = HERE / "template_compile_measure_cuda.txt"
NCU_TEMPLATE = HERE / "template_compile_measure_profile_cuda.txt"

DEF_PLACEHOLDER = "<definition_name>"
UUID_PLACEHOLDER = "<workload_uuid>"


def render(template_text: str, definition_id: str, workload_uuid: str) -> str:
    return template_text.replace(DEF_PLACEHOLDER, definition_id).replace(
        UUID_PLACEHOLDER, workload_uuid
    )


def load_rows(config_path: Path) -> list[dict]:
    with config_path.open() as f:
        reader = csv.DictReader(f)
        required = {"subdir", "definition_id", "workload_uuid"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{config_path}: missing required columns: {sorted(missing)}"
            )
        rows = []
        for i, row in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in row.items()}
            if not row["subdir"]:
                continue
            for col in ("subdir", "definition_id", "workload_uuid"):
                if not row[col]:
                    raise SystemExit(
                        f"{config_path}:{i}: empty value for required column '{col}'"
                    )
            rows.append(row)
    return rows


def write_if_changed(path: Path, content: str, *, dry_run: bool) -> str:
    if path.exists() and path.read_text() == content:
        return "unchanged"
    if dry_run:
        return "would-write"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return "wrote"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without modifying any files.",
    )
    args = parser.parse_args()

    plain_tpl = PLAIN_TEMPLATE.read_text()
    ncu_tpl = NCU_TEMPLATE.read_text()

    rows = load_rows(args.config)
    if not rows:
        print(f"No rows found in {args.config}", file=sys.stderr)
        return 1

    for row in rows:
        subdir = HERE / row["subdir"]
        if not subdir.is_dir():
            raise SystemExit(f"Subdir does not exist: {subdir}")

        def_id = row["definition_id"]
        uuid = row["workload_uuid"]
        stem = f"{def_id}_{uuid}"

        plain_path = subdir / f"{stem}.py"
        ncu_path = subdir / f"ncu_{stem}.py"

        plain_status = write_if_changed(
            plain_path, render(plain_tpl, def_id, uuid), dry_run=args.dry_run
        )
        ncu_status = write_if_changed(
            ncu_path, render(ncu_tpl, def_id, uuid), dry_run=args.dry_run
        )

        print(f"[{plain_status:>11}] {plain_path.relative_to(HERE)}")
        print(f"[{ncu_status:>11}] {ncu_path.relative_to(HERE)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
