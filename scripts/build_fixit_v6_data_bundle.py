#!/usr/bin/env python3
"""Build a relocatable Fixit-v6 source-data archive from a legacy data tree."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path

PATH_COLUMNS = (
    "test_path",
    "wrong_kernel_path",
    "wrong_log_path",
    "wrong_trajectory_path",
    "correct_kernel_path",
    "plan_path",
    "turn_csv",
)
COPIED_PATH_COLUMNS = tuple(column for column in PATH_COLUMNS if column != "test_path")
PAIRS_NAME = "fixit-v5-gemini-kernel-pairs.csv"
ARCHIVE_ROOT = Path("ptxbench-data")
MANIFEST_NAME = "fixit-v6-source-manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to(path: Path, root: Path, *, label: str) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is outside {root}: {path}") from exc


def portable_value(path: Path, *, data_root: Path, mini_root: Path) -> str:
    try:
        relative = relative_to(path, data_root, label="data path")
    except ValueError:
        relative = relative_to(path, mini_root, label="source path")
        return f"${{MINI_PTX_AGENT_ROOT}}/{relative.as_posix()}"
    return f"${{PTXBENCH_DATA_ROOT}}/{relative.as_posix()}"


def mapped_data_path(path: Path, *, data_root: Path) -> Path:
    return relative_to(path, data_root, label="data path")


def required_data_files(rows: list[dict[str, str]]) -> set[Path]:
    required: set[Path] = set()
    for row_no, row in enumerate(rows, 2):
        for column in COPIED_PATH_COLUMNS:
            value = (row.get(column) or "").strip()
            if not value:
                raise ValueError(f"row {row_no}: blank {column}")
            required.add(Path(value).expanduser())

        exp_dir = Path((row.get("exp_dir") or "").strip()).expanduser()
        trajectory_id = (row.get("trajectory_id") or "").strip()
        if not str(exp_dir) or not trajectory_id:
            raise ValueError(f"row {row_no}: blank exp_dir or trajectory_id")
        required.add(exp_dir / "trajectories" / f"{trajectory_id}.json")

        correct_kernel = Path(row["correct_kernel_path"]).expanduser()
        required.add(correct_kernel.parent / "record.json")
    return required


def portable_rows(
    rows: list[dict[str, str]], *, data_root: Path, mini_root: Path
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in rows:
        portable = dict(row)
        portable["exp_dir"] = portable_value(
            Path(row["exp_dir"]).expanduser(), data_root=data_root, mini_root=mini_root
        )
        for column in PATH_COLUMNS:
            portable[column] = portable_value(
                Path(row[column]).expanduser(), data_root=data_root, mini_root=mini_root
            )
        result.append(portable)
    return result


def csv_bytes(fieldnames: list[str], rows: Iterable[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def add_bytes(archive: tarfile.TarFile, relative_path: Path, payload: bytes) -> None:
    archive_path = ARCHIVE_ROOT / relative_path
    archive.addfile(tar_info(archive_path.as_posix(), len(payload)), io.BytesIO(payload))


def add_file(archive: tarfile.TarFile, relative_path: Path, source: Path) -> None:
    archive_path = ARCHIVE_ROOT / relative_path
    with source.open("rb") as file:
        archive.addfile(tar_info(archive_path.as_posix(), source.stat().st_size), file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        required=True,
        help=f"Historical {PAIRS_NAME}.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing every AccRL-exps data path referenced by the CSV.",
    )
    parser.add_argument(
        "--mini-agent-root",
        type=Path,
        required=True,
        help="Root corresponding to historical AccRL test_path values.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs_csv = args.pairs_csv.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    mini_root = args.mini_agent_root.expanduser().resolve()
    if not pairs_csv.is_file():
        raise SystemExit(f"missing pairs CSV: {pairs_csv}")

    with pairs_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required_columns = {"exp_dir", "trajectory_id", *PATH_COLUMNS}
    missing_columns = required_columns.difference(fieldnames)
    if missing_columns:
        raise SystemExit(
            f"{pairs_csv}: missing columns: {', '.join(sorted(missing_columns))}"
        )

    source_files = required_data_files(rows)
    missing_files = sorted(path for path in source_files if not path.is_file())
    if missing_files:
        preview = "\n".join(str(path) for path in missing_files[:20])
        raise SystemExit(
            f"Fixit-v6 source closure has {len(missing_files)} missing files:\n{preview}"
        )

    mapped_files: dict[Path, Path] = {}
    for source in source_files:
        relative = mapped_data_path(source, data_root=data_root)
        prior = mapped_files.get(relative)
        if prior is not None and prior.resolve() != source.resolve():
            raise SystemExit(f"two source files map to {relative}: {prior} and {source}")
        mapped_files[relative] = source

    project_relative = mapped_data_path(pairs_csv.parent, data_root=data_root)
    portable_pairs_path = project_relative / PAIRS_NAME
    portable_pairs = csv_bytes(
        fieldnames,
        portable_rows(rows, data_root=data_root, mini_root=mini_root),
    )
    if portable_pairs_path in mapped_files:
        raise SystemExit(f"generated pairs CSV collides with source file: {portable_pairs_path}")

    file_records = [
        {
            "path": relative.as_posix(),
            "sha256": sha256_file(source),
            "size": source.stat().st_size,
        }
        for relative, source in sorted(mapped_files.items())
    ]
    file_records.append(
        {
            "path": portable_pairs_path.as_posix(),
            "sha256": sha256_bytes(portable_pairs),
            "size": len(portable_pairs),
        }
    )
    manifest = {
        "format_version": 1,
        "experiment": "fixit-v6",
        "archive_root": ARCHIVE_ROOT.as_posix(),
        "pairs_csv": portable_pairs_path.as_posix(),
        "source_pairs_sha256": sha256_file(pairs_csv),
        "rows": len(rows),
        "files": sorted(file_records, key=lambda record: record["path"]),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_output, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw_output, mtime=0
    ) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
        for relative, source in sorted(mapped_files.items()):
            add_file(archive, relative, source)
        add_bytes(archive, portable_pairs_path, portable_pairs)
        add_bytes(archive, Path(MANIFEST_NAME), manifest_bytes)

    print(
        f"wrote {len(mapped_files)} source files and {len(rows)} portable pairs "
        f"({output.stat().st_size} bytes) -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
