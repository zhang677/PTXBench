#!/usr/bin/env python3
"""Build a deterministic, relocatable KernelGen source-data archive."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RUNS = ROOT / "experiments" / "kernelgen" / "source-runs.csv"
COLLECTOR = (
    ROOT
    / "packages"
    / "mini-ptx-agent"
    / "fib_runtime"
    / "multiturn"
    / "collect_kernels"
    / "collect_correct_kernels.py"
)
ENRICHER = (
    ROOT
    / "packages"
    / "mini-ptx-agent"
    / "fib_runtime"
    / "multiturn"
    / "task_to_correct_kernels"
    / "enrich_correct_kernels_for_reasoning.py"
)
ARCHIVE_ROOT = Path("ptxbench-data")
MANIFEST_NAME = "kernelgen-source-manifest.json"
EXPECTED_SOURCE_RUNS = 12
EXPECTED_SELECTED_ROWS = 521


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


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def portable_value(path: Path, *, data_root: Path, mini_root: Path) -> str:
    try:
        relative = relative_to(path, data_root, label="data path")
    except ValueError:
        relative = relative_to(path, mini_root, label="source path")
        return f"${{MINI_PTX_AGENT_ROOT}}/{relative.as_posix()}"
    return f"${{PTXBENCH_DATA_ROOT}}/{relative.as_posix()}"


def resolve_token_path(value: str, *, data_root: Path, mini_root: Path) -> Path:
    return Path(
        value.replace("${PTXBENCH_DATA_ROOT}", str(data_root)).replace(
            "${MINI_PTX_AGENT_ROOT}", str(mini_root)
        )
    ).expanduser()


def portable_csv(
    path: Path, *, data_root: Path, mini_root: Path
) -> bytes:
    fieldnames, rows = read_csv(path)
    path_columns = (
        "exp_dir",
        "test_path",
        "kernel_path",
        "correct_kernel_path",
        "turn_csv",
        "trajectory_path",
    )
    for row in rows:
        for column in path_columns:
            if not row.get(column):
                continue
            row[column] = portable_value(
                Path(row[column]).expanduser(),
                data_root=data_root,
                mini_root=mini_root,
            )
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
    parser.add_argument("--source-runs-csv", type=Path, default=DEFAULT_SOURCE_RUNS)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root substituted for PTXBENCH_DATA_ROOT in source-runs.csv.",
    )
    parser.add_argument(
        "--mini-agent-root",
        type=Path,
        required=True,
        help="Root substituted for MINI_PTX_AGENT_ROOT in source-runs.csv.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def generated_source_rows(
    source_runs_csv: Path, *, data_root: Path, mini_root: Path, output_dir: Path
) -> tuple[Path, Path]:
    correct = output_dir / "correct-kernels.csv"
    enriched = output_dir / "correct-kernels.enriched.csv"
    env = {
        **os.environ,
        "PTXBENCH_DATA_ROOT": str(data_root),
        "MINI_PTX_AGENT_ROOT": str(mini_root),
    }
    subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            str(source_runs_csv),
            "--min-speedup",
            "0",
            "--output",
            str(correct),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ENRICHER),
            "--input-csv",
            str(correct),
            "--output-csv",
            str(enriched),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return correct, enriched


def main() -> int:
    args = parse_args()
    source_runs_csv = args.source_runs_csv.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    mini_root = args.mini_agent_root.expanduser().resolve()
    if not source_runs_csv.is_file():
        raise SystemExit(f"missing source-run registry: {source_runs_csv}")

    _, source_runs = read_csv(source_runs_csv)
    if len(source_runs) != EXPECTED_SOURCE_RUNS:
        raise SystemExit(
            f"{source_runs_csv}: expected {EXPECTED_SOURCE_RUNS} runs, "
            f"found {len(source_runs)}"
        )

    with tempfile.TemporaryDirectory(prefix="ptxbench-kernelgen-bundle.") as temp:
        correct, enriched = generated_source_rows(
            source_runs_csv,
            data_root=data_root,
            mini_root=mini_root,
            output_dir=Path(temp),
        )
        _, enriched_rows = read_csv(enriched)
        if len(enriched_rows) != EXPECTED_SELECTED_ROWS:
            raise SystemExit(
                f"expected {EXPECTED_SELECTED_ROWS} enriched rows, "
                f"found {len(enriched_rows)}"
            )

        required_files: set[Path] = set()
        for row in source_runs:
            exp_dir = resolve_token_path(
                row["exp_dir"], data_root=data_root, mini_root=mini_root
            )
            required_files.add(exp_dir / "figures" / "turn_correctness_arch.csv")
            if (exp_dir / "plan.json").is_file():
                required_files.add(exp_dir / "plan.json")
            elif (exp_dir / "summary.json").is_file():
                required_files.add(exp_dir / "summary.json")
            else:
                raise SystemExit(f"missing plan.json or summary.json: {exp_dir}")

        for row in enriched_rows:
            required_files.add(Path(row["kernel_path"]).expanduser())
            required_files.add(Path(row["trajectory_path"]).expanduser())

        missing = sorted(path for path in required_files if not path.is_file())
        if missing:
            preview = "\n".join(str(path) for path in missing[:20])
            raise SystemExit(
                f"KernelGen source closure has {len(missing)} missing files:\n{preview}"
            )

        mapped_files: dict[Path, Path] = {}
        for source in required_files:
            relative = relative_to(source, data_root, label="data path")
            prior = mapped_files.get(relative)
            if prior is not None and prior.resolve() != source.resolve():
                raise SystemExit(
                    f"two source files map to {relative}: {prior} and {source}"
                )
            mapped_files[relative] = source

        project_relative = Path(
            "sft_experiments/mha-8def-single-turn-qwen36-27b-gemini-glm"
        )
        portable_correct = portable_csv(
            correct, data_root=data_root, mini_root=mini_root
        )
        portable_enriched = portable_csv(
            enriched, data_root=data_root, mini_root=mini_root
        )
        generated_files = {
            project_relative / "correct-kernels.csv": portable_correct,
            project_relative / "correct-kernels.enriched.csv": portable_enriched,
        }

        file_records = [
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(source),
                "size": source.stat().st_size,
            }
            for relative, source in sorted(mapped_files.items())
        ]
        file_records.extend(
            {
                "path": relative.as_posix(),
                "sha256": sha256_bytes(payload),
                "size": len(payload),
            }
            for relative, payload in sorted(generated_files.items())
        )
        manifest = {
            "format_version": 1,
            "experiment": "kernelgen",
            "archive_root": ARCHIVE_ROOT.as_posix(),
            "source_runs": len(source_runs),
            "selected_rows": len(enriched_rows),
            "source_runs_sha256": sha256_file(source_runs_csv),
            "generated_correct_kernels_sha256": sha256_file(correct),
            "generated_enriched_sha256": sha256_file(enriched),
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
            for relative, payload in sorted(generated_files.items()):
                add_bytes(archive, relative, payload)
            add_bytes(archive, Path(MANIFEST_NAME), manifest_bytes)

    print(
        f"wrote {len(mapped_files)} source files and "
        f"{EXPECTED_SELECTED_ROWS} portable rows "
        f"({output.stat().st_size} bytes) -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
