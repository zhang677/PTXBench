#!/usr/bin/env python3
"""Build a narrow PTXBench source archive from the experiment closures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = ROOT / "packages" / "mini-ptx-agent"
PROVENANCE_FILES = (
    ROOT / "experiments" / "fixit-v6" / "provenance.json",
    ROOT / "experiments" / "sft-v4" / "provenance.json",
)
TREE_ROOTS = (
    "configs",
    "docker",
    "experiments",
    "scripts",
    "tests",
    "packages/fibserve",
    "packages/mini-ptx-agent/mini_ptx_agent",
)
ROOT_FILES = (
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "README.md",
    "RELEASING.md",
    "data/README.md",
    "data/datasets/.gitkeep",
    "packages/mini-ptx-agent/LICENSE",
    "packages/mini-ptx-agent/README.md",
    "packages/mini-ptx-agent/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
)


def source_files() -> set[str]:
    process = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return {line for line in process.stdout.splitlines() if line}


def prompt_fragments(provenance: dict) -> set[str]:
    multiturn = MINI_ROOT / "fib_runtime" / "multiturn"
    hub = json.loads((multiturn / "prompt_configs" / "hub.json").read_text())
    fragments: set[str] = set()
    seen: set[str] = set()

    def visit(tag: str) -> None:
        if tag in seen:
            return
        seen.add(tag)
        if tag not in hub:
            raise ValueError(f"prompt tag {tag!r} is absent from prompt_configs/hub.json")
        for item in hub[tag]:
            if "/" in item:
                fragments.add(f"packages/mini-ptx-agent/fib_runtime/{item}")
            else:
                visit(item)

    for tag in provenance["required_prompt_tags"]:
        visit(tag)
    return fragments


def parent_init_files(path: str, tracked: set[str]) -> set[str]:
    parents: set[str] = set()
    current = Path(path).parent
    while current != Path("."):
        candidate = str(current / "__init__.py")
        if candidate in tracked:
            parents.add(candidate)
        current = current.parent
    return parents


def release_files(available: set[str]) -> list[str]:
    selected: set[str] = set(ROOT_FILES)
    for tree_root in TREE_ROOTS:
        prefix = tree_root.rstrip("/") + "/"
        selected.update(path for path in available if path.startswith(prefix))

    for provenance_path in PROVENANCE_FILES:
        provenance = json.loads(provenance_path.read_text())
        selected.update(provenance["required_source_files"])
        selected.update(prompt_fragments(provenance))
        if provenance_path.parent.name == "fixit-v6":
            selected.update(
                f"experiments/fixit-v6/{stage}"
                for stage in provenance["ordered_stages"]
            )

    for path in tuple(selected):
        selected.update(parent_init_files(path, available))

    missing = sorted(path for path in selected if path not in available)
    if missing:
        raise SystemExit("release closure contains missing files:\n" + "\n".join(missing))
    return sorted(selected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tar_info(name: str, size: int, *, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def add_file(archive: tarfile.TarFile, relative_path: str) -> None:
    source = ROOT / relative_path
    mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
    with source.open("rb") as file:
        archive.addfile(
            tar_info(f"PTXBench/{relative_path}", source.stat().st_size, mode=mode),
            file,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "ptxbench-source.tar.gz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = release_files(source_files())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = "".join(f"{sha256(ROOT / path)}  {path}\n" for path in files)
    with (
        args.output.open("wb") as raw_output,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for relative_path in files:
            add_file(archive, relative_path)
        payload = manifest.encode()
        archive.addfile(
            tar_info("PTXBench/RELEASE-MANIFEST.sha256", len(payload)),
            io.BytesIO(payload),
        )
    print(f"wrote {len(files)} source files -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
