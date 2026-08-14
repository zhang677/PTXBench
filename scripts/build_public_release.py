#!/usr/bin/env python3
"""Build a PTXBench source archive containing the runnable pipelines."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE_ROOTS = (
    "configs",
    "docker",
    "experiments",
    "scripts",
    "tests",
    "packages/fibserve",
    "packages/mini-ptx-agent/accrl",
    "packages/mini-ptx-agent/fib_runtime/multiturn/construct_eval_scripts",
    "packages/mini-ptx-agent/fib_runtime/multiturn/fix_kernels",
    "packages/mini-ptx-agent/fib_runtime/multiturn/prompt_configs",
    "packages/mini-ptx-agent/fib_runtime/multiturn/task_to_correct_kernels",
    "packages/mini-ptx-agent/mini_ptx_agent",
)
SOURCE_FILES = (
    "packages/mini-ptx-agent/benchmark/export_turn_correctness_arch.py",
    "packages/mini-ptx-agent/fib_runtime/mini_swe_agent_docker/envs/example.cu",
    "packages/mini-ptx-agent/fib_runtime/mini_swe_agent_docker/envs/triton_example.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0413-1611/gemm_n7168_k5120_94920358-01a8-4c5b-9209-3103fd490e94.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0413-1611/gemm_n7168_k5120_94920358-01a8-4c5b-9209-3103fd490e94_triton.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d128_bc38b351-d595-451b-9153-8e225702e53b.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d128_causal_6d2f67a7-225a-4af5-87d3-cbb99b496325.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d64_7d2575a0-bcc2-42a0-812f-6a7e9a57d97f.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d64_causal_b69f7675-568f-40f2-9a4b-8bbe374b4a59.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_38c3b07c-f006-5f5e-9860-ba214c805a6b.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_causal_c119b3f0-c051-5e96-9c2a-2268d992fe1a.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_f5645ae3-e24f-5534-9d30-c46b68a8ffea.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d64_causal_5799ea50-77aa-56cb-9f62-a4c1f5473770.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d64_d3bcb902-6a13-5ada-9251-fa841b10cd0b.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/analyze_kernel_per_turn.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/analyze_pattern.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/build_doc_v2.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/create_triton_test.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/collect_kernels/collect_correct_kernels.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/collect_notes/note_feedback.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/common.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/launcher_utils.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/resume_utils.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/run_parallel_v2.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/run_v2.py",
    "packages/mini-ptx-agent/fib_runtime/multiturn/template_compile_measure_triton.txt",
    "packages/mini-ptx-agent/fib_runtime/multiturn/user_template.txt",
    "packages/mini-ptx-agent/fib_runtime/multiturn/user_template_triton.txt",
    "packages/mini-ptx-agent/fib_runtime/structural_doc/document/triton_knowledge_sm90_plus.md",
    "packages/mini-ptx-agent/fib_runtime/structural_doc/document/triton_knowledge_sm100_plus.md",
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
    return {
        line
        for line in process.stdout.splitlines()
        if line and (ROOT / line).is_file()
    }


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
    selected: set[str] = set(ROOT_FILES) | set(SOURCE_FILES)
    for tree_root in TREE_ROOTS:
        prefix = tree_root.rstrip("/") + "/"
        selected.update(path for path in available if path.startswith(prefix))

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
