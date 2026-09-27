from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_INSTS = {
    "arch=compute_90a,code=sm_90a": {"cp.async.bulk.tensor", "wgmma."},
    "arch=compute_100a,code=sm_100a": {"cp.async.bulk.tensor", "tcgen05"},
}


def compile_source(
    source: str,
    *,
    nvcc_gencode: str,
    tvm_ffi_dir: str | None = None,
) -> dict[str, Any]:
    """Compile untrusted candidate source inside the controlled endpoint."""
    with tempfile.TemporaryDirectory(prefix="ptxbench-compile-") as directory:
        kernel_path = Path(directory) / "kernel.cu"
        kernel_path.write_text(source)
        return compile_kernel(
            kernel_path,
            nvcc_gencode=nvcc_gencode,
            tvm_ffi_dir=tvm_ffi_dir,
        )


def compile_kernel(
    kernel_path: Path,
    *,
    nvcc_gencode: str,
    tvm_ffi_dir: str | None = None,
) -> dict[str, Any]:
    tvm_ffi_dir = tvm_ffi_dir or os.environ.get(
        "PTXBENCH_TVM_FFI_DIR", "/usr/local/lib/python3.12/dist-packages/tvm_ffi"
    )
    workdir = kernel_path.resolve().parent
    for artifact in (workdir / "kernel.so", workdir / "kernel.ptx"):
        artifact.unlink(missing_ok=True)
    command = [
        "nvcc",
        "-shared",
        "-O3",
        "-gencode",
        nvcc_gencode,
        kernel_path.name,
        "-lineinfo",
        "--ptxas-options=-v",
        "-Xcompiler",
        "-fPIC,-fvisibility=hidden",
        "-lcuda",
        f"-I{tvm_ffi_dir}/include",
        "-std=c++17",
        f"-L{tvm_ffi_dir}/lib",
        "-ltvm_ffi",
        "-o",
        "kernel.so",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "output": str(exc),
            "isa_warning": "",
            "command": command,
            "infrastructure_error": True,
        }
    output = result.stdout or ""
    if result.returncode != 0:
        return {
            "ok": False,
            "output": output,
            "isa_warning": "",
            "command": command,
            "infrastructure_error": False,
        }
    warning = _instruction_warning(kernel_path, nvcc_gencode, tvm_ffi_dir)
    return {
        "ok": True,
        "output": output,
        "isa_warning": warning,
        "command": command,
        "infrastructure_error": False,
    }


def _instruction_warning(kernel_path: Path, nvcc_gencode: str, tvm_ffi_dir: str) -> str:
    if nvcc_gencode not in REQUIRED_INSTS:
        return ""
    ptx_gencode = nvcc_gencode.replace("code=sm_", "code=compute_")
    ptx_path = kernel_path.resolve().parent / "kernel.ptx"
    result = subprocess.run(
        [
            "nvcc",
            "--ptx",
            "-O3",
            "-gencode",
            ptx_gencode,
            kernel_path.name,
            "-lineinfo",
            f"-I{tvm_ffi_dir}/include",
            "-std=c++17",
            "-o",
            ptx_path.name,
        ],
        cwd=kernel_path.resolve().parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        return ""
    text = ptx_path.read_text(errors="replace")
    missing = [instruction for instruction in REQUIRED_INSTS[nvcc_gencode] if instruction not in text]
    if not missing:
        return ""
    return (
        "INSTRUCTION WARNING: the kernel might not be fully exercising the specialized "
        "hardware units. Please try these instructions if they can accelerate the kernel: "
        f"{REQUIRED_INSTS[nvcc_gencode]}."
    )
