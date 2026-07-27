#!/usr/bin/env python3
"""Prepare safetensors inputs + definitions + workloads for fp8_gemm_nt_1d2d.

DeepGEMM `fp8_gemm_nt` (SM90, 1D2D recipe): per-token A SF, per-128x128-block B SF.
Source tensors are sampled bf16 randn; FP8 inputs are produced by DeepGEMM's own
`per_token_cast_to_fp8` / `per_block_cast_to_fp8` (see DeepGEMM/tests/generators.py
:265-288 generate_normal). The bf16 source is also stored alongside the FP8 inputs
so a downstream solution can re-quantize if it wants to test 1D1D vs 1D2D.

"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import torch
from safetensors.torch import save_file

import deep_gemm
from deep_gemm.utils.math import per_block_cast_to_fp8, per_token_cast_to_fp8

BASE_SEED = 20260427
GRAN_K = 128

NK_pairs = [
    (24576, 1536),
    (32768, 512),
    (7168, 16384),
    (4096, 7168),
    (7168, 2048),
]
M = [4096]


def definition_name(n_val: int, k_val: int) -> str:
    return f"fp8_gemm_nt_1d2d_n{n_val}_k{k_val}"


def workload_path(n_val: int, k_val: int) -> Path:
    return Path(f"workloads/gemm/{definition_name(n_val, k_val)}.jsonl")


def definition_path(n_val: int, k_val: int) -> Path:
    return Path(f"definitions/gemm/{definition_name(n_val, k_val)}.json")


def blob_dir(n_val: int, k_val: int) -> Path:
    return Path(f"blob/workloads/gemm/{definition_name(n_val, k_val)}")


def check_dims(n_val: int, k_val: int) -> None:
    if n_val % GRAN_K != 0:
        raise ValueError(f"N must be divisible by {GRAN_K}, got {n_val}")
    if k_val % GRAN_K != 0:
        raise ValueError(f"K must be divisible by {GRAN_K}, got {k_val}")


REFERENCE_SOURCE = """\
import torch

import deep_gemm


def run(A, SF_A, B, SF_B):
    D = torch.empty(A.shape[0], B.shape[0], dtype=torch.bfloat16, device=A.device)
    deep_gemm.fp8_gemm_nt(a=(A, SF_A), b=(B, SF_B), d=D)
    return D
"""


def build_definition_dict(n_val: int, k_val: int) -> dict:
    check_dims(n_val, k_val)
    definition = definition_name(n_val, k_val)
    return {
        "name": definition,
        "description": f"""
SM90 FP8 dense GEMM, 1D2D recipe (per-token A SF, per-128x128-block B SF).
NT layout: A and B both K-major. N={n_val}, K={k_val} fixed; M is the variable axis. Output D is bf16.
The pseudo-code below shows the computation.
```python
# A: [M, K] e4m3,    SF_A: [M, K/128] f32          (per-token-A)
# B: [N, K] e4m3,    SF_B: [N/128, K/128] f32      (per-128x128-block-B)
# C, D : [M, N]                                         (out_dtype = bf16 or fp32)
for i in range(M):
    for j in range(N):
        acc = float(C[i, j]) if accumulate else 0.0
        for kb in range(K // 128):
            sa = SF_A[i,        kb]
            sb = SF_B[j // 128, kb]
            scale = sa * sb # sa * sb should be calculated first
            partial = 0.0
            for kk in range(128):
                k = kb * 128 + kk
                partial += float(A[i, k]) * float(B[j, k])  # fp32 accumulate
            acc += partial * scale   # SF folded per K-block
        D[i, j] = cast_to(acc, out_dtype)
```
""",
        "op_type": "gemm",
        "tags": [
            "status:reference",
            "source:deepgemm",
            "ref:fp8_gemm_nt",
            "recipe:1d2d",
            "arch:sm90",
            "layout:nt",
        ],
        "axes": {
            "m": {"type": "var", "description": "M dimension"},
            "n": {"type": "const", "value": n_val},
            "k": {"type": "const", "value": k_val},
            "num_nblock": {"type": "const", "value": n_val // GRAN_K},
            "num_kblock": {"type": "const", "value": k_val // GRAN_K},
        },
        "inputs": {
            "A": {"dtype": "float8_e4m3fn", "shape": ["m", "k"]},
            "SF_A": {"dtype": "float32", "shape": ["m", "num_kblock"]},
            "B": {"dtype": "float8_e4m3fn", "shape": ["n", "k"]},
            "SF_B": {"dtype": "float32", "shape": ["num_nblock", "num_kblock"]},
        },
        "outputs": {
            "D": {"dtype": "bfloat16", "shape": ["m", "n"]},
        },
        "reference": REFERENCE_SOURCE,
    }


def write_definition(dataset_root: Path, n_val: int, k_val: int, overwrite: bool) -> None:
    out = dataset_root / definition_path(n_val, k_val)
    if out.exists() and not overwrite:
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(build_definition_dict(n_val, k_val), f, indent=2)
        f.write("\n")
    print(f"wrote {out}")


def build_workload_entry(m_val: int, n_val: int, k_val: int, blob_relpath: str) -> dict:
    check_dims(n_val, k_val)
    definition = definition_name(n_val, k_val)
    wl_uuid = str(uuid.uuid4())
    safetensor_path = f"./{blob_relpath}/{definition}_{wl_uuid.replace('-', '')}.safetensors"
    return {
        "definition": definition,
        "solution": None,
        "workload": {
            "uuid": wl_uuid,
            "axes": {
                "m": m_val,
                "n": n_val,
                "k": k_val,
                "num_nblock": n_val // GRAN_K,
                "num_kblock": k_val // GRAN_K,
            },
            "inputs": {
                "A": {"type": "safetensors", "path": safetensor_path, "tensor_key": "A"},
                "SF_A": {"type": "safetensors", "path": safetensor_path, "tensor_key": "SF_A"},
                "B": {"type": "safetensors", "path": safetensor_path, "tensor_key": "B"},
                "SF_B": {"type": "safetensors", "path": safetensor_path, "tensor_key": "SF_B"}
            },
            "outputs": {
                "D": {"type": "safetensors", "path": safetensor_path, "tensor_key": "D"}
            }
        },
        "evaluation": None,
    }


def output_path(dataset_root: Path, item: dict) -> Path:
    rel = item["workload"]["inputs"]["A"]["path"]
    if not rel.startswith("./"):
        raise ValueError(f"expected relative ./blob path, got {rel!r}")
    return dataset_root / rel[2:]


def generate_tensors(
    m_val: int, n_val: int, k_val: int, device: torch.device, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Sample bf16 randn source, then FP8-quantize via DeepGEMM's own casts.

    Mirrors `generate_normal` in DeepGEMM/tests/generators.py:265-288 for
    `kernel_type=1D2D`, `accumulate=False`, `out_dtype=bf16`.
    """
    check_dims(n_val, k_val)
    a_bf16 = torch.empty((m_val, k_val), dtype=torch.bfloat16, device=device).normal_(
        0.0, 1.0, generator=generator
    )
    b_bf16 = torch.empty((n_val, k_val), dtype=torch.bfloat16, device=device).normal_(
        0.0, 1.0, generator=generator
    )

    a_fp8, sf_a = per_token_cast_to_fp8(a_bf16, use_ue8m0=False, gran_k=GRAN_K)
    b_fp8, sf_b = per_block_cast_to_fp8(b_bf16, use_ue8m0=False, gran_k=GRAN_K)

    assert a_fp8.shape == (m_val, k_val) and a_fp8.dtype == torch.float8_e4m3fn
    assert sf_a.shape == (m_val, k_val // GRAN_K) and sf_a.dtype == torch.float32
    assert b_fp8.shape == (n_val, k_val) and b_fp8.dtype == torch.float8_e4m3fn
    assert sf_b.shape == (n_val // GRAN_K, k_val // GRAN_K) and sf_b.dtype == torch.float32

    return {
        "A": a_fp8,
        "SF_A": sf_a,
        "B": b_fp8,
        "SF_B": sf_b,
        "A_bf16": a_bf16,
        "B_bf16": b_bf16,
    }


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """DeepGEMM's cosine-distance metric (deep_gemm/testing/numeric.py:5)."""
    x = x.double()
    y = y.double()
    denom = (x * x + y * y).sum()
    if denom.item() == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def prepare_one(
    dataset_root: Path,
    item: dict,
    device: torch.device,
    check: bool,
    overwrite: bool,
    tolerance: float,
) -> None:
    """Generate inputs, run the reference once for D_ref, time it, save blob."""
    axes = item["workload"]["axes"]
    m_val = int(axes["m"])
    n_val = int(axes["n"])
    k_val = int(axes["k"])
    check_dims(n_val, k_val)
    out_path = output_path(dataset_root, item)

    seed = BASE_SEED + m_val * 1_000_003 + n_val * 1_009 + k_val
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    tensors = generate_tensors(m_val, n_val, k_val, device, generator)
    a_bf16 = tensors["A_bf16"]
    b_bf16 = tensors["B_bf16"]

    # Always run DeepGEMM once to obtain D_ref (it goes into the safetensors blob and
    # will be loaded by the evaluator as the precomputed reference output).
    a_pair = (tensors["A"], tensors["SF_A"])
    b_pair = (tensors["B"], tensors["SF_B"])
    d_ref = torch.empty((m_val, n_val), dtype=torch.bfloat16, device=device)
    deep_gemm.fp8_gemm_nt(a=a_pair, b=b_pair, d=d_ref)
    torch.cuda.synchronize()

    if check:
        ref_bf16 = (a_bf16.float() @ b_bf16.float().t()).to(torch.bfloat16)
        diff = calc_diff(d_ref, ref_bf16)
        ok = diff < tolerance
        print(
            f"check n={n_val} k={k_val} m={m_val}: "
            f"calc_diff={diff:.6g} tolerance={tolerance:.0e} ok={ok}"
        )
        if not ok:
            raise AssertionError(f"calc_diff {diff:.6g} >= tolerance {tolerance:.0e}")

    # Time the actual REFERENCE_SOURCE.run() path — this is what the evaluator's
    # speedup metric is measured against.
    ref_ns: dict = {}
    exec(REFERENCE_SOURCE, ref_ns)
    ref_run = ref_ns["run"]

    n_warmup, n_iters = 5, 50
    for _ in range(n_warmup):
        ref_run(tensors["A"], tensors["SF_A"], tensors["B"], tensors["SF_B"])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        ref_run(tensors["A"], tensors["SF_A"], tensors["B"], tensors["SF_B"])
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / n_iters
    latency_s = elapsed_ms * 1e-3
    flops = 2.0 * m_val * n_val * k_val
    tflops = flops / latency_s / 1e12
    bytes_moved = (
        tensors["A"].numel() * tensors["A"].element_size()
        + tensors["SF_A"].numel() * tensors["SF_A"].element_size()
        + tensors["B"].numel() * tensors["B"].element_size()
        + tensors["SF_B"].numel() * tensors["SF_B"].element_size()
        + d_ref.numel() * d_ref.element_size()
    )
    gbs = bytes_moved / latency_s / 1e9
    print(
        f"bench n={n_val} k={k_val} m={m_val}: {elapsed_ms} ms | "
        f" {flops / 1e12} TFLOPS | {tflops:6.1f} TFLOPs/s | {gbs:6.1f} GB/s"
    )

    if out_path.exists() and not overwrite:
        print(f"skip existing safetensors blob {out_path} (latency still re-measured)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_blob = {
        "A": tensors["A"].detach().cpu().contiguous(),
        "SF_A": tensors["SF_A"].detach().cpu().contiguous(),
        "B": tensors["B"].detach().cpu().contiguous(),
        "SF_B": tensors["SF_B"].detach().cpu().contiguous(),
        "A_bf16": tensors["A_bf16"].detach().cpu().contiguous(),
        "B_bf16": tensors["B_bf16"].detach().cpu().contiguous(),
        "D": d_ref.detach().cpu().contiguous(),
    }
    save_file(save_blob, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MiB)")


def write_workloads(
    dataset_root: Path, n_val: int, k_val: int, items: list[dict], overwrite: bool
) -> None:
    out = dataset_root / workload_path(n_val, k_val)
    if out.exists() and not overwrite:
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"wrote {out} ({len(items)} workloads)")


def iter_or_build_workloads(
    dataset_root: Path, n_val: int, k_val: int, overwrite: bool
) -> list[dict]:
    """Reuse an existing workload jsonl (so safetensor paths/uuids stay stable
    across re-runs), otherwise build a fresh one from the M list."""
    out = dataset_root / workload_path(n_val, k_val)
    if out.exists() and not overwrite:
        items: list[dict] = []
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        print(f"reusing {out} ({len(items)} workloads)")
        return items
    items = [
        build_workload_entry(m_val, n_val, k_val, str(blob_dir(n_val, k_val)))
        for m_val in M
    ]
    write_workloads(dataset_root, n_val, k_val, items, overwrite=True)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true",
                        help="compare DeepGEMM FP8 GEMM against bf16-fp32 reference via calc_diff")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="calc_diff tolerance (DeepGEMM's QuantConfig.max_diff() = 0.001 for FP8xFP8)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: DeepGEMM FP8 kernels are SM90+ device kernels")

    dataset_root = args.dataset_root.resolve()
    device = torch.device("cuda")
    torch.manual_seed(BASE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"selected {len(NK_pairs)} NK pairs and {len(M)} M values")
    for n_val, k_val in NK_pairs:
        write_definition(dataset_root, n_val, k_val, args.overwrite)

        items = iter_or_build_workloads(dataset_root, n_val, k_val, args.overwrite)
        print(f"selected {len(items)} workloads for n={n_val} k={k_val}")

        for item in items:
            prepare_one(
                dataset_root,
                item,
                device,
                args.check,
                args.overwrite,
                args.tolerance,
            )

        write_workloads(dataset_root, n_val, k_val, items, overwrite=True)


if __name__ == "__main__":
    main()
