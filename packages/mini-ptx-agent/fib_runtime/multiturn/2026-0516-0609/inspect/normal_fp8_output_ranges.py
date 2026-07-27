#!/usr/bin/env python3
"""Generate non-spiky normal FP8 MHA inputs and inspect O/LSE output ranges.

This mirrors scripts/prepare_fp8_mha.py except Q/K/V are sampled from N(0, 1)
only, with no Bernoulli spike component. Outputs are produced by the same cuDNN
sdpa_fp8 reference helper and summarized without writing the large tensors.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


PREPARE_SCRIPT = Path(
    "/home/ubuntu/AccRL/fib_runtime/multiturn/2026-0516-0609/scripts/prepare_fp8_mha.py"
)
B, H = 4, 48
D_VALUES = (64, 128, 256)
S_VALUES = (512, 1024, 2048, 4096, 8192, 16384)
BASE_SEED = 20260529


@dataclass
class TensorStats:
    definition: str
    causal: bool
    S: int
    D: int
    tensor: str
    dtype: str
    shape: list[int]
    numel: int
    finite_count: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    zero_count: int
    min: float
    max: float
    mean: float
    std: float
    absmax: float
    absmax_count: int
    q001: float
    q01: float
    q50: float
    q99: float
    q999: float
    descale_q: float
    descale_k: float
    descale_v: float


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_fp8_mha", PREPARE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {PREPARE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normal_bf16(shape, device, generator):
    return torch.empty(shape, dtype=torch.float32, device=device).normal_(
        0.0, 1.0, generator=generator
    ).to(torch.bfloat16)


def even_indices(n: int, count: int, device: torch.device) -> torch.Tensor:
    count = min(n, count)
    idx = torch.arange(count, device=device, dtype=torch.int64)
    if count > 1:
        idx = idx * (n - 1) // (count - 1)
    return idx


def collect_stats(
    definition: str,
    causal: bool,
    s_val: int,
    d_val: int,
    name: str,
    tensor: torch.Tensor,
    descales: tuple[float, float, float],
    sample_values: int,
) -> TensorStats:
    values = tensor.float().reshape(-1)
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    abs_values = finite.abs()
    absmax = float(abs_values.max().item())
    sample = finite.index_select(0, even_indices(int(finite.numel()), sample_values, finite.device))
    quantiles = torch.quantile(
        sample,
        torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], device=sample.device),
    )
    return TensorStats(
        definition=definition,
        causal=causal,
        S=s_val,
        D=d_val,
        tensor=name,
        dtype=str(tensor.dtype).replace("torch.", ""),
        shape=[int(dim) for dim in tensor.shape],
        numel=int(values.numel()),
        finite_count=int(finite.numel()),
        nan_count=int(torch.count_nonzero(torch.isnan(values)).item()),
        posinf_count=int(torch.count_nonzero(values == float("inf")).item()),
        neginf_count=int(torch.count_nonzero(values == float("-inf")).item()),
        zero_count=int(torch.count_nonzero(values == 0).item()),
        min=float(finite.min().item()),
        max=float(finite.max().item()),
        mean=float(finite.mean().item()),
        std=float(finite.std(unbiased=False).item()),
        absmax=absmax,
        absmax_count=int(torch.count_nonzero(abs_values == absmax).item()),
        q001=float(quantiles[0].item()),
        q01=float(quantiles[1].item()),
        q50=float(quantiles[2].item()),
        q99=float(quantiles[3].item()),
        q999=float(quantiles[4].item()),
        descale_q=descales[0],
        descale_k=descales[1],
        descale_v=descales[2],
    )


def definition_name(d_val: int, causal: bool) -> str:
    return f"normal_fp8_mha_with_lse_d{d_val}{'_causal' if causal else ''}"


def run_case(prepare, device, d_val: int, s_val: int, causal: bool, sample_values: int):
    seed = BASE_SEED + s_val + 100_000 * d_val + 10_000_000 * int(causal)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (B, H, s_val, d_val)

    q_bf16 = normal_bf16(shape, device, generator)
    k_bf16 = normal_bf16(shape, device, generator)
    v_bf16 = normal_bf16(shape, device, generator)
    q_fp8, dq = prepare.fp8_quantize(q_bf16)
    k_fp8, dk = prepare.fp8_quantize(k_bf16)
    v_fp8, dv = prepare.fp8_quantize(v_bf16)

    descale_q = torch.tensor([[[[dq]]]], dtype=torch.float32, device=device)
    descale_k = torch.tensor([[[[dk]]]], dtype=torch.float32, device=device)
    descale_v = torch.tensor([[[[dv]]]], dtype=torch.float32, device=device)
    one = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=device)

    out, lse = prepare.cudnn_fp8_forward(
        q_fp8,
        k_fp8,
        v_fp8,
        descale_q,
        descale_k,
        descale_v,
        one,
        one,
        one,
        causal,
        device,
    )
    torch.cuda.synchronize(device)
    descales = (dq, dk, dv)
    rows = [
        collect_stats(definition_name(d_val, causal), causal, s_val, d_val, "O", out, descales, sample_values),
        collect_stats(definition_name(d_val, causal), causal, s_val, d_val, "LSE", lse, descales, sample_values),
    ]
    del q_bf16, k_bf16, v_bf16, q_fp8, k_fp8, v_fp8, out, lse
    torch.cuda.empty_cache()
    return rows


def summarize(rows: list[TensorStats]) -> None:
    print(
        "definition\ttensor\tcases\tglobal_min\tglobal_max\tmax_abs\t"
        "max_abs_count\tmax_q999\tmean_range\tstd_range\tnonfinite"
    )
    for d_val in D_VALUES:
        for causal in (False, True):
            definition = definition_name(d_val, causal)
            for tensor in ("O", "LSE"):
                group = [row for row in rows if row.definition == definition and row.tensor == tensor]
                nonfinite = sum(row.nan_count + row.posinf_count + row.neginf_count for row in group)
                max_abs_row = max(group, key=lambda row: row.absmax)
                print(
                    "\t".join(
                        [
                            definition,
                            tensor,
                            str(len(group)),
                            f"{min(row.min for row in group):.6g}",
                            f"{max(row.max for row in group):.6g}",
                            f"{max_abs_row.absmax:.6g}",
                            f"{max_abs_row.absmax_count}/{max_abs_row.numel}",
                            f"{max(row.q999 for row in group):.6g}",
                            f"{min(row.mean for row in group):.6g}..{max(row.mean for row in group):.6g}",
                            f"{min(row.std for row in group):.6g}..{max(row.std for row in group):.6g}",
                            str(nonfinite),
                        ]
                    )
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-values", type=int, default=500_000)
    parser.add_argument("--json", type=Path, default=Path("normal_fp8_output_ranges.json"))
    parser.add_argument("--csv", type=Path, default=Path("normal_fp8_output_ranges.csv"))
    parser.add_argument("--max-s", type=int, default=max(S_VALUES))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    prepare = load_prepare_module()

    rows: list[TensorStats] = []
    for causal in (False, True):
        for d_val in D_VALUES:
            for s_val in S_VALUES:
                if s_val > args.max_s:
                    continue
                print(f"running D={d_val} S={s_val} causal={causal}", flush=True)
                rows.extend(run_case(prepare, device, d_val, s_val, causal, args.sample_values))

    args.json.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n")
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    summarize(rows)
    print(f"Wrote {args.json}")
    print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
