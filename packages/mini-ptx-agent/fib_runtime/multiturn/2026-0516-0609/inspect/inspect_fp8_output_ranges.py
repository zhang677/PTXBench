#!/usr/bin/env python3
"""Inspect stored output ranges for FP8 MHA+LSE safetensors workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_WORKLOAD_ROOT = Path("/home/ubuntu/accrl-training/blob/workloads/attention")
FP8_DEFS = (
    "fp8_mha_with_lse_d64",
    "fp8_mha_with_lse_d64_causal",
    "fp8_mha_with_lse_d128",
    "fp8_mha_with_lse_d128_causal",
    "fp8_mha_with_lse_d256",
    "fp8_mha_with_lse_d256_causal",
)
OUTPUT_NAMES = ("O", "LSE")


@dataclass
class TensorStats:
    definition: str
    file: str
    tensor: str
    dtype: str
    shape: list[int]
    numel: int
    finite_count: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    zero_count: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    absmax: float | None
    absmax_count: int
    q001: float | None
    q01: float | None
    q50: float | None
    q99: float | None
    q999: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--sample-values", type=int, default=500_000)
    parser.add_argument("--json", type=Path, default=Path("fp8_output_ranges.json"))
    parser.add_argument("--csv", type=Path, default=Path("fp8_output_ranges.csv"))
    return parser.parse_args()


def even_indices(n: int, count: int, device: torch.device) -> torch.Tensor:
    count = min(n, count)
    idx = torch.arange(count, device=device, dtype=torch.int64)
    if count > 1:
        idx = idx * (n - 1) // (count - 1)
    return idx


def tensor_to_float_flat(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bool:
        return tensor.to(torch.float32).reshape(-1)
    if torch.is_floating_point(tensor):
        return tensor.to(torch.float32).reshape(-1)
    return tensor.to(torch.float32).reshape(-1)


def collect_stats(definition: str, path: Path, name: str, sample_values: int) -> TensorStats:
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
        dtype = str(tensor.dtype).replace("torch.", "")
        shape = [int(dim) for dim in tensor.shape]
        flat = tensor_to_float_flat(tensor)

    numel = int(flat.numel())
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    finite_count = int(finite.numel())
    nan_count = int(torch.count_nonzero(torch.isnan(flat)).item())
    posinf_count = int(torch.count_nonzero(flat == float("inf")).item())
    neginf_count = int(torch.count_nonzero(flat == float("-inf")).item())
    zero_count = int(torch.count_nonzero(flat == 0).item())

    min_value = max_value = mean = std = absmax = None
    absmax_count = 0
    q001 = q01 = q50 = q99 = q999 = None
    if finite_count:
        min_value = float(finite.min().item())
        max_value = float(finite.max().item())
        finite64 = finite.to(torch.float64)
        mean = float(finite64.mean().item())
        std = float(finite64.std(unbiased=False).item())
        abs_values = finite.abs()
        absmax = float(abs_values.max().item())
        absmax_count = int(torch.count_nonzero(abs_values == absmax).item())

        if sample_values > 0:
            sample = finite.index_select(0, even_indices(finite_count, sample_values, finite.device))
            q001 = float(torch.quantile(sample, 0.001).item())
            q01 = float(torch.quantile(sample, 0.01).item())
            q50 = float(torch.quantile(sample, 0.50).item())
            q99 = float(torch.quantile(sample, 0.99).item())
            q999 = float(torch.quantile(sample, 0.999).item())

    return TensorStats(
        definition=definition,
        file=str(path),
        tensor=name,
        dtype=dtype,
        shape=shape,
        numel=numel,
        finite_count=finite_count,
        nan_count=nan_count,
        posinf_count=posinf_count,
        neginf_count=neginf_count,
        zero_count=zero_count,
        min=min_value,
        max=max_value,
        mean=mean,
        std=std,
        absmax=absmax,
        absmax_count=absmax_count,
        q001=q001,
        q01=q01,
        q50=q50,
        q99=q99,
        q999=q999,
    )


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def summarize(rows: list[TensorStats]) -> None:
    print(
        "definition\ttensor\tfiles\tglobal_min\tglobal_max\tmax_abs\t"
        "max_abs_count\tmax_q999\tmean_range\tstd_range\tnonfinite"
    )
    for definition in FP8_DEFS:
        for tensor in OUTPUT_NAMES:
            group = [row for row in rows if row.definition == definition and row.tensor == tensor]
            if not group:
                continue
            nonfinite = sum(row.nan_count + row.posinf_count + row.neginf_count for row in group)
            global_min = min(row.min for row in group if row.min is not None)
            global_max = max(row.max for row in group if row.max is not None)
            max_abs_row = max(group, key=lambda row: row.absmax or -math.inf)
            max_q999 = max(row.q999 for row in group if row.q999 is not None)
            mean_min = min(row.mean for row in group if row.mean is not None)
            mean_max = max(row.mean for row in group if row.mean is not None)
            std_min = min(row.std for row in group if row.std is not None)
            std_max = max(row.std for row in group if row.std is not None)
            print(
                "\t".join(
                    [
                        definition,
                        tensor,
                        str(len(group)),
                        fmt(global_min),
                        fmt(global_max),
                        fmt(max_abs_row.absmax),
                        f"{max_abs_row.absmax_count}/{max_abs_row.numel}",
                        fmt(max_q999),
                        f"{fmt(mean_min)}..{fmt(mean_max)}",
                        f"{fmt(std_min)}..{fmt(std_max)}",
                        str(nonfinite),
                    ]
                )
            )


def main() -> int:
    args = parse_args()
    rows: list[TensorStats] = []
    for definition in FP8_DEFS:
        directory = args.workload_root / definition
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
            for output_name in OUTPUT_NAMES:
                if output_name in keys:
                    rows.append(collect_stats(definition, path, output_name, args.sample_values))

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
