#!/usr/bin/env python3
"""Recreate the two MHA backward input blobs used by this dataset subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cudnn
import torch
from safetensors.torch import save_file


BASE_SEED = 20260427
EXPECTED_SHA256 = {
    "mha_bwd_d128": "c13f08ac9488481cecfb4a119235a6fdd05bb1f36de0acaa32445d2982dbd46b",
    "mha_bwd_d128_causal": "a2f25c87a1a20bc01a00cbc6e9db8677c3f5d80615a3ff6e548b792798a1b794",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workload(dataset_root: Path, definition: str) -> tuple[Path, tuple[int, ...]]:
    manifest = dataset_root / "workloads" / "attention" / f"{definition}.jsonl"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise ValueError(f"expected one workload in {manifest}, got {len(rows)}")
    row = json.loads(rows[0])
    axes = row["workload"]["axes"]
    shape = tuple(int(axes[key]) for key in ("B", "H", "S", "d"))
    if row["definition"] != definition or shape != (4, 48, 4096, 128):
        raise ValueError(f"unexpected workload in {manifest}: {row['definition']}, {shape}")
    inputs = row["workload"]["inputs"]
    if any(inputs[key]["type"] != "safetensors" or inputs[key]["tensor_key"] != key
           for key in ("Q", "K", "V", "O", "L")):
        raise ValueError(f"unexpected tensor keys in {manifest}")
    paths = {inputs[key]["path"] for key in ("Q", "K", "V", "O", "L")}
    if len(paths) != 1 or inputs["dO"] != {"type": "random"}:
        raise ValueError(f"unexpected input layout in {manifest}")
    relative_path = paths.pop()
    uuid = row["workload"]["uuid"].replace("-", "")
    expected_path = f"./blob/workloads/attention/{definition}/{definition}_{uuid}.safetensors"
    if relative_path != expected_path:
        raise ValueError(f"unexpected blob path: {relative_path}")
    destination = dataset_root / relative_path[2:]
    return destination, shape


def cudnn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    b, h, s, d = q.shape
    output = torch.empty_like(q)
    lse = torch.empty((b, h, s, 1), dtype=torch.float32, device=q.device)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = graph.tensor_like(q.detach())
    k_t = graph.tensor_like(k.detach())
    v_t = graph.tensor_like(v.detach())
    o_t, s_t = graph.sdpa(
        name="sdpa", q=q_t, k=k_t, v=v_t, is_inference=False,
        attn_scale=1.0 / math.sqrt(d), use_causal_mask=causal,
    )
    o_t.set_output(True).set_dim(output.shape).set_stride(output.stride())
    s_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device=q.device, dtype=torch.uint8)
    graph.execute({q_t: q, k_t: k, v_t: v, o_t: output, s_t: lse}, workspace)
    return output, lse


def generate(destination: Path, shape: tuple[int, ...], causal: bool) -> str:
    b, h, s, d = shape
    seed = BASE_SEED + s + 100_000 * d + 10_000_000 * int(causal)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    tensors = [
        torch.empty((b, h, s, d), dtype=torch.bfloat16, device="cuda").normal_(
            0.0, 1.0, generator=generator
        )
        for _ in range(3)
    ]
    q, k, v = tensors
    output, lse = cudnn_forward(q, k, v, causal)
    materialized = {
        "Q": q.cpu().contiguous(),
        "K": k.cpu().contiguous(),
        "V": v.cpu().contiguous(),
        "O": output.cpu().contiguous(),
        "L": lse.squeeze(-1).cpu().contiguous().float(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".safetensors", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(materialized, str(temporary))
        digest = sha256(temporary)
        expected = EXPECTED_SHA256[destination.parent.name]
        if digest != expected:
            raise RuntimeError(
                f"generated {destination.name} has SHA-256 {digest}; expected {expected}. "
                "Check GPU, PyTorch, cuDNN frontend, and safetensors versions."
            )
        if destination.exists():
            raise FileExistsError(f"destination appeared while generating: {destination}")
        temporary.rename(destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path(__file__).resolve().parent,
        help="directory containing workloads/ and blob/ (default: this script's directory)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for cuDNN SDPA forward")
    torch.manual_seed(BASE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset_root = args.dataset_root.resolve()
    for definition, expected in EXPECTED_SHA256.items():
        destination, shape = workload(dataset_root, definition)
        if destination.exists():
            digest = sha256(destination)
            if digest != expected:
                raise RuntimeError(
                    f"existing {destination} has SHA-256 {digest}; expected {expected}. "
                    "Move or remove it before regenerating."
                )
            print(f"verified {destination}: {digest}", flush=True)
            continue
        digest = generate(destination, shape, definition.endswith("_causal"))
        print(f"generated {destination}: {digest}", flush=True)


if __name__ == "__main__":
    main()
