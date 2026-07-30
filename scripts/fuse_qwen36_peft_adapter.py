#!/usr/bin/env python3
"""Make Tinker Qwen3.6 LoRA adapters loadable by standard Transformers PEFT.

Tinker trains separate ``in_proj_q``, ``in_proj_k``, and ``in_proj_v`` LoRA
modules in Qwen3.6 linear-attention layers. Hugging Face Transformers stores
the corresponding base weight as one ``in_proj_qkv`` module. This converter
represents the three independent rank-r updates as one exactly equivalent
rank-3r block-diagonal LoRA update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

Q_SUFFIX = ".in_proj_q.lora_A.weight"


def fuse_adapter(source: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(output)
    source_config = json.loads((source / "adapter_config.json").read_text())
    if source_config.get("use_rslora"):
        raise ValueError("RS-LoRA scaling is not supported by this exact fusion")

    weights = load_file(source / "adapter_model.safetensors", device="cpu")
    fused = dict(weights)
    prefixes = sorted(key[: -len(Q_SUFFIX)] for key in weights if key.endswith(Q_SUFFIX))
    if not prefixes:
        raise ValueError("no split in_proj_q/k/v LoRA groups found")

    ranks: set[int] = set()
    for prefix in prefixes:
        components = {}
        for projection in ("q", "k", "v"):
            a_key = f"{prefix}.in_proj_{projection}.lora_A.weight"
            b_key = f"{prefix}.in_proj_{projection}.lora_B.weight"
            if a_key not in weights or b_key not in weights:
                raise ValueError(f"incomplete split-QKV group at {prefix}")
            components[projection] = (weights[a_key], weights[b_key])

        a_tensors = [components[name][0] for name in ("q", "k", "v")]
        b_tensors = [components[name][1] for name in ("q", "k", "v")]
        rank = a_tensors[0].shape[0]
        ranks.add(rank)
        if any(a.shape != a_tensors[0].shape for a in a_tensors):
            raise ValueError(f"incompatible LoRA A shapes at {prefix}")
        if any(b.shape[1] != rank for b in b_tensors):
            raise ValueError(f"incompatible LoRA B ranks at {prefix}")

        fused_a = torch.cat(a_tensors, dim=0)
        fused_b = torch.zeros(
            (sum(tensor.shape[0] for tensor in b_tensors), 3 * rank),
            dtype=b_tensors[0].dtype,
        )
        row = 0
        for block, tensor in enumerate(b_tensors):
            next_row = row + tensor.shape[0]
            fused_b[row:next_row, block * rank : (block + 1) * rank] = tensor
            row = next_row

        # Structural equality proves B_fused @ A_fused is the vertical
        # concatenation of B_q @ A_q, B_k @ A_k, and B_v @ A_v.
        for block, tensor in enumerate(a_tensors):
            assert torch.equal(fused_a[block * rank : (block + 1) * rank], tensor)
        row = 0
        for block, tensor in enumerate(b_tensors):
            next_row = row + tensor.shape[0]
            assert torch.equal(
                fused_b[row:next_row, block * rank : (block + 1) * rank], tensor
            )
            assert torch.count_nonzero(fused_b[row:next_row, : block * rank]) == 0
            assert (
                torch.count_nonzero(fused_b[row:next_row, (block + 1) * rank :])
                == 0
            )
            row = next_row

        for projection in ("q", "k", "v"):
            del fused[f"{prefix}.in_proj_{projection}.lora_A.weight"]
            del fused[f"{prefix}.in_proj_{projection}.lora_B.weight"]
        fused[f"{prefix}.in_proj_qkv.lora_A.weight"] = fused_a
        fused[f"{prefix}.in_proj_qkv.lora_B.weight"] = fused_b

    if len(ranks) != 1:
        raise ValueError(f"expected one split-QKV rank, found {sorted(ranks)}")
    rank = ranks.pop()
    original_alpha = int(source_config["lora_alpha"])
    target_modules = set(source_config["target_modules"])
    target_modules.difference_update({"in_proj_q", "in_proj_k", "in_proj_v"})
    target_modules.add("in_proj_qkv")
    source_config["target_modules"] = sorted(target_modules)
    source_config["rank_pattern"] = {
        **source_config.get("rank_pattern", {}),
        "in_proj_qkv": 3 * rank,
    }
    source_config["alpha_pattern"] = {
        **source_config.get("alpha_pattern", {}),
        "in_proj_qkv": 3 * original_alpha,
    }
    source_config["inference_mode"] = True

    output.mkdir(parents=True)
    save_file(
        fused,
        output / "adapter_model.safetensors",
        metadata={
            "format": "pt",
            "qwen36_split_qkv_fusion": "exact_block_diagonal",
        },
    )
    (output / "adapter_config.json").write_text(
        json.dumps(source_config, indent=2, sort_keys=True) + "\n"
    )
    (output / "FUSION_PROVENANCE.json").write_text(
        json.dumps(
            {
                "method": "exact_block_diagonal_lora",
                "source_tensor_count": len(weights),
                "output_tensor_count": len(fused),
                "fused_groups": len(prefixes),
                "source_rank": rank,
                "fused_rank": 3 * rank,
                "source_alpha": original_alpha,
                "fused_alpha": 3 * original_alpha,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return len(prefixes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = fuse_adapter(args.source, args.output)
    print(f"fused_groups={groups}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
