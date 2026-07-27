#!/usr/bin/env python3
"""Write the reference definitions and random-input workloads for MHA + LSE.

This is the reference-only construction from ../make_attention_with_lse_problem.py:
definitions use torch.ops.aten._scaled_dot_product_cudnn_attention and return
both attention output O and natural-log logsumexp LSE.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any


B_VAL, H_VAL = 4, 48
D_VALUES = [64, 96, 128, 256]
S_VALUES = [512, 1024, 2048, 4096, 8192, 16384]


_DEF_REFERENCE_NONCAUSAL = """\
import torch


def run(Q, K, V):
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(
        Q, K, V,
        None,    # attn_bias
        True,    # compute_log_sumexp
        0.0,     # dropout_p
        False,   # is_causal
        False,   # return_debug_mask
    )
    return out[0], out[1].squeeze(-1)
"""


_DEF_REFERENCE_CAUSAL = """\
import torch


def run(Q, K, V):
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(
        Q, K, V,
        None,    # attn_bias
        True,    # compute_log_sumexp
        0.0,     # dropout_p
        True,    # is_causal
        False,   # return_debug_mask
    )
    return out[0], out[1].squeeze(-1)
"""


def build_definition(causal: bool, d_val: int) -> dict[str, Any]:
    name = (
        f"mha_with_lse_d{d_val}_causal"
        if causal
        else f"mha_with_lse_d{d_val}"
    )
    descr = (
        "Causal multi-head attention forward returning O and LSE, bf16 inputs."
        if causal
        else "Non-causal multi-head attention forward returning O and LSE, bf16 inputs."
    )
    descr += (
        " Q/K/V share shape (B,H,S,D). Outputs: O = softmax(Q@K^T/sqrt(D))@V"
        " (bf16, [B,H,S,D]) and LSE = max(P) + log(sum(exp(P - max(P)))) where"
        " P = Q@K^T/sqrt(D) (fp32 natural-log, [B,H,S])"
    )
    descr += (
        ", with a lower-triangular mask over (query_pos, key_pos)."
        if causal
        else "."
    )
    descr += (
        " Reference is torch.ops.aten._scaled_dot_product_cudnn_attention;"
        " LSE is squeezed from (B,H,S,1) to (B,H,S)."
    )
    return {
        "name": name,
        "description": descr,
        "op_type": "attention",
        "tags": [
            "status:verified",
            "source:tritonbench",
            "bench:flash_attention",
            "ref:aten_cudnn_attention",
            "returns:o_and_lse",
            "causal:true" if causal else "causal:false",
        ],
        "axes": {
            "B": {"type": "const", "value": B_VAL, "description": "batch size"},
            "H": {"type": "const", "value": H_VAL, "description": "number of attention heads"},
            "D": {"type": "const", "value": d_val, "description": "head dimension"},
            "S": {"type": "var", "description": "sequence length (Q and KV share this axis)"},
        },
        "inputs": {
            "Q": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
            "K": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
            "V": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
        },
        "outputs": {
            "O": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
            "LSE": {"shape": ["B", "H", "S"], "dtype": "float32"},
        },
        "reference": _DEF_REFERENCE_CAUSAL if causal else _DEF_REFERENCE_NONCAUSAL,
    }


def build_workload_line(defn_name: str, s_val: int, d_val: int) -> dict[str, Any]:
    return {
        "definition": defn_name,
        "solution": None,
        "workload": {
            "uuid": str(uuid.uuid4()),
            "axes": {"B": B_VAL, "H": H_VAL, "D": d_val, "S": s_val},
            "inputs": {
                "Q": {"type": "random"},
                "K": {"type": "random"},
                "V": {"type": "random"},
            },
        },
        "evaluation": None,
    }


def write_definition(dataset_root: Path, definition: dict[str, Any], overwrite: bool) -> None:
    out = dataset_root / "definitions" / "attention" / f"{definition['name']}.json"
    if out.exists() and not overwrite:
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(definition, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")


def write_workloads(dataset_root: Path, defn_name: str, d_val: int, overwrite: bool) -> None:
    out = dataset_root / "workloads" / "attention" / f"{defn_name}.jsonl"
    if out.exists() and not overwrite:
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s_val in S_VALUES:
            f.write(json.dumps(build_workload_line(defn_name, s_val, d_val), separators=(",", ":")) + "\n")
    print(f"wrote {out} ({len(S_VALUES)} workloads)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    for causal in (False, True):
        for d_val in D_VALUES:
            definition = build_definition(causal, d_val)
            write_definition(dataset_root, definition, args.overwrite)
            write_workloads(dataset_root, definition["name"], d_val, args.overwrite)


if __name__ == "__main__":
    main()
