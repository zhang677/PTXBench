"""Generate two attention Definitions + matching helion Solutions.

Definitions (`reference` = F.scaled_dot_product_attention, single-line PyTorch):
  mha_h48_d128.json          non-causal MHA
  mha_h48_d128.jsonl         6 workloads (S = 128..4096)
  mha_h48_d128_causal.json   causal MHA
  mha_h48_d128_causal.jsonl  6 workloads (S = 128..4096)

Helion Solutions (separate files, pinned tuned config):
  solutions/mha_h48_d128/helion_mha_h48_d128.json
  solutions/mha_h48_d128_causal/helion_mha_h48_d128_causal.json

Shapes: tritonbench `flash_attention` operator under helion's benchmark config
(d_head=128, num_inputs=6) → BATCH=4, H=48, D=128, S in {128,256,512,1024,2048,4096}.
"""

import importlib.util
import json
import os
import sys
import tempfile
import uuid
from typing import Dict, Any

import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))

B_VAL, H_VAL, D_VAL = 4, 48, 128
S_VALUES = [128, 256, 512, 1024, 2048, 4096]


# ----- Definition reference (F.scaled_dot_product_attention) -----------

_DEF_REFERENCE_NONCAUSAL = """\
import torch
import torch.nn.functional as F


def run(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=False)
"""

_DEF_REFERENCE_CAUSAL = """\
import torch
import torch.nn.functional as F


def run(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
"""


# ----- Helion Solution kernel template ----------------------------------
# Verbatim non-causal kernel from /home/ubuntu/helion/examples/attention.py:35-90
# with a hand-tuned config and an optional in-tile causal mask block.

_SOLUTION_TEMPLATE = """\
from __future__ import annotations

import math

import torch

import helion
import helion.language as hl


@helion.kernel(
    # Pinned tuned config (block_sizes=[1,128,128], num_warps=8, num_stages=3,
    # block_ptr indexing). 1.5x faster than helion's test_attention_pointer
    # config across S=128..4096 on H100 while still compiling on all shapes.
    config=helion.Config(
        block_sizes=[1, 128, 128],
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
)
def attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
__CAUSAL_BLOCK__            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def run(Q, K, V):
    return attention(Q, K, V)
"""

_CAUSAL_BLOCK = """\
            # Causal mask: zero out positions where k_pos > q_pos by setting
            # their pre-softmax scores to -inf.
            q_pos = tile_m.begin + hl.arange(tile_m.block_size)
            k_pos = tile_n.begin + hl.arange(tile_n.block_size)
            causal_mask = q_pos[:, None] >= k_pos[None, :]
            qk = torch.where(
                causal_mask[None, :, :],
                qk,
                torch.full_like(qk, float("-inf")),
            )
"""


def build_solution_source(causal: bool) -> str:
    repl = _CAUSAL_BLOCK if causal else ""
    return _SOLUTION_TEMPLATE.replace("__CAUSAL_BLOCK__", repl)


# ----- Builders ---------------------------------------------------------

def build_definition(causal: bool) -> Dict[str, Any]:
    name = "mha_h48_d128_causal" if causal else "mha_h48_d128"
    descr = (
        "Causal multi-head attention (forward only), bf16."
        if causal
        else "Non-causal multi-head attention (forward only), bf16."
    )
    descr += (
        " Q/K/V share shape (B,H,S,D). Output O = softmax(Q@K^T/sqrt(D))@V"
    )
    descr += (
        " with a lower-triangular mask over (query_pos, key_pos)."
        if causal
        else "."
    )
    descr += (
        " Shapes match tritonbench `flash_attention` operator under helion's"
        " benchmark config (d_head=128, num_inputs=6). Reference is"
        " F.scaled_dot_product_attention."
    )
    return {
        "name": name,
        "description": descr,
        "op_type": "attention",
        "tags": [
            "status:verified",
            "source:tritonbench",
            "bench:flash_attention",
            "ref:torch_sdpa",
            "causal:true" if causal else "causal:false",
        ],
        "axes": {
            "B": {"type": "const", "value": B_VAL, "description": "batch size"},
            "H": {"type": "const", "value": H_VAL, "description": "number of attention heads"},
            "D": {"type": "const", "value": D_VAL, "description": "head dimension"},
            "S": {"type": "var", "description": "sequence length (Q and KV share this axis)"},
        },
        "inputs": {
            "Q": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
            "K": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
            "V": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
        },
        "outputs": {
            "O": {"shape": ["B", "H", "S", "D"], "dtype": "bfloat16"},
        },
        "reference": _DEF_REFERENCE_CAUSAL if causal else _DEF_REFERENCE_NONCAUSAL,
    }


def build_workload_line(defn_name: str, s_val: int) -> Dict[str, Any]:
    return {
        "definition": defn_name,
        "solution": None,
        "workload": {
            "uuid": str(uuid.uuid4()),
            "axes": {"B": B_VAL, "H": H_VAL, "D": D_VAL, "S": s_val},
            "inputs": {
                "Q": {"type": "random"},
                "K": {"type": "random"},
                "V": {"type": "random"},
            },
        },
        "evaluation": None,
    }


def build_solution(causal: bool) -> Dict[str, Any]:
    defn_name = "mha_h48_d128_causal" if causal else "mha_h48_d128"
    sol_name = f"helion_{defn_name}"
    return {
        "name": sol_name,
        "definition": defn_name,
        "author": "AccRL",
        "spec": {
            "language": "python",
            "target_hardware": ["NVIDIA H100"],
            "entry_point": "main.py::run",
            "dependencies": ["helion"],
            "destination_passing_style": False,
        },
        "sources": [
            {"path": "main.py", "content": build_solution_source(causal)},
        ],
        "description": (
            "Helion flash-attention kernel adapted from helion/examples/attention.py "
            "with a hand-tuned config (block_sizes=[1,128,128], num_warps=8, "
            "num_stages=3, indexing='block_ptr')"
            + (" and an in-tile causal mask." if causal else ".")
        ),
    }


# ----- FA2 baseline Solution (PyTorch ATen flash-attention forward).
# Single-output to match the mha_h48_d128 / mha_h48_d128_causal Definitions
# (the with-LSE family in 2026-0426-1410 returns (O, LSE) instead).

_FA2_SOURCE_TEMPLATE = """\
import torch


def run(Q, K, V):
    o, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        Q, K, V, is_causal=__IS_CAUSAL__)
    return o
"""


def build_fa2_source(causal: bool) -> str:
    return _FA2_SOURCE_TEMPLATE.replace("__IS_CAUSAL__", "True" if causal else "False")


def build_fa2_solution(causal: bool) -> Dict[str, Any]:
    defn_name = "mha_h48_d128_causal" if causal else "mha_h48_d128"
    sol_name = f"fa2_{defn_name}"
    return {
        "name": sol_name,
        "definition": defn_name,
        "author": "AccRL",
        "spec": {
            "language": "python",
            "target_hardware": ["NVIDIA H100"],
            "entry_point": "main.py::run",
            "dependencies": [],
            "destination_passing_style": False,
        },
        "sources": [
            {"path": "main.py", "content": build_fa2_source(causal)},
        ],
        "description": (
            "PyTorch ATen FlashAttention-2 (torch.ops.aten._scaled_dot_product_flash_attention),"
            " returning O. Baseline for comparing against the F.sdpa reference."
            + (" Causal." if causal else "")
        ),
    }


# ----- Local writes -----------------------------------------------------

def _write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


def write_local_artifacts(causal: bool):
    defn = build_definition(causal)
    sol = build_solution(causal)
    fa2_sol = build_fa2_solution(causal)

    defn_path = os.path.join(HERE, f"{defn['name']}.json")
    _write_json(defn_path, defn)

    wl_path = os.path.join(HERE, f"{defn['name']}.jsonl")
    with open(wl_path, "w") as f:
        for s in S_VALUES:
            line = build_workload_line(defn["name"], s)
            f.write(json.dumps(line, separators=(",", ":")) + "\n")
    print(f"wrote {wl_path} ({len(S_VALUES)} lines)")

    sol_path = os.path.join(HERE, f"{sol['name']}.json")
    _write_json(sol_path, sol)

    fa2_path = os.path.join(HERE, f"{fa2_sol['name']}.json")
    _write_json(fa2_path, fa2_sol)

    return defn, sol, fa2_sol


# ----- Local correctness check ------------------------------------------

def _load_solution_module(sol: Dict[str, Any]):
    src = sol["sources"][0]["content"]
    tmpdir = tempfile.mkdtemp(prefix="mha_sol_")
    mod_path = os.path.join(tmpdir, "main.py")
    with open(mod_path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"_mha_sol_{sol['name']}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def local_correctness_check(defn: Dict[str, Any], sol: Dict[str, Any]):
    """Sanity check: helion solution matches F.sdpa reference on each S."""
    print(f"\n--- local check: {sol['name']} vs {defn['name']}.reference ---")
    assert torch.cuda.is_available()
    is_causal = "causal:true" in defn["tags"]
    mod = _load_solution_module(sol)
    run_sol = mod.run

    for s in S_VALUES:
        torch.manual_seed(0)
        shape = (B_VAL, H_VAL, s, D_VAL)
        Q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        O_sol = run_sol(Q, K, V)
        O_ref = F.scaled_dot_product_attention(
            Q.float(), K.float(), V.float(), is_causal=is_causal
        ).to(torch.bfloat16)
        max_abs = (O_sol.float() - O_ref.float()).abs().max().item()
        denom = O_ref.float().abs().max().item() + 1e-9
        max_rel = max_abs / denom
        ok_shape = tuple(O_sol.shape) == shape
        ok_dtype = O_sol.dtype == torch.bfloat16
        ok_finite = bool(torch.isfinite(O_sol).all().item())
        ok_num = max_rel < 1e-1
        status = "PASS" if (ok_shape and ok_dtype and ok_finite and ok_num) else "FAIL"
        print(
            f"[S={s:>4d}] {status} | shape={tuple(O_sol.shape)} | dtype={O_sol.dtype}"
            f" | finite={ok_finite} | max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        )
        if status == "FAIL":
            raise AssertionError(f"local check failed at S={s}")


def main():
    artifacts = []
    for causal in (False, True):
        artifacts.append(write_local_artifacts(causal))
    for defn, sol, fa2_sol in artifacts:
        local_correctness_check(defn, sol)
        local_correctness_check(defn, fa2_sol)
    print("\nLOCAL CHECKS PASSED.")


if __name__ == "__main__":
    main()
