"""Generate two `mha_with_lse` Definitions + matching Solutions.

Pattern mirrors /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0424-1043/
but the definitions return BOTH attention output O (bf16, [B,H,S,D]) and
the natural-log logsumexp LSE (fp32, [B,H,S]).

Definitions (`reference` = aten._scaled_dot_product_cudnn_attention, returns o,lse):
  mha_with_lse_h48_d128.json          non-causal MHA + LSE
  mha_with_lse_h48_d128.jsonl         6 workloads (S = 128..4096)
  mha_with_lse_h48_d128_causal.json   causal MHA + LSE
  mha_with_lse_h48_d128_causal.jsonl  6 workloads (S = 128..4096)

Solutions (separate files, return tuple):
  helion_mha_with_lse_h48_d128.json          # pinned tuned helion kernel
  helion_mha_with_lse_h48_d128_causal.json
  fa2_mha_with_lse_h48_d128.json             # FA2 baseline (~0.55x cuDNN)
  fa2_mha_with_lse_h48_d128_causal.json

Shapes: tritonbench `flash_attention` operator under helion's benchmark config
(d_head=128, num_inputs=6) -> BATCH=4, H=48, D=128, S in {128,256,512,1024,2048,4096}.
"""

import importlib.util
import json
import os
import sys
import tempfile
import uuid
from typing import Dict, Any

import torch


HERE = os.path.dirname(os.path.abspath(__file__))

B_VAL, H_VAL, D_VAL = 4, 48, 128
S_VALUES = [512, 1024, 2048, 4096, 8192, 16384]


# ----- Definition reference: aten cudnn attention returns (o, lse[B,H,S,1]) -----
# cuDNN flash attention is the SDPA backend PyTorch picks by default on H100,
# ~1.7-1.8x faster than FA2 on bf16/d128 for these shapes. Its `logsumexp` has
# a trailing singleton (B,H,S,1); we squeeze to (B,H,S) so it matches FA2 and
# the helion kernel. `lse` is fp32, natural-log: max(QK^T*sm_scale) +
# log(sum(exp(QK^T*sm_scale - max))). Op signature (torch 2.10):
#   _scaled_dot_product_cudnn_attention(
#     Q, K, V, attn_bias, compute_log_sumexp, dropout_p,
#     is_causal, return_debug_mask, *, scale=None) -> (o, lse, ...)

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


# ----- Helion Solution kernel template ----------------------------------
# Adapted from /home/ubuntu/helion/examples/attention.py:35-90 with LSE
# export. The kernel already accumulates log2-base LSE into `m_i` after
# the inner loop; multiplying by ln(2) converts to natural log.
# Copyright (c) Meta Platforms, Inc. and affiliates. The adapted portion is
# distributed under BSD-3-Clause; see ../../../licenses/helion.LICENSE.txt.

_SOLUTION_TEMPLATE = """\
# Portions adapted from pytorch/helion, examples/attention.py.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name Meta nor the names of its contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math

import torch

import helion
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        block_sizes=[1, 128, 128],
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
)
def attention_with_lse(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    lse_out = torch.empty([q_view.size(0), m_dim], dtype=torch.float32, device=q_in.device)
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
        # log2-base LSE -> natural-log LSE
        lse_out[tile_b, tile_m] = m_i * 0.6931471805599453
    return out.view(q_in.size()), lse_out.view(q_in.shape[0], q_in.shape[1], m_dim)


def run(Q, K, V):
    return attention_with_lse(Q, K, V)
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
    name = f"mha_with_lse_h{H_VAL}_d{D_VAL}_causal" if causal else f"mha_with_lse_h{H_VAL}_d{D_VAL}"
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
            "LSE": {"shape": ["B", "H", "S"], "dtype": "float32"},
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
    defn_name = "mha_with_lse_h48_d128_causal" if causal else "mha_with_lse_h48_d128"
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
            "to also output LSE (fp32 natural-log) alongside O. Hand-tuned config "
            "(block_sizes=[1,128,128], num_warps=8, num_stages=3, indexing='block_ptr')"
            + (" with an in-tile causal mask." if causal else ".")
        ),
    }


# ----- FA2 baseline Solution (kept as a competing solution under the cuDNN
# reference). Same signature: (O, LSE).

_FA2_SOURCE_TEMPLATE = """\
import torch


def run(Q, K, V):
    o, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        Q, K, V, is_causal=__IS_CAUSAL__)
    return o, lse
"""


def build_fa2_source(causal: bool) -> str:
    return _FA2_SOURCE_TEMPLATE.replace("__IS_CAUSAL__", "True" if causal else "False")


def build_fa2_solution(causal: bool) -> Dict[str, Any]:
    defn_name = "mha_with_lse_h48_d128_causal" if causal else "mha_with_lse_h48_d128"
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
            " returning (O, LSE). Baseline for comparing against the cuDNN reference."
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
    tmpdir = tempfile.mkdtemp(prefix="mha_lse_sol_")
    mod_path = os.path.join(tmpdir, "main.py")
    with open(mod_path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"_mha_lse_sol_{sol['name']}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _aten_reference(Q, K, V, causal: bool):
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(
        Q, K, V, None, True, 0.0, causal, False)
    return out[0], out[1].squeeze(-1)


def local_correctness_check(defn: Dict[str, Any], sol: Dict[str, Any]):
    """Sanity check: helion solution matches aten reference on each S."""
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

        O_sol, LSE_sol = run_sol(Q, K, V)
        O_ref, LSE_ref = _aten_reference(Q, K, V, is_causal)

        # O checks
        o_max_abs = (O_sol.float() - O_ref.float()).abs().max().item()
        o_denom = O_ref.float().abs().max().item() + 1e-9
        o_max_rel = o_max_abs / o_denom

        # LSE checks
        lse_max_abs = (LSE_sol.float() - LSE_ref.float()).abs().max().item()
        lse_denom = LSE_ref.float().abs().max().item() + 1e-9
        lse_max_rel = lse_max_abs / lse_denom

        ok_o_shape = tuple(O_sol.shape) == shape
        ok_o_dtype = O_sol.dtype == torch.bfloat16
        ok_lse_shape = tuple(LSE_sol.shape) == (B_VAL, H_VAL, s)
        ok_lse_dtype = LSE_sol.dtype == torch.float32
        ok_finite = bool(torch.isfinite(O_sol).all().item() and torch.isfinite(LSE_sol).all().item())
        ok_o_num = o_max_rel < 1e-1
        ok_lse_num = lse_max_rel < 1e-1

        all_ok = ok_o_shape and ok_o_dtype and ok_lse_shape and ok_lse_dtype \
            and ok_finite and ok_o_num and ok_lse_num
        status = "PASS" if all_ok else "FAIL"
        print(
            f"[S={s:>4d}] {status} | O: shape={tuple(O_sol.shape)} dtype={O_sol.dtype}"
            f" max_abs={o_max_abs:.3e} max_rel={o_max_rel:.3e}"
            f" | LSE: shape={tuple(LSE_sol.shape)} dtype={LSE_sol.dtype}"
            f" max_abs={lse_max_abs:.3e} max_rel={lse_max_rel:.3e}"
            f" | finite={ok_finite}"
        )
        if not all_ok:
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
