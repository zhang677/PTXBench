#!/usr/bin/env python3
"""Prepare safetensors inputs for mha_bwd_h48_d128.

This script is intentionally CUDA-only for full workload generation. The
largest workload is S=16384, and the cuDNN forward/backward path (called via
the cuDNN python ``cudnn.pygraph`` API) is the reference. This is the same
path used by FlashAttention-3's ``hopper/benchmark_attn.py`` to produce
Fig. 6 of the FA3 paper.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path

import cudnn
import torch
from safetensors.torch import save_file


DEFINITION = "mha_bwd_wd_h48_d128"
WORKLOAD = Path("workloads/attention/mha_bwd_wd_h48_d128.jsonl")
DEFINITION_PATH = Path("definitions/attention/mha_bwd_wd_h48_d128.json")
BLOB_DIR = Path("blob/workloads/attention/mha_bwd_wd_h48_d128")
BASE_SEED = 20260427
B_VAL, H_VAL, D_VAL = 4, 48, 128
S_VALUES = [512, 1024, 2048, 4096, 8192, 16384]


REFERENCE_SOURCE = """import math

import cudnn
import torch


def _cudnn_dtype(t):
    if t == torch.float16:
        return cudnn.data_type.HALF
    if t == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if t == torch.float32:
        return cudnn.data_type.FLOAT
    raise ValueError(t)


_GRAPH_CACHE = {}


def _build_graph(Q, K, V, O, dO, lse, dQ, dK, dV):
    b, h, s, d = Q.shape
    graph = cudnn.pygraph(
        io_data_type=_cudnn_dtype(Q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = graph.tensor_like(Q.detach())
    k_t = graph.tensor_like(K.detach())
    v_t = graph.tensor_like(V.detach())
    o_t = graph.tensor_like(O.detach())
    g_t = graph.tensor_like(dO.detach())
    s_t = graph.tensor_like(lse.detach())
    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q_t, k=k_t, v=v_t, o=o_t, dO=g_t, stats=s_t,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=False,
    )
    dq.set_output(True).set_dim(dQ.shape).set_stride(dQ.stride())
    dk.set_output(True).set_dim(dK.shape).set_stride(dK.stride())
    dv.set_output(True).set_dim(dV.shape).set_stride(dV.stride())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device=Q.device, dtype=torch.uint8)
    return graph, (q_t, k_t, v_t, o_t, g_t, s_t, dq, dk, dv), workspace


def _cache_key(*tensors):
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors)


def run(Q, K, V, O, dO, L, D):
    # cuDNN sdpa_backward via the python pygraph API. Same path as
    # FlashAttention-3 hopper/benchmark_attn.py (FA3 paper Fig. 6).
    # Graph + workspace are built once per (shape, stride, dtype, device)
    # key so repeated invocations during timing measure graph.execute only,
    # matching FA3's cudnn_spda_bwd_setup protocol. D = rowsum(dO * O) is
    # part of the FA3 backward interface but is recomputed internally by
    # cuDNN; we accept it for schema compatibility. The torch.cuda.device
    # wrapper aligns current_device with the input tensors' device — cuDNN's
    # pygraph tensor_like asserts they match, and flashinfer-bench's
    # build_baseline path runs us in the parent process whose current_device
    # is not pinned to the worker's GPU.
    _ = D
    with torch.cuda.device(Q.device):
        lse = L.unsqueeze(-1) if L.dim() == 3 else L
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        key = _cache_key(Q, K, V, O, dO, lse)
        cached = _GRAPH_CACHE.get(key)
        if cached is None:
            cached = _build_graph(Q, K, V, O, dO, lse, dQ, dK, dV)
            _GRAPH_CACHE[key] = cached
        graph, (q_t, k_t, v_t, o_t, g_t, s_t, dq, dk, dv), workspace = cached

        graph.execute({
            q_t: Q, k_t: K, v_t: V, o_t: O,
            g_t: dO, s_t: lse,
            dq: dQ, dk: dK, dv: dV,
        }, workspace)
        return dQ, dK, dV
"""


DEFINITION_DICT: dict = {
    "name": f"mha_bwd_wd_h{H_VAL}_d{D_VAL}",
    "description": (
        "Non-causal multi-head attention backward, bf16. Inputs are Q, K, V, "
        "forward output O, upstream gradient dO, log-sum-exp L, and precomputed "
        "D = rowsum(dO * O). Shapes match mha_h48_d128. Reference uses cuDNN "
        "SDPA backward and returns dQ, dK, dV."
    ),
    "op_type": "attention",
    "tags": [
        "status:reference",
        "source:flashattention3",
        "ref:cudnn_pygraph_sdpa_backward",
        "causal:false",
        "direction:backward",
    ],
    "axes": {
        "B": {"type": "const", "value": B_VAL, "description": "batch size"},
        "H": {"type": "const", "value": H_VAL, "description": "number of attention heads"},
        "d": {"type": "const", "value": D_VAL, "description": "head dimension"},
        "S": {
            "type": "var",
            "description": "sequence length; Q and KV share this axis",
        },
    },
    "inputs": {
        "Q": {"shape": ["B", "H", "S", "d"], "dtype": "bfloat16"},
        "K": {"shape": ["B", "H", "S", "d"], "dtype": "bfloat16"},
        "V": {"shape": ["B", "H", "S", "d"], "dtype": "bfloat16"},
        "O": {
            "shape": ["B", "H", "S", "d"],
            "dtype": "bfloat16",
            "description": "forward attention output",
        },
        "dO": {
            "shape": ["B", "H", "S", "d"],
            "dtype": "bfloat16",
            "description": "upstream gradient for O",
        },
        "L": {
            "shape": ["B", "H", "S"],
            "dtype": "float32",
            "description": (
                "natural-log logsumexp of QK^T/sqrt(d), squeezed from cuDNN's "
                "trailing singleton dimension"
            ),
        },
        "D": {
            "shape": ["B", "H", "S"],
            "dtype": "float32",
            "description": "rowsum(dO * O), as used by the FlashAttention backward algorithm",
        },
    },
    "outputs": {
        "dQ": {"shape": ["B", "H", "S", "d"], "dtype": "bfloat16"},
        "dK": {"shape": ["B", "H", "S", "d"], "dtype": "bfloat16"},
        "dV": {"shape": ["B", "H", "S", "d"], "dtype": "bfloat16"},
    },
    "reference": REFERENCE_SOURCE,
}


def write_definition(dataset_root: Path, overwrite: bool) -> None:
    out = dataset_root / DEFINITION_PATH
    if out.exists() and not overwrite:
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(DEFINITION_DICT, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")


def _workload_uuid(s_val: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"accrl://{DEFINITION}/S={s_val}"))


def build_workload_entry(s_val: int) -> dict:
    wl_uuid = _workload_uuid(s_val)
    safetensor_path = f"./{BLOB_DIR}/{DEFINITION}_{wl_uuid.replace('-', '')}.safetensors"
    inputs = {}
    for name in ["Q", "K", "V", "O", "dO", "L", "D"]:
        inputs[name] = {"type": "safetensors", "path": safetensor_path, "tensor_key": name}
    return {
        "definition": DEFINITION,
        "solution": None,
        "workload": {
            "uuid": wl_uuid,
            "axes": {"B": B_VAL, "H": H_VAL, "S": s_val, "d": D_VAL},
            "inputs": inputs,
        },
        "evaluation": None,
    }


def build_workloads() -> list[dict]:
    return [build_workload_entry(s_val) for s_val in S_VALUES]


def write_workloads(dataset_root: Path, items: list[dict]) -> None:
    out = dataset_root / WORKLOAD
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({len(items)} workloads)")


def spiky_normal(shape: tuple[int, ...], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """torch.normal(0, 1) + torch.normal(0, 100) * torch.bernoulli(0.001)."""
    base = torch.empty(shape, dtype=torch.float32, device=device).normal_(0.0, 1.0, generator=generator)
    spikes = torch.empty(shape, dtype=torch.float32, device=device).normal_(0.0, 100.0, generator=generator)
    mask = torch.empty(shape, dtype=torch.float32, device=device).bernoulli_(0.001, generator=generator)
    return (base + spikes * mask).to(torch.bfloat16)


def plain_randn(shape: tuple[int, ...], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    """torch.randn(0, 1), straight bf16 — well-conditioned for pointwise allclose checks."""
    return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(0.0, 1.0, generator=generator)


SAMPLERS = {"spiky": spiky_normal, "randn": plain_randn}


def _cudnn_dtype(t: torch.dtype):
    if t == torch.float16:
        return cudnn.data_type.HALF
    if t == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if t == torch.float32:
        return cudnn.data_type.FLOAT
    raise ValueError(f"unsupported dtype {t!r}")


def cudnn_forward(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """cuDNN sdpa forward via the python pygraph API. Returns (O, lse).

    lse has shape (B, H, S, 1) float32 — exactly the form
    cudnn.pygraph.sdpa_backward expects. Same path as
    flash-attention-cutedsl/hopper/benchmark_attn.py::cudnn_spda_setup.
    """
    b, h, s, d = Q.shape
    O = torch.empty_like(Q)
    lse = torch.empty(b, h, s, 1, dtype=torch.float32, device=Q.device)
    graph = cudnn.pygraph(
        io_data_type=_cudnn_dtype(Q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = graph.tensor_like(Q.detach())
    k_t = graph.tensor_like(K.detach())
    v_t = graph.tensor_like(V.detach())
    o_t, s_t = graph.sdpa(
        name="sdpa",
        q=q_t, k=k_t, v=v_t,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=False,
    )
    o_t.set_output(True).set_dim(O.shape).set_stride(O.stride())
    s_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({q_t: Q, k_t: K, v_t: V, o_t: O, s_t: lse}, workspace)
    return O, lse


def cudnn_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """cuDNN sdpa_backward via the python pygraph API (FA3 paper Fig. 6 path)."""
    b, h, s, d = Q.shape
    lse = L.unsqueeze(-1) if L.dim() == 3 else L
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    graph = cudnn.pygraph(
        io_data_type=_cudnn_dtype(Q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = graph.tensor_like(Q.detach())
    k_t = graph.tensor_like(K.detach())
    v_t = graph.tensor_like(V.detach())
    o_t = graph.tensor_like(O.detach())
    g_t = graph.tensor_like(dO.detach())
    s_t = graph.tensor_like(lse.detach())
    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q_t, k=k_t, v=v_t, o=o_t, dO=g_t, stats=s_t,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=False,
    )
    dq.set_output(True).set_dim(dQ.shape).set_stride(dQ.stride())
    dk.set_output(True).set_dim(dK.shape).set_stride(dK.stride())
    dv.set_output(True).set_dim(dV.shape).set_stride(dV.stride())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({
        q_t: Q, k_t: K, v_t: V, o_t: O,
        g_t: dO, s_t: lse,
        dq: dQ, dk: dK, dv: dV,
    }, workspace)
    return dQ, dK, dV


def pytorch_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Algorithmic fp32 reference. We do NOT use F.scaled_dot_product_attention
    # because PyTorch dispatches it to cuDNN by default — that would be a
    # cuDNN-vs-cuDNN comparison, and cuDNN's backward kernel mixes the saved
    # philox_seed/offset into its arithmetic even at dropout=0, so two
    # independent forwards (manual + autograd's internal) yield bit-different
    # outputs and a misleading "mismatch".
    D_head = Q.shape[-1]
    scale = 1.0 / math.sqrt(D_head)
    q = Q.detach().float().clone().requires_grad_(True)
    k = K.detach().float().clone().requires_grad_(True)
    v = V.detach().float().clone().requires_grad_(True)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    out.backward(dO.float())
    assert q.grad is not None and k.grad is not None and v.grad is not None
    return q.grad, k.grad, v.grad


def compare(
    name: str,
    cudnn: torch.Tensor,
    torch_grad: torch.Tensor,
    rtol: float,
    atol: float,
    metric: str,
) -> None:
    """Compare bf16 cuDNN backward against the fp32 algorithmic reference.

    metric="allclose"  → pointwise torch.allclose(rtol, atol). Strict; passes
                          for randn-distributed inputs but fails on spiky
                          inputs where bf16 rounding amplifies a handful of
                          extreme elements.
    metric="relnorm"   → ‖err‖₂/‖ref‖₂ ≤ rtol + atol. Robust to outliers; the
                          right metric for spiky stress data.
    """
    cudnn_f32 = cudnn.float()
    torch_f32 = torch_grad.float()
    diff = cudnn_f32 - torch_f32
    max_abs = diff.abs().max().item()
    denom = torch_f32.abs().clamp_min(1e-6)
    max_rel = (diff.abs() / denom).max().item()
    rel_norm = diff.norm().item() / max(torch_f32.norm().item(), 1e-12)
    if metric == "allclose":
        ok = torch.allclose(cudnn_f32, torch_f32, rtol=rtol, atol=atol)
    elif metric == "relnorm":
        ok = rel_norm <= rtol + atol
    else:
        raise ValueError(f"unknown metric {metric!r}")
    print(
        f"{name}: ok={ok} metric={metric} "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
        f"‖err‖/‖ref‖={rel_norm:.4g}"
    )
    if not ok:
        raise AssertionError(f"{name} failed {metric} check (rtol={rtol}, atol={atol})")


def output_path(dataset_root: Path, item: dict) -> Path:
    first_input = next(iter(item["workload"]["inputs"].values()))
    rel = first_input["path"]
    if not rel.startswith("./"):
        raise ValueError(f"expected relative ./blob path, got {rel!r}")
    return dataset_root / rel[2:]


def prepare_one(
    dataset_root: Path,
    item: dict,
    device: torch.device,
    check: bool,
    rtol: float,
    atol: float,
    overwrite: bool,
    distribution: str,
    metric: str,
) -> None:
    axes = item["workload"]["axes"]
    B = int(axes["B"])
    H = int(axes["H"])
    S = int(axes["S"])
    D_head = int(axes["d"])
    shape = (B, H, S, D_head)
    out_path = output_path(dataset_root, item)

    if out_path.exists() and not overwrite:
        print(f"skip existing {out_path}")
        return

    seed = BASE_SEED + S
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    sampler = SAMPLERS[distribution]
    Q = sampler(shape, device, generator)
    K = sampler(shape, device, generator)
    V = sampler(shape, device, generator)
    dO = sampler(shape, device, generator)

    O, lse = cudnn_forward(Q, K, V)
    L = lse.squeeze(-1)  # (B, H, S, 1) float32 → (B, H, S) for storage
    D_rowsum = torch.sum(dO.float() * O.float(), dim=-1)

    if check:
        dQ_cudnn, dK_cudnn, dV_cudnn = cudnn_backward(Q, K, V, O, dO, L)
        dQ_torch, dK_torch, dV_torch = pytorch_backward(Q, K, V, dO)
        print(f"compare S={S} distribution={distribution} metric={metric}")
        compare("dQ", dQ_cudnn, dQ_torch, rtol, atol, metric)
        compare("dK", dK_cudnn, dK_torch, rtol, atol, metric)
        compare("dV", dV_cudnn, dV_torch, rtol, atol, metric)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "Q": Q.detach().cpu().contiguous(),
        "K": K.detach().cpu().contiguous(),
        "V": V.detach().cpu().contiguous(),
        "O": O.detach().cpu().contiguous(),
        "dO": dO.detach().cpu().contiguous(),
        "L": L.detach().cpu().contiguous().float(),
        "D": D_rowsum.detach().cpu().contiguous().float(),
    }
    save_file(tensors, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.1f} MiB)")


def print_operator_schemas() -> None:
    # Both forward and backward now go through cudnn.pygraph (cuDNN python API);
    # there are no aten schemas to print.
    print("forward:  cudnn.pygraph.sdpa")
    print("backward: cudnn.pygraph.sdpa_backward")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="compare cuDNN backward with PyTorch autograd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument(
        "--input-distribution",
        choices=sorted(SAMPLERS.keys()),
        default="randn",
        help="input sampler for Q/K/V/dO. 'spiky' is the FA3-style stress test "
        "(default, materialized data); 'randn' is well-conditioned and lets "
        "pointwise allclose pass at rtol=atol=1e-2.",
    )
    parser.add_argument(
        "--metric",
        choices=("allclose", "relnorm"),
        default=None,
        help="check metric. Defaults: allclose for randn, relnorm for spiky.",
    )
    parser.add_argument("--print-schemas", action="store_true")
    args = parser.parse_args()
    if args.metric is None:
        args.metric = "allclose" if args.input_distribution == "randn" else "relnorm"

    if args.print_schemas:
        print_operator_schemas()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: cuDNN SDPA forward/backward is not available on CPU")

    dataset_root = args.dataset_root.resolve()
    device = torch.device("cuda")
    torch.manual_seed(BASE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False

    write_definition(dataset_root, args.overwrite)

    workloads = build_workloads()
    write_workloads(dataset_root, workloads)
    if not workloads:
        raise SystemExit(f"no {DEFINITION} workloads selected")

    total_elems = sum(math.prod(int(v) for v in w["workload"]["axes"].values()) for w in workloads)
    print(f"selected {len(workloads)} workloads; axis-product checksum={total_elems}")
    for item in workloads:
        prepare_one(
            dataset_root,
            item,
            device,
            args.check,
            args.rtol,
            args.atol,
            args.overwrite,
            args.input_distribution,
            args.metric,
        )


if __name__ == "__main__":
    main()
