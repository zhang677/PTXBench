#!/usr/bin/env python3
"""Prepare safetensors inputs + definition + workloads for FP8 MHA + LSE.

FP8 multi-head attention forward, returning O (fp8 e4m3) and LSE (fp32).
B=4, H=48, D in {64, 96, 128, 256}, S in {512, 1024, 2048, 4096, 8192, 16384}.

Inputs Q/K/V are quantized from FA3-style spiky bf16 normals (FA3 paper Fig. 6
input distribution): base ~ N(0,1) + N(0,100) * Bernoulli(0.001), then
per-tensor amax → scale = 448/amax, FP8 = (x_bf16 * scale).to(e4m3fn).

descale_q, descale_k, descale_v are populated as 1/scale per tensor; descale_s,
scale_s, scale_o are left at 1.0 (matches FA3's `cudnn_spda_setup` benchmark
which uses ones for these — see flash-attention/hopper/benchmark_flash_attention_fp8.py:113).

Reference uses cuDNN sdpa_fp8 via the pygraph API; graph is cached per
(shape, stride, dtype, device) so the timed reference measures only
graph.execute (matches FA3 paper Fig. 6 timing convention).
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


B, H = 4, 48
IS_CAUSAL = [False, True]
D_VALUES = [64, 96, 128, 256]
S_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
BASE_SEED = 20260516

DATASET_DEFAULT = Path("/home/ubuntu/accrl-training")


REFERENCE_SOURCE = '''import math

import cudnn
import torch


# scale_s / descale_s (intermediate S = QK^T quantization) and scale_o
# (output O quantization) are pinned to 1.0 in this definition, so they are
# carried by these module-level tensors rather than passed as inputs.
_ONE_GPU = {}


def _ones_tensor(device):
    t = _ONE_GPU.get(device)
    if t is None:
        t = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=device)
        _ONE_GPU[device] = t
    return t


def _build_graph(Q, K, V, descale_q, descale_k, descale_v,
                 descale_s, scale_s, scale_o, O_fp8, LSE,
                 use_causal_mask):
    b, h, s, d = Q.shape
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = graph.tensor_like(Q.detach())
    k_t = graph.tensor_like(K.detach())
    v_t = graph.tensor_like(V.detach())
    dq_t = graph.tensor_like(descale_q.detach())
    dk_t = graph.tensor_like(descale_k.detach())
    dv_t = graph.tensor_like(descale_v.detach())
    ds_t = graph.tensor_like(descale_s.detach())
    ss_t = graph.tensor_like(scale_s.detach())
    so_t = graph.tensor_like(scale_o.detach())
    o_t, stats_t, amax_s_t, amax_o_t = graph.sdpa_fp8(
        name="sdpa_fp8",
        q=q_t, k=k_t, v=v_t,
        descale_q=dq_t, descale_k=dk_t, descale_v=dv_t,
        descale_s=ds_t, scale_s=ss_t, scale_o=so_t,
        generate_stats=True,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=use_causal_mask,
    )
    o_t.set_output(True).set_dim(O_fp8.shape).set_stride(O_fp8.stride())
    stats_lse_dim = (b, h, s, 1)
    stats_lse_stride = (h * s, s, 1, 1)
    stats_t.set_output(True).set_dim(stats_lse_dim).set_stride(stats_lse_stride).set_data_type(cudnn.data_type.FLOAT)
    amax_s_t.set_output(False).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1])
    amax_o_t.set_output(False).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1])
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device=Q.device, dtype=torch.uint8)
    return graph, (q_t, k_t, v_t, dq_t, dk_t, dv_t, ds_t, ss_t, so_t,
                   o_t, stats_t, amax_s_t, amax_o_t), workspace


_GRAPH_CACHE = {}


def _cache_key(*tensors):
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, str(t.device)) for t in tensors)


def run(Q, K, V, descale_q, descale_k, descale_v):
    with torch.cuda.device(Q.device):
        b, h, s, d = Q.shape
        device = Q.device
        O_fp8 = torch.empty((b, h, s, d), dtype=torch.float8_e4m3fn, device=device)
        LSE_4d = torch.empty((b, h, s, 1), dtype=torch.float32, device=device)
        amax_s = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
        amax_o = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
        ones = _ones_tensor(device)  # acts as descale_s, scale_s, scale_o (all 1.0)

        use_causal_mask = __USE_CAUSAL_MASK__
        key = _cache_key(Q, K, V) + (use_causal_mask,)
        cached = _GRAPH_CACHE.get(key)
        if cached is None:
            cached = _build_graph(Q, K, V, descale_q, descale_k, descale_v,
                                ones, ones, ones,
                                O_fp8, LSE_4d, use_causal_mask)
            _GRAPH_CACHE[key] = cached
        graph, (q_t, k_t, v_t, dq_t, dk_t, dv_t, ds_t, ss_t, so_t,
                o_t, stats_t, amax_s_t, amax_o_t), workspace = cached

        graph.execute({
            q_t: Q, k_t: K, v_t: V,
            dq_t: descale_q, dk_t: descale_k, dv_t: descale_v,
            ds_t: ones, ss_t: ones, so_t: ones,
            o_t: O_fp8, stats_t: LSE_4d,
            amax_s_t: amax_s, amax_o_t: amax_o,
        }, workspace)
        return O_fp8, LSE_4d.squeeze(-1)
'''


def make_reference_source(use_causal_mask: bool) -> str:
    return REFERENCE_SOURCE.replace("__USE_CAUSAL_MASK__", "True" if use_causal_mask else "False")


def definition_name(d_val: int, use_causal_mask: bool) -> str:
    return (
        f"fp8_mha_with_lse_d{d_val}_causal"
        if use_causal_mask
        else f"fp8_mha_with_lse_d{d_val}"
    )


def make_definition_dict(name: str, d_val: int, use_causal_mask: bool) -> dict:
    desc = (
            "FP8 (E4M3) multi-head attention forward returning O (fp8) and LSE (fp32). "
            "Q/K/V share shape (B,H,L,D) and are FP8 quantized via per-tensor scales "
            "(descale_q/k/v supplied). Output O is fp8_e4m3fn, LSE = max(S) + "
            "log(sum(exp(S - max(S)))) where S = Q@K^T * descale_q * descale_k /sqrt(D), "
            "squeezed from (B,H,L,1) to (B,H,L). "
            "cuDNN's intermediate descale_s / scale_s and the output scale_o are pinned to 1.0 "
            "in this definition (so O is naively cast to FP8 with no rescale); they are not "
            "exposed as inputs. All the matrix multiplications in the kernel should take in FP8."
        )

    if use_causal_mask:
        desc += " Causal mask is applied to S (softmax input)."

    return {
        "name": name,
        "description": desc,
        "op_type": "attention",
        "tags": [
            "status:reference",
            "ref:cudnn_pygraph_sdpa_fp8",
            "dtype:fp8_e4m3",
            "returns:o_and_lse",
            "causal:true" if use_causal_mask else "causal:false",
        ],
        "axes": {
            "B": {"type": "const", "value": B, "description": "batch size"},
            "H": {"type": "const", "value": H, "description": "number of attention heads"},
            "D": {"type": "const", "value": d_val, "description": "head dimension"},
            "L": {"type": "var", "description": "sequence length; Q and KV share this axis"},
            "one": {"type": "const", "value": 1, "description": "scalar dimension (per-tensor scale factors are (1,1,1,1))"},
        },
        "inputs": {
            "Q": {"shape": ["B", "H", "L", "D"], "dtype": "float8_e4m3fn"},
            "K": {"shape": ["B", "H", "L", "D"], "dtype": "float8_e4m3fn"},
            "V": {"shape": ["B", "H", "L", "D"], "dtype": "float8_e4m3fn"},
            "descale_q": {"shape": ["one", "one", "one", "one"], "dtype": "float32",
                          "description": "descale factor for Q: dequantized Q ≈ Q.float() * descale_q"},
            "descale_k": {"shape": ["one", "one", "one", "one"], "dtype": "float32"},
            "descale_v": {"shape": ["one", "one", "one", "one"], "dtype": "float32"},
        },
        "outputs": {
            "O": {"shape": ["B", "H", "L", "D"], "dtype": "float8_e4m3fn",
                  "description": "O = softmax(Q@K^T * descale_q * descale_k * scale_s / sqrt(D)) @ V * descale_v * descale_s * scale_o, "
                                 "quantized to FP8 E4M3 by cuDNN with scale_o (scale_s/descale_s/scale_o are 1.0 in this definition)."},
            "LSE": {"shape": ["B", "H", "L"], "dtype": "float32"},
        },
        "reference": make_reference_source(use_causal_mask),
    }


def spiky_bf16(shape, device, generator):
    """FA3 paper Fig. 6 spiky distribution: N(0,1) + N(0,100)*Bernoulli(0.001) in bf16."""
    base = torch.empty(shape, dtype=torch.float32, device=device).normal_(0.0, 1.0, generator=generator)
    spikes = torch.empty(shape, dtype=torch.float32, device=device).normal_(0.0, 100.0, generator=generator)
    mask = torch.empty(shape, dtype=torch.float32, device=device).bernoulli_(0.001, generator=generator)
    return (base + spikes * mask).to(torch.bfloat16)


def fp8_quantize(x_bf16):
    """Per-tensor amax → FP8 E4M3. Returns (x_fp8, descale) where descale = 1/scale = amax/448."""
    FP8_MAX = 448.0
    amax = x_bf16.abs().float().max().clamp_min(1e-6)
    scale = FP8_MAX / amax
    descale = amax / FP8_MAX
    x_fp8 = (x_bf16.float() * scale).to(torch.float8_e4m3fn)
    return x_fp8, float(descale.item())


def build_graph(Q, K, V, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
                use_causal_mask, device):
    b, h, s, d = Q.shape
    O = torch.empty((b, h, s, d), dtype=torch.float8_e4m3fn, device=device)
    LSE_4d = torch.empty((b, h, s, 1), dtype=torch.float32, device=device)
    amax_s_buf = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)
    amax_o_buf = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=device)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = graph.tensor_like(Q.detach())
    k_t = graph.tensor_like(K.detach())
    v_t = graph.tensor_like(V.detach())
    dq_t = graph.tensor_like(descale_q.detach())
    dk_t = graph.tensor_like(descale_k.detach())
    dv_t = graph.tensor_like(descale_v.detach())
    ds_t = graph.tensor_like(descale_s.detach())
    ss_t = graph.tensor_like(scale_s.detach())
    so_t = graph.tensor_like(scale_o.detach())
    o_t, stats_t, amax_s_t, amax_o_t = graph.sdpa_fp8(
        name="sdpa_fp8",
        q=q_t, k=k_t, v=v_t,
        descale_q=dq_t, descale_k=dk_t, descale_v=dv_t,
        descale_s=ds_t, scale_s=ss_t, scale_o=so_t,
        generate_stats=True,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=use_causal_mask,
    )
    o_t.set_output(True).set_dim(O.shape).set_stride(O.stride())
    stats_t.set_output(True).set_dim(LSE_4d.shape).set_stride(LSE_4d.stride()).set_data_type(cudnn.data_type.FLOAT)
    amax_s_t.set_output(False).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1])
    amax_o_t.set_output(False).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1])
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device=device, dtype=torch.uint8)
    return (graph, q_t, k_t, v_t, dq_t, dk_t, dv_t, ds_t, ss_t, so_t,
            o_t, stats_t, amax_s_t, amax_o_t, workspace, O, LSE_4d, amax_s_buf, amax_o_buf)


def cudnn_fp8_forward(Q_fp8, K_fp8, V_fp8,
                      descale_q, descale_k, descale_v,
                      descale_s, scale_s, scale_o,
                      use_causal_mask, device):
    pkg = build_graph(Q_fp8, K_fp8, V_fp8,
                      descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
                      use_causal_mask, device)
    (graph, q_t, k_t, v_t, dq_t, dk_t, dv_t, ds_t, ss_t, so_t,
     o_t, stats_t, amax_s_t, amax_o_t, workspace, O, LSE_4d, amax_s_buf, amax_o_buf) = pkg
    graph.execute({
        q_t: Q_fp8, k_t: K_fp8, v_t: V_fp8,
        dq_t: descale_q, dk_t: descale_k, dv_t: descale_v,
        ds_t: descale_s, ss_t: scale_s, so_t: scale_o,
        o_t: O, stats_t: LSE_4d,
        amax_s_t: amax_s_buf, amax_o_t: amax_o_buf,
    }, workspace)
    torch.cuda.synchronize()
    return O, LSE_4d.squeeze(-1)


def fp32_reference(Q_fp8, K_fp8, V_fp8,
                   descale_q, descale_k, descale_v,
                   use_causal_mask, chunk_size=1024):
    """FP32 algorithmic reference: dequantize FP8, compute SDPA in FP32.

    Streamed over Q rows to keep attention scores bounded to a single
    chunk_size x S slab in memory (so we can validate S=16384 within 80 GB).
    """
    b, h, s, d = Q_fp8.shape
    q = Q_fp8.float() * descale_q.float()
    k = K_fp8.float() * descale_k.float()
    v = V_fp8.float() * descale_v.float()
    scale = 1.0 / math.sqrt(d)
    o_ref = torch.empty_like(q)
    lse_ref = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    for start in range(0, s, chunk_size):
        end = min(start + chunk_size, s)
        scores = (q[:, :, start:end] @ k.transpose(-2, -1)) * scale  # (b,h,c,s)
        if use_causal_mask:
            row_idx = torch.arange(start, end, device=q.device).view(-1, 1)
            col_idx = torch.arange(s, device=q.device).view(1, -1)
            mask = col_idx > row_idx
            scores = scores.masked_fill(mask, float("-inf"))
        lse_chunk = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        o_ref[:, :, start:end] = probs @ v
        lse_ref[:, :, start:end] = lse_chunk
        del scores, probs
    return o_ref, lse_ref


def build_workload_entry(definition_name: str, s_val: int, d_val: int, blob_dir: str) -> dict:
    wl_uuid = str(uuid.uuid4())
    safetensor_path = f"./{blob_dir}/{definition_name}_{wl_uuid.replace('-', '')}.safetensors"
    inputs = {}
    for name in ["Q", "K", "V", "descale_q", "descale_k", "descale_v"]:
        inputs[name] = {"type": "safetensors", "path": safetensor_path, "tensor_key": name}
    outputs = {
        "O": {"type": "safetensors", "path": safetensor_path, "tensor_key": "O"},
        "LSE": {"type": "safetensors", "path": safetensor_path, "tensor_key": "LSE"},
    }
    return {
        "definition": definition_name,
        "solution": None,
        "workload": {
            "uuid": wl_uuid,
            "axes": {"B": B, "H": H, "D": d_val, "L": s_val, "one": 1},
            "inputs": inputs,
            "outputs": outputs,
        },
        "evaluation": None,
    }


def output_path(dataset_root: Path, item: dict) -> Path:
    rel = item["workload"]["inputs"]["Q"]["path"]
    if not rel.startswith("./"):
        raise ValueError(f"expected relative ./blob path, got {rel!r}")
    return dataset_root / rel[2:]


def write_definition(dataset_root: Path, definition_dict: dict, overwrite: bool) -> None:
    out = dataset_root / "definitions" / "attention" / f"{definition_dict['name']}.json"
    if out.exists() and not overwrite:
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(definition_dict, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")


def write_workloads(dataset_root: Path, definition_name: str, items: list, overwrite: bool) -> None:
    out = dataset_root / "workloads" / "attention" / f"{definition_name}.jsonl"
    if out.exists() and not overwrite:
        # Still rewrite — workloads carry uuids so re-running with --overwrite re-uses cached uuids
        # if you want stability. Without --overwrite we don't touch existing jsonl.
        print(f"skip existing {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"wrote {out} ({len(items)} workloads)")


def iter_or_build_workloads(dataset_root: Path, definition_name: str, d_val: int,
                            blob_dir: str, overwrite: bool) -> list[dict]:
    out = dataset_root / "workloads" / "attention" / f"{definition_name}.jsonl"
    if out.exists() and not overwrite:
        items = []
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        print(f"reusing {out} ({len(items)} workloads)")
        return items
    items = [
        build_workload_entry(definition_name, s_val, d_val, blob_dir)
        for s_val in S_VALUES
    ]
    write_workloads(dataset_root, definition_name, items, overwrite=True)
    return items


def prepare_one(dataset_root: Path, item: dict, use_causal_mask: bool,
                device: torch.device, check: bool, overwrite: bool) -> None:
    axes = item["workload"]["axes"]
    s_val = int(axes["L"])
    d_val = int(axes["D"])
    out_path = output_path(dataset_root, item)

    seed = BASE_SEED + s_val + 100_000 * d_val + 10_000_000 * int(use_causal_mask)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    shape = (B, H, s_val, d_val)
    Q_bf16 = spiky_bf16(shape, device, generator)
    K_bf16 = spiky_bf16(shape, device, generator)
    V_bf16 = spiky_bf16(shape, device, generator)

    Q_fp8, dq = fp8_quantize(Q_bf16)
    K_fp8, dk = fp8_quantize(K_bf16)
    V_fp8, dv = fp8_quantize(V_bf16)

    descale_q = torch.tensor([[[[dq]]]], dtype=torch.float32, device=device)
    descale_k = torch.tensor([[[[dk]]]], dtype=torch.float32, device=device)
    descale_v = torch.tensor([[[[dv]]]], dtype=torch.float32, device=device)
    descale_s = torch.tensor([[[[1.0]]]], dtype=torch.float32, device=device)
    scale_s = torch.tensor([[[[1.0]]]], dtype=torch.float32, device=device)
    scale_o = torch.tensor([[[[1.0]]]], dtype=torch.float32, device=device)

    O_fp8, LSE = cudnn_fp8_forward(
        Q_fp8, K_fp8, V_fp8,
        descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        use_causal_mask, device,
    )

    if check:
        O_ref, LSE_ref = fp32_reference(
            Q_fp8, K_fp8, V_fp8, descale_q, descale_k, descale_v,
            use_causal_mask,
        )
        diff_lse = (LSE - LSE_ref).norm() / max(LSE_ref.norm().item(), 1e-12)
        diff_o = (O_fp8.float() - O_ref).norm() / max(O_ref.norm().item(), 1e-12)
        print(f"  check S={s_val} causal={use_causal_mask}: "
              f"|LSE_cudnn-LSE_ref|/|LSE_ref|={diff_lse:.4g}, "
              f"|O_cudnn-O_ref|/|O_ref|={diff_o:.4g}")

    if out_path.exists() and not overwrite:
        print(f"skip existing safetensors blob {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_blob = {
        "Q": Q_fp8.detach().cpu().contiguous(),
        "K": K_fp8.detach().cpu().contiguous(),
        "V": V_fp8.detach().cpu().contiguous(),
        "descale_q": descale_q.detach().cpu().contiguous(),
        "descale_k": descale_k.detach().cpu().contiguous(),
        "descale_v": descale_v.detach().cpu().contiguous(),
        "O": O_fp8.detach().cpu().contiguous(),
        "LSE": LSE.detach().cpu().contiguous(),
    }
    save_file(save_blob, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {out_path} ({size_mb:.1f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--check", action="store_true",
                        help="compare cuDNN FP8 SDPA against fp32 dequantized reference")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--desc-only", action="store_true",
                        help="only write the definition JSONs; skip workload + safetensor generation (no CUDA required)")
    args = parser.parse_args()

    if not args.desc_only and not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dataset_root = args.dataset_root.resolve()
    device = torch.device("cuda") if not args.desc_only else None
    if not args.desc_only:
        torch.manual_seed(BASE_SEED)
        torch.backends.cuda.matmul.allow_tf32 = False

    for use_causal_mask in IS_CAUSAL:
        for d_val in D_VALUES:
            name = definition_name(d_val, use_causal_mask)
            print(f"=== definition {name!r} (causal={use_causal_mask}, D={d_val}) ===")
            defn_dict = make_definition_dict(name, d_val, use_causal_mask)
            write_definition(dataset_root, defn_dict, args.overwrite)

            if args.desc_only:
                continue

            blob_dir = f"blob/workloads/attention/{name}"
            items = iter_or_build_workloads(dataset_root, name, d_val, blob_dir, args.overwrite)

            for item in items:
                prepare_one(dataset_root, item, use_causal_mask, device, args.check, args.overwrite)

            write_workloads(dataset_root, name, items, overwrite=True)


if __name__ == "__main__":
    main()
