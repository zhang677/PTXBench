"""Profile cuDNN attention backward using the EXACT setup from FA3 paper Fig. 6.

Source: /home/ubuntu/flash-attention-cutedsl/hopper/benchmark_attn.py
        (Dao-AILab/flash-attention, hopper/benchmark_attn.py)

The FA3 paper's Fig. 6 is produced by `benchmark_attn.py` with:
    dtype = torch.bfloat16
    causal = False
    headdim = 128
    bs_seqlen_vals = [(32,512),(16,1024),(8,2048),(4,4096),(2,8192),(1,16384)]
    nheads = dim // headdim = 2048 // 128 = 16
TFLOPS reported as: 2.5 * nFLOPS / latency, where
    nFLOPS = batch * nheads * 2 * seqlen_q * seqlen_k * (headdim + headdim_v)
i.e. 2 * 2 * B * H * S * S * D = 4 * B * H * S * S * D for non-causal.
The 2.5 multiplier is the paper's bwd accounting (≈ 5x mat-mul of fwd's 2x).

We run cuDNN's sdpa_backward directly using the same graph builder and the
same do_bench timing the FA3 benchmark uses, so the TFLOPS number is
apples-to-apples with what the paper plots.
"""
from __future__ import annotations

import math

import cudnn
import torch
from triton.testing import do_bench


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    raise ValueError(torch_type)


def cudnn_spda_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=-1):
    """Verbatim from flash-attention-cutedsl/hopper/benchmark_attn.py."""
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert v.shape == (b, nheads_k, seqlen_k, headdim)
    assert g.shape == (b, nheads, seqlen_q, headdim)
    assert o.shape == (b, nheads, seqlen_q, headdim)
    assert lse.shape == (b, nheads, seqlen_q, 1)

    q_gpu, k_gpu, v_gpu, o_gpu, g_gpu = q, k, v, o, g
    dq_gpu = torch.empty_like(q_gpu)
    dk_gpu = torch.empty_like(k_gpu)
    dv_gpu = torch.empty_like(v_gpu)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q_t = graph.tensor_like(q_gpu.detach())
    k_t = graph.tensor_like(k_gpu.detach())
    v_t = graph.tensor_like(v_gpu.detach())
    o_t = graph.tensor_like(o_gpu.detach())
    g_t = graph.tensor_like(g_gpu.detach())
    stats_t = graph.tensor_like(lse.detach())

    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q_t,
        k=k_t,
        v=v_t,
        o=o_t,
        dO=g_t,
        stats=stats_t,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal or window_size_left >= 0,
        sliding_window_length=window_size_left if window_size_left >= 0 and not causal else None,
    )

    dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
    dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
    dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q_t: q_gpu, k_t: k_gpu, v_t: v_gpu, o_t: o_gpu,
        g_t: g_gpu, stats_t: lse,
        dq: dq_gpu, dk: dk_gpu, dv: dv_gpu,
    }
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run():
        graph.execute(variant_pack, workspace)
        return dq_gpu, dk_gpu, dv_gpu

    return run


def flops_bwd(batch, nheads, seqlen, headdim):
    """FA3 benchmark's bwd FLOPs: 2.5 * (2 * 2 * B * H * S * S * D)."""
    nFLOPS = batch * nheads * 2 * seqlen * seqlen * (headdim + headdim)
    return 2.5 * nFLOPS


def bench_one(batch, nheads, seqlen, headdim, dtype=torch.bfloat16, repeats=30):
    device = "cuda"
    # Tensors in (B, H, S, D) layout (cuDNN's expected layout — the FA3 script
    # constructs (B,S,H,D) and then transpose(1,2)s before passing to cudnn).
    q = torch.randn(batch, nheads, seqlen, headdim, device=device, dtype=dtype)
    k = torch.randn(batch, nheads, seqlen, headdim, device=device, dtype=dtype)
    v = torch.randn(batch, nheads, seqlen, headdim, device=device, dtype=dtype)
    o = torch.randn(batch, nheads, seqlen, headdim, device=device, dtype=dtype)
    g = torch.randn(batch, nheads, seqlen, headdim, device=device, dtype=dtype)
    lse = torch.randn(batch, nheads, seqlen, 1, device=device, dtype=torch.float32)

    run = cudnn_spda_bwd_setup(q, k, v, o, g, lse, causal=False)
    # Warm + timing identical to FA3 benchmark's time_fwd: do_bench(warmup=3, rep=repeats).
    ms = do_bench(run, warmup=3, rep=repeats)
    tflops = flops_bwd(batch, nheads, seqlen, headdim) / (ms * 1e-3) * 1e-12
    return ms, tflops


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__}, cuDNN(py)={cudnn.__version__}, "
          f"cuDNN(rt)={torch.backends.cudnn.version()}")
    print()

    print("=== FA3 paper Fig. 6 config (bf16, headdim=128, nheads=16, non-causal) ===")
    print(f"{'B':>3} {'S':>6} | {'lat (ms)':>10} {'TFLOPS':>8}")
    fa3_grid = [(4, 512), (4, 1024), (4, 2048), (4, 4096), (4, 8192), (4, 16384)]
    for batch, seqlen in fa3_grid:
        ms, tf = bench_one(batch, nheads=16, seqlen=seqlen, headdim=128)
        print(f"{batch:>3} {seqlen:>6} | {ms:>10.3f} {tf:>8.1f}")

    print()
    print("=== User's workload (bf16, B=4, H=48, D=128) ===")
    print(f"{'S':>6} | {'lat (ms)':>10} {'TFLOPS':>8}")
    for s in (128, 256, 512, 1024, 2048, 4096):
        ms, tf = bench_one(batch=4, nheads=48, seqlen=s, headdim=128)
        print(f"{s:>6} | {ms:>10.3f} {tf:>8.1f}")


if __name__ == "__main__":
    main()
