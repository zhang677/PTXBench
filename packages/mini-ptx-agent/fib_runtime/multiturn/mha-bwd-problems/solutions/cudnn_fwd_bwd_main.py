"""cuDNN fwd+bwd via cudnn.pygraph with per-shape graph cache.

Mirrors the fa_fwd_bwd structure (forward + backward) but uses the cuDNN
python pygraph API directly instead of aten ops. Both the forward (sdpa)
and backward (sdpa_backward) graphs are cached by tensor
shape/stride/dtype/device, so repeated invocations only pay for
graph.execute. The forward recomputes O and stats so the backward consumes
values produced by cuDNN's own numerics, matching the cuDNN-pygraph
reference path used by FlashAttention-3 (FA3 paper Fig. 6).
"""
import math

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


_FWD_CACHE = {}
_BWD_CACHE = {}


def _build_fwd_graph(Q, K, V, O_out, stats_out):
    _, _, _, d = Q.shape
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
    o_t.set_output(True).set_dim(O_out.shape).set_stride(O_out.stride())
    s_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    s_t.set_dim(stats_out.shape).set_stride(stats_out.stride())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device=Q.device, dtype=torch.uint8)
    return graph, (q_t, k_t, v_t, o_t, s_t), workspace


def _build_bwd_graph(Q, K, V, O, dO, stats, dQ, dK, dV):
    _, _, _, d = Q.shape
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
    s_t = graph.tensor_like(stats.detach())
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


def run(Q, K, V, O, dO, L):
    # D = rowsum(dO * O) is part of the FA3 backward interface but cuDNN
    # recomputes the equivalent internally; accept it for schema compatibility.
    _ = O  # cuDNN-recomputed O is used so numerics match cuDNN's bwd expectations.
    _ = L  # likewise for stats.

    b, h, s, _ = Q.shape
    O_local = torch.empty_like(Q)
    stats_local = torch.empty((b, h, s, 1), dtype=torch.float32, device=Q.device)

    # Forward: cached per (Q, K, V, O_local, stats_local) layout. O_local and
    # stats_local are freshly allocated each call so they're not part of the key
    # — only their shape/stride/dtype/device matter, which match Q's layout.
    fwd_key = _cache_key(Q, K, V) + (
        (tuple(O_local.shape), tuple(O_local.stride()), O_local.dtype),
        (tuple(stats_local.shape), tuple(stats_local.stride()), stats_local.dtype),
    )
    fwd = _FWD_CACHE.get(fwd_key)
    if fwd is None:
        fwd = _build_fwd_graph(Q, K, V, O_local, stats_local)
        _FWD_CACHE[fwd_key] = fwd
    fwd_graph, (qf, kf, vf, of, sf), fwd_ws = fwd

    fwd_graph.execute(
        {qf: Q, kf: K, vf: V, of: O_local, sf: stats_local},
        fwd_ws,
    )

    # Backward: cached per (Q, K, V, O_local, dO, stats_local) layout.
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    bwd_key = _cache_key(Q, K, V, O_local, dO, stats_local)
    bwd = _BWD_CACHE.get(bwd_key)
    if bwd is None:
        bwd = _build_bwd_graph(Q, K, V, O_local, dO, stats_local, dQ, dK, dV)
        _BWD_CACHE[bwd_key] = bwd
    bwd_graph, (qb, kb, vb, ob, gb, sb, dqb, dkb, dvb), bwd_ws = bwd

    bwd_graph.execute(
        {qb: Q, kb: K, vb: V, ob: O_local, gb: dO, sb: stats_local,
         dqb: dQ, dkb: dK, dvb: dV},
        bwd_ws,
    )
    return dQ, dK, dV
