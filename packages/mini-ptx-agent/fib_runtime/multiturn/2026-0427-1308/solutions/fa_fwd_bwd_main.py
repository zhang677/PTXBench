"""FlashAttention backward via aten._scaled_dot_product_flash_attention[_backward].

The aten flash backward needs auxiliary tensors (cum_seq_q, cum_seq_k,
max_q, max_k, philox_seed, philox_offset) that are only produced by the
matching flash forward. The workload-provided O/L/D were saved from cuDNN's
forward and are not interchangeable with FA's saved state, so we re-run the
flash forward here -- analogous to how the cuDNN reference re-runs the cuDNN
forward to recover its own metadata.
"""
import torch


def run(Q, K, V, O, dO, L):
    fwd = torch.ops.aten._scaled_dot_product_flash_attention(
        Q, K, V,
        0.0,    # dropout_p
        False,  # is_causal
        False,  # return_debug_mask
    )
    out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, _ = fwd

    dQ, dK, dV = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        dO, Q, K, V, out, logsumexp,
        cum_seq_q, cum_seq_k, max_q, max_k,
        0.0,    # dropout_p
        False,  # is_causal
        philox_seed, philox_offset,
    )
    return dQ, dK, dV
