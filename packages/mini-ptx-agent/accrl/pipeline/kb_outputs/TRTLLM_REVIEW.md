# TRT-LLM Definitions Review

**Date**: 2026-03-20
**Reviewed by**: Chengze (automated validator + manual review)
**Target hardware**: B200 (183GB HBM, ~4TB/s effective BW, ~500 TFLOPS effective bf16)

---

## Summary

**21 definitions across 4 tiers. Two systemic issues:**

### Issue 1: Workload sizes are WAY too small for B200

**Every single definition** has its smallest workloads running in <0.01ms. Even the **largest** workloads for most definitions are under 1ms. This violates the FEEDBACK.md 1-15ms target.

The core problem: these ranges were probably designed for H100 (~3.35 TB/s BW, ~990 TFLOPS), but B200 is ~2-4x faster. Even on H100, most of these would be under 1ms.

For bandwidth-bound ops (normalization, elementwise, quantization), you need **at least 500M-2B elements** to hit 1ms on B200. Most definitions max out at 30-230M elements.

| Definition | Largest workload elements | Est. latency | Target |
|---|---|---|---|
| trtllm_layernorm | 235M | 0.24ms | 1-15ms ❌ |
| trtllm_per_token_quant_fp8 | 235M | 0.24ms | ❌ |
| trtllm_moe_routing_topk2 | 2M | 0.002ms | ❌❌❌ |
| trtllm_per_channel_scale | 235M | 0.24ms | ❌ |
| trtllm_fused_qknorm_rope | 151M | 0.15ms | ❌ |
| trtllm_causal_conv1d_silu | 2.1B | 2.1ms | ✅ |
| trtllm_cumsum | 4M | 0.004ms | ❌❌❌ |
| trtllm_group_rmsnorm | 17M | 0.017ms | ❌❌ |
| trtllm_block_fp8_quant | 34M | 0.034ms | ❌❌ |
| trtllm_selective_scan | 67M | 0.067ms | ❌ |
| trtllm_sage_attention_quant | 4.3B | 4.3ms | ✅ |
| trtllm_ring_attention_recovery | 4.3B | 4.3ms | ✅ |

**Fix**: Increase `num_tokens` upper range to 32768-65536, increase `hidden_size` to 16384-28672 (Llama 405B intermediate size). For MoE routing, it's inherently tiny — may need to accept or batch multiple sequences.

### Issue 2: 3 definitions fail FEEDBACK.md complexity criterion

| Definition | Operation | Problem |
|---|---|---|
| `trtllm_per_channel_scale` | `output = input * scale` | Pure broadcast multiply. No reduction, no fusion. Literally `torch.mul` with broadcasting. |
| `trtllm_causal_attention_mask` | `torch.tril(ones(S,S))` | Just generating a constant boolean mask. Zero computation on the input. |
| `trtllm_beam_score_update` | `log_probs + cum_log_probs.unsqueeze(-1)` | Pure broadcast addition. |

These are so trivial that any correct implementation will match PyTorch performance. There's no optimization challenge. **Recommend removing these 3.**

---

## Per-Tier Details

### Tier 1 (6 definitions): 1 trivial, all sizing issues

| Definition | Complexity | Sizing |
|---|---|---|
| `trtllm_layernorm` | ✅ Good fusion (reduction + scale + shift) | ❌ Max 235M elems = 0.24ms |
| `trtllm_per_token_quant_fp8` | ✅ Good (absmax reduction + quantize) | ❌ Max 235M elems = 0.24ms |
| `trtllm_moe_routing_topk2` | ✅ Complex (softmax + topk + renorm) | ❌❌ Max 2M elems = 0.002ms |
| `trtllm_moe_routing_topk8` | ✅ Same | ❌❌ Max 2M elems = 0.002ms |
| `trtllm_per_channel_scale` | ❌ **TRIVIAL** (broadcast multiply) | ❌ |
| `trtllm_fused_qknorm_rope` | ✅ Great fusion (RMS + norm + trig) | ❌ Max 151M elems = 0.15ms |

**MoE routing** is a special case: the tensor is inherently small (num_experts is 8-128). To hit 1ms you'd need num_tokens > 1M which isn't realistic. Options: (a) accept it as a sub-ms kernel, (b) batch multiple requests, (c) remove.

### Tier 2 (6 definitions): 0 trivial, most sizing issues

| Definition | Complexity | Sizing |
|---|---|---|
| `trtllm_causal_conv1d_silu` | ✅ Good (conv + activation) | ✅ Max 2.1B elems = 2.1ms |
| `trtllm_topk_last_dim_k32/k128` | ✅ Partial sort | ❌ Max 131M elems = 0.13ms |
| `trtllm_cumsum` | ⚠️ Borderline (single op but non-trivial algorithm) | ❌❌ Max 4M elems = 0.004ms |
| `trtllm_embedding_lookup` | ✅ Gather + scale (random access) | ❌ Max 67M elems = 0.07ms |
| `trtllm_penalty_application` | ✅ Multiple conditional ops | ❌ Max 39M elems = 0.04ms |

### Tier 3 (6 definitions): 1 trivial, all sizing issues

| Definition | Complexity | Sizing |
|---|---|---|
| `trtllm_causal_attention_mask` | ❌ **TRIVIAL** (tril mask) | ⚠️ |
| `trtllm_ban_repeat_ngram` | ✅ Good (unfold + match + scatter) | ❌ Max 8M elems |
| `trtllm_group_rmsnorm` | ✅ Good (dual RMSNorm fusion) | ❌ Max 17M elems |
| `trtllm_relative_attention_bias` | ✅ Good (bucketing + gather) | needs checking |
| `trtllm_block_fp8_quant` | ✅ Good (block reduction + quant) | ❌ Max 34M elems |
| `trtllm_fused_relu2_fp8_quant` | ✅ Good (activation + quant fusion) | ❌ Max 34M elems |

### Tier 4 (5 definitions): 1 trivial, mixed sizing

| Definition | Complexity | Sizing |
|---|---|---|
| `trtllm_selective_scan` | ✅ Great (sequential SSM recurrence) | ❌ Max 67M elems |
| `trtllm_lru_recurrence` | ✅ Great (gated recurrence) | ❌ Max 67M elems |
| `trtllm_sage_attention_quant` | ✅ Good (4D block quant) | ✅ Max 4.3B elems |
| `trtllm_ring_attention_recovery` | ✅ Good (online softmax merge) | ✅ Max 4.3B elems |
| `trtllm_beam_score_update` | ❌ **TRIVIAL** (broadcast add) | ⚠️ |

---

## Recommended Actions

### Remove (3 definitions)
1. `trtllm_per_channel_scale` — trivial broadcast multiply
2. `trtllm_causal_attention_mask` — trivial mask generation
3. `trtllm_beam_score_update` — trivial broadcast add

### Fix workload ranges (18 definitions)
For all remaining definitions, increase the upper end of workload ranges:
- `num_tokens`: max should be **32768 or 65536** (not 16384)
- `hidden_size`: add **16384, 28672** (Llama 405B intermediate)
- `batch` for conv/SSM: add **128, 256**
- `seq_len` for conv/SSM: add **8192**
- Also increase lower bounds: remove anything under **1024** for token/batch dims

### Consider for MoE routing
MoE routing tensors are inherently small. Either:
- Accept as a <1ms problem (some optimization challenges exist even at small scales)
- Add a `num_groups` axis that batches multiple routing decisions
- Remove from the training set if sub-ms kernels aren't useful

### Optional: add cumsum variants
`trtllm_cumsum` is borderline trivial but the parallel prefix sum algorithm (Blelloch scan) IS interesting. Consider making it larger (batch=4096, length=65536) or adding segmented prefix sum variant.
