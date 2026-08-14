# Optimization Pattern: Streaming Attention Forward in Standard Triton on Blackwell

Map one Triton program to one query tile for one batch item and query head. Keep that query tile and
its online-softmax state live while streaming over key/value tiles. This gives a standard-Triton
implementation of the FlashAttention forward algorithm without promising manual TMEM placement,
producer/consumer warp roles, or explicit MMA/softmax overlap.

## Program and storage contract

For dense Q, K, and V stored as `[batch, heads, sequence, head_dim]`, a typical grid is:

```python
grid = (
    triton.cdiv(query_length, BLOCK_M),
    batch_size,
    query_heads,
)
```

Inside the kernel, `tl.program_id(0)` chooses the query tile, axis 1 chooses the batch, and axis 2
chooses the query head. For grouped-query attention, map the query head to its shared KV head with
integer division and keep the query-head remainder only when the algorithm needs it.

Use device-created tensor descriptors when the physical last dimension is contiguous and all
descriptor alignment rules hold:

```python
q_desc = tl.make_tensor_descriptor(
    q_base,
    shape=[query_length, head_dim],
    strides=[stride_qm, 1],
    block_shape=[BLOCK_M, HEAD_DIM_ROUNDED],
    padding_option="zero",
)
k_desc = tl.make_tensor_descriptor(
    k_base,
    shape=[kv_length, head_dim],
    strides=[stride_kn, 1],
    block_shape=[BLOCK_N, HEAD_DIM_ROUNDED],
    padding_option="zero",
)
v_desc = tl.make_tensor_descriptor(
    v_base,
    shape=[kv_length, value_dim],
    strides=[stride_vn, 1],
    block_shape=[BLOCK_N, VALUE_DIM_ROUNDED],
    padding_option="zero",
)
```

Load Q once before the KV loop. Reuse the descriptor objects across loop iterations and change only
their offsets. Keep a masked pointer-tensor path when descriptor stride or alignment requirements do
not match the public input contract.

## Online-softmax loop

Use FP32 for the running row maximum, normalization sum, and output accumulator. Converting scores
to base-2 lets the loop use Triton's native `tl.math.exp2` path:

```python
RCP_LN2: tl.constexpr = 1.4426950408889634
m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
l_i = tl.zeros((BLOCK_M,), tl.float32)
acc = tl.zeros((BLOCK_M, VALUE_DIM_ROUNDED), tl.float32)

for kv_tile in range(0, tl.cdiv(kv_length, BLOCK_N)):
    offset_n = (kv_tile * BLOCK_N).to(tl.int32)
    k = k_desc.load([offset_n, 0])
    v = v_desc.load([offset_n, 0])

    scores = tl.dot(q, k.T) * softmax_scale
    # Apply causal, window, and sequence-tail masks before either reduction.
    scores = tl.where(valid_score, scores * RCP_LN2, -float("inf"))

    m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
    safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
    alpha = tl.math.exp2(m_i - safe_m_ij)
    p = tl.math.exp2(scores - safe_m_ij[:, None])

    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(q.dtype), v.to(q.dtype), acc)
    m_i = m_ij
```

The `valid_score` mask must protect both query and KV tails and must include the causal or window
condition. A descriptor's zero padding is not a score mask: padded K values can produce a finite dot
product and must still become `-inf` before row max and exponential.

Finalize fully masked rows explicitly, normalize once, and keep the LSE convention consistent with
backward:

```python
safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
output = acc / safe_l_i[:, None]
lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
```

If the public LSE contract uses natural logarithms, convert the final value and convert it back to
the same base in backward. Do not mix natural-log LSE with base-2 exponentials.

## Compiler controls and tuning

- Tune `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages` together. Head-dimension rounding changes
  descriptor shapes, dot legality, padding masks, and register pressure.
- Test descriptor and pointer paths independently. TMA setup is not automatically profitable for
  small sequence domains.
- Use `tl.range(..., num_stages=...)` only after the basic loop compiles and passes. Loop staging is
  distinct from the kernel launch's `num_stages` option.
- Do not put `warp_specialize=True` on this full online-softmax loop. Triton 3.7.1 limits automatic
  warp specialization to simple matmul loops; attention carries reductions, exponentials, masks,
  and several loop-carried tensors.
- Source-level descriptors and `tl.dot` do not prove a particular TMA, TMEM, or `tcgen05` lowering.
  Inspect generated artifacts when that distinction matters.

## Validation checklist

- Compare causal and noncausal modes, full and partial query/KV tiles, and fully masked rows.
- Test GQA/MQA head mapping and any broadcast batch mapping explicitly.
- Check both output and LSE using the declared log base and tolerance.
- Cover sequence lengths smaller than one block and lengths not divisible by either block size.
- Measure pointer and descriptor variants across the complete workload shape set.
