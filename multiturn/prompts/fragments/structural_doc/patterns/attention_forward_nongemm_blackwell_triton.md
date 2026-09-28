# Optimization Pattern: Standard-Triton Non-GEMM Attention Forward Refinements

Apply these refinements only after the streaming online-softmax kernel is correct. They target index,
mask, reduction, and exponential work around the two `tl.dot` operations. Standard Triton does not
expose source-level control over Blackwell MUFU-versus-FMA instruction assignment or a dedicated
softmax warpgroup, so do not promise those FA4 mechanisms.

## Pre-scale Q in base-2 units

When the scale is invariant for the kernel specialization, fold both softmax scaling and the
natural-to-base-2 conversion into Q once, outside the KV loop:

```python
RCP_LN2: tl.constexpr = 1.4426950408889634
q = (q * softmax_scale * RCP_LN2).to(MATMUL_DTYPE)

# In the loop:
scores = tl.dot(q, k.T)
```

This removes one full score-tile multiply per KV block. It changes rounding before the dot, so keep
the post-dot FP32 scaling path as the numerical baseline and compare output/LSE errors across the
actual tolerance suite.

## Separate full blocks from partial blocks

Do not execute boundary and mask work on blocks known to be fully valid. A common structure is one
loop or helper for full blocks and one helper for blocks that need a tail, causal, window, or custom
mask. Both helpers must implement the same online-softmax update and log-base convention.

```python
if FULL_BLOCKS_ONLY:
    # No sequence-tail predicate in the score update.
    scores = tl.dot(q, k.T)
else:
    scores = tl.dot(q, k.T)
    scores = tl.where(valid_score, scores, -float("inf"))
```

When block indices are contiguous, advance the KV offset arithmetically. Load an index array only
for genuinely sparse or noncontiguous traversal. Hoist invariant batch/head offsets, descriptor
construction, Q loads, and head-mapping arithmetic outside the KV loop.

## Safe-row specialization

If the launch contract proves that every active query row has at least one valid key, specialize a
fast path that omits the `m_ij == -inf` repair. Keep the safe path for arbitrary masks and empty KV
ranges. This must be a proven launch invariant, not an inference from common sequence lengths.

## Exponential and rescaling boundaries

- Prefer `tl.math.exp2` after base-2 score conversion. Do not inject an unvalidated polynomial
  approximation or bit-level exponent construction into a correctness-critical prompt.
- Always rescale both `l_i` and `acc` when the running maximum changes. Skipping small rescale steps
  is an algorithmic approximation and requires an explicit error budget; it is not a default Triton
  optimization.
- Keep FP32 running state until the final output conversion. Downcasting P for the PV dot is a
  separate, measured precision choice.
- Profile instruction mix and register pressure. Source ordering does not guarantee that vector
  exponentials overlap Tensor Core work.

Retain the unoptimized streaming kernel as a correctness and latency baseline. Enable pre-scaling,
full-block specialization, safe-row specialization, and sparse-index elimination independently so
a regression can be attributed to one transformation.
