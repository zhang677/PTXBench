# Optimization Pattern: Single-Owner Attention Backward in Standard Triton

Start from a one-program-per-query-tile backward kernel. The program keeps Q, dO, the forward LSE,
and the rowwise delta live while streaming over KV tiles. It computes its complete dQ tile locally;
dK and dV contributions from different query programs require atomic accumulation or a separate
ownership decomposition.

This is a standard-Triton baseline. It does not use source-level TMEM columns, distributed shared
memory, two-CTA MMA, remote barriers, or manual producer/consumer warpgroups.

## Forward state and log-base contract

Precompute the rowwise delta in a small destination-passing kernel:

```python
delta_i = tl.sum(dO_i.to(tl.float32) * O_i.to(tl.float32), axis=1)
```

Backward must consume the same LSE convention produced by forward. For base-2 LSE and base-2 scores:

```python
p = tl.math.exp2(scores_log2 - lse_log2[:, None])
```

If forward stores natural-log LSE, use the corresponding natural-log formula or explicitly convert
both operands. Treat fully masked rows consistently with forward so `-inf`, zero probabilities, and
zero gradients do not become NaNs.

## Per-KV-block gradient equations

For Q and dO tiles shaped `[BLOCK_M, D]` and K and V tiles shaped `[BLOCK_N, D]`:

```python
scores = tl.dot(q, k.T) * softmax_scale
scores = tl.where(valid_score, scores, -float("inf"))
p = tl.math.exp2(scores * RCP_LN2 - lse_log2[:, None])

dp = tl.dot(do, v.T)
ds = p * (dp - delta_i[:, None]) * softmax_scale
ds = tl.where(valid_score, ds, 0.0)

dq += tl.dot(ds.to(q.dtype), k)
dk_partial = tl.dot(ds.T.to(q.dtype), q)
dv_partial = tl.dot(p.T.to(q.dtype), do)
```

Apply `softmax_scale` exactly once to dS, as shown above, or equivalently to both completed dQ and
dK accumulators. Do not scale dV. Mask dS to zero after reconstructing probabilities; `-inf` is
correct for invalid scores but not for gradient matrix multiplication.

The program owns dQ and stores it once after completing the KV loop. If the grid has several query
tiles that contribute to the same dK/dV tile, update those outputs with `tl.atomic_add` in an
accumulation dtype whose numerical and alignment contracts are supported. Initialize accumulation
outputs before launch and document nondeterminism from atomic ordering.

## Descriptor and pointer paths

Device-created descriptors can load Q, K, V, and dO blocks when their physical last dimensions are
contiguous and aligned. Descriptor padding does not replace causal or tail masks. Keep pointer loads
for layouts that violate descriptor invariants, and mask head-dimension tails independently from
sequence tails.

Do not enable automatic warp specialization on the full backward loop. It contains multiple dots,
exponentials, masks, and loop-carried accumulators and is outside Triton 3.7.1's simple-matmul
contract.

## Validation checklist

- Compare dQ, dK, and dV independently for causal and noncausal attention.
- Cover partial query/KV blocks, fully masked rows, and head dimensions requiring rounding.
- Verify that softmax scaling and the LSE log base match forward exactly.
- Test repeated runs when atomic dK/dV is allowed and quantify nondeterministic error.
- Use this single-owner version as the correctness baseline for a split-ownership implementation.
