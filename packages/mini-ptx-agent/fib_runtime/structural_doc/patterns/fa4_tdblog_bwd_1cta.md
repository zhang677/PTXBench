# Attention backward pass (FA-4 for Blackwell)
For M=N=D=128, here are the feeds on B200 (per SM):
Tensor Cores (BF16): 8192 ops/cycle
Exponential unit: 16 ops/cycle
Shared Memory traffic: 128 bytes/cycle
And the speeds (clock-cycles per tile):
Backward (5 MMAs + MN exp) — 1-CTA:
Tensor Cores: 2560
Exp: 1024
SMEM: 3328
Takeaway: Backward is bottlenecked by shared memory bandwidth.
# 1-CTA Backward pass: Where shared memory traffic dominates

Optimizing FlashAttention backward can feel like stuffing an oversized rug into a room: flatten one corner and another pops up. Backward computes about 2.5x the tensor core work of the forward pass, chaining five MMA operations to recompute S and run the QK and PV gradient MMAs for dQ, dK, dP, and dV, plus the element-wise work for P and dS. On Blackwell, FLOPs are not the limiter for backward; shared memory bandwidth is.

## Pipeline: Overlap MMAs with softmax

Hopper-era FlashAttention-3 keeps MMA accumulators in registers, so register pressure often forces a more serial schedule. On Blackwell, accumulators live in TMEM, which makes it practical to keep multiple MMAs in flight while the CUDA cores handle the element-wise work for P and dS. Since exponential throughput is comparable to two MMAs in our roofline, hiding it is worth it.

The key overlap is simple: while we compute softmax for tile j, we already issue the dK and dQ MMAs for tile j-1.

To reduce shared memory traffic, the backward pass recomputes S and P in a transposed tile relative to the forward pass, so the intermediate is already $S^T$ and $P^T$. We can then store $P^T$ (and later $S^T$) directly in TMEM in the exact operand A layout consumed by the dV and dK MMAs respectively.

TMEM cannot hold five full accumulators and intermediates at once, so FA4 reuses TMEM columns across stages: S and P share one set of columns, and dP, dS, and dQ share another.

Phase 1: Prologue

The prologue initializes the first tile of the pipeline.
D_i = rowsum(dO_i ⊙ O_i) has been precomputed.
For the first query tile Q_0 and the current key/value tile K, V:
- Load or stage through SMEM: Q_0, K, V, dO_0, LSE_0, D_0
- Computes:
    S_0^T = K @ Q_0^T
    P_0^T = exp(S_0^T - LSE_0)
    dP_0^T = V @ dO_0^T
    dS_0^T = P_0^T ⊙ (dP_0^T - D_0^T)
    dV_0 += P_0^T · dO_0
For TMEM columns at tcgen05.mma, S_0^T and P_0^T stay at [0, 128), dV_0 stays at [128, 256), dP_0^T and dS_0^T stay at [256, 384).

Phase 2: Main loop over query blocks
The main loop iterates over query blocks: j = 1...B
Each iteration computes the current block’s softmax reconstruction and dV contribution, while consuming the previous block’s dS to compute dK and dQ.

For current query block Q_j:
- Load or stage through SMEM: Q_j, Q_{j-1}, K, dO_j, K, dS_{j-1}, V, LSE_j, D_j
- Computes:
    S_j^T = K @ Q_j^T
    dK_{j-1} += dS_{j-1}^T @ Q_{j-1}
    dQ_{j-1} = dS_{j-1} @ K (dQ_{j-1} also does AtomicAdd to the global mem)
    P_j^T = exp(S_j^T - LSE_j)
    dP_j^T = V @ dO_j^T
    dS_j^T = P_j^T ⊙ (dP_j^T - D_j^T)
    dV_j += P_j^T · dO_j
For TMEM columns at tcgen05.mma, S_j^T and P_j^T stay at [0, 128), dV_j stays at [128, 256), dQ_{j-1}, dP_j^T and dS_j^T stay at [256, 384), dK_{j-1} stays at [384, 512)

Phase 3: Tail for final dK and dQ
After the main loop finishes, the last score-gradient tile dS_B has been computed but not yet consumed for dK and dQ.

- Load or stage through SMEM: Q_B, K, dS_B
- Computes:
    dK_B += dS_B^T @ Q_B
    dQ_B = dS_B @ K (dQ_B also does AtomicAdd to the global mem)

For TMEM columns at tcgen05.mma, dQ_B stays at [256, 384), dK_B stays at [384, 512)

