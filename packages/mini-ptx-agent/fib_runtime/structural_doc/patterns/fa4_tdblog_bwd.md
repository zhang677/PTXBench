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
 
# 2-CTA backward pass: Reducing shared memory traffic and global atomic adds

Shared memory traffic. Even with the improved pipeline and with two of the ten GEMM operands kept in tensor memory, the backward pass is still limited by shared memory bandwidth. We mitigate this with Blackwell 2-CTA MMA mode, which partitions the output accumulator across the CTA pair. With M=256 and N=K=128, the two CTAs cooperate as one tile: each CTA stages half of operand B and keeps only its own accumulator slice. This roughly halves shared memory traffic for operand B.

For 2-CTA MMA tile shape: (M, N, K), the MMA for S, dP, dV, dK is (256, 128, 128); the MMA for dQ is (128, 128, 256). Logically:
- S^T = K @ Q^T: CTA-0 holds K[0:128][:] and Q^T[:][0:64], CTA-1 holds K[128:256][:] and Q^T[:][64:128]
- dP^T = V @ dO^T: CTA-0 holds V[0:128][:] and dO^T[:][0:64], CTA-1 holds V[128:256][:] and dO^T[:][64:128]
- dV += P^T @ dO: CTA-0 holds P^T[0:128][:] and dO[:][0:64], CTA-1 holds P^T[128:256][:] and dO[:][64:128]
- dK += dS^T @ Q: CTA-0 holds dS^T[0:128][:] and Q[:][0:64], CTA-1 holds dS^T[128:256][:] and Q[:][64:128]
- dQ = dS @ K: CTA-0 holds dS[:][0:128] and K[0:128][:], CTA-1 holds dS[:][128:256] and K[128:256][:]

Reduction axis conflict. We use M=256 and N=K=128 MMA tile across the five backward GEMMs to cut B traffic, but the nature of dQ MMA introduces a mismatch. In FlashAttention backward, each CTA owns a fixed KV tile (outer loop parallelized across N CTAs) and iterates over M tiles in the inner loop. The dQ update reduces over the KV sequence in the outer loop. 2-CTA MMA splits the output tile, not the reduction, and the dQ reduction dimension is N, which is already split across the CTA pair. Each CTA still needs the full reduction for the rows it owns.

Solution: DSMEM exchange. We resolve this by exchanging half of dS between the two CTAs using distributed shared memory within the cluster. This repacks dS so it is partitioned along the non-reduction axis: each CTA owns M/2 rows while holding the full 2N reduction. The per-CTA dQ MMA becomes (M/2, 2N)(2N, d), accumulating an (M/2, d) tile in tensor memory. In 2-CTA mode, the S, dP, dV, and dK MMAs keep M=256, while dQ uses M=128 with doubled reduction 2N=256. We then reorder the pipeline to hide DSMEM latency: compute dP for the current tile before computing dQ for the previous tile. Since the dQ tile fits in TMEM alongside P, it can reuse the TMEM region used for S, so dP and dQ no longer share a region as in 1-CTA mode. With this ordering, element-wise dS for the current tile overlaps with the dQ MMA from the previous iteration.

dQ atomic adds. As a side benefit, the dQ decomposition halves the number of global atomic reductions. Atomics are nondeterministic and expensive, and they occur in every inner-loop iteration. Consequently, in the 2-CTA backward pass each CTA writes only half of the dQ tile and performs half as many global atomic reductions as the 1-CTA counterpart.

Deterministic mode
The source of nondeterminism is the global atomic accumulation for dQ. FA4 provides a deterministic mode that serializes the global reductions with a semaphore-style lock and memory fence to enforce a fixed accumulation order. However, determinism does not have to mean “everything stops.” FA4 reduces lock contention with CTA swizzling, and uses a shortest-processing-time-first (SPT) ordering for causal masking to reduce stalls. In practice, deterministic backward reaches up to about 85-90% of the nondeterministic throughput in our benchmarks.

