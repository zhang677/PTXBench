# Attention forward pass (FA-4 for Blackwell)
## Analysis
For M=N=D=128, here are the feeds on B200 (per SM):
Tensor Cores (BF16): 8192 ops/cycle
Exponential unit: 16 ops/cycle
Shared Memory traffic: 128 bytes/cycle
And the speeds (clock-cycles per tile) for Forward (2 MMAs + MN exp):
Tensor Cores: 1024
Exp: 1024
SMEM: 768

Takeaway: Forward is bottlenecked by compute and exponential

The forward pass has two matmuls, QK^T and PV. On Blackwell, tensor cores got much faster, but the exponential unit (MUFU.EX2) did not. So softmax is no longer “just the thing between the two matmuls” — it is a bottleneck that must be carefully pipelined.

The FWD pass in short:
- Ping-pong schedule 2x Q and 2x O tiles per CTA: maximize overlap between MMA and Softmax
- 2x softmax warpgroups: per-tile softmax with synchronization to not overlap when computing exponential
- Store P in TMEM in stages: mitigate register pressure
- Correction warpgroup: designated “correction” warpgroup to perform rescaling to remove from critical path
- Online softmax (conditional) rescaling: Rescale less frequently to minimize non-matmul operations

## Pipeline: Ping-pong Q tiles plus a dedicated correction stage

FlashAttention-4 computes two query tiles per CTA — $Q^H$ and $Q^L$ — each covering 128 query tokens, and alternates them in a ping-pong schedule.

Blackwell changes the softmax mapping. The accumulator tile for `S = QK^T` is 128x128 and lives in tensor memory; however, upon being read into registers, we have one thread per row for the partitioning of the tile as dictated by the hardware. We use two 128-thread warpgroups, one per Q tile, and each softmax warpgroup executes the following sequence of operations:

1. Each thread loads one 128-element row of `S` from tensor memory into registers
2. Reduce `rowmax` and `rowsum`
3. Compute `P = softmax(S)` and convert to BF16 precision
4. Store `P` back to tensor memory in stages to relieve register pressure (as opposed to holding 128 elements of S and 64 BF16 elements of P simultaneously)
5. Trigger the corresponding `PV` matmul as soon as a 3/4 chunk of `P` is stored

The critical detail is that exp is the bottlenecked section. We explicitly synchronize the two softmax warpgroups so they do not evaluate exp at the same time, thereby reducing MUFU contention.

To keep rescaling off the critical path, the kernel assigns it to a dedicated warpgroup. The correction warpgroup computes:

Only rescale when the max jump is large:

$$O_j = \begin{cases}\exp(m_{j-1}-m_j)\,O_{j-1} + \exp(S_j-m_j)\,V_j, & \text{if } m_j - m_{j-1} > \tau,\\O_{j-1} + \exp(S_j-m_{j-1})\,V_j, & \text{otherwise.}\end{cases}$$

Apply the final normalization at the end of the iteration $O_{\text{final}} = \frac{O}{l_{\text{final}}}$.

At the end we still normalize using the true final statistics, so skipping small rescale steps preserves the final output while deleting many vector computations from the critical path. We make the decision at warp granularity to avoid divergence.
