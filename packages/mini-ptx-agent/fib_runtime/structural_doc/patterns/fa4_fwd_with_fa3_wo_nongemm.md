# Optimization pattern: Producer-Consumer asynchrony through warp-specialization and pingpong scheduling (FA-3)

### Warp-specialization

As with FlashAttention-2, the forward pass of **FlashAttention-3** is embarrassingly parallel in the batch size, number of heads, and query sequence length. Thus, it will suffice to give a CTA-level view of the algorithm, which operates on a tile $Q_i$ of the query matrix to compute the corresponding tile $O_i$ of the output. To simplify the description, we first give the warp-specialization scheme with a circular SMEM buffer that does not have in addition the GEMM–softmax overlapping. Let $d$ be the head dimension, $N$ the sequence length, and fix a query block size $B_r$ to divide $Q$ into $T_r = \lceil N / B_r \rceil$ blocks $Q_1, \ldots, Q_{T_r}$.

```latex
\begin{algorithm}
\caption{Algorithm 1: FlashAttention-3 forward pass without intra-consumer overlapping -- CTA view}
\begin{algorithmic}[1]
\Require Matrices $Q_i \in \mathbb{R}^{B_r \times d}$ and $K, V \in \mathbb{R}^{N \times d}$ in HBM, key block size $B_c$ with $T_c = \lceil N / B_c \rceil$.
\State Initialize pipeline object to manage barrier synchronization with $s$-stage circular SMEM buffer.
\If{in producer warpgroup}
    \State Deallocate predetermined number of registers.
    \State Issue load $Q_i$ from HBM to shared memory.
    \State Upon completion, commit to notify consumer of the load of $Q_i$.
    \For{$0 \le j < T_c$}
        \State Wait for the $(j \bmod s)$th stage of the buffer to be consumed.
        \State Issue loads of $K_j, V_j$ from HBM to shared memory at the $(j \bmod s)$th stage of the buffer.
        \State Upon completion, commit to notify consumers of the loads of $K_j, V_j$.
    \EndFor
\Else
    \State Reallocate predetermined number of registers as function of number of consumer warps.
    \State On-chip, initialize $O_i = (0) \in \mathbb{R}^{B_r \times d}$ and $\ell_i, m_i = (0), (-\infty) \in \mathbb{R}^{B_r}$.
    \State Wait for $Q_i$ to be loaded in shared memory.
    \For{$0 \le j < T_c$}
        \State Wait for $K_j$ to be loaded in shared memory.
        \State Compute $S_i^{(j)} = Q_i K_j^{\top}$ (SS-GEMM). Commit and wait.
        \State Store $m_i^{\text{old}} = m_i$ and compute $m_i = \max\!\left(m_i^{\text{old}},\, \mathrm{rowmax}(S_i^{(j)})\right)$.
        \State Compute $\widetilde{P}_i^{(j)} = \exp\!\left(S_i^{(j)} - m_i\right)$ and $\ell_i = \exp(m_i^{\text{old}} - m_i)\,\ell_i + \mathrm{rowsum}(\widetilde{P}_i^{(j)})$.
        \State Wait for $V_j$ to be loaded in shared memory.
        \State Compute $O_i = \mathrm{diag}(\exp(m_i^{\text{old}} - m_i))^{-1} O_i + \widetilde{P}_i^{(j)} V_j$ (RS-GEMM). Commit and wait.
        \State Release the $(j \bmod s)$th stage of the buffer for the producer.
    \EndFor
    \State Compute $O_i = \mathrm{diag}(\ell_i)^{-1} O_i$ and $L_i = m_i + \log(\ell_i)$.
    \State Write $O_i$ and $L_i$ to HBM as the $i$th block of $O$ and $L$.
\EndIf
\end{algorithmic}
\end{algorithm}
```

For our implementation of Algorithm 1 on Hopper, we use `setmaxnreg_inc_sync_fn` and `setmaxnreg_dec_sync_fn` for (de)allocations, TMA for loads of $Q_i$ and $\{K_j, V_j\}_{0 \le j < T_c}$, and WGMMA to execute the GEMMs in the consumer mainloop, with the SS or RS prefix indicating whether the first operand is sourced from shared memory or register file. For interpreting the execution flow of Algorithm 1, note that issuing TMA loads does not stall on the completion of other loads due to asynchrony. Moreover, in the producer mainloop, no waits will be issued for the first $s$ iterations as the buffer gets filled.

### Pingpong scheduling

The asynchronous nature of WGMMA and TMA, along with warp-specialization, opens up the opportunity to overlap the softmax computation of one warpgroup with the GEMM of another warpgroup. To motivate this, notice that non-matmul operations have much lower throughput than matmul operations on modern hardware accelerators. As an example, the H100 SXM5 GPU has 989 TFLOPS of FP16 matmul but only 3.9 TFLOPS of special functions such as exponential[^1] (necessary for softmax). For the attention forward pass in FP16 with head dimension 128, there are $512\times$ more matmul FLOPs compared to exponential operations, but the exponential has $256\times$ lower throughput, so exponential can take $50\%$ of the cycle compared to matmul. The situation is even worse with FP8, where the matmul throughput doubles but the exponential throughput stays the same.

Since the exponential is performed by a separate hardware unit (the multi-function unit), ideally we'd want the exponential calculation to be scheduled when the Tensor Cores are performing the matmul. To do so, we use synchronization barriers (`named_barrier_sync_fn` primitives) to force the GEMMs (GEMM1 — $PV$ of one iteration, and GEMM0 — $QK^{\top}$ of the next iteration) of warpgroup 1 to be scheduled before the GEMMs of warpgroup 2. As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs. Then the roles swap, with warpgroup 2 doing softmax while warpgroup 1 doing GEMMs (hence, "pingpong" scheduling). We generally find pingpong scheduling to improve performance (e.g., from 570 TFLOPS to 620–640 TFLOPS for FP16 forward with head dimension 128 and sequence length 8192).

### Attention variants

For multi-query attention and grouped query attention, we follow the approach in **FlashAttention-2** and adjust the tensor indexing to avoid duplicating $K$ and $V$ in HBM.

[^1]: The CUDA programming guide specifies that 16 operations of special functions can be performed per streaming multiprocessor (SM) per clock cycle. We multiply 16 by 132 SMs and 1830 MHz clock speed to get 3.9 TFLOPS of special functions.


## New pipeline to overlap matmul and softmax on Blackwell

Since the Blackwell architecture doubled the tensor core flops again, taking care to overlap softmax and tensor core operations is even more crucial than on Hopper. We follow a ping-pong schedule similar to FA-3, where two tiles of the output are computed per thread block. While one tile's tensor core operations are executed, the other tile computes softmax. While Hopper tensor cores hold the accumulator in registers, with four threads per row in an interleaved pattern, Blackwell tensor cores hold their accumulators in tensor memory. Additionally, a single accumulator tile on Blackwell is 128 by 128 elements large, where Hopper's tile size was 64 by 128.

The natural way to distribute work across these tiles is then to have two warpgroups of 128 threads each, with each thread processing an entire row. This eliminates the need for inter-warp shuffles to reduce the row max, and for multiple statistics registers per thread. Just like with FA-3, we explicitly synchronize the two softmax warpgroups to not overlap in their critical section, which is the part of exponential computation. Each softmax warpgroup proceeds by first loading the entire row into registers, then computing the maximum, then computing the softmax (i.e., subtract the max, rescale, exponentiate, convert to input precision), and finally computing the row sum.

Another difference from FA-3 is that since we transfer P via tensor memory rather than register file, we can decouple the rescaling of the output to a separate "correction" warpgroup and thus take it out of the critical path.

Several tensor memory partitionings are possible to achieve this pipeline overlap. All must allocate two tiles worth of output, leaving (at head dimension 128) half the tensor memory to store S and P. That memory can store two copies of S or four copies of P (assuming the input of the FP16 or BF16 tensor core). This leaves us with roughly two partitioning options for the remaining tensor memory: one tile of S and two tiles of P, or two tiles of S that overlap with P. We choose the latter because it allows us to start our software pipeline by immediately computing two S tiles. It also leaves some tensor memory to communicate rescale statistics to the correction warpgroup.

One issue of the larger Blackwell tile sizes and the chosen thread assignment is that, unless we re-load from tensor memory, we must hold an entire row of 128 elements in register. Given that we use two softmax warpgroups, one correction warpgroup, and one warpgroup to drive tensor cores and TMA units, assigning sufficient registers to softmax and preventing register spills is critical. For BF16 input data types, we need to hold 128 registers for the input, and potentially 64 registers for the output (plus miscellaneous and temporary registers). To reduce register pressure, we stage out storing P: The first three quarters are stored once (and trigger the corresponding MMA operations), and the last quarter is stored separately.
