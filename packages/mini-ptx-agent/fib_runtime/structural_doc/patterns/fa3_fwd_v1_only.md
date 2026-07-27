# Optimization pattern: Producer-Consumer asynchrony through warp-specialization and pingpong scheduling

### Warp-specialization

As with FlashAttention-2, the forward pass of **FlashAttention-3** is embarrassingly parallel in the batch size, number of heads, and query sequence length. Thus, it will suffice to give a CTA-level view of the algorithm, which operates on a tile $Q_i$ of the query matrix to compute the corresponding tile $O_i$ of the output. We give the warp-specialization scheme with a circular SMEM buffer together with the intra-consumer two-stage[^2] GEMM–softmax overlapping. Let $d$ be the head dimension, $N$ the sequence length, and fix a query block size $B_r$ to divide $Q$ into $T_r = \lceil N / B_r \rceil$ blocks $Q_1, \ldots, Q_{T_r}$.

```latex
\begin{algorithm}
\caption{Algorithm 1: FlashAttention-3 forward pass with intra-consumer GEMM-softmax overlapping -- CTA view}
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
    \State Wait for $Q_i$ and $K_0$ to be loaded in shared memory.
    \State Compute $S_{cur} = Q_i K_0^{\top}$ using WGMMA. Commit and wait.
    \State Release the $0$th stage of the buffer for $K$.
    \State Compute $m_i$, $\widetilde{P}_{cur}$ and $\ell_i$ based on $S_{cur}$.
    \For{$1 \le j < T_c$}
        \State Wait for $K_j$ to be loaded in shared memory.
        \State Compute $S_{next} = Q_i K_j^{\top}$ using WGMMA. Commit but do not wait.
        \State Wait for $V_{j-1}$ to be loaded in shared memory.
        \State Compute $O_i = O_i + \widetilde{P}_{cur} V_{j-1}$ using WGMMA. Commit but do not wait.
        \State Wait for the WGMMA $Q_i K_j^{\top}$.
        \State Save $m_i^{old} \leftarrow m_i$
        \State Update $m_i$ and $\ell_i$ online, and compute $\widetilde{P}_{next} = \exp(S_{next} - m_i)$.
        \State Wait for the WGMMA $\widetilde{P}_{cur} V_{j-1}$ and then rescale $O_i$ as $O_i = \mathrm{diag}(\exp(m_i^{old} - m_i)) O_i$.
        \State Release the $(j \bmod s)$th, resp.\ $((j-1) \bmod s)$th stage of the buffer for $K$, resp.\ $V$.
        \State Copy $S_{next}$ to $S_{cur}$ and $\widetilde{P}_{next}$ to $\widetilde{P}_{cur}$
    \EndFor
    \State Wait for $V_{T_c - 1}$ to be loaded in shared memory.
    \State Compute $O_i = O_i + \widetilde{P}_{cur} V_{T_c - 1}$ using WGMMA. Commit and wait.
    \State \textbf{Epilogue:} Rescale $O_i$ with $O_i = \mathrm{diag}(\ell_i)^{-1} O_i$. Compute $L_i = m_i + \log(\ell_i)$. Write $O_i$ and $L_i$ to HBM as the $i$-th block of $O$ and $L$.
\EndIf
\end{algorithmic}
\end{algorithm}
```

For our implementation of Algorithm 1 on Hopper, we use `setmaxnreg_inc_sync_fn` and `setmaxnreg_dec_sync_fn` for (de)allocations, TMA for loads of $Q_i$ and $\{K_j, V_j\}_{0 \le j < T_c}$, and WGMMA to execute the GEMMs in the pipelined consumer mainloop: $Q_i K_j^{\top}$ is an SS-GEMM (both operands in shared memory) while $\widetilde{P}_{cur} V_{j-1}$ is an RS-GEMM (first operand in registers). For interpreting the execution flow of Algorithm 1, note that issuing TMA loads does not stall on the completion of other loads due to asynchrony. Moreover, in the producer mainloop, no waits will be issued for the first $s$ iterations as the buffer gets filled.

### Pingpong scheduling

The asynchronous nature of WGMMA and TMA, along with warp-specialization, opens up the opportunity to overlap the softmax computation of one warpgroup with the GEMM of another warpgroup. To motivate this, notice that non-matmul operations have much lower throughput than matmul operations on modern hardware accelerators. As an example, the H100 SXM5 GPU has 989 TFLOPS of FP16 matmul but only 3.9 TFLOPS of special functions such as exponential[^1] (necessary for softmax). For the attention forward pass in FP16 with head dimension 128, there are $512\times$ more matmul FLOPs compared to exponential operations, but the exponential has $256\times$ lower throughput, so exponential can take $50\%$ of the cycle compared to matmul. The situation is even worse with FP8, where the matmul throughput doubles but the exponential throughput stays the same.

Since the exponential is performed by a separate hardware unit (the multi-function unit), ideally we'd want the exponential calculation to be scheduled when the Tensor Cores are performing the matmul. To do so, we use synchronization barriers (`named_barrier_sync_fn` primitives) to force the GEMMs (GEMM1 — $PV$ of one iteration, and GEMM0 — $QK^{\top}$ of the next iteration) of warpgroup 1 to be scheduled before the GEMMs of warpgroup 2. As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs. Then the roles swap, with warpgroup 2 doing softmax while warpgroup 1 doing GEMMs (hence, "pingpong" scheduling). We generally find pingpong scheduling to improve performance (e.g., from 570 TFLOPS to 620–640 TFLOPS for FP16 forward with head dimension 128 and sequence length 8192).

### Attention variants

For multi-query attention and grouped query attention, we follow the approach in **FlashAttention-2** and adjust the tensor indexing to avoid duplicating $K$ and $V$ in HBM.

### Intra-warpgroup overlapping GEMMs and softmax

Even within one warpgroup, the consumer mainloop has sequential dependencies that impede parallelization within a single iteration: (local) softmax depends on the output $S_i^{(j)}$ of the first GEMM, while the second GEMM consumes the softmax output $\widetilde{P}_i^{(j)}$. Naively waited, these dependencies serialize softmax and the two GEMMs. The two-stage[^2] pipelined consumer branch in Algorithm 1 breaks these dependencies by carrying $S_{next}$ in registers across iterations: at iteration $j$, the second WGMMA ($\widetilde{P}_{cur} V_{j-1}$) is issued without waiting and overlaps with the first WGMMA ($Q_i K_j^{\top}$) and the softmax that follows it, so that the second WGMMA of iteration $j$ runs concurrently with the softmax of iteration $j + 1$. We use WGMMA here as a metonym for asynchronous GEMM.

While the pipelined structure offers theoretical performance gains, there are several practical aspects to consider:

**Compiler reordering.** The pseudocode represents an idealized execution order but the compiler (NVCC) often rearranges instructions for optimization. This can disrupt the carefully crafted WGMMA and non-WGMMA operation pipelining sequence, potentially leading to unexpected behavior or diminished performance gains. An analysis of the SASS code shows that the compiler generates overlapped code as expected (Section B.2).

**Register pressure.** To maintain optimal performance, register spilling should be minimized. However, the 2-stage pipeline requires additional registers to store intermediate results and maintain context between stages. Specifically, an extra $S_{next}$ must be kept in registers, leading to extra register usage of size $B_r \times B_c \times \mathrm{sizeof}(\texttt{float})$ per threadblock. This increased register demand may conflict with using larger block sizes (another common optimization), which is also register-hungry. In practice, trade-offs should be made based on profiling results.


[^1]: The CUDA programming guide specifies that 16 operations of special functions can be performed per streaming multiprocessor (SM) per clock cycle. We multiply 16 by 132 SMs and 1830 MHz clock speed to get 3.9 TFLOPS of special functions.
[^2]: Note that the number of stages of the overlapping scheme is bounded by, but need not equal, the number $s$ of stages in the circular SMEM buffer.
