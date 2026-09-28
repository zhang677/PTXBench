# Optimization pattern: Producer-Consumer asynchrony through warp-specialization and pingpong scheduling

### Warp-specialization

As with FlashAttention-2, the forward pass of **FlashAttention-3** is embarrassingly parallel in the batch size, number of heads, and query sequence length. Thus, it will suffice to give a CTA-level view of the algorithm, which operates on a tile $Q_i$ of the query matrix to compute the corresponding tile $O_i$ of the output. To simplify the description, we first give the warp-specialization scheme with a circular SMEM buffer that does not have in addition the GEMM–softmax overlapping. Let $d$ be the head dimension, $N$ the sequence length, and fix a query block size $B_r$ to divide $Q$ into $T_r = \lceil N / B_r \rceil$ blocks $Q_1, \ldots, Q_{T_r}$.

```latex
\begin{algorithm}
\caption{Algorithm 1: FlashAttention-3 forward pass without intra-consumer overlapping -- CTA view}
\begin{algorithmic}[1]
\Require Matrices $Q_i \in \mathbb{R}^{B_r \times d}$ and $K, V \in \mathbb{R}^{N \times d}$ in HBM, key block size $B_c$ with $T_c = \lceil N / B_c \rceil$, scale $\tau = 1/\sqrt{d}$.
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
        \State Store $m_i^{\text{old}} = m_i$ and compute $S_i^{(j)} = S_i^{(j)} \times \tau$, $m_i = \max\!\left(m_i^{\text{old}},\, \mathrm{rowmax}(S_i^{(j)})\right)$.
        \State Compute $\widetilde{P}_i^{(j)} = \exp\!\left(S_i^{(j)} - m_i\right)$ and $\ell_i = \exp(m_i^{\text{old}} - m_i)\,\ell_i + \mathrm{rowsum}(\widetilde{P}_i^{(j)})$.
        \State Wait for $V_j$ to be loaded in shared memory.
        \State Compute $O_i = \mathrm{diag}(\exp(m_i^{\text{old}} - m_i)) O_i + \widetilde{P}_i^{(j)} V_j$ (RS-GEMM). Commit and wait.
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


# Optimization pattern: Asynchrony Through Warp Specialization for the Backward Pass

Similar to the forward pass, FlashAttention-3 uses warp specialization to handle # Optimization pattern: Asynchrony Through Warp Specialization for the Backward Pass
FlashAttention-3 uses warp specialization to handle asynchrony. When computing `dKdV` each thread block produces a local contribution to `dQ`, which must be accumulated into the global `dQ`. Therefore, we use a 2-kernel strategy, one for `dKdV` and the other for `dQ`. The below two algorithms just describe the computation process without good overlapping.

```latex
\begin{algorithm}
\caption{Algorithm 3: FlashAttention-3 backward pass with warp specialization (dKdV kernel)}
\begin{algorithmic}[1]
\Require Matrices $Q, K, V, O, dO \in \mathbb{R}^{N \times d}$ in HBM, logsumexp vector $L \in \mathbb{R}^N$ in HBM, block sizes $B_c, B_r$, scale $\tau = 1/\sqrt{d}$ (i.e. $L_i$ is the logsumexp of $\tau Q_i K^\top$).
\State In a preprocessing kernel, compute $D = \mathrm{rowsum}(dO \circ O) \in \mathbb{R}^N$ (pointwise multiply), write $D$ to HBM, and divide it into $T_r$ blocks $D_1,\ldots,D_{T_r}$ of size $B_r$ each.
\State Divide $Q$ into $T_r = \lceil N / B_r \rceil$ blocks $Q_1,\ldots,Q_{T_r}$ of size $B_r \times d$ each, and divide $K,V$ into $T_c = \lceil N / B_c \rceil$ blocks $K_1,\ldots,K_{T_c}$ and $V_1,\ldots,V_{T_c}$ of size $B_c \times d$ each.
\State Divide $dO$ into $T_r$ blocks $dO_1,\ldots,dO_{T_r}$ of size $B_r \times d$ each, and divide $L$ into $T_r$ blocks $L_1,\ldots,L_{T_r}$ of size $B_r$ each.
\State Initialize pipeline object to manage barrier synchronization with $s$-stage circular SMEM buffer.
\If{in producer warpgroup}
    \State Deallocate predetermined number of registers.
    \State Issue load $K_j$ and $V_j$ from HBM to shared memory.
    \State Upon completion, commit to notify consumer of the load of $K_j$ and $V_j$.
    \For{$1 \le i \le T_r$}
        \State Wait for the $(i \bmod s)$th stage of the buffer to be consumed.
        \State Issue loads of $Q_i,dO_i$ from HBM to shared memory at the $(i \bmod s)$th stage of the buffer.
        \State Upon completion, commit to notify consumers of the loads of $Q_i,dO_i$.
    \EndFor
\ElsIf{in consumer warpgroups}
    \State Reallocate predetermined number of registers as function of number of consumer warps.
    \State On-chip, initialize $dK_j = (0)_{B_c \times d}$ and $dV_j = (0)_{B_c \times d}$.
    \State Wait for $K_j$ and $V_j$ to be loaded in shared memory.
    \For{$1 \le i \le T_r$}
        \State Wait for $Q_i$ to be loaded in shared memory.
        \State Load $L_i,D_i$ from HBM to on-chip SRAM.
        \State On chip, compute $S_i^{(j)} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). Commit.
        \State Wait for $dO_i$ to be loaded in shared memory.
        \State On chip, compute $dP_i^{(j)} = dO_i V_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). Commit.
        \State On chip, wait for $S_i^{(j)}$, then compute $S_i^{(j)} = S_i^{(j)} \times \tau$, $P_i^{(j)} = \exp(S_i^{(j)} - L_i) \in \mathbb{R}^{B_r \times B_c}$.
        \State On chip, wait for $dP_i^{(j)}$, then compute $dS_i^{(j)} = P_i^{(j)} \circ (dP_i^{(j)} - D_i) \in \mathbb{R}^{B_r \times B_c}$, $dS_i^{(j)} = dS_i^{(j)}\times \tau$
        \State On chip, compute $dV_j \leftarrow dV_j + (P_i^{(j)})^\top dO_i \in \mathbb{R}^{B_c \times d}$ (RS-GEMM). Commit.
        \State On chip, compute $dK_j \leftarrow dK_j + (dS_i^{(j)})^\top Q_i \in \mathbb{R}^{B_c \times d}$ (RS-GEMM, $dS_i^{(j)}$ has been scale by $\tau$)). Commit and wait for both $dV_j$ and $dK_j$.
    \EndFor
    \State Write $dK_j$ and $dV_j$ to HBM as the $j$th block of $dK$ and $dV$.
\EndIf
\end{algorithmic}
\end{algorithm}
```

```latex
\begin{algorithm}
\caption{Algorithm 3: FlashAttention-3 backward pass with warp specialization (dQ kernel)}
\begin{algorithmic}[1]
\Require Matrices $Q, K, V, O, dO \in \mathbb{R}^{N \times d}$ in HBM, logsumexp vector $L \in \mathbb{R}^N$ in HBM, block sizes $B_c, B_r$, scale $\tau = 1/\sqrt{d}$ (i.e. $L_i$ is the logsumexp of $\tau Q_i K^\top$).
\State In a preprocessing kernel, compute $D = \mathrm{rowsum}(dO \circ O) \in \mathbb{R}^N$ (pointwise multiply), write $D$ to HBM, and divide it into $T_r$ blocks $D_1,\ldots,D_{T_r}$ of size $B_r$ each.
\State Divide $Q$ into $T_r = \lceil N / B_r \rceil$ blocks $Q_1,\ldots,Q_{T_r}$ of size $B_r \times d$ each, and divide $K,V$ into $T_c = \lceil N / B_c \rceil$ blocks $K_1,\ldots,K_{T_c}$ and $V_1,\ldots,V_{T_c}$ of size $B_c \times d$ each.
\State Divide $dO$ into $T_r$ blocks $dO_1,\ldots,dO_{T_r}$ of size $B_r \times d$ each, and divide $L$ into $T_r$ blocks $L_1,\ldots,L_{T_r}$ of size $B_r$ each.
\State Initialize pipeline object to manage barrier synchronization with $s$-stage circular SMEM buffer.
\If{in producer warpgroup}
    \State Deallocate predetermined number of registers.
    \State Issue load $Q_i$ and $dO_i$ from HBM to shared memory.
    \State Upon completion, commit to notify consumer of the load of $Q_i$ and $dO_i$.
    \For{$1 \le j \le T_c$}
        \State Wait for the $(j \bmod s)$th stage of the buffer to be consumed.
        \State Issue loads of $K_j,V_j$ from HBM to shared memory at the $(j \bmod s)$th stage of the buffer.
        \State Upon completion, commit to notify consumers of the loads of $K_j,V_j$.
    \EndFor
\ElsIf{in consumer warpgroups}
    \State Reallocate predetermined number of registers as function of number of consumer warps.
    \State On-chip, initialize $dQ_i = (0)_{B_r \times d}$.
    \State Load $L_i,D_i$ from HBM to on-chip SRAM.
    \State Wait for $Q_i$ and $dO_i$ to be loaded in shared memory.
    \For{$1 \le j \le T_c$}
        \State Wait for $K_j,V_j$ to be loaded in shared memory.
        \State On chip, compute $S_i^{(j)} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). Commit.
        \State Wait for $dO_i$ to be loaded in shared memory.
        \State On chip, compute $dP_i^{(j)} = dO_i V_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). Commit.
        \State On chip, wait for $S_i^{(j)}$, then compute $S_i^{(j)} = S_i^{(j)} \times \tau$, $P_i^{(j)} = \exp(S_i^{(j)} - L_i) \in \mathbb{R}^{B_r \times B_c}$.
        \State On chip, wait for $dP_i^{(j)}$, then compute $dS_i^{(j)} = P_i^{(j)} \circ (dP_i^{(j)} - D_i) \in \mathbb{R}^{B_r \times B_c}$, $dS_i^{(j)} = dS_i^{(j)}\times \tau$
        \State On chip, compute $dQ_i \leftarrow dQ_i + dS_i^{(j)}K_j \in \mathbb{R}^{B_r \times d}$ (RS-GEMM, $dS_i^{(j)}$ has been scale by $\tau$)). Commit and wait for $dQ_i$.
    \EndFor
    \State Write $dQ_i$ to HBM as the $i$th block of $dQ$.
\EndIf
\end{algorithmic}
\end{algorithm}
```

