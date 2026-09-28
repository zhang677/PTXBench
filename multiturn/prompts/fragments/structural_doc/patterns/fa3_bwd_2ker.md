# Optimization pattern: Asynchrony Through Warp Specialization for the Backward Pass
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
