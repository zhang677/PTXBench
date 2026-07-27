# Optimization Pattern: Postpone the division
\begin{algorithm}[t]
\caption{\textsc{FlashAttention-2} forward pass}
\label{alg:flashattention2-forward}
\begin{algorithmic}[1]
\Require Matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ in HBM, block sizes $B_c, B_r$.
\State Divide $\mathbf{Q}$ into $T_r = \left\lceil \frac{N}{B_r} \right\rceil$ blocks $\mathbf{Q}_1, \ldots, \mathbf{Q}_{T_r}$ of size $B_r \times d$ each, and divide $\mathbf{K}, \mathbf{V}$ into $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks $\mathbf{K}_1, \ldots, \mathbf{K}_{T_c}$ and $\mathbf{V}_1, \ldots, \mathbf{V}_{T_c}$, of size $B_c \times d$ each.
\State Divide the output $\mathbf{O} \in \mathbb{R}^{N \times d}$ into $T_r$ blocks $\mathbf{O}_i, \ldots, \mathbf{O}_{T_r}$ of size $B_r \times d$ each, and divide the logsumexp $L$ into $T_r$ blocks $L_i, \ldots, L_{T_r}$ of size $B_r$ each.
\For{$1 \leq i \leq T_r$}
    \State Load $\mathbf{Q}_i$ from HBM to on-chip SRAM.
    \State On chip, initialize $\mathbf{O}_i^{(0)} = (0)_{B_r \times d} \in \mathbb{R}^{B_r \times d}$, $\ell_i^{(0)} = (0)_{B_r} \in \mathbb{R}^{B_r}$, $m_i^{(0)} = (-\infty)_{B_r} \in \mathbb{R}^{B_r}$.
    \For{$1 \leq j \leq T_c$}
        \State Load $\mathbf{K}_j, \mathbf{V}_j$ from HBM to on-chip SRAM.
        \State On chip, compute $\mathbf{S}_i^{(j)} = \mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c}$.
        \State On chip, compute $m_i^{(j)} = \max(m_i^{(j-1)}, \mathrm{rowmax}(\mathbf{S}_i^{(j)})) \in \mathbb{R}^{B_r}$, $\tilde{\mathbf{P}}_i^{(j)} = \exp(\mathbf{S}_i^{(j)} - m_i^{(j)}) \in \mathbb{R}^{B_r \times B_c}$ pointwise, $\ell_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} \ell_i^{(j-1)} + \mathrm{rowsum}(\tilde{\mathbf{P}}_i^{(j)}) \in \mathbb{R}^{B_r}$.
        \State On chip, compute $\mathbf{O}_i^{(j)} = \mathrm{diag}(e^{m_i^{(j-1)} - m_i^{(j)}})^{-1}\mathbf{O}_i^{(j-1)} + \tilde{\mathbf{P}}_i^{(j)}\mathbf{V}_j$.
    \EndFor
    \State On chip, compute $\mathbf{O}_i = \mathrm{diag}(\ell_i^{(T_c)})^{-1}\mathbf{O}_i^{(T_c)}$.
    \State On chip, compute $L_i = m_i^{(T_c)} + \log(\ell_i^{(T_c)})$.
    \State Write $\mathbf{O}_i$ to HBM as the $i$-th block of $\mathbf{O}$.
    \State Write $L_i$ to HBM as the $i$-th block of $L$.
\EndFor
\State \Return the output $\mathbf{O}$ and the logsumexp $L$.
\end{algorithmic}
\end{algorithm}

# Optimization Pattern: Work partition between warps
Split Q across 4 warps while keeping K and V accessible by all warps. After each warp performs matrix multiply to get a slice of QK⊤, they just need to multiply with their shared slice of V to get their corresponding slice of the output. There is no need for communication between warps. The reduction in shared memory reads/writes could yield speedup
