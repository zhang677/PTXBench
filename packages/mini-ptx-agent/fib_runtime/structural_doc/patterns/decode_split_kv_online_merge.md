# Optimization Pattern: Split-KV decode with online-softmax merge

Decode underutilizes the GPU when one CTA owns a whole long context. Split the KV sequence across multiple CTAs, compute partial attention states independently, and merge the states with the same online-softmax algebra used inside FlashAttention.

## High-value kernel strategy

Split-KV decode is useful when the context is long, the active batch is small, or paged metadata reduces memory locality. The first kernel creates multiple partial states per `(batch row, query head)`. The second kernel reduces those states to the final output.

Keep the partial state as:

$$m = \max_t s_t,\qquad l = \sum_t \exp(s_t - m),\qquad A = \sum_t \exp(s_t - m)v_t.$$

Do not store normalized partial output alone unless you also store its denominator. The stable merge needs both `m` and `l`, and the accumulator is easiest to merge in unnormalized form.

## Per-split algorithm

```latex
\begin{algorithm}
\caption{Split-KV decode partial state}
\begin{algorithmic}[1]
\Require Query $q$, paged keys and values, split range $[a,b)$.
\State $m \leftarrow -\infty$, $l \leftarrow 0$, $A \leftarrow 0$.
\For{$t=a$ to $b-1$}
    \State Load $k_t,v_t$ through the page table.
    \State $s_t \leftarrow q \cdot k_t / \sqrt{d}$.
    \State $m' \leftarrow \max(m,s_t)$.
    \State $A \leftarrow \exp(m-m')A + \exp(s_t-m')v_t$.
    \State $l \leftarrow \exp(m-m')l + \exp(s_t-m')$.
    \State $m \leftarrow m'$.
\EndFor
\State Write partial tuple $(m,l,A)$.
\end{algorithmic}
\end{algorithm}
```

## Merge algorithm

```latex
\begin{algorithm}
\caption{Stable merge of split-KV decode partial states}
\begin{algorithmic}[1]
\Require Partial states $(m_r,l_r,A_r)$ for $r=1,\ldots,R$.
\State $m \leftarrow -\infty$, $l \leftarrow 0$, $A \leftarrow 0$.
\For{$r=1$ to $R$}
    \State $m' \leftarrow \max(m,m_r)$.
    \State $A \leftarrow \exp(m-m')A + \exp(m_r-m')A_r$.
    \State $l \leftarrow \exp(m-m')l + \exp(m_r-m')l_r$.
    \State $m \leftarrow m'$.
\EndFor
\State $O \leftarrow A/l$.
\State Write $O$.
\end{algorithmic}
\end{algorithm}
```

## Implementation notes

- A separate merge kernel is simple and robust. A fused persistent kernel is possible if all splits for a row/head can synchronize or use a deterministic owner CTA.
- Choose split count from context length and active batch. Too many splits add global-memory traffic for partial states and can make the merge kernel visible.
- Store partial `A` in FP32 when correctness tolerance is tight. The final output can be cast after the division.
- For GQA, split by KV head group and reuse one K/V stream for multiple query heads before writing partials.
- For MLA, the partial `A` may be an accumulator in compressed-value space rather than full value-head space. Merge before any final projection when that saves bandwidth.
- Split boundaries are logical token ranges, not physical page ranges. The page table maps logical tokens to physical pages.

## Good source anchors

- Flash-Decoding: split sequence length, produce per-split logsumexp/partial outputs, then reduce.
- FlashAttention online softmax: stable state update and merge algebra.
- FlashInfer decode: configurable split-KV and batch decode planning.
