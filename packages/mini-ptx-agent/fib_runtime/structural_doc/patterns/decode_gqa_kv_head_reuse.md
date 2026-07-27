# Optimization Pattern: GQA decode by KV-head reuse

Grouped-query attention has more query heads than KV heads. The query-to-KV grouping ratio is:

$$G = H_q / H_{kv}.$$

The high-value optimization is to load each paged K/V token once per KV head and reuse it across the query heads that share that KV head.

## High-value kernel strategy

Naive per-query-head decode reloads the same K/V vectors for every query head in a group. That wastes bandwidth on the dominant data path. Instead, map work around KV heads:

- Task owner: `(batch row, kv head, sequence split)`.
- Inside the task, compute the query-head group: `qh = G * kvh + r` for `0 <= r < G`.
- Load each paged `k_t, v_t` once, then update one independent online-softmax state per query head.
- Keep one `(m,l,A)` state per query head.
- Write either partial states or final outputs for each query head.

## CTA-level algorithm

```latex
\begin{algorithm}
\caption{Grouped-query paged decode with KV reuse}
\begin{algorithmic}[1]
\Require Query heads $\{q_{Gh+r}\}_{r=0}^{G-1}$, KV head $h$, paged KV cache, range $[a,b)$.
\For{$r=0$ to $G-1$}
    \State $m_r \leftarrow -\infty$, $l_r \leftarrow 0$, $A_r \leftarrow 0$.
\EndFor
\For{$t=a$ to $b-1$}
    \State Resolve page-table address for logical token $t$.
    \State Load $k_{h,t}$ and $v_{h,t}$ once.
    \For{$r=0$ to $G-1$}
        \State $s_r \leftarrow q_{Gh+r} \cdot k_{h,t} / \sqrt{d}$.
        \State $m'_r \leftarrow \max(m_r,s_r)$.
        \State $A_r \leftarrow \exp(m_r-m'_r)A_r + \exp(s_r-m'_r)v_{h,t}$.
        \State $l_r \leftarrow \exp(m_r-m'_r)l_r + \exp(s_r-m'_r)$.
        \State $m_r \leftarrow m'_r$.
    \EndFor
\EndFor
\For{$r=0$ to $G-1$}
    \State Write partial state $(m_r,l_r,A_r)$ or final output $A_r/l_r$.
\EndFor
\end{algorithmic}
\end{algorithm}
```

## Implementation notes

- For large head dimensions, a whole warp or half-warp can cooperate on the dot product for one query head. A CTA can process the grouped query heads serially, with warp groups, or with register tiling depending on occupancy.
- The best reuse point is the K/V vector load. Do not duplicate page-table lookup or K/V address arithmetic across query heads in the group.
- Keep `q` resident in registers or shared memory for the duration of the KV loop.
- If using split-KV, write partial states indexed by `(batch, query head, split)`, not `(batch, kv head, split)`, because the merge is per query head.
- Prioritize coalesced vector loads within each token vector and L2-friendly task order across rows and KV heads.
- Avoid a prefill-style query block. Decode query length is one; the parallelism comes from heads, batch rows, and KV sequence splits.

## Good source anchors

- TensorRT-LLM XQA/GQA decode: query-head groups sharing one KV head.
- FlashInfer decode: batch decode with grouped-query head mapping.
- PersistentKV: task mapping by KV-head group and sequence split.
