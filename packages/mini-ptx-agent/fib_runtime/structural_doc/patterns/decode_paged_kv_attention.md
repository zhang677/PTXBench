# Optimization Pattern: Native paged-KV decode traversal

## High-value kernel strategy

Decode has a tiny query length, usually one token per request, and a long KV length. The bottleneck is moving KV/cache metadata and maintaining enough parallelism, not computing a large square attention tile.

- Treat the page table as part of the hot path. Cache or stage page indices when several query heads reuse the same KV token.
- Fuse page-table lookup, K/V load, score update, and output accumulation in one loop over KV tokens or token blocks.
- Avoid expanding paged KV into a dense temporary buffer.
- Prefer a compact CTA task shape such as `(batch row, kv head, sequence split)` rather than a full 2D prefill-style `(query block, key block)` grid.
- Use vectorized global loads for K/V vectors. TMA is most useful when the page table yields contiguous spans or when staging vectors into shared memory amortizes the setup cost.
- Keep per-row online-softmax state in registers: running max `m`, running denominator `l`, and unnormalized output accumulator `A`.

## CTA-level decode algorithm

```latex
\begin{algorithm}
\caption{Paged-KV decode for one row/head/split task}
\begin{algorithmic}[1]
\Require Query vector $q$, paged KV cache, page table $P$, token range $[t_0,t_1)$, head dimension $d$.
\State Initialize $m \leftarrow -\infty$, $l \leftarrow 0$, $A \leftarrow 0_d$.
\For{$t = t_0$ to $t_1-1$}
    \State Read page id $p \leftarrow P[t]$ and compute the physical address of token $t$.
    \State Load $k_t, v_t$ from the paged KV cache.
    \State Compute score $s \leftarrow q \cdot k_t / \sqrt{d}$.
    \State $m_{\mathrm{new}} \leftarrow \max(m, s)$.
    \State $\alpha \leftarrow \exp(m - m_{\mathrm{new}})$, $\beta \leftarrow \exp(s - m_{\mathrm{new}})$.
    \State $A \leftarrow \alpha A + \beta v_t$.
    \State $l \leftarrow \alpha l + \beta$.
    \State $m \leftarrow m_{\mathrm{new}}$.
\EndFor
\State Write partial state $(m,l,A)$ for this split, or write $O=A/l$ if this task covers the whole KV range.
\end{algorithmic}
\end{algorithm}
```

## Implementation notes

- The prompt should explicitly ask for direct paged addressing, not a fake contiguous layout.
- If several query heads share one KV head, compute page addresses once per KV head and reuse them across the query-head group.
- Group consecutive logical tokens when possible. A warp can cooperatively load the page ids for a token block, then load K/V vectors.
- Separate metadata bounds checks from the inner vector loop when possible: compute valid token count for the current split before the vectorized K/V loads.
- For variable lengths, use `last_page_len` or equivalent metadata to avoid reading padding.

## Good source anchors

- FlashInfer paged attention and decode APIs: native paged KV layout, batch decode wrappers, and CUDA graph compatible planning.
- PagedAttention/vLLM: logical block table to physical KV page mapping.
- PersistentKV: page-aware decode task scheduling and native block-table traversal.
