# Optimization Pattern: MLA paged decode over compressed KV and positional key embedding

This is not ordinary MHA/GQA. The prompt should tell the model to reason about MLA's compressed cache directly rather than expanding every token into full per-head K and V vectors.

## High-value kernel strategy

MLA decode reduces KV-cache bandwidth by storing a compressed latent vector and a smaller positional component. The decode kernel should keep the cache compressed as long as possible:

- Load compressed latent cache `c_t` and positional key embedding `kpe_t` through the page table.
- For each query head, compute the attention score from a content term plus a positional term.
- Accumulate values in the compressed or absorbed value space when possible.
- Apply any final projection after the online-softmax merge, not inside every token iteration, if that reduces bandwidth and repeated work.
- Use split-KV when context length is long; merge partial MLA states with the same `(m,l,A)` algebra as standard attention.

## Abstract MLA decode algorithm

Let each query head have a content query $q^c_h$ and positional query $q^p_h$. Let the paged cache store compressed latent vector $c_t$ and positional key embedding $k^p_t$. The implementation-specific learned projection from compressed latent to key/value space can be handled either by absorbing it into the query/output projections or by materializing per-token fragments in registers.

```latex
\begin{algorithm}
\caption{Paged MLA decode for one row/head/split task}
\begin{algorithmic}[1]
\Require Query components $q^c_h,q^p_h$, paged compressed cache $c_t$, paged positional key embedding $k^p_t$, range $[a,b)$.
\State Initialize $m \leftarrow -\infty$, $l \leftarrow 0$, compressed accumulator $A_c \leftarrow 0$.
\For{$t=a$ to $b-1$}
    \State Resolve page-table address for token $t$.
    \State Load compressed latent vector $c_t$.
    \State Load positional key embedding $k^p_t$.
    \State Compute content score $s^c_t$ using the MLA content projection or absorbed query.
    \State Compute positional score $s^p_t \leftarrow q^p_h \cdot k^p_t$.
    \State $s_t \leftarrow (s^c_t + s^p_t) / \sqrt{d_{\mathrm{score}}}$.
    \State $m' \leftarrow \max(m,s_t)$.
    \State $A_c \leftarrow \exp(m-m')A_c + \exp(s_t-m')c_t$.
    \State $l \leftarrow \exp(m-m')l + \exp(s_t-m')$.
    \State $m \leftarrow m'$.
\EndFor
\State Write partial state $(m,l,A_c)$ or compute final projected output from $A_c/l$.
\end{algorithmic}
\end{algorithm}
```

## Implementation notes

- Do not reuse a GQA prompt unchanged. MLA's compressed cache changes the dominant tradeoff: less KV bandwidth, more projection or absorbed-score computation.
- Avoid repeated page lookup across heads when they share the same row and token range.
- If the score projection is compute-heavy, assign enough work per CTA to keep arithmetic units busy rather than optimizing only for raw memory bandwidth.
- If the value projection is shared across heads, accumulate in compressed space and project once per head after the split merge.
- Keep the positional key-embedding path separate from the compressed latent path. It is smaller, may have different precision sensitivity, and should not force full expansion of compressed KV.
- For low precision variants, keep online-softmax state in FP32 and be conservative with the positional component; MLA papers report heterogeneous sensitivity between latent and positional parts.

## Good source anchors

- FlashMLA and FlashInfer MLA decode: paged MLA decode kernels and compressed-cache traversal.
- DeepSeek MLA descriptions: compressed latent KV cache and decoupled positional component.
- Hardware-centric MLA and SnapMLA analyses: decode tradeoffs between bandwidth savings, projection compute, and positional precision.
