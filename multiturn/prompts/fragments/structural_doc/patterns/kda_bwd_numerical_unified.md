# KDA backward operator and numerical reference

Implement the direct KDA backward operator with the saved-input semantics,
output shapes, and dtypes described below. All six outputs must meet the
workload evaluator's numerical accuracy requirements. Bitwise reproduction of
the reference or its intermediate values is not required.

The arithmetic recipe below explains the reference's rounding-sensitive
behavior; it is a correctness guide, not an exclusive implementation contract.
Alternative formulations, reduction groupings, operation associations, and
fusion are allowed when they preserve the operator semantics and meet the
required output accuracy. Use the detailed recipe to diagnose numerical
mismatches rather than assuming every intermediate must be reproduced exactly.

Kernel decomposition, physical tile sizes, thread layouts, storage,
recomputation, and pipelining are implementation choices. A completed numerical
intermediate expresses a dependency; it does not require a separate launch,
memory write, or whole-block synchronization. Independent work may overlap,
and values may remain in registers or other suitable storage across operations.
The logical groups and loop orders describe the reference arithmetic, not a
required execution schedule.

The equations are parameterized by batch size `B`, sequence length `T`, heads
`H`, widths `K,V`, and chunk size `C`. The reference workload uses
`B=4, T=4096, H=96, K=V=128, C=64`, with V-first state, `safe_gate=True`,
recomputation enabled, no packed sequences or context parallelism, and
`scale=1/sqrt(K)`.

## Direct operator contract

```text
dq, dk, dv, db, dg, dh0 = run(
    q, k, v, g, beta, Aqk, Akk, initial_state, do, dht, scale
)
```

| Name | Shape | Dtype | Meaning |
|---|---:|---|---|
| `q`, `k` | `[B,T,H,K]` | BF16 | Saved, already-normalized query and key |
| `v` | `[B,T,H,V]` | BF16 | Saved value |
| `g` | `[B,T,H,K]` | FP32 | Saved chunk-local cumulative base-2 log decay |
| `beta` | `[B,T,H]` | BF16 | Saved activated update coefficient |
| `Aqk` | `[B,T,H,C]` | BF16 | Saved scaled causal Q/K factor |
| `Akk` | `[B,T,H,C]` | BF16 | Saved row-wise WY inverse factor |
| `initial_state` | `[B,H,V,K]` | FP32 | Initial state, V-first and K-last |
| `do` | `[B,T,H,V]` | BF16 | Upstream output gradient |
| `dht` | `[B,H,V,K]` | FP32 | Upstream final-state gradient |
| `scale` | scalar | FP32 | Output scale |
| `dq`, `dk` | `[B,T,H,K]` | FP32 | Raw direct query and key gradients |
| `dv` | `[B,T,H,V]` | BF16 | Value gradient |
| `db` | `[B,T,H]` | FP32 | Gradient of activated `beta` |
| `dg` | `[B,T,H,K]` | FP32 | Gradient before the natural-log cumulative sum |
| `dh0` | `[B,H,V,K]` | FP32 | Initial-state gradient |

Consume saved `g`, `Aqk`, and `Akk` directly. Do not reconstruct them or apply
Q/K normalization, beta sigmoid, raw-gate activation, or parameter VJPs. This
direct contract keeps `dq`, `dk`, `db`, and `dg` in FP32; a public autograd
wrapper may cast them and therefore has a different output boundary.

The active workload has one value head per Q/K head, so there is no grouped-head
reduction. Require `T % C == 0` for this dense direct contract.

## Layout, gates, and numerical notation

Suppress batch and head indices for one chunk. Use

```text
Q, Kd : [C,K]
Vv    : [C,V]
S     : [V,K]       # external and live state, V-first / K-last
Aqk, Akk : [C,C]    # logical chunk views of the saved tensors
```

Token rows are `i,j in [0,C)`, key coordinates are `d`, and value coordinates
are `r`. All scans and cumulative sums reset at logical-sequence boundaries;
chunks must not cross such boundaries.

Let `ell_i[d]` be the natural-log decay increment before the forward chunk-local
cumulative sum. The saved tensor and its derived factors are

```text
g_i[d]     = (sum_{s<=i} ell_s[d]) / ln(2)
gamma_i[d] = E(g_i[d])
eta_i[d]   = E(g_{C-1}[d] - g_i[d])
R_ij[d]    = E(g_i[d] - g_j[d])
```

where the decay is independently defined for every key coordinate. The
reference arithmetic uses the following notation:

```text
F(x_bf)       exact promotion of BF16 x_bf to FP32
RBF(x)        one round-to-nearest-even conversion of FP32 x to BF16
E(x)          base-2 exponential on FP32 x
BFMM(X,Y)     BF16 tensor-core operands, FP32 accumulation and result
BFMMA(X,Y,Acc) BF16 tensor-core product accumulated into existing FP32 Acc
TFMM(X,Y)     FP32 operands entering tensor cores at TF32 input precision
lower(X)      inclusive lower triangle, j <= i
strict(X)     strict lower triangle, j < i
```

In this recipe, `TFMM` uses TF32 input precision. Reductions accumulate in
FP32. `BFMMA` continues an existing tensor-core accumulator, whereas standalone
`BFMM` starts from zero. A separately completed product followed by scalar
arithmetic, including scalar FP32 FMA, rounds differently from using the scalar
term as the tensor-core accumulator. Folding scalar arithmetic into a matrix
accumulator, or changing grouping across the displayed rounding boundaries, can
therefore change the result.

## Semantic chunk algebra

The token recurrence, with state `S [V,K]`, is

```text
alpha_t[d] = exp(ell_t[d])
Sbar_t     = S_{t-1} * alpha_t[None,:]
residual_t = beta_t * (v_t - Sbar_t @ k_t)
S_t        = Sbar_t + residual_t[:,None] * k_t[None,:]
o_t        = S_t @ (scale * q_t)
```

For the parallel chunk form define

```text
L_ij  = beta_i * sum_d(k_i[d] * k_j[d] * R_ij[d])  if j < i else 0
A     = inverse(I + L)
P0_ij = sum_d(q_i[d] * k_j[d] * R_ij[d])            if j <= i else 0

Kgamma = Kd * gamma
Keta   = Kd * eta
Qgamma = Q * gamma
Bv     = beta[:,None] * Vv
Bk     = beta[:,None] * Kgamma
U      = A @ Bv
W      = A @ Bk
X      = U - W @ S.T
O      = scale * (Qgamma @ S.T + P0 @ X)
Snext  = gamma_{C-1}[None,:] * S + X.T @ Keta
```

`Aqk` is a BF16 materialization of `lower(scale*P0)`. `Akk` is the supplied
BF16 materialization of the row-wise inverse `A`; its forward construction may
have used mixed-precision block solves, so it is not interchangeable with a
fresh exact inverse. These statements describe the saved inputs, not work to
repeat inside backward.

The returned `dg` is with respect to `ell`, not the stored numeric `g`. After
forming the cumulative adjoint, return

```text
dg_j[d] = sum_{i=j..C-1} bar_ell_i[d]
```

with no extra `ln(2)` or `1/ln(2)` factor.

## Reference arithmetic recipe

Suffixes `_bf` and `_fp` describe the precision of these reference values at
a dependency boundary, not mandatory storage. Retain these distinctions when
using the recipe directly. An optimized alternative may change internal
precision, grouping, or fusion, provided all outputs still meet the required
accuracy; greater intermediate precision alone does not guarantee agreement.

### WY operands and forward state scan

For every chunk row:

```text
Bv_bf[i,r]       = RBF(F(beta[i]) * F(Vv[i,r]))
key_beta_bf[i,d] = RBF(F(Kd[i,d]) * F(beta[i]))
Bk_bf[i,d]       = RBF(F(key_beta_bf[i,d]) * E(g[i,d]))
Qgamma_bf[i,d]   = RBF(F(Q[i,d]) * E(g[i,d]))
Keta_bf[i,d]     = RBF(F(Kd[i,d]) * E(g[C-1,d] - g[i,d]))

U_bf = RBF(BFMM(Akk, Bv_bf))
W_bf = RBF(BFMM(Akk, Bk_bf))
```

`Bk_bf` has two BF16 boundaries: the BF16-by-BF16 `K*beta` product rounds
before it is promoted for multiplication by the FP32 exponential, and the
result rounds again before the matrix product. `Qgamma_bf` and `Keta_bf`
are independently rounded BF16 operands that may be produced where consumed.
The safe-gate column path independently forms the same `key_beta_bf` value
before applying its anchored exponential.

Initialize `S_live_fp` from FP32 `initial_state`. For chunks in increasing
order:

```text
H_bf      = RBF(S_live_fp)                     # save chunk-entry state
pred_fp   = BFMM(W_bf, H_bf.T)
X_fp      = F(U_bf) - pred_fp
X_bf      = RBF(X_fp)                          # save corrected value
update_fp = BFMM(X_bf.T, Keta_bf)              # complete product from zero
S_live_fp = E(g[C-1])[None,:] * S_live_fp + update_fp
```

The update consumes BF16 `X_bf`; the live state remains FP32 between chunks.

The reference completes `update_fp` from a zero tensor-core accumulator and
then combines it with the live state; the scalar update may use FP32 FMA.
Using `gamma_end * S_live_fp` as the product's initial accumulator, as in
`BFMMA(X_bf.T, Keta_bf, gamma_end * S_live_fp)`, has different rounding.
A final `dg` failure can originate in the forward `H_bf` and `X_bf` snapshots;
passing `dv` and `dh0` does not establish that those snapshots match.

### Output interaction and reverse state scan

The VJP through the local output interaction is

```text
dP0_fp        = lower(scale * BFMM(do, X_bf.T))
barX_local_bf = RBF(BFMM(Aqk.T, do))
```

`dP0_fp` is the FP32 adjoint of unscaled `P0`. Saved `Aqk` already contains
`scale`, so do not apply another scale in the `barX_local` path.

Initialize `dS_live_fp` from FP32 `dht`. For chunks in decreasing order:

```text
dH_bf  = RBF(dS_live_fp)
barX_fp = F(barX_local_bf) + BFMM(Keta_bf, dH_bf.T)
barX_bf = RBF(barX_fp)

add_fp = scale * BFMM(do.T, Qgamma_bf) - BFMM(barX_bf.T, W_bf)
dS_live_fp = E(g[C-1])[None,:] * dS_live_fp + add_fp
```

`dH_bf` feeds both the corrected-value and later key-gradient paths. Keep the
live adjoint FP32; after processing the first sequence chunk, store it directly
as FP32 `dh0 [V,K]`.

### Direct state/output gradients

```text
dq_state_fp = BFMM(do, H_bf) * E(g) * scale
dk_state_fp = BFMM(X_bf, dH_bf) * E(g[C-1] - g)

dg_state_fp[i,d] = F(Q[i,d])  * dq_state_fp[i,d]
                   - F(Kd[i,d]) * dk_state_fp[i,d]

state_dot_fp[d] = sum_r(F(H_bf[r,d]) * F(dH_bf[r,d]))
end_k[d] = state_dot_fp[d] * E(g[C-1,d])
           + sum_i(F(Kd[i,d]) * dk_state_fp[i,d])
dg_state_fp[C-1,d] += end_k[d]
```

`state_dot_fp[d]` contracts the BF16 chunk-entry state snapshot with the BF16
chunk-exit adjoint snapshot over the value axis, giving one FP32 value per key
coordinate. Promote both operands to FP32 before multiplying; every
BF16-by-BF16 product is then exact in FP32, and products and partial sums are
accumulated in FP32. The reduction order, value tiling, and FMA contraction are
not numerically significant here. Do not round the products, partial sums, or
the result to BF16, and do not substitute the FP32 live states for the BF16
snapshots. The Q/K and `barW` matrix products over V continue their FP32
tensor-core accumulators across value groups.

### WY right-hand side and saved-inverse VJP

```text
barW_bf  = -RBF(BFMM(barX_bf, H_bf))
barBv_fp = BFMM(Akk.T, barX_bf)
barBk_fp = BFMM(Akk.T, barW_bf)

dv_bf[i,r] = RBF(F(beta[i]) * barBv_fp[i,r])
gamma_beta[i,d] = E(g[i,d]) * F(beta[i])
Kgamma_fp[i,d] = F(Kd[i,d]) * E(g[i,d])
dk_rhs_fp[i,d] = barBk_fp[i,d] * gamma_beta[i,d]

db_rhs_fp[i] = 0
for each value tile vt in increasing order:
    db_rhs_fp[i] += sum_{r in vt}(barBv_fp[i,r] * F(Vv[i,r]))
for each key tile kt in increasing order:
    db_rhs_fp[i] += sum_{d in kt}(barBk_fp[i,d] * Kgamma_fp[i,d])
dg_rhs_fp[i,d] = (Kgamma_fp[i,d] * barBk_fp[i,d]) * F(beta[i])
```

Keep `barBv`, `barBk`, `dk_rhs`, `db_rhs`, and `dg_rhs` in FP32. `dv_bf` is the
final BF16 value gradient. The value and key tiles above are FP32 partial
groups whose width depends on the reference's selected configuration; they do
not fix physical storage or transfer tiles, and regrouping these FP32 partials
is subject only to output accuracy. The intra-chunk pair reductions below use
logical groups of 32 key coordinates.
Preserve the unrounded `Kgamma_fp` in the scalar VJPs and the displayed factor
association.

For the inverse VJP, first form a separately rounded key operand without beta:

```text
Kgamma_bf = RBF(Kgamma_fp)

barA0_fp = 0
for each value tile vt in increasing order:
    barA0_fp = BFMMA(barX_bf[:,vt], Vv[:,vt].T, barA0_fp)
for each key tile kt in increasing order:
    barA0_fp = BFMMA(barW_bf[:,kt], Kgamma_bf[:,kt].T, barA0_fp)

D_bf[i,j] = RBF(barA0_fp[i,j] * F(beta[j]))  if j < i
              BF16(0)                         otherwise
tmp_bf    = RBF(BFMM(D_bf, Akk.T))
dL_fp     = strict(-BFMM(Akk.T, tmp_bf))
```

The value and key products share one FP32 tensor-core accumulator. Add all
value tiles first, then each key tile in order; do not form independent full
value/key products and add once.
Both `D_bf` and `tmp_bf` are checker-visible BF16 boundaries; do not retain both
inverse products in FP32.

### Safe-gate intra-chunk VJP

The pair dependencies below use `R_ij` as algebraic shorthand. Evaluate the
ratios using the anchored schedule below, with FP32 `dP0_fp` and `dL_fp`:

```text
dq_pair[i,d] = sum_{j<=i} dP0[i,j] * Kd[j,d] * R_ij[d]

dk_first_unscaled[i,d] = sum_{j<i} dL[i,j] * Kd[j,d] * R_ij[d]
dk_first[i,d] = dk_first_unscaled[i,d] * F(beta[i])

dk_second[j,d] = sum_{i>=j} dP0[i,j] * Q[i,d] * R_ij[d]
                 + sum_{i>j} dL[i,j] * F(key_beta_bf[i,d]) * R_ij[d]

db_inv_tile[kt,i] = sum_{d in kt}(dk_first_unscaled[i,d] * F(Kd[i,d]))

dg_pair[i,d] = F(Q[i,d]) * dq_pair[i,d]
               + (dk_first[i,d] - dk_second[i,d]) * F(Kd[i,d])
```

For the active safe-gate arithmetic, group the 64 rows into four logical `BC=16`
subchunks and the 128 key coordinates into four logical `BK=32` tiles. These
groups determine gate anchors and numerical partials, not thread-block or
memory-transfer shapes. Diagonal blocks use row 8 as their midpoint gate anchor;
cross-subchunk row paths use the first row of the destination block, and column
paths use the last row of the source block. Factor each exponential ratio around
its anchor. For example,

```text
# One anchored row-path block; a is its gate-anchor row.
Kanch_fp[j,d] = F(Kd[j,d]) * E(g[a,d] - g[j,d])

dq_block_fp = TFMM(dP0_block_fp, Kanch_fp)
dk_first_unscaled_block_fp = TFMM(dL_block_fp, Kanch_fp)

dq_block_fp[i,d] *= E(g[i,d] - g[a,d])
dk_first_unscaled_block_fp[i,d] *= E(g[i,d] - g[a,d])
```

The analogous column-path contractions also use `TFMM`; cross-subchunk loops
continue their existing FP32 tensor-core accumulators.

The anchored factors are algebraically equal to `R_ij[d]` but have different
rounding from a direct exponent difference. In both diagonal and cross-block
column paths, form `key_beta_bf = RBF(Kd * beta)` first, promote it, and then
multiply by the anchored exponential. Keep that scaled operand FP32 at the
TF32 matrix-product boundary; do not replace the intermediate BF16 product
with `F(Kd) * F(beta)` or cast the scaled operand back to BF16. Row paths
instead apply beta to `dk_first_unscaled` after the contractions.

```text
# Key-beta contribution to one column-path block.
key_beta_bf[i,d] = RBF(F(Kd[i,d]) * F(beta[i]))
key_col_fp[i,d] = F(key_beta_bf[i,d]) * E(g[i,d] - g[a,d])

dk_second_L_block_fp = TFMM(dL_block_fp.T, key_col_fp)
dk_second_L_block_fp[j,d] *= E(g[a,d] - g[j,d])
```

The FP32 matrix operands enter tensor cores at TF32 input precision. Pair
contributions remain FP32 when combined with the state/WY gradients; they
are not separately stored BF16 tensors.

Reduce `dk_first_unscaled * Kd` before multiplying the first key path by beta.
Compute `db_inv_tile` independently for each of the four key tiles, yielding four
FP32 partials. Preserve the FP32 reduction grouping of those partials and
then add `db_rhs_fp` once. Do not cast either term to BF16.

`dg_pair` names the algebraic contribution. Preserve the final expression
below when combining it with the existing state/WY gradient, since regrouping
the terms changes FP32 rounding and scalar FMA contraction.

### Final accumulation and outputs

```text
dq_fp     = dq_pair + dq_state_fp
dk_base_fp = dk_state_fp + dk_rhs_fp
dk_fp     = (dk_first + dk_base_fp) + dk_second
db_fp     = reduce_key_tile_partials_fp32(db_inv_tile) + db_rhs_fp
dg_base_fp = dg_state_fp + dg_rhs_fp
dg_cum_fp = F(Q) * dq_pair + ((dk_first - dk_second) * F(Kd) + dg_base_fp)
dg_fp     = chunk_local_reverse_cumsum_fp32(dg_cum_fp, C)

return dq_fp, dk_fp, dv_bf, db_fp, dg_fp, dh0_fp
```

The reverse cumulative sum is inclusive, independent for every key coordinate,
and resets at each chunk boundary. There is no head reduction or output
postprocessing. The output dtypes are FP32 for `dq`, `dk`, `db`, `dg`, and
`dh0`, and BF16 only for `dv`.
