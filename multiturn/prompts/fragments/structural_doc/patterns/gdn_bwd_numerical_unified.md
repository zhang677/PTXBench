# GDN backward operator and numerical reference

Implement the direct GDN backward operator with the saved-input semantics,
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

Use symbolic batch size `B`, sequence length `T`, shared Q/K head count `H`,
and value/state head count `HV`. The key and value widths are `K=V=128`;
the saved inverse and cumulative gates use chunk size `C=64`.

## Direct operator contract

```text
dq, dk, dv, db, dg, dh0 = run(
    q, k, v, g, beta, A, initial_state, do, dht, scale
)
```

| Name | Shape | Dtype | Meaning |
|---|---:|---|---|
| `q`, `k` | `[B,T,H,K]` | BF16 | Saved query and key inputs |
| `v` | `[B,T,HV,V]` | BF16 | Saved value input |
| `g` | `[B,T,HV]` | FP32 | Saved chunk-local cumulative base-2 log decay |
| `beta` | `[B,T,HV]` | BF16 | Saved activated update gate |
| `A` | `[B,T,HV,C]` | BF16 | Saved row-wise WY inverse |
| `initial_state` | `[B,HV,V,K]` | FP32 | Initial state, V-first and K-last |
| `do` | `[B,T,HV,V]` | BF16 | Upstream output gradient |
| `dht` | `[B,HV,V,K]` | FP32 | Upstream final-state gradient |
| `scale` | scalar | FP32 | Fixed output scale |
| `dq`, `dk` | `[B,T,H,K]` | BF16 | Query and key gradients after head reduction |
| `dv` | `[B,T,HV,V]` | BF16 | Value gradient |
| `db` | `[B,T,HV]` | BF16 | Gradient of activated `beta` |
| `dg` | `[B,T,HV]` | FP32 | Gradient before the natural-log cumulative sum |
| `dh0` | `[B,HV,V,K]` | FP32 | Initial-state gradient |

Consume saved `g` and `A` directly. Do not reconstruct the cumulative sum or
inverse, normalize Q/K, or differentiate through a raw-gate activation. Require
positive head counts, `T % C == 0`, and `HV % H == 0`. For value head `hv`, gather Q/K head
`hv // (HV/H)`. Reduce only `dq` and `dk` to `H` heads; keep `dv`, `db`, and
`dg` at `HV` heads.

## Layout, gates, and numerical notation

Suppress batch and head indices for one chunk. Use

```text
Q, Kc : [C,K]       # BF16 Q/K rows gathered for this value head
Vv, dO : [C,V]     # BF16 value and upstream output-gradient rows
beta  : [C]        # BF16
g     : [C]        # FP32
A     : [C,C]      # BF16 logical chunk view of the saved inverse
S     : [K,V]      # internal live state
```

Transpose the last two logical axes when loading or storing external
`initial_state`, `dht`, and `dh0 [V,K]`. Token rows are `i,j in [0,C)`, key
coordinates are `k`, and value coordinates are `v`. Live state scans carry
across chunks within a sequence; gate cumulative sums reset at every chunk.

Let `lambda_i` be the natural-log decay increment before the forward
chunk-local cumulative sum:

```text
ell_i     = sum_{j<=i} lambda_j = ln(2) * g_i
gamma_i   = E(g_i)
gamma_end = E(g_{C-1})
eta_i     = E(g_{C-1} - g_i)
R_ij      = E(g_i - g_j)
```

The reference arithmetic uses the following notation:

```text
F(x_bf)      exact promotion of stored BF16 x_bf to FP32
RBF(x)       one round-to-nearest-even conversion of FP32 x to BF16
E(x)         approximate base-2 exponential on FP32 x,
             flushing subnormal inputs and results to zero
BFMM(X,Y)    BF16 tensor-core operands, FP32 accumulation and result
BFMMA(X,Y,Acc) BF16 tensor-core product accumulated into existing FP32 Acc
TFMM(X,Y)    raw FP32 operands entering tensor cores at TF32 input precision
MUL32(a,b)  FP32 multiply, one round-to-nearest-even result
ADD32(a,b)  FP32 add, one round-to-nearest-even result
FMA32(a,b,c) scalar FP32 fused multiply-add, one round-to-nearest-even result
lower(X)     inclusive lower triangle, j <= i; zero elsewhere
strict(X)    strict lower triangle, j < i; zero elsewhere
```

In this recipe, `TFMM` uses TF32 input precision without explicit TF32
pre-rounding. Accumulation is FP32. `BFMMA`
continues an existing accumulator, whereas standalone `BFMM` starts from zero.
`FMA32` combines completed FP32 operands with one rounding; `MUL32` followed by
`ADD32` rounds twice. Folding scalar arithmetic into a matrix accumulator or
adding separately rounded products can therefore change the result.

The recipe promotes scalar BF16 operands to FP32 and rounds at named `RBF` boundaries. Suffixes `_bf` and `_fp` describe
these reference values, not mandatory storage. Retain these distinctions when
using the recipe directly. An optimized alternative may change internal
precision, grouping, or fusion, provided all outputs still meet the required
accuracy; greater intermediate precision alone does not guarantee agreement.

## Semantic chunk algebra

The saved inverse corresponds to

```text
G = Kc @ Kc.T
L = strict(Diag(beta) @ (G * R))
A = inverse(I + L)
```

Using the supplied `A`, the semantic forward quantities are

```text
Bv    = Diag(beta) @ Vv
Bk    = Diag(beta * gamma) @ Kc
U     = A @ Bv
W     = A @ Bk
X     = U - W @ S
N     = Q @ Kc.T
P_sem = lower(N * R)
O     = scale * (Diag(gamma) @ Q @ S + P_sem @ X)
Z     = Diag(eta) @ X
Snext = gamma_end * S + Kc.T @ Z
```

These equations explain the dependencies; the reference recipe below shows
rounding and folds `scale` into selected BF16 intermediates. Use the supplied
`A`.
The returned gate gradient is with respect to `lambda`, not the saved numeric
base-2 tensor `g`:

```text
dg_j = sum_{i=j..C-1} bar_ell_i
```

Apply this inclusive reverse cumulative sum independently within each chunk,
with no extra `ln(2)` or `1/ln(2)` factor.

## Reference arithmetic recipe

### WY operands and forward state scan

For every chunk row:

```text
Bv[i,v]       = RBF(F(beta[i]) * F(Vv[i,v]))
key_beta[i,k] = RBF(F(beta[i]) * F(Kc[i,k]))
Bk_fwd[i,k]   = RBF(F(key_beta[i,k]) * gamma_i)
beta_gamma_i = F(beta[i]) * gamma_i
Kbg_fp[i,k]   = F(Kc[i,k]) * beta_gamma_i
Kbg_bwd[i,k]  = RBF(Kbg_fp[i,k])
U             = RBF(BFMM(A, Bv))
W             = RBF(BFMM(A, Bk_fwd))
```

The reference forward key path has two BF16 rounds, while `Kbg_bwd` has
one final round for the saved-WY `barA` contraction. These values need not
agree. The recipe uses unrounded `Kbg_fp` for the WY gate gradient and
`key_beta` for the second inverse key path; merging these paths can affect
output accuracy.

The reference starts with FP32 `S_live = transpose(initial_state)` and visits
chunks in increasing order:

```text
S_snap_bf = RBF(S_live)
WS_fp     = BFMM(W, S_snap_bf)
x_fp      = F(U) - WS_fp
X_bf      = RBF(x_fp)
Z_bf      = RBF(eta[:,None] * x_fp)
KS_fp     = BFMM(Kc.T, Z_bf)
S_live    = FMA32(gamma_end, S_live, KS_fp)
```

Backward uses the values represented by `S_snap_bf` and `X_bf`, which may
be retained or recomputed. In the recipe, `Z_bf` uses unrounded `x_fp`, and
the live state stays FP32 between chunks. `KS_fp` completes from a zero
accumulator before the state FMA. Starting that matrix product from
`gamma_end * S_live` has different rounding; assess such changes by their
output accuracy. These dependencies do not prescribe storage or synchronization.

### Output interaction and reverse state scan

For every chunk:

```text
N_fp         = BFMM(Q, Kc.T)
P_bf         = RBF(lower(scale * N_fp * R))
barX_local_bf = RBF(BFMM(P_bf.T, dO))
```

The recipe applies the inclusive mask before rounding `P_bf`, which already
contains `scale`; applying another scale to `barX_local_bf` would change the
operator.

The reference starts with FP32 `dS_live = transpose(dht)` and visits chunks in
decreasing order:

```text
dS_snap_bf  = RBF(dS_live)
barZ_fp     = BFMM(Kc, dS_snap_bf)
barX_fp     = FMA32(eta[:,None], barZ_fp, F(barX_local_bf))
barX_bf     = RBF(barX_fp)

state_path_fp  = BFMM(W.T, barX_bf)
output_path_fp = TFMM((F(Q) * gamma[:,None]).T, F(dO))
add_fp         = FMA32(scale, output_path_fp, -state_path_fp)
dS_live        = FMA32(gamma_end, dS_live, add_fp)      # H100 reference
# B200 reference: dS_live = ADD32(MUL32(gamma_end, dS_live), add_fp)
```

The reference forms `add_fp` from completed contractions with one FP32 FMA,
and keeps `Q*gamma` in FP32 until the TF32 operand boundary; `barX_fp` uses
one FP32 FMA. The reverse live-state update depends on the reference's code
generation: the H100 reference contracts it into one FP32 FMA, while the B200
reference rounds the decayed state with `MUL32` before `ADD32`. Either form is
acceptable on either GPU when all outputs meet the required accuracy; the
distinction is useful when diagnosing state-gradient mismatches. After chunk index `0`, the resulting `transpose(dS_live)` is the
FP32 output `dh0`.

### Direct state/output gradients

```text
barP_fp = BFMM(dO, X_bf.T)
barN_bf = RBF(lower(barP_fp * R) * scale)

dq_main_fp = BFMM(dO, S_snap_bf.T)
dq_main_fp[i,:] = (dq_main_fp[i,:] * gamma_i) * scale
dk_main_fp = BFMM(X_bf, dS_snap_bf.T)
dk_main_fp[i,:] *= eta_i
dk_state_fp = value of dk_main_fp at this point
Deta_i = sum_k(dk_state_fp[i,k] * F(Kc[i,k]))

dq_main_fp = BFMMA(barN_bf, Kc, dq_main_fp)
dk_main_fp = BFMMA(barN_bf.T, Q, dk_main_fp)

dg0_i = sum_k(dq_main_fp[i,k] * F(Q[i,k]))
        - sum_k(dk_main_fp[i,k] * F(Kc[i,k]))
eta_end_fp = sum_{i,k}(dk_state_fp[i,k] * F(Kc[i,k]))
state_dot_fp = sum_{k,v}(F(S_snap_bf[k,v]) * F(dS_snap_bf[k,v]))
end_fp = FMA32(gamma_end, state_dot_fp, eta_end_fp)
dg0_{C-1} += end_fp
```

In the recipe, `barN_bf` is the causal score-gradient boundary and the causal
products continue the FP32 Q/K accumulators. `dg0` uses the main gradients
before saved-WY contributions or output casts. Row dots and the endpoint
reduction use FP32 accumulation. The reference endpoint reduces the complete
`[C,K]` product; summing separately rounded `Deta` rows can give a different
result. Alternative reductions remain subject to output accuracy.

The `-k*dk_state` term already contributes `-Deta_i`. Restore `sum_i(Deta_i)`
at the final row because every `eta_i` depends on the chunk-end gate. The
recipe keeps both endpoint terms and `dg0` in FP32 before the suffix scan.

### Chunk-end state dot

`state_dot_fp` contracts the BF16 chunk-entry state snapshot with the BF16
chunk-exit adjoint snapshot over all `[K,V]` elements and returns one FP32
scalar per chunk and value head. Promote both operands to FP32 before
multiplying; every BF16-by-BF16 product is then exact in FP32, and products and
partial sums are accumulated in FP32. The reduction order, tiling over `K` or
`V`, and FMA contraction are not numerically significant here. Do not round
the products, partial sums, or the result to BF16, and do not substitute the
FP32 live states for the BF16 snapshots.

Add the `gamma_end`-scaled contribution once per chunk/head, even if the Q/K
gradient computation is split into key tiles.

### WY right-hand-side VJP

```text
barW_bf  = -RBF(BFMM(barX_bf, S_snap_bf.T))
barBv_fp = BFMM(A.T, barX_bf)
barBk_fp = BFMM(A.T, barW_bf)

dv_bf[i,v] = RBF(F(beta[i]) * barBv_fp[i,v])
dk_rhs_fp[i,k] = barBk_fp[i,k] * beta_gamma_i
dk_rhs_bf[i,k] = RBF(dk_rhs_fp[i,k])

dbeta_rhs_i = 0
dg_rhs_i = 0
for each logical 64-coordinate key group kt in increasing order:
    dbeta_rhs_i += sum_{k in kt}((barBk_fp[i,k] * F(Kc[i,k])) * gamma_i)
    dg_rhs_i += sum_{k in kt}(barBk_fp[i,k] * Kbg_fp[i,k])
for each logical 64-coordinate value group vt in increasing order:
    dbeta_rhs_i += sum_{v in vt}(barBv_fp[i,v] * F(Vv[i,v]))
```

The recipe negates `barW` after its BF16 cast and keeps `barBv` and `barBk`
FP32. Its 64-coordinate groups describe the reference partials, with key
partials added before value partials; they do not fix physical tiles or rule
out other accurate reductions. The direct WY gate term uses unrounded
`Kbg_fp` and already represents an adjoint of the cumulative natural-log
decay, so an extra gamma factor would change the operator. `dk_rhs_bf`
feeds the later inverse VJP.

### Saved-inverse VJP

```text
barA_fp = 0
for each logical 64-coordinate key group kt in increasing order:
    barA_fp = BFMMA(barW_bf[:,kt], Kbg_bwd[:,kt].T, barA_fp)
for each logical 64-coordinate value group vt in increasing order:
    barA_fp = BFMMA(barX_bf[:,vt], Bv[:,vt].T, barA_fp)
barA_lower_bf = RBF(strict(barA_fp))
T_bf          = RBF(BFMM(barA_lower_bf, A.T))
Ltmp_fp       = BFMM(A.T, T_bf)
C_L_fp        = strict(-Ltmp_fp)
D_bf          = RBF(strict(C_L_fp * R))

G_fp      = BFMM(Kc, Kc.T)
first_fp  = BFMM(D_bf, Kc)
second_fp = BFMM(D_bf.T, key_beta)

dbeta_fp = dbeta_rhs
for each logical 64-coordinate key group kt in increasing order:
    dbeta_fp[i] += sum_{k in kt}(first_fp[i,k] * F(Kc[i,k]))
dk_inv_fp[i,k] = FMA32(F(beta[i]), first_fp[i,k], second_fp[i,k])
dk_wy_fp[i,k]  = dk_inv_fp[i,k] + F(dk_rhs_bf[i,k])
rbar[i,j] = F(D_bf[i,j]) * (G_fp[i,j] * F(beta[i]))
```

The reference accumulates key products followed by value products into one
FP32 `barA` accumulator. `G_fp` also continues one accumulator across key
groups. Independently rounded full products followed by addition can differ.
The recipe rounds strict-lower `barA` and the first inverse product to BF16;
the second inverse product stays FP32. Its decay ratio precedes the `D_bf`
rounding, and `beta` enters at the displayed later operations.

The reference adds inverse key-group partials directly to `dbeta`, and forms
`dk_wy_fp` as `FMA32(F(beta), first_fp, second_fp) + F(dk_rhs_bf)` from completed
products. Accumulating `beta * first_fp` inside `second_fp`, or combining
main/WY branches before their output reductions, changes rounding. These are
numerical considerations when selecting an alternative, not a requirement to
use separate buffers, launches, or synchronization for each expression.

### Final accumulation and outputs

The direct `gamma` and `eta` paths are already in `dg0`. For each row, form
the remaining saved-WY decay contribution in FP32:

```text
row_i = sum_j(rbar[i,j])
col_i = sum_j(rbar[j,i])
re_i = row_i - col_i
dg2_i = dg_rhs_i + re_i
dg_cum_fp[i] = dg0_i + dg2_i
```

`rbar` already contains `R` through `D_bf`; do not multiply by the ratio again.
The reference reduces rows and columns separately in FP32 before subtracting;
an interleaved add/subtract reduction has different rounding and needs to meet
the same output accuracy requirements.
Any separate `etabar` accumulator stays zero, because adding that path again
would double-count the `eta` terms in `dg0`.

Perform one chunk-local FP32 inclusive reverse cumulative sum:

```text
dg_fp = chunk_local_reverse_cumsum_fp32(dg_cum_fp, C)
```

The sum resets at each chunk boundary and introduces no logarithm-base factor.
Store `dg_fp` directly as FP32 `dg`. Let `r=HV/H`; the value heads belonging
to Q/K head `h` are `hv=h*r .. (h+1)*r-1`. Define the elementwise reduction

```text
head_reduce_bf16(x) = RBF(sum_fp32_{hv=h*r..(h+1)*r-1}(F(RBF(x[...,hv,:]))))

dq[...,h,:] = head_reduce_bf16(dq_main_fp)
dk_main_head_bf = head_reduce_bf16(dk_main_fp)
dk_wy_head_bf   = head_reduce_bf16(dk_wy_fp)
dk[...,h,:] = RBF(F(dk_main_head_bf) + F(dk_wy_head_bf))

dv[...,hv,:]    = dv_bf
db[...,hv]      = RBF(dbeta_fp)
dg[...,hv]      = dg_fp
dh0[...,hv,:,:] = transpose(dS_live)

return dq, dk, dv, db, dg, dh0
```

The reference rounds each value-head Q/K contribution to BF16, reduces
main/WY branches independently to BF16, and adds the two reduced `dk` values
with a final BF16 rounding. This explains accuracy differences that can arise
from fusing the reductions; it does not require intermediate storage or
exclude another formulation that meets the output accuracy requirements.
Output dtypes remain BF16 for `dq`, `dk`, `dv`, and `db`, and FP32 for `dg`
and `dh0`.
