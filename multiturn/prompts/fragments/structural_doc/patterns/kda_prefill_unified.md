# Optimization Pattern: Kimi Delta Attention prefill

Implement the packed KDA operator with the shapes, gate definitions, sequence
semantics, and numerical accuracy described below. Correctness applies to both
the token output and the final state; bitwise reproduction of a particular
kernel is not required.

Kernel decomposition, physical tile sizes, reduction trees, triangular-solve
algorithm, storage placement, fusion, and pipelining are implementation choices.
Alternative formulations are allowed when both returned tensors meet the
required numerical tolerances. Named intermediates describe values and
precision boundaries, not mandatory buffers or separate kernel launches.

## Packed operator contract

```text
output, final_state = run(
    q, k, v, g, beta, A_log, dt_bias, initial_state,
    cu_seqlens, scale, lower_bound
)
```

Let `T` be the packed-token count, `N` the sequence count, and
`Hq = Hk = Hv = H` the symbolic head count. The key and value widths are
`K = V = 128`.

| Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `q`, `k` | `[T,H,K]` | BF16 | Raw queries and keys |
| `v` | `[T,H,V]` | BF16 | Values |
| `g` | `[T,H,K]` | BF16 | Raw coordinate-wise decay projection |
| `beta` | `[T,H]` | BF16 | Raw update logits |
| `A_log` | `[H]` | FP32 | Per-head log scale for the decay gate |
| `dt_bias` | `[H,K]` | FP32 | Per-head, per-coordinate decay bias |
| `initial_state` | `[N,H,V,K]` | FP32 | Initial k-last state |
| `cu_seqlens` | `[N+1]` | INT64 | Packed sequence boundaries |
| `scale` | scalar | FP32 | Supplied output scale |
| `lower_bound` | scalar | FP32 | Supplied negative bounded-gate parameter |
| `output` | `[T,H,V]` | BF16 | Packed token outputs |
| `final_state` | `[N,H,V,K]` | FP32 | Final k-last state |

Inputs are contiguous tensors on the same CUDA device. Boundaries are
nondecreasing, start at zero, and end at `T`. Each sequence starts from its
own initial state. State updates, cumulative decays, and causal interactions
must never mix logical sequences. Mask partial tiles without treating padded
positions as real tokens.

Use the supplied numeric `scale` and `lower_bound`. Initial state is required;
`None` values and a softplus decay parameterization are outside this contract.

## Input transforms and recurrence

Normalize Q and K independently over their key coordinates. Use FP32
accumulation with epsilon `1e-6`, multiply by the FP32 reciprocal square root,
and round the normalized vectors to BF16 before subsequent products. Activate
the raw gates in FP32:

```text
Q_t        = RBF(FP32(q_t) * rsqrt(sum_fp32(q_t * q_t) + 1e-6))
K_t        = RBF(FP32(k_t) * rsqrt(sum_fp32(k_t * k_t) + 1e-6))
SIG(x)     = FMA32(TANH_APPROX_F32(FP32(0.5 * x)), 0.5, 0.5)
beta_t     = SIG(FP32(beta_raw_t))
gate_scale = FP32(lower_bound * log2(e))      # once, on the host
x_t[d]     = FP32(exp(A_log) * FP32(FP32(g_t[d]) + dt_bias[d]))
ell2_t[d]  = FP32(gate_scale * SIG(x_t[d]))   # base-2 log decay
alpha_t    = exp2(ell2_t)
```

The square products in the norm are formed after promoting the inputs to
FP32. Apply normalization and gate activation once. `TANH_APPROX_F32` denotes
the native approximate FP32 hyperbolic tangent (`tanh.approx.f32`).
`FMA32(a,b,c)` evaluates the exact `a*b+c` with one FP32 round-to-nearest-even
rounding. Both the decay and the beta activations use this tanh-based sigmoid;
a reciprocal/exponential sigmoid is not numerically interchangeable with it.
The reference folds `log2(e)` into `lower_bound` before multiplying by the
sigmoid, so `ell2` is already a base-2 log decay; forming the natural-log decay
first and converting it afterwards rounds differently. Other transcendental
implementations may vary within the required numerical accuracy.

Suppress sequence/head indices and transpose external `[V,K]` state to
`S [K,V]` for the equations. The real-number recurrence is

```text
Sbar_t = alpha_t[:,None] * S_{t-1}
r_t    = beta_t * (v_t - K_t @ Sbar_t)
S_t    = Sbar_t + K_t[:,None] * r_t[None,:]
o_t    = (scale * Q_t) @ S_t
```

Decay acts separately on every key-coordinate row of the state. It cannot
be replaced by a scalar gate or moved outside a dot product as a scalar.

## Chunk formulation

For token rows `i,j` within a chunk, define `G_i = sum_{r<=i} ell2_r` and
`D_ij = exp2(G_i - G_j)`, both coordinate-wise. Then

```text
L_ij = beta_i * dot(K_i * D_ij, K_j)       if j < i, else 0
R    = inverse(I + L)
P_ij = dot(Q_i * D_ij, K_j)                if j <= i, else 0
Kd_i = K_i * exp2(G_i)
Qd_i = Q_i * exp2(G_i)
Kr_i = K_i * exp2(G_last - G_i)

X     = R @ (beta[:,None] * (Vv - Kd @ S_in))
O     = scale * (Qd @ S_in + P @ X)
S_out = exp2(G_last)[:,None] * S_in + Kr.T @ X
```

These equations expose matrix work without requiring an explicit inverse or
a particular solve algorithm. A fused solve may compute the corrected values
directly. Preserve the strictly lower mask in `L`, the inclusive causal mask
in `P`, and the incoming-state contribution.

## Numerically significant boundaries

The finite-precision form uses 16-token logical chunks and BF16 state
rounding between them. This is a numerical boundary, not a physical tile or
launch-size requirement. FP32 external state is rounded to BF16 initially;
the final BF16 state is promoted to FP32. For an empty sequence, return the
BF16-rounded initial state promoted to FP32.

Use `F` for exact BF16-to-FP32 promotion, `RBF` for round-to-nearest-even
BF16 conversion, and `MM` for BF16 operands with FP32 accumulation.
`BMUL`, `BADD`, and `BSUB` mean the corresponding operation followed by
BF16 rounding. `FMA32(a,b,c)` is the FP32 fused operation `a*b+c`.

In the following form, `G2` is the FP32 chunk-local cumulative sum of
`ell2`, accumulated sequentially over the chunk's rows. The last valid token
determines the total decay. Missing rows contribute zero log-decay increments and zero
Q/K/V contributions.

```text
gamma_end = exp2(G2_last)                     # FP32
e_i       = RBF(exp2(G2_i))
einv_i    = RBF(exp2(-G2_i))
end_bf    = RBF(gamma_end)
scale_bf  = RBF(scale)

Kd_i = BMUL(K_i, e_i)
Qd_i = BMUL(BMUL(Q_i, e_i), scale_bf)
Ki_i = BMUL(K_i, einv_i)
Kr_i = BMUL(Ki_i, end_bf)

L_fp = strict_lower(MM(Kd, Ki.T) * beta[:,None])
P_bf = lower(RBF(MM(Qd, Ki.T)))
```

Construct the unit-lower inverse factor from `L_fp` with sufficient accuracy,
and use its BF16 representation `R_bf` in the products below. The reference
inverts the two 8x8 diagonal blocks by FP32 forward substitution, merges them
into the 16x16 inverse with BF16-operand matrix products, and stores `R_bf`.
The inverse's block sizes, pivot traversal, and internal precision are not
prescribed.
A direct solve may replace factor construction if it preserves the resulting
corrected values to the required accuracy.

The important downstream rounding points are

```text
S_bf     = RBF(initial_state.transpose(-1,-2))  # once per sequence
beta_bf  = RBF(beta)

# For each logical chunk:
pred_bf  = RBF(MM(Kd, S_bf))
res_bf   = BSUB(Vv, pred_bf)
rhs_bf   = BMUL(res_bf, beta_bf[:,None])
X_bf     = RBF(MM(R_bf, rhs_bf))

inter_bf = RBF(MM(Qd, S_bf))
intra_bf = RBF(MM(P_bf, X_bf))
O_bf     = BADD(inter_bf, intra_bf)

delta_fp = MM(Kr.T, X_bf)
S_bf     = RBF(FMA32(F(S_bf), gamma_end[:,None], delta_fp))
```

The reference keeps the live state in BF16 between 16-token chunks, as shown;
an FP32 live state is permitted when both outputs meet the required accuracy.
Query scaling is already included in `Qd`; do not apply it again.
The restored key has rounded inverse-decay and total-decay factors.
The residual is rounded before beta multiplication, the two output terms
are rounded before their BF16 addition, and state is rounded after its update.
Removing these boundaries or retaining a different live-state precision can
change the result even if the real-number recurrence is unchanged.

These boundaries guide numerical accuracy. Fusion, reassociation, alternative
solves, and different
physical chunking are permitted when they retain the required accuracy of
both output and final state.
