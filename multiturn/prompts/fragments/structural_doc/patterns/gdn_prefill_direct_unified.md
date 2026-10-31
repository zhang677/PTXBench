# Optimization Pattern: Gated DeltaNet prefill

Implement the packed GDN operator with the shapes, input definitions, sequence
semantics, and numerical accuracy described below. Correctness applies to both
the token output and the final state; bitwise reproduction of a particular
kernel is not required.

Kernel decomposition, physical tile sizes, reduction trees, triangular-solve
algorithm, context partitioning, fusion, and pipelining are implementation
choices. Alternative formulations are allowed when both returned tensors meet
the required numerical tolerances. Named intermediates describe values and
precision boundaries, not mandatory buffers or separate kernel launches.

## Packed operator contract

```text
output, new_state = run(
    q, k, v, state, g, beta, cu_seqlens, scale
)
```

Let `T` be the packed-token count, `N` the sequence count, and `Hq,Hk,Hv`
the symbolic query, key, and value/state head counts. Require positive head
counts with `Hv % Hq == 0` and `Hv % Hk == 0`. The key and value widths
are `K = V = 128`.

| Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `q` | `[T,Hq,K]` | BF16 | Queries, consumed without additional normalization |
| `k` | `[T,Hk,K]` | BF16 | Keys, consumed without additional normalization |
| `v` | `[T,Hv,V]` | BF16 | Values |
| `state` | `[N,Hv,V,K]` | FP32 | Initial k-last state |
| `g` | `[T,Hv]` | FP32 | Multiplicative decay coefficient in `[0,1]` |
| `beta` | `[T,Hv]` | FP32 | Update coefficient in `[0,1]` |
| `cu_seqlens` | `[N+1]` | INT64 | Packed sequence boundaries |
| `scale` | scalar | FP32 | Accepted but ignored; effective scale is `1/sqrt(K)` |
| `output` | `[T,Hv,V]` | BF16 | Packed token outputs |
| `new_state` | `[N,Hv,V,K]` | FP32 | Final k-last state |

Inputs are contiguous tensors on the same CUDA device. Boundaries are
nondecreasing, start at zero, and end at `T`. Each sequence starts from its
own initial state. State updates, cumulative decays, and causal interactions
must never mix logical sequences. Mask partial tiles without treating padded
positions as real tokens.

For value head `h`, use query head `h // (Hv / Hq)` and key head
`h // (Hv / Hk)`. Preserve the initial FP32 state for an empty sequence,
which produces no token output.

Use the effective output scale `s = 1/sqrt(K)`; the `scale` argument does
not alter it. Do not add Q/K normalization.

## Decay and recurrence

For each token and value head, `g` is the FP32 multiplicative decay
coefficient and `beta` is the FP32 update coefficient. Both lie in `[0,1]`
and are used as provided in the equations below.

Retain the stabilized decay conversion:

```text
l2 = log2(FP32(g + 1e-10))
```

The FP32 addition precedes the logarithm. Preserve its behavior for zero
and very small `g`; using `log2(g)` alone removes the stabilization.
Input generation is outside the timed operator. Stabilized logarithms,
cumulative decays, chunk factors, state propagation, and both returned
outputs remain inside it.

Suppress sequence/head indices and transpose external `[V,K]` state into
`S [K,V]` for the equations. With effective decay `rho_t = exp2(l2_t)`,
the real-number recurrence is

```text
Sbar_t = rho_t * S_{t-1}
r_t    = beta_t * (v_t - k_t @ Sbar_t)
S_t    = Sbar_t + k_t[:,None] * r_t[None,:]
o_t    = (s * q_t) @ S_t
```

GDN uses one scalar decay per token/head. The coefficients, stabilized logarithms,
and decay scans use FP32 precision, but no particular exponential instruction
or scan tree is required.

## Chunk formulation

For one sequence/head, arrange token rows as `Q,Kd [C,K]` and
`Vv [C,V]`. Here `Kd` is the key matrix, not a decayed key. Define

```text
G_i      = sum_{r<=i} l2_r                    # chunk-local FP32 scan
gamma_i  = exp2(G_i)
ratio_ij = exp2(G_i - G_j)                    # causal pairs only
eta_i    = exp2(G_last - G_i)

L_ij = beta_i * dot(Kd_i, Kd_j) * ratio_ij    if j < i, else 0
R    = inverse(I + L)
A    = R @ diag(beta)
P_ij = dot(Q_i, Kd_j) * ratio_ij              if j <= i, else 0

X     = A @ (Vv - gamma[:,None] * (Kd @ S_in))
O     = s * (gamma[:,None] * (Q @ S_in) + P @ X)
S_out = gamma_last * S_in + Kd.T @ (eta[:,None] * X)
```

Preserve the strictly lower mask in `L`, the inclusive causal mask in
`P`, and the incoming-state contribution. Beta scales rows of the lower
system `L` and columns of its inverse in `A`. Form decay ratios from log
differences to avoid division by underflowed cumulative products.

The equations allow a direct triangular solve, explicit inverse, or another
factorization. They do not require any specific inverse hierarchy, intermediate
inverse dtype, kernel count, or context-parallel dispatch rule.

## Numerically significant boundaries

The following 64-token formulation identifies important precision boundaries.
It does not prescribe physical tiles or require a sequential implementation.
Use `F` for exact BF16-to-FP32 promotion, `RBF` for round-to-nearest-even
BF16 conversion, and `MM` for BF16 operands with FP32 accumulation.

Keep the live state and returned final state in FP32. Use a BF16 chunk-entry
state operand for the Q/K state products. Let `A_bf` be a sufficiently accurate
BF16 representation of the factor `A` above, with beta already included. The
reference inverts the ungated system `I + strict(Diag(beta) Kd Kd.T)` in FP16,
rounds `-beta * inverse` to BF16, and then applies the decay ratio
`exp2(G_i - G_j)` before rounding `A_bf` again; the gated and ungated inverses
are related exactly by that diagonal similarity.

```text
S_operand_bf = RBF(S_fp)
KS_fp       = MM(Kd, S_operand_bf)
pred_bf     = RBF(KS_fp * gamma[:,None])
residual_bf = RBF(F(Vv) - F(pred_bf))

X_fp = MM(A_bf, residual_bf)
X_bf = RBF(X_fp)
end_values_bf = RBF(X_fp * eta[:,None])        # see the per-GPU note below

P_bf     = RBF(lower((MM(Q, Kd.T) * ratio) * s))
inter_fp = (MM(Q, S_operand_bf) * gamma[:,None]) * s
O_bf     = RBF(inter_fp + MM(P_bf, X_bf))

S_fp = gamma_last * S_fp + MM(Kd.T, end_values_bf)
```

FP32 sums here may use accumulator continuation or other fusion that meets
the required numerical accuracy. The significant boundaries are the rounded
state operand, prediction and residual, BF16 output factors, and separate
decayed end values used for the state update. Carrying only a BF16 live state
discards information required by the FP32 final state.

For a partial chunk, padded positions have neutral decay and zero update
contribution. The total decay is that of the last valid token. Store the
final state in external `[V,K]` orientation after all tokens of the sequence.

An implementation may resolve incoming states through a scan or partition
composition instead of a sequential chunk loop. It must retain each sequence's
initial-state contribution and produce both outputs to the required accuracy.
A partition boundary is not a state reset. No particular partition length,
transfer-matrix precision, or state-fixup algorithm is mandated.

State-dependent BF16 rounding of the prediction and residual is part of the
recurrence. An affine transfer derived from real-valued equations does not
automatically preserve these rounding boundaries. Composed incoming states
must still satisfy the accuracy requirement for both outputs.

## Per-GPU reference differences

The FlashInfer reference uses the same 64-token boundaries above on H100 and
B200, including context-parallel partitioning of this workload's sequences.
It differs between GPUs only on the state side:

- On B200 the end values are `RBF(X_fp * eta)`, as displayed. On H100 they
  are formed from the rounded corrected values, `RBF(F(X_bf) * eta)`.
- The partition-entry state fixup uses TF32 tensor-core products with FP32
  accumulation on B200 and FP32 scalar arithmetic on H100.
- On B200 the final state comes from the partition-composition path; on H100
  it comes from the chunk recurrence over the last partition.

Either variant is acceptable on either GPU when both outputs meet the
required accuracy. These differences mainly affect the final state, not the
token output.
