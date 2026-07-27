# KernelBench Level 1 Definitions Review

**Date**: 2026-03-20
**Files reviewed**: `kb_L1_1.json` through `kb_L1_10.json` (10 files, 18 definitions)
**Source**: KernelBench problems #1-10 (matmul/linear algebra variants)

---

## 1. Summary

**Overall quality: 5/10**

The definitions are structurally correct and well-described, but suffer from three systemic problems:
1. **Heavy duplication** -- many definitions are trivially identical in their reference implementation (just `torch.matmul(A, B)`) and differ only in workload ranges, which should be handled by workload variation, not separate definitions.
2. **Workload ranges violate the 1-15ms target** -- the smallest workloads for nearly every definition will run in microseconds (launch-overhead dominated), and the largest workloads in some batched/large-K definitions will OOM or exceed 15ms.
3. **FEEDBACK.md criterion #2 violated** -- most definitions are pure matmul with no vector operations or additional complexity. The rubric says definitions must contain "both matrix multiplication and vector operations" or have sufficiently complex vector operations.

---

## 2. Per-Definition Issues

### File: kb_L1_1.json

#### `kb_square_matmul`
- **Reference**: Correct.
- **Axes/shapes**: Correct.
- **Workload ranges**: N=256 and N=512 will run in <0.01ms (launch-overhead dominated). N=16384 is ~9ms compute + 1.5GB memory, acceptable. **Remove N=256, N=512.**
- **FEEDBACK.md**: Pure matmul, no vector ops. Likely rejected under criterion #2.

### File: kb_L1_2.json

#### `kb_rectangular_matmul`
- **Reference**: Correct.
- **Axes/shapes**: Correct.
- **Workload ranges**: Smallest combo (M=256,K=256,N=256) runs in <0.001ms. Largest combo (M=8192,K=16384,N=8192) ~2.2ms, fine. **Remove 256 from all axes.**
- **Deduplication**: `kb_square_matmul` is a strict subset of this definition (just set M=K=N). **Keep `kb_rectangular_matmul`, drop `kb_square_matmul`** or constrain square to only square workloads that aren't representable here.

### File: kb_L1_3.json

#### `kb_batched_matmul`
- **Reference**: Uses `torch.bmm`, correct.
- **Workload ranges**: Smallest (B=1,M=128,K=128,N=128) is ~0.000004ms. Largest (B=128,M=4096,K=4096,N=4096) is ~17.8ms and 12GB memory -- will OOM on single-GPU benchmarking. **Remove B=1 and small dims; cap largest workloads to avoid OOM.**
- **Naming**: Uses lowercase `m`, `k`, `n` for axes while other definitions use uppercase `M`, `K`, `N`. **Inconsistent -- standardize to uppercase.**

#### `kb_batched_matmul_transposed_b`
- **Reference**: Correct (`torch.bmm(A, B.transpose(-2, -1))`).
- **Axes naming**: Same lowercase inconsistency.
- **Workload ranges**: Same small-end problem.
- **Value**: Meaningfully different optimization problem (transposed memory access). Worth keeping.

#### `kb_batched_matmul_fp32_accum`
- **Reference**: Converts to float32 then back. Correct but note: this makes the reference ~2-4x slower than native bf16 matmul, so the "speedup" an agent gets is misleading since the reference is artificially slow.
- **Issue**: The reference implementation is not the standard PyTorch path. `torch.bmm` on bf16 already uses fp32 accumulation internally on H100 tensor cores. This definition's reference does an explicit cast which doubles memory traffic. The agent could "win" by simply removing the cast. **This is a deceptive benchmark -- consider removing or fixing the reference to use `torch.bmm` directly.**
- **Deduplication**: If fp32 accum reference is fixed to just `torch.bmm`, it becomes identical to `kb_batched_matmul`. **Drop this or redefine with a genuinely different operation.**

### File: kb_L1_4.json

#### `kb_matvec`
- **Reference**: Correct (`torch.mv`).
- **Workload ranges**: M=512,K=512 runs in <0.001ms. Even M=16384,K=131072 is only ~2.5ms bandwidth-limited. OK at the high end.
- **Remove**: M=512, K=512, and other small combos.

#### `kb_matvec_tall_skinny`
- **Reference**: Identical to `kb_matvec` (`torch.mv`).
- **Workload ranges**: M=256000,K=8192 uses ~3.9GB, runs ~2ms. Reasonable.
- **Deduplication**: Same reference as `kb_matvec`, just different workload ranges. **Merge into `kb_matvec` by adding these ranges as workloads, not a separate definition.**

#### `kb_matvec_short_wide`
- **Reference**: Identical to `kb_matvec`.
- **Deduplication**: Same issue -- merge into `kb_matvec`.

#### `kb_matvec_transposed`
- **Reference**: `torch.mv(A.t(), x)` -- meaningfully different (columnar access pattern).
- **Workload ranges**: Acceptable range.
- **Value**: Worth keeping as a separate definition since the memory access pattern is genuinely different.

### File: kb_L1_5.json

#### `kb_matrix_scalar_multiply`
- **Reference**: `A * s` -- correct.
- **FEEDBACK.md**: This is a trivial elementwise op with no matmul or complex vector operations. **Should be rejected under criterion #2.**
- **Workload ranges**: M=1024,N=1024 runs in ~0.002ms. Even M=65536,N=16384 is only ~2ms. Most workloads are too small. **If kept, needs much larger sizes.**

#### `kb_matrix_scalar_fma`
- **Reference**: `A * s + b` -- correct but trivially close to scalar multiply.
- **FEEDBACK.md**: Same rejection as above.
- **Deduplication**: Nearly identical to `kb_matrix_scalar_multiply`. **Drop both or merge.**

### File: kb_L1_6.json

#### `kb_gemm_large_k`
- **Reference**: `torch.matmul(A, B)` -- correct.
- **Workload ranges**: M=512,K=524288,N=512 is only 0.3ms compute but 1GB memory. The compute is far below 1ms for all workloads. This is a **bandwidth-bound problem** not a compute-bound one, but even bandwidth-wise the small M,N means little output data. **All workloads run <1ms -- violates the 1-15ms target.**
- **Deduplication**: This is just `kb_rectangular_matmul` with constrained ranges. **Merge.**

#### `kb_gemm_very_large_k`
- **Reference**: Correct.
- **Workload ranges**: M=256,K=4194304,N=256 uses 4GB memory but only 0.6ms compute. M=64,K=1048576,N=64 runs in ~0.01ms. **All workloads below 1ms. OOM risk at high end.**
- **op_type**: Uses `gemm` while similar definitions use `matmul`. **Inconsistent.**
- **Issue**: K=4194304 (4M) is unrealistic for any real workload and creates numerical issues. **Drop this definition.**

### File: kb_L1_7.json

#### `kb_matmul_small_k`
- **Reference**: Correct.
- **Workload ranges**: M=32768,N=32768,K=16 -- output is 32K x 32K = 1B elements = 2GB output. Compute is tiny (2*32768*32768*16 = 34B flops = 0.03ms). This is purely memory-bound writing output. The large M,N with tiny K makes this essentially a memory copy benchmark, not a matmul benchmark.
- **Value**: Interesting edge case but runs <0.1ms compute. May be OK if memory-bound timing reaches 1-15ms range due to output write.

### File: kb_L1_8.json

#### `kb_matmul_irregular_shapes`
- **Reference**: Correct.
- **Value**: Genuinely useful -- tests boundary handling and non-power-of-2 tiling. Worth keeping.
- **Workload ranges**: Reasonable (1K-8K range).
- **Deduplication**: Functionally identical reference to `kb_rectangular_matmul`. Could be merged as workload variants. **However, the irregular shapes are a meaningfully different optimization challenge, so keeping as separate definition is justified.**

### File: kb_L1_9.json

#### `kb_tall_skinny_matmul_m_much_greater_n`
- **Reference**: Correct.
- **op_type**: Uses `gemm` instead of `matmul`. Inconsistent.
- **Workload ranges**: M=65536,K=4096,N=16 runs in <0.01ms compute, <0.3ms bandwidth. **Too small.** Even the largest combo is under 1ms.
- **Deduplication**: Same reference as `kb_rectangular_matmul` with constrained ranges.

#### `kb_tall_skinny_matmul_n_much_greater_m`
- **Reference**: `torch.matmul(B, A)` -- **this is confusing**. The definition names inputs A and B, but the reference computes B@A. The shapes are A=[K,M], B=[N,K], output=[N,M]. This is mathematically correct but the naming is backwards (A is the smaller matrix, B is the larger one). **Rename inputs or clarify in description.**
- **op_type**: `gemm` inconsistency again.
- **Workload ranges**: Same too-small issue.
- **Deduplication**: Effectively the same operation as `kb_tall_skinny_matmul_m_much_greater_n` with transposed perspective.

### File: kb_L1_10.json

#### `kb_3d_tensor_matmul_broadcast`
- **Reference**: `torch.matmul(A, B)` where A is 3D and B is 2D -- correct (PyTorch broadcasts).
- **Value**: Interesting variant -- broadcast means B is shared across batch, unlike `kb_batched_matmul`.
- **Workload ranges**: N=8,M=512,K=512,L=512 runs in ~0.002ms. **Too small at low end.** N=64,M=4096,K=4096,L=2048 is ~8.5ms. Acceptable at high end.
- **Remove small N values (8, 16).**

---

## 3. Deduplication Recommendations

### Definitions to DROP (7 definitions)

| Definition | Reason |
|---|---|
| `kb_square_matmul` | Strict subset of `kb_rectangular_matmul` (M=K=N) |
| `kb_matvec_tall_skinny` | Same reference as `kb_matvec`, only different ranges |
| `kb_matvec_short_wide` | Same reference as `kb_matvec`, only different ranges |
| `kb_batched_matmul_fp32_accum` | Deceptive reference (explicit cast that PyTorch does internally); identical to `kb_batched_matmul` if fixed |
| `kb_matrix_scalar_multiply` | Fails FEEDBACK.md criterion #2 (too trivial) |
| `kb_matrix_scalar_fma` | Fails FEEDBACK.md criterion #2 (too trivial); near-duplicate of scalar multiply |
| `kb_gemm_very_large_k` | Unrealistic K values (4M), all workloads <1ms compute, OOM risk |

### Definitions to MERGE (3 -> 1)

| Merge Target | Absorbed Definitions |
|---|---|
| `kb_rectangular_matmul` | `kb_gemm_large_k`, `kb_tall_skinny_matmul_m_much_greater_n`, `kb_tall_skinny_matmul_n_much_greater_m` (all just rectangular matmul with different aspect ratios -- handle via workloads) |

### Definitions to KEEP (8 definitions after dedup)

1. `kb_rectangular_matmul` -- general matmul (absorbing square, large-K, tall-skinny)
2. `kb_batched_matmul` -- batched matmul
3. `kb_batched_matmul_transposed_b` -- batched matmul with transpose (different access pattern)
4. `kb_matvec` -- matrix-vector multiply (absorbing tall-skinny and short-wide variants as workloads)
5. `kb_matvec_transposed` -- transposed matvec (genuinely different access pattern)
6. `kb_matmul_small_k` -- interesting edge case (bandwidth-bound matmul)
7. `kb_matmul_irregular_shapes` -- non-power-of-2 tiling challenge
8. `kb_3d_tensor_matmul_broadcast` -- broadcast matmul (shared B across batch)

---

## 4. Missing Variants

Valuable definitions NOT present that would diversify the set:

1. **Matmul + bias (GEMM + epilogue)**: `C = A @ B + bias` -- fused epilogue is a key optimization target and satisfies FEEDBACK.md criterion #2 (matmul + vector op)
2. **Matmul + ReLU**: `C = relu(A @ B)` -- fused activation epilogue
3. **Matmul + residual add**: `C = A @ B + D` -- residual connections ubiquitous in transformers
4. **Symmetric matmul**: `C = A @ A^T` -- exploits symmetry for 2x speedup potential
5. **Strided/sliced matmul**: Inputs are non-contiguous slices of larger tensors -- tests memory layout handling
6. **Batched matmul with varying batch**: Different batch sizes per group (grouped GEMM)
7. **Chain matmul**: `D = A @ B @ C` -- tests fusion of sequential matmuls

---

## 5. Workload Range Fixes

### Ranges that are too small (will run < 0.1ms)

| Definition | Fix |
|---|---|
| `kb_square_matmul` | Remove N=256, N=512. Min should be N=1024. |
| `kb_rectangular_matmul` | Remove 256 from all axes. Min should be 512 or 1024. |
| `kb_batched_matmul` | Remove B=1 and dims 128. Min combo should be B=8,M=512. |
| `kb_batched_matmul_transposed_b` | Same fix as batched_matmul. |
| `kb_matvec` | Remove M=512,K=512. Min should be M=2048,K=4096. |
| `kb_matrix_scalar_*` | If kept, min M*N should be ~500M elements. |
| `kb_3d_tensor_matmul_broadcast` | Remove N=8, N=16. Min N=32. |
| `kb_gemm_large_k` | All workloads <1ms. Increase M,N to at least 1024. |
| `kb_tall_skinny_*` | All workloads <1ms. Need larger total FLOP count. |

### Ranges that are too large (will OOM or exceed 15ms)

| Definition | Fix |
|---|---|
| `kb_batched_matmul` | B=128,M=4096,K=4096,N=4096 uses 12GB and ~18ms. Cap at B=64 or reduce M/K/N. |
| `kb_gemm_very_large_k` | K=4194304 uses 4GB. Unrealistic. Drop definition. |
| `kb_matvec_tall_skinny` | M=256000,K=8192 uses 3.9GB. Acceptable but borderline. |

### Naming inconsistencies

| Issue | Fix |
|---|---|
| `kb_batched_matmul*` uses lowercase axes (`m`, `k`, `n`, `batch_size`) | Change to uppercase `M`, `K`, `N`, `B` |
| Mixed `op_type`: some use `matmul`, others `gemm`, some `elementwise` | Standardize: use `gemm` for all matrix multiply variants, `elementwise` for scalar ops |

---

## 6. Top 3 Recommendations

### 1. Apply FEEDBACK.md criteria before generating
The pipeline should check criterion #2 ("must contain both matrix multiplication and vector operations") BEFORE producing definitions. This would have prevented the 2 trivial scalar multiply definitions and flagged the pure-matmul definitions as needing epilogue fusion (bias, activation, etc.).

### 2. Add a workload range validator
Build an automated check that estimates compute time (FLOPS / peak_throughput) and memory footprint (tensor sizes) for each workload. Reject workloads outside the 1-15ms window and flag OOM risks (>40GB for single GPU, >10GB for safety margin). This would catch the ~50% of workloads that currently violate the timing rubric.

### 3. Deduplicate at the reference level, not the name level
Multiple definitions share the exact same `def run(A, B): return torch.matmul(A, B)` reference. The pipeline should detect when two definitions have identical references and merge them, differentiating via workload ranges only. This would reduce 18 definitions to ~8 meaningfully distinct ones.
