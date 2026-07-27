# Debugging FP8 GEMM numerical discrepancy (kernel_v0 vs kernel_v1)

Context: `fib_runtime/mini_swe_agent_docker/isolated/fp8_gemm_nt_1d2d_n4096_k7168/`
- `kernel_v0.cu`: reported `max_relative_error=0.4536`, `max_absolute_error=1.0`
- `kernel_v1.cu`: reported `max_relative_error=0.0`,    `max_absolute_error=0.0`

Both kernels are standard Hopper WGMMA + TMA warp-specialized FP8 GEMMs, computing
`D = A @ B^T` with per-token A scales and per-128x128-block B scales (1D2D recipe).

## TL;DR

The bug in `kernel_v0.cu` is **not** in the matmul, the TMA loads, the WGMMA
descriptors, the warp specialization, or the SF indexing — those are all
equivalent in v0 and v1. The single line that produces the divergence is:

```cpp
// kernel_v0.cu:280 (and copy at :301)
global_acc[i] += temp_acc[prev_tc][i] * sa * sb;
```

C++ `*` is left-associative, so this evaluates as `((temp * sa) * sb)` — two
fp32 multiplies, each rounded. The DeepGEMM reference (and `kernel_v1.cu`)
pre-multiplies the two scale factors once:

```cpp
// kernel_v1.cu:260-261, 273
float sa_A = smem.sf_a[prev_stage][row_A] * sb;     // (sa * sb) rounded once
...
global_acc[0][i] += t0 * sa_x;                       // temp * (sa*sb)
```

Result: `(temp * sa) * sb` and `temp * (sa * sb)` are mathematically equal but
have different fp32 rounding chains. Over 56 K-block scalings, the fp32 acc
drifts by a few LSBs, occasionally landing on the other side of a bf16
rounding boundary on the final cast to D.

Patching `:280` and `:301` to `temp_acc[...][i] * (sa * sb)` makes v0 bit-exact
with the reference. No other change required.

## Evidence

### Output diff (v0 vs precomputed `blob["D"]` from
`scripts/prepare_fp8_gemm_nt_1d2d_n4096_k7168.py`)

- **1819 / 16,777,216 elements differ (0.011 %)**, scattered uniformly, not
  concentrated in any tile.
- All differences are 1-bf16-ULP off:
  - 17 elements differ by exactly 1.0 (1 ULP at magnitudes ~128–256, e.g.
    v0=130 vs ref=129)
  - 49 by 0.5, 96 by 0.125, 120 by 0.0625, etc.
  - 0 differ by ≥ 2.0
- v1 vs ref: **0** mismatches.

### Confirming the hypothesis

Patched `kernel_v0_patched.cu` = `kernel_v0.cu` with **only** these two edits:

```diff
- global_acc[i] += temp_acc[prev_tc][i] * sa * sb;
+ global_acc[i] += temp_acc[prev_tc][i] * (sa * sb);
```
```diff
- global_acc[i] += temp_acc[last_tc][i] * sa * sb;
+ global_acc[i] += temp_acc[last_tc][i] * (sa * sb);
```

Run against the same safetensors workload (M=4096, N=4096, K=7168):

```
patched v0: max_abs=0.000000  max_rel=0.000000   mismatches: 0 / 16777216
```

Bit-exact. The two-character fix is sufficient.

### Matching the DeepGEMM reference

`DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh:328-340`:

```cpp
float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
...
shifted_accum[i*4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4 + 0];
```

DeepGEMM pre-multiplies `scale_a * scale_b` exactly once and then multiplies
the WGMMA partial by that pre-rounded combined scale. The on-disk reference
output we compare against (`blob["D"]` produced by
`deep_gemm.fp8_gemm_nt(...)` in `prepare_fp8_gemm_nt_1d2d_n4096_k7168.py`) was
written by this exact code path, so any kernel that matches DeepGEMM's
multiplication order matches bit-exactly. `kernel_v1.cu` does; `kernel_v0.cu`
does not.

## Why the test status was `INCORRECT_NUMERICAL` even though `(abs > atol) AND (rel > rtol)` is empty

The flashinfer-bench `DefaultEvaluator`
(`flashinfer_bench/bench/utils.py:113`) flags an element only when **both**
thresholds are exceeded:

```python
exceeds_tol_mask = (abs_error > cfg.atol) & (rel_error > cfg.rtol)
```

With `atol = rtol = 1e-2`:

- The 1.0-magnitude disagreements occur where `|ref| ≈ 129` → `rel_err ≈
  0.008 < rtol`, so they pass.
- The 0.45 relative error occurs where `|ref| ≈ 2.8e-5` → `abs_err ≈ 1.3e-5
  < atol`, so it also passes.

Re-running `test.py` over `kernel_v0.cu` produced `status: PASSED` with the
same `max_relative_error=0.4536, max_absolute_error=1.0`. The
`INCORRECT_NUMERICAL` row in the stored `traces.json` is stale — either
older evaluator semantics or a tighter tolerance config. **kernel_v0 was
mathematically correct.**

## Things that were NOT the bug (worth ruling out, all verified equivalent
between v0 and v1)

- WGMMA descriptors: same `make_wgmma_desc(ptr, 1, 1024, 1)` with 128B
  swizzle in both. LBO/SBO are correct for 128B-swizzled K-major FP8.
- TMA descriptors and transaction bytes: A=16384, B=8192 in v0 (B is 64×128
  bytes for the N=64 block tile); B=16384 in v1 (B is 128×128 for the
  N=128 block tile). Both correct.
- Block-tile factoring: v0 uses `M=128, N=64` with grid (M/128, N/64);
  v1 uses `M=128, N=128`. Both decompositions cover the full output.
- SF_B indexing: `SF_B[(bn / 128) * 56 + kb]`. v0 calls this with `bn =
  blockIdx.y * 64`, so two consecutive blocks share the same SF_B[n_block,
  kb] (correct — SF_B is per-128 N-block).
- SF_A indexing: same `gm * 56 + kb` loop with 32 producer threads each
  fetching 4 rows of SF_A.
- WGMMA register layout: `get_coord_n64_` in v0 produces row =
  `t1 + t2*16 + r1*8`, col = `t0*2 + r0 + r2*8`, which matches the canonical
  Hopper m64nN layout that v1 hard-codes inline via `row_A`/`row_B`. SF_A is
  loaded for the same per-register row positions in both versions.
- Warp specialization correctness: v0 has 12 warps = 1 producer + 3 idle +
  2 consumer warpgroups. Idle warps return before WGMMA is issued; the two
  consumer warpgroups (WG_B, WG_C) each have all 4 warps active when they
  issue `wgmma.fence` / `wgmma.commit` / `wgmma.wait`. WGMMA aligned-sync
  semantics are satisfied. v1 reorganizes to 9 warps = 2 consumer
  warpgroups (8 warps) + 1 producer warp = 288 threads, eliminating idle
  warps for slightly better occupancy/scheduling, but this is a
  performance change, not a correctness one.

## Lessons for future numerical-correctness debugging

1. **Bit-exactness with a precomputed safetensors reference requires
   reproducing the reference's fp32 rounding chain, not just its math.**
   Pre-multiplying scales (`sa * sb`) vs left-to-right (`temp * sa * sb`)
   are mathematically identical but produce different fp32 results, which
   sometimes shows up in the bf16 cast.

2. **`max_relative_error` alone is misleading near zero.** When `|ref|` is
   tiny (here ~3e-5), even sub-ULP fp32 noise inflates relative error
   above 0.4 while absolute error stays in the 1e-5 range. The
   flashinfer-bench evaluator's `(abs > atol) AND (rel > rtol)` rule
   correctly ignores this; trust the AND-rule, not the headline rel error.

3. **Compare patterns before fixing.** The output diff was:
   `1819 mismatches scattered uniformly, all 1-bf16-ULP off`. That
   signature ruled out structural bugs (wrong tile, wrong SF row, race
   condition between producer and consumer, etc.) — those would produce
   blocky errors of arbitrary magnitude, not uniform single-ULP noise.
   The signature pointed straight at fp32 rounding order.

4. **C++ operator associativity matters in fp arithmetic.** `a * b * c` is
   `(a * b) * c` and is NOT free to be reassociated by the compiler under
   IEEE semantics (nvcc respects this unless `-ffast-math` /
   `--use_fast_math` is set, which neither kernel uses). Whenever a kernel
   needs to match a specific reference's bit pattern, write the
   multiplication order explicitly — pre-multiply the scales, or use
   parentheses.

## Reproduction commands

From `fib_runtime/mini_swe_agent_docker/isolated/fp8_gemm_nt_1d2d_n4096_k7168/`:

```bash
# compile both
TVM_FFI_DIR=/home/ubuntu/miniconda3/envs/acc/lib/python3.12/site-packages/tvm_ffi
for v in v0 v1 v0_patched; do
  nvcc -shared -O3 -gencode arch=compute_90a,code=sm_90a kernel_${v}.cu \
    -lineinfo --ptxas-options=-v \
    -Xcompiler -fPIC,-fvisibility=hidden -lcuda \
    -I${TVM_FFI_DIR}/include -std=c++17 \
    -L${TVM_FFI_DIR}/lib -ltvm_ffi \
    -o kernel_${v}.so
done

# diff against precomputed reference
CUDA_VISIBLE_DEVICES=0 python run_compare.py
```

`run_compare.py` loads the safetensors blob at
`/home/ubuntu/accrl-training/blob/workloads/gemm/fp8_gemm_nt_1d2d_n4096_k7168/...safetensors`,
runs each `.so` via `tvm_ffi.load_module`, and dumps the per-tile error
pattern.
