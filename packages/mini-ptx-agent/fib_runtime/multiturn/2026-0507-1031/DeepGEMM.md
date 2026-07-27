# FP8 GEMM on Hopper — DeepGEMM eligible candidates

Survey of every FP8 kernel in [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (commit `891d57b`, library version `2.5.0`) that targets Hopper (SM90), framed as definition-skeletons in the same shape as `2026-0427-1308/PREPARE_ATTN_BWD.md`. Use this as the menu when authoring `prepare_*.py` scripts and `definitions/gemm/*.json` for FP8-Hopper workloads.

The Hopper FP8 device kernels live at:

- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh`
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`

…and the host-side launchers / dispatch logic at `csrc/apis/gemm.hpp`, `csrc/apis/attention.hpp`. Five GEMM API entry points and two MQA-logits entry points compile on SM90.

## Common notation

- `kFp8 = float8_e4m3fn`.
- `SF_a` = scaling factor for A, FP32 on SM90, MN-major and TMA-aligned (use `deep_gemm.get_mn_major_tma_aligned_tensor`).
- `SF_b` = scaling factor for B, FP32 on SM90.
- `gran_k = 128` is the only granularity SM90 supports.
- Layout is K-major on both A and B (i.e. `nt`) for every kernel below except `k_grouped_fp8_gemm_tn_contiguous`, which is TN.
- Two SF recipes are valid on Hopper:
  - **1D2D** — per-token A (`SF_a: [M, K/128]`), per-128×128 block B (`SF_b: [N/128, K/128]`). DeepSeek-V3 forward recipe.
  - **1D1D** — per-token both sides (`SF_a: [M, K/128]`, `SF_b: [N, K/128]`). Forced when `accumulate=True` (wgrad) or with the FP8 backward pass on SM90 (`tests/generators.py:145-147`).

## 1. `fp8_gemm_nt` — non-grouped dense FP8 GEMM

The plain FP8 GEMM, NT-only on SM90. Two distinct shape regimes are tested separately — forward (single `M`, varied `N,K`) and backward (`M=4096`, `N,K` permuted into dgrad and wgrad).

```yaml
name: fp8_gemm_nt_dense_fwd
op_type: gemm
axes:
  M:      var    # tested at 1, 128, 4096
  N:      var
  K:      var
  recipe: const  "1d2d"
inputs:
  A:    [M, K]              kFp8
  SF_a: [M, K/128]          f32     # MN-major, TMA-aligned
  B:    [N, K]              kFp8
  SF_b: [N/128, K/128]      f32
outputs:
  D:    [M, N]              bf16    # or float32 for "fp32-output" shapes
reference:  deep_gemm.fp8_gemm_nt(a=(A,SF_a), b=(B,SF_b), d=D)
```

```yaml
name: fp8_gemm_nt_dense_bwd
op_type: gemm
axes:
  M:        const  4096
  N, K:     var
  variant:  const  in {dgrad, wgrad}
  recipe:   const  "1d1d"               # SM90 fp8 bwd forces 1D1D + K-major
inputs:
  # dgrad:  D[M,K] = dO[M,N] @ W[N,K]^T
  # wgrad:  D[N,K] = dO[M,N]^T @ X[M,K]   (accumulate=True, fp32 out)
outputs:
  # bf16 dgrad; bf16 and fp32 wgrad both tested
reference:  deep_gemm.fp8_gemm_nt(...)
```

**Realistic shape sweep** (`tests/generators.py::enumerate_normal`, line 114):

| role | M | (N, K) candidates |
|---|---|---|
| fwd | 1, 128, 4096 | `(2112,7168), (576,7168), (24576,1536), (32768,512), (7168,16384), (4096,7168), (7168,2048)` |
| bwd | 4096 | same `(N,K)` set; dgrad and wgrad each emit one permutation |

## 2. `m_grouped_fp8_gemm_nt_contiguous` — MoE forward / prefill

Contiguous M-grouped GEMM. Token segments per expert are concatenated along M, padded to `get_mk_alignment_for_contiguous_layout()` (128 on SM90).

```yaml
name: fp8_m_grouped_gemm_nt_contig
op_type: grouped_gemm
axes:
  G:        var          # num experts
  Mg:       var          # expected M per group; actual ∈ 0.7..1.3 × Mg, padded to 128
  N, K:     var
  recipe:   const "1d2d"
inputs:
  A:               [ΣMi, K]            kFp8     # concatenated along M
  SF_a:            [ΣMi, K/128]         f32
  B:               [G, N, K]            kFp8
  SF_b:            [G, N/128, K/128]    f32
  grouped_layout:  [ΣMi]                int32   # expert id per row, or [G] PSum form
outputs:
  D:               [ΣMi, N]            bf16
reference:  deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a=(A,SF_a), b=(B,SF_b), d=D, m_indices=grouped_layout)
```

**Sweep** (`tests/generators.py::enumerate_m_grouped_contiguous`, line 153):

| (G, expected_M_per_group) | (N, K) |
|---|---|
| (4, 8192), (8, 4096) | `(6144,7168), (7168,3072), (4096,4096), (4096,2048)` |

## 3. `m_grouped_fp8_gemm_nt_masked` — MoE decode (CUDA-graph)

Same shape language as #2 but with a per-group `masked_m` tensor in place of `grouped_layout`. Designed for the case where token-counts-per-expert are GPU-resident (decode under CUDA graphs).

```yaml
name: fp8_m_grouped_gemm_nt_masked
op_type: grouped_gemm
axes:
  G:        var
  max_M:    const 4096
  Mg:       var          # expected M per group; masked_m[i] sampled per call
  N, K:     var
  recipe:   const "1d2d"
inputs:
  A:        [G, max_M, K]            kFp8
  SF_a:     [G, max_M, K/128]         f32
  B:        [G, N, K]                kFp8
  SF_b:     [G, N/128, K/128]         f32
  masked_m: [G]                       int32
outputs:
  D:        [G, max_M, N]            bf16
reference:  deep_gemm.m_grouped_fp8_gemm_nt_masked(
                a=(A,SF_a), b=(B,SF_b), d=D, masked_m=masked_m, expected_m=Mg)
```

**Sweep** (`tests/generators.py::enumerate_m_grouped_masked`, line 169):

| (G, Mg) | (N, K) |
|---|---|
| (32, 192), (6, 1024), (32, 20), (6, 20) | `(6144,7168), (7168,3072), (4096,4096), (4096,2048)` |

## 4. `k_grouped_fp8_gemm_tn_contiguous` — MoE wgrad

K-grouped, **TN** layout (the only TN FP8 kernel on SM90). M and N are fixed across groups; per-group K is variable. Always `accumulate=True`, fp32 output, 1D1D recipe.

```yaml
name: fp8_k_grouped_gemm_tn_contig
op_type: grouped_gemm
axes:
  G:      var
  M, N:   var
  Kg:     var       # expected K per group; per-group K[i] sampled, padded to 128
  recipe: const "1d1d"
inputs:
  A:        [M, ΣKi]              kFp8     # K-major; TN means K-stride on A is 1
  SF_a:     [M, ΣKi/128]           f32
  B:        [N, ΣKi]              kFp8
  SF_b:     [N, ΣKi/128]           f32
  ks:       [G]                    int32
  ks_tensor:[G]                    int32   # device-side cumulative offsets
outputs:
  D:        [G, M, N]              f32     # wgrad accumulates per expert
reference:  deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                a=(A,SF_a), b=(B,SF_b), d=D, ks=ks, ks_tensor=ks_tensor)
```

**Sweep** (`tests/generators.py::enumerate_k_grouped_contiguous`, line 185 — modeled on EP{16,32,64}):

| (G, M, N, Kg) | role |
|---|---|
| `(4, 4096, 7168, 8192)`, `(4, 7168, 2048, 8192)`   | EP64 |
| `(8, 4096, 7168, 4096)`, `(8, 7168, 2048, 4096)`   | EP32 |
| `(16, 4096, 7168, 2048)`, `(16, 7168, 2048, 2048)` | EP16 |

> Note: `k_grouped_fp8_gemm_nt_contiguous` (NT variant) is **SM100-only** — see `csrc/apis/gemm.hpp:308`. Skip on Hopper.

## 5. `fp8_gemm_nt_skip_head_mid` — attention output-projection GEMM

Specialization of `fp8_gemm_nt` that writes only the left and right head slices and zeros the middle. Used in DeepSeek's MLA-flavored attention output projection.

```yaml
name: fp8_gemm_nt_skip_head_mid
op_type: gemm
axes:
  M:           var       # tested at 128, 4096
  N:           const     # 32768 or 8192 (= num_heads * (left+right))
  K:           const     # 512
  head_splits: const     # (left=128, mid=64, right=128)
inputs:
  A:    [M, K]                  kFp8
  SF_a: [M, K/128]              f32
  B:    [N_full, K]             kFp8
  SF_b: [N_full/128, K/128]     f32
outputs:
  D:    [M, N_full + num_heads*mid]   bf16   # middle slice is zeroed
reference:  deep_gemm.fp8_gemm_nt_skip_head_mid(
                a=(A,SF_a), b=(B,SF_b), d=D, head_splits=(128,64,128))
```

**Sweep** (`tests/test_attention.py::test_gemm_skip_head_mid`):

| M | (N, K) |
|---|---|
| 128, 4096 | `(32768, 512)`, `(8192, 512)` |

## Adjacent FP8 kernels (not GEMM-shaped, but use FP8 MMAs)

If the workload sweep should also cover the lightning-indexer kernels, the two Hopper-eligible ones are:

- `fp8_mqa_logits` — non-paged scoring (`csrc/apis/attention.hpp:389`, device kernel `sm90_fp8_mqa_logits.cuh`).
- `fp8_paged_mqa_logits` — paged-KV scoring (`csrc/apis/attention.hpp:401`, device kernel `sm90_fp8_paged_mqa_logits.cuh`).

Inputs (both): `q [seq_len, num_heads, head_dim] kFp8`, `kv = (kFp8 [seq_len_kv, head_dim], f32 [seq_len_kv])`, `weights [seq_len, num_heads] f32`, plus the per-token KV index ranges. Worth a separate `attention/`-prefixed definition family rather than living under `gemm/`.

## Not eligible on Hopper (drop)

For completeness, do **not** include in the FP8-Hopper batch — these all require SM100:

- `fp8_fp4_gemm_{nt,nn,tn,tt}` — `csrc/apis/gemm.hpp:95`.
- `m_grouped_fp8_fp4_*` (contiguous and masked).
- `k_grouped_fp8_gemm_nt_contiguous` (NT variant only) — `csrc/apis/gemm.hpp:308`.
- `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits`.
- `fp8_fp4_mega_moe`.

## Recommended ordering for preparing definitions

1. **`fp8_gemm_nt_dense_fwd`** — simplest, single shape-axis story, biggest reward.
2. **`fp8_gemm_nt_dense_bwd`** — same kernel, different recipe (`1D1D`) and `accumulate=True`; reuse the same prepare script with a flag.
3. **`fp8_m_grouped_gemm_nt_contig`** — first MoE shape; introduces `grouped_layout`.
4. **`fp8_m_grouped_gemm_nt_masked`** — adds `masked_m` / `expected_m`; otherwise mirrors #3.
5. **`fp8_k_grouped_gemm_tn_contig`** — different layout (TN) and recipe (1D1D); deserves its own script.
6. **`fp8_gemm_nt_skip_head_mid`** — small / specialized; can piggyback on #1.

Each prepare script can follow the `prepare_mha_bwd_h48_d128.py` skeleton: embed `DEFINITION_DICT` + `REFERENCE_SOURCE`, write `definitions/gemm/<name>.json`, and materialize per-workload safetensors with `(A, SF_a, B, SF_b, [grouped_layout|masked_m|ks], D_ref)`. Construct the FP8 inputs via `deep_gemm.utils.per_token_cast_to_fp8` / `per_block_cast_to_fp8` from a bf16 source tensor; use the bf16 matmul as the fp32 ground truth for the verification metric (`relnorm` is the safe default for FP8 stress data, mirroring the `--metric relnorm` choice in `prepare_mha_bwd_h48_d128.py`).

## PyTorch reference (the "ground truth" DeepGEMM checks against)

DeepGEMM does **not** verify against `torch._scaled_mm` or any FP8-native PyTorch op. Its reference is the **fp32 matmul of the un-quantized bf16 source tensors**, computed before the FP8 cast — i.e. the algorithmic ideal that the FP8 path would converge to in infinite precision. The diff is scored with `deep_gemm.testing.calc_diff`, a cosine-distance-style metric (`numeric.py:5`):

```python
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return 0.0 if denom == 0 else (1 - 2 * (x * y).sum() / denom)
```

Per-shape thresholds vary (FP8 1e-4..1e-2 typical, BF16 1e-5). The flow shared by all five FP8-Hopper kernels is:

```python
# 1. Sample bf16 source.
a_bf16 = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
b_bf16 = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)

# 2. Compute reference in fp32 (this is the "PyTorch counterpart").
ref_d = (a_bf16.float() @ b_bf16.float().t()).to(out_dtype)   # + c if accumulate

# 3. Quantize to FP8 with the kernel-appropriate recipe.
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8, per_channel_cast_to_fp8
a_fp8, sf_a = per_token_cast_to_fp8(a_bf16, use_ue8m0=False, gran_k=128)
b_fp8, sf_b = per_block_cast_to_fp8(b_bf16, use_ue8m0=False, gran_k=128)   # 1D2D
# or per_token_cast_to_fp8 on b for 1D1D / accumulate paths

# 4. Run the kernel.
deep_gemm.fp8_gemm_nt(a=(a_fp8, sf_a), b=(b_fp8, sf_b), d=d)

# 5. Score.
assert deep_gemm.testing.calc_diff(d, ref_d) < tolerance
```

Per-kernel reference recipes (all from `tests/generators.py`):

| kernel | bf16 reference computation (fp32 ground truth) | A cast | B cast | source |
|---|---|---|---|---|
| `fp8_gemm_nt` (fwd, 1D2D) | `(A.float() @ B.float().t() + c).to(out_dtype)` | `per_token_cast_to_fp8` | `per_block_cast_to_fp8` | `generate_normal:271-288` |
| `fp8_gemm_nt` (bwd, 1D1D, `accumulate=True`) | same, with `c` populated and added | `per_token_cast_to_fp8` | `per_token_cast_to_fp8` | `generate_normal:271-288` (kernel_type=1D1D, `use_block_cast_for_fp8=False`) |
| `m_grouped_fp8_gemm_nt_contiguous` | per-group: `ref_d[start:aligned_end] = A[start:aligned_end] @ B[i].t()` (bf16, then `.to(bf16)`) | `per_token_cast_to_fp8` | per-group `per_block_cast_to_fp8` (via `grouped_cast_fp8_fp4_with_major`, `use_block_cast_for_fp8=True`) | `generate_m_grouped_contiguous:307-318` |
| `m_grouped_fp8_gemm_nt_masked` | `torch.einsum('gmk,gnk->gmn', A, B)` (bf16) | grouped `per_token_cast_to_fp8` | grouped `per_block_cast_to_fp8` | `generate_m_grouped_masked:346-365` |
| `k_grouped_fp8_gemm_tn_contiguous` | per-group: `ref_d[i] = c[i] + (A[s:e].T @ B[s:e])` (fp32, accumulate) | `per_channel_cast_to_fp8` (per-128-row column SF) | `per_channel_cast_to_fp8` | `generate_k_grouped_contiguous:373-407` |
| `fp8_gemm_nt_skip_head_mid` | same as `fp8_gemm_nt` fwd, then `apply_skip_head_mid(ref_d, head_splits)` zeros the middle head slice | `per_token_cast_to_fp8` | `per_block_cast_to_fp8` (or `per_token` for 1D1D) | `tests/test_attention.py::test_gemm_skip_head_mid` |

For the two adjacent attention kernels:

| kernel | reference |
|---|---|
| `fp8_mqa_logits` | `ref_logits[i, j] = (q[i].float() @ (kv[0][j].float() * kv[1][j])).relu() @ weights[i].float()` for `j ∈ [cu_seq_len_k_start[i], cu_seq_len_k_end[i])`, else `-inf` if `clean_logits=True`. See `tests/test_attention.py::test_mqa_logits` and the README block on `fp8_mqa_logits`. |
| `fp8_paged_mqa_logits` | Identical formula but `kv` is fetched through `block_table`/`context_lens` paged indexing. See `tests/test_attention.py::test_paged_mqa_logits`. |

## How the FP8 path is actually computed

The bf16-vs-fp32 reference table above is what DeepGEMM compares against, but it does not describe what the kernel itself does. To write a correct (slow) re-implementation you need the FP8 dataflow: **values are stored in `e4m3` and SF tensors are stored in fp32; dequantization is folded into the inner-product accumulation, which itself happens in fp32.** No fp8 op runs on the SFs themselves, and no per-tile fp32 dequantize-then-bf16-matmul shortcut produces bit-equivalent results — the SF must be applied at accumulation granularity, not after.

The four building blocks of the SF layouts (all from `deep_gemm/utils/math.py`, all with `gran_k = 128` on Hopper):

| recipe | A SF shape | B SF shape | semantics |
|---|---|---|---|
| **1D2D** (default fwd) | `[M, K/128]` per-token-A | `[N/128, K/128]` per-128×128-block-B | one SF per (row of A, K-block); one SF per (N-block, K-block) of B |
| **1D1D** (`accumulate=True` / fp8 bwd) | `[M, K/128]` per-token-A | `[N, K/128]` per-token-B | one SF per (row, K-block) on **both** sides |
| **per-channel** (only `k_grouped_*_tn`) | `[K/128, M]` per-128-K-row-A | `[K/128, N]` per-128-K-row-B | A is stored K-contiguous (TN); SF is shared across each block of 128 K-rows for every column |

With those layouts, every kernel is a variant of one core inner loop: dequantize `A_fp8 * SF_a` and `B_fp8 * SF_b` while accumulating into an fp32 buffer, write out as bf16 (or fp32 for accumulate paths). Per-kernel pseudocode:

### `fp8_gemm_nt` (1D2D — fwd recipe)

```python
# A_fp8: [M, K] e4m3,    SF_a: [M, K/128] f32          (per-token-A)
# B_fp8: [N, K] e4m3,    SF_b: [N/128, K/128] f32      (per-128x128-block-B)
# C, D : [M, N]                                         (out_dtype = bf16 or fp32)
for i in range(M):
    for j in range(N):
        acc = float(C[i, j]) if accumulate else 0.0
        for kb in range(K // 128):
            sa = SF_a[i,        kb]
            sb = SF_b[j // 128, kb]
            partial = 0.0
            for kk in range(128):
                k = kb * 128 + kk
                partial += float(A_fp8[i, k]) * float(B_fp8[j, k])  # fp32 accumulate
            acc += partial * sa * sb                                # SF folded per K-block
        D[i, j] = cast_to(acc, out_dtype)
```

### `fp8_gemm_nt` (1D1D — bwd / `accumulate=True`)

```python
# Same as 1D2D but B has per-token SF (one row per N).
# Replace SF_b indexing with: sb = SF_b[j, kb]
acc += partial * SF_a[i, kb] * SF_b[j, kb]
```

### `m_grouped_fp8_gemm_nt_contiguous`

```python
# A_fp8:          [sum(M_g), K] e4m3
# SF_a:           [sum(M_g), K/128] f32
# B_fp8:          [G, N, K] e4m3       (one expert weight per group)
# SF_b:           [G, N/128, K/128] f32
# m_indices:      [sum(M_g)] int32     (-1 for padding rows; expert id otherwise)
# D:              [sum(M_g), N] bf16
for i in range(sum(M_g)):
    g = m_indices[i]
    if g < 0:
        continue                       # padding row — skip
    for j in range(N):
        acc = 0.0
        for kb in range(K // 128):
            sa = SF_a[i, kb]
            sb = SF_b[g, j // 128, kb]
            partial = sum(float(A_fp8[i, kb*128 + kk]) *
                          float(B_fp8[g, j, kb*128 + kk]) for kk in range(128))
            acc += partial * sa * sb
        D[i, j] = cast_to(acc, bfloat16)
```

The PSum-layout variant (`use_psum_layout=True`) replaces `m_indices: [sum(M_g)]` (per-row expert id) with `m_indices: [G]` holding cumulative aligned-M offsets — semantically identical, just a different way to encode "which row belongs to which expert".

### `m_grouped_fp8_gemm_nt_masked`

```python
# A_fp8:    [G, max_M, K] e4m3         (each group is its own contiguous M-segment)
# SF_a:     [G, max_M, K/128] f32
# B_fp8:    [G, N, K] e4m3
# SF_b:     [G, N/128, K/128] f32
# masked_m: [G] int32                  (number of valid rows per group)
# D:        [G, max_M, N] bf16
for g in range(G):
    for i in range(masked_m[g]):       # rows beyond masked_m[g] are NOT computed
        for j in range(N):
            acc = 0.0
            for kb in range(K // 128):
                sa = SF_a[g, i,        kb]
                sb = SF_b[g, j // 128, kb]
                partial = sum(float(A_fp8[g, i, kb*128+kk]) *
                              float(B_fp8[g, j, kb*128+kk]) for kk in range(128))
                acc += partial * sa * sb
            D[g, i, j] = cast_to(acc, bfloat16)
    # D[g, masked_m[g]:, :] is left undefined — caller must respect the mask.
```

### `k_grouped_fp8_gemm_tn_contiguous`

K is the **concatenated** axis. A is stored K-contiguous (TN); per-group K segments live back-to-back. Recipe is 1D1D-equivalent via `per_channel_cast_to_fp8` (one SF per 128 K-rows, per column).

```python
# A_fp8: [sum(K_g), M] e4m3,   SF_a: [sum(K_g)/128, M] f32
# B_fp8: [sum(K_g), N] e4m3,   SF_b: [sum(K_g)/128, N] f32
# ks:    [G] int32,            ks_tensor: [G] int32 (cumulative offsets, on device)
# C, D:  [G, M, N] f32         (always accumulate, always fp32 output)
offset = 0
for g in range(G):
    K_g = ks[g]
    assert K_g % 128 == 0
    for m in range(M):
        for n in range(N):
            acc = float(C[g, m, n])
            for kb in range(K_g // 128):
                k_start = offset + kb * 128
                sa = SF_a[k_start // 128, m]
                sb = SF_b[k_start // 128, n]
                partial = sum(float(A_fp8[k_start+kk, m]) *
                              float(B_fp8[k_start+kk, n]) for kk in range(128))
                acc += partial * sa * sb
            D[g, m, n] = acc
    offset += K_g
```

### `fp8_gemm_nt_skip_head_mid`

Compute the dense `fp8_gemm_nt` output, then zero the middle slice of every head:

```python
# head_splits = (left, mid, right);   N_full = num_heads * (left + right)
D_dense = fp8_gemm_nt(A, B)                          # shape [M, N_full]
out = zeros([M, num_heads, left + mid + right])      # mid columns stay zero
out[:, :, :left]      = D_dense.view(M, num_heads, left + right)[:, :, :left]
out[:, :, -right:]    = D_dense.view(M, num_heads, left + right)[:, :, -right:]
D = out.view(M, num_heads * (left + mid + right))
```

In practice the kernel fuses the slicing into the epilogue (it never writes the mid columns), but a correct re-implementation can do it as a post-pass.

### Why not a "dequantize-then-bf16-GEMM" shortcut?

It is tempting to write `(A_fp8.float() * SF_a) @ (B_fp8.float() * SF_b).t()` and call it a day. This is **almost** correct and is fine as an algorithmic reference, but it is not bit-equivalent to the kernel: the kernel folds SF into the accumulation per K-block (each 128-element partial sum is multiplied by `sa * sb` once and added to a single fp32 accumulator), whereas the shortcut materializes `A * SF` element-wise and then runs the matmul, which changes the order of operations and the dynamic range of the multiplicands. Both forms are within `calc_diff < 1e-4` of each other for typical inputs, so either is acceptable as a verification reference — but if you are trying to match DeepGEMM bit-for-bit (e.g. for a regression test), use the per-K-block accumulation form above.

### Notes for prepare scripts

- **Use `randn` source data, not the spiky distribution.** `prepare_mha_bwd_h48_d128.py` defaults to spiky for attention because the FA3 paper uses it; for FP8 GEMMs the test sweeps use plain `torch.randn` (see `generate_normal:271-272`) and `calc_diff` is robust to that choice. Keep `randn` to mirror DeepGEMM's own tests.
- **Save both the bf16 source and the FP8-quantized inputs** in the safetensors blob. Quantization is lossy and recipe-dependent; storing only the FP8 tensors locks the workload to one recipe. The blob keys should be roughly `{A_bf16, B_bf16, A_fp8, SF_a, B_fp8, SF_b, [grouped_layout|masked_m|ks], C, D_ref}` so a downstream solution can re-quantize if it wants to test 1D1D vs 1D2D.
- **`calc_diff` thresholds** observed in tests: `< 1e-4` for FP8 forward, `< quant_config.max_diff()` (≈ 1e-3) for FP8 with FP4 mixed inputs (not relevant on Hopper), `< 1e-5` for BF16. Document the threshold per workload alongside the reference, similar to `--rtol`/`--atol` in the attention-bwd script.
- **`torch._scaled_mm` is not a useful baseline here.** PyTorch's `torch._scaled_mm` only supports per-tensor or per-row scalar SF, not the per-token-A/per-block-B layout DeepGEMM uses, and it does not exist for grouped/masked/k-grouped variants. Stick with the bf16-fp32 algorithmic reference.
