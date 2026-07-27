# Qwen3.6-27B Pair-Notes Fixit Reasoning and Kernel Delta Analysis

Run root: `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-pair-notes-fixit`

## Scope
- Planned fix tasks: `159`
- Qwen trajectories analyzed: `159`
- Qwen final kernels compared: `159`
- Paired Gemini success kernels found: `159`
- Qwen success files in this run: `0`

## Outcome
Final trajectory feedback labels:
- `runtime_error`: `46` (28.9%)
- `timeout`: `42` (26.4%)
- `compile_error`: `39` (24.5%)
- `numerical_error`: `31` (19.5%)
- `other`: `1` (0.6%)

All feedback labels across turns:
- `compile_error`: `264`
- `runtime_error`: `247`
- `timeout`: `160`
- `numerical_error`: `120`
- `other`: `4`

By definition:
- `mha_bwd_d128`: `timeout`=13, `compile_error`=9, `runtime_error`=8, `numerical_error`=4
- `mha_bwd_d128_causal`: `runtime_error`=15, `timeout`=10, `compile_error`=9, `numerical_error`=3, `other`=1
- `mha_with_lse_d128`: `timeout`=13, `numerical_error`=12, `compile_error`=11, `runtime_error`=9
- `mha_with_lse_d128_causal`: `runtime_error`=14, `numerical_error`=12, `compile_error`=10, `timeout`=6

## Kernel Similarity
- Qwen final vs paired Gemini correct kernel line similarity: median 0.333, mean 0.355, min 0.048, max 0.873
- Wrong input vs Qwen final kernel line similarity: median 0.405, mean 0.437, min 0.047, max 0.903
- Per-definition Qwen-vs-Gemini line similarity:
  - `mha_bwd_d128`: median 0.263, mean 0.281, min 0.083, max 0.603
  - `mha_bwd_d128_causal`: median 0.265, mean 0.287, min 0.115, max 0.873
  - `mha_with_lse_d128`: median 0.452, mean 0.426, min 0.112, max 0.746
  - `mha_with_lse_d128_causal`: median 0.380, mean 0.402, min 0.048, max 0.792

Feature mismatches between final Qwen kernel and paired Gemini kernel:
- `has_cluster_tma`: `81` pairs (50.9%)
- `has_setmaxnreg`: `49` pairs (30.8%)
- `has_atomic_float`: `14` pairs (8.8%)
- `has_cta_tma`: `12` pairs (7.5%)
- `has_cutensor_map`: `9` pairs (5.7%)
- `has_tma`: `8` pairs (5.0%)
- `has_cuda_func_attr`: `7` pairs (4.4%)
- `has_mbarrier`: `6` pairs (3.8%)
- `has_wgmma`: `5` pairs (3.1%)

Lowest-similarity examples:
- `exp_157` `mha_with_lse_d128_causal` final=`compile_error` qwen/gemini ratio=`0.048` wrong/qwen ratio=`0.067`
- `exp_078` `mha_bwd_d128` final=`compile_error` qwen/gemini ratio=`0.0832` wrong/qwen ratio=`0.1224`
- `exp_062` `mha_bwd_d128` final=`timeout` qwen/gemini ratio=`0.0966` wrong/qwen ratio=`0.0731`
- `exp_118` `mha_with_lse_d128_causal` final=`compile_error` qwen/gemini ratio=`0.102` wrong/qwen ratio=`0.0982`
- `exp_025` `mha_with_lse_d128` final=`compile_error` qwen/gemini ratio=`0.1117` wrong/qwen ratio=`0.3089`

Highest-similarity examples:
- `exp_094` `mha_bwd_d128_causal` final=`timeout` qwen/gemini ratio=`0.873` wrong/qwen ratio=`0.8674`
- `exp_156` `mha_with_lse_d128_causal` final=`numerical_error` qwen/gemini ratio=`0.7923` wrong/qwen ratio=`0.8741`
- `exp_015` `mha_with_lse_d128` final=`timeout` qwen/gemini ratio=`0.7464` wrong/qwen ratio=`0.751`
- `exp_147` `mha_with_lse_d128_causal` final=`numerical_error` qwen/gemini ratio=`0.731` wrong/qwen ratio=`0.7538`
- `exp_142` `mha_with_lse_d128_causal` final=`runtime_error` qwen/gemini ratio=`0.7087` wrong/qwen ratio=`0.8042`

## Reasoning Categories
Category hits are regex-based over assistant `reasoning_content` plus assistant messages. They are diagnostic, not mutually exclusive.
- `wgmma_descriptor_layout`: `159` trajectories (100.0%); finals: runtime_error=46, timeout=42, compile_error=39, numerical_error=31, other=1
- `mbarrier_tma_sequence`: `159` trajectories (100.0%); finals: runtime_error=46, timeout=42, compile_error=39, numerical_error=31, other=1
- `softmax_lse_reduction`: `159` trajectories (100.0%); finals: runtime_error=46, timeout=42, compile_error=39, numerical_error=31, other=1
- `indexing_block_mapping`: `159` trajectories (100.0%); finals: runtime_error=46, timeout=42, compile_error=39, numerical_error=31, other=1
- `compile_symbol_patch`: `159` trajectories (100.0%); finals: runtime_error=46, timeout=42, compile_error=39, numerical_error=31, other=1
- `minimal_change_fixation`: `159` trajectories (100.0%); finals: runtime_error=46, timeout=42, compile_error=39, numerical_error=31, other=1
- `uncertainty_or_guessing`: `159` trajectories (100.0%); finals: runtime_error=46, timeout=42, compile_error=39, numerical_error=31, other=1
- `tma_tensor_map_coords`: `158` trajectories (99.4%); finals: runtime_error=46, timeout=42, compile_error=38, numerical_error=31, other=1
- `performance_timeout`: `154` trajectories (96.9%); finals: runtime_error=45, timeout=41, compile_error=36, numerical_error=31, other=1

Reasoning length:
- Reasoning chars: median `107388`, mean `107569`, max `295080`
- Assistant visible-content chars: median `80544`, mean `83590`, max `155472`

## Representative Reasoning Snippets
### wgmma_descriptor_layout
- `exp_000` final=`timeout`: 01[32], acc_10[32], acc_11[32]; #pragma unroll for(int i=0;i<32;i++){acc_00[i]=acc_01[i]=acc_10[i]=acc_11[i]=0.0f;} uint32_t swizzle = 1; // 128B swizzle uint32_t lbo = 1; // K-major swizzled: LBO=1 uint32_t sbo = 1024; // K-major swizzled: SBO=1024 // Q descriptor offsets for 2x2 tiling uint64_t da_base_0 = make_wgmma_desc(sQ, lbo, sbo, swizzle); uint64_t da_base_1 = make_wgmma_desc((void*)((char
- `exp_001` final=`runtime_error`: eps. This matches note 1: `boxDim={1, 128, 64}` or similar. Let's adjust: `globalDim = {B*H, S, D}`. `globalStrides = {S*D*2, D*2}`. `boxDim = {1, BLOCK_M, 64}`. `swizzle = 128B`. LBO/SBO for swizzle 128B MN-major: From docs: `SBO = 128 * 8 = 1024` (8 spans along outer dim within swizzle band). `LBO` is not used for swizzle, assume 1. Wait, note 1 says `globalDim={128, S, B*H}`. That's [D, S, BH]. I'll stick to [D, S, BH] to match notes ex
- `exp_002` final=`numerical_error`: and B to simplify, or adjust accordingly. The prompt example uses `trans_a=1, trans_b=1` but note 5 says "Do not cargo-cult trans_a=1, trans_b=1". For 128B swizzle K-major layout, LBO=1, SBO=1024. If we load directly into that layout via TMA, we can use `trans_a=0, trans_b=0`. Wait, TMA layout for K-major with 128B swizzle: `boxDim[0] = 64` (inner), `boxDim[1] = 128` (outer) for a 64x128 tile. Actually, let's look at the reference for TMA

### mbarrier_tma_sequence
- `exp_000` final=`timeout`: roup running WGMMA m64n64k16 tiles. - The TMA copy functions used are incorrect or mismatched with the provided helpers. The provided helpers use `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` but the note says to use `cuTensorMapEncodeTiled` and `cp.async.bulk.tensor.2d`. However, to keep it simple and matching the provided helper structure, I can stick to `cp.async.bulk` if done correctly, or implement the TMA tensor copy pro
- `exp_001` final=`runtime_error`: g `cp.async.bulk.tensor`. The collected repair notes suggest replacing raw 1D `cp.async.bulk` copies with TMA tensor loads (`cp.async.bulk.tensor.2d` or `.3d`) backed by explicit `mbarrier expect-tx` accounting, grid-constant `CUtensorMap` descriptors, and swizzled WGMMA-compatible shared-memory layouts. Let's look at the task: Inputs: Q, K, V of shape [B, H, S, D] where B=4, H=48, D=128, S=var (4096 in eval). Outputs: O [B, H, S, D] bf16, LSE
- `exp_002` final=`numerical_error`: ta.global` and basic barriers, which is causing a timeout and failing an instruction check (`cp.async.bulk.tensor`). I need to refactor it to use `cp.async.bulk.tensor.2d` (TMA), `mbarrier.arrive.expect_tx`, correct WGMMA descriptors, and follow the collected repair notes. Key changes needed based on the prompt: 1. Use `CUtensorMap` with `__grid_constant__` qualifier passed by value to the kernel. 2. Create TMA descriptors on the host using `c

### tma_tensor_map_coords
- `exp_000` final=`timeout`: ions used are incorrect or mismatched with the provided helpers. The provided helpers use `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` but the note says to use `cuTensorMapEncodeTiled` and `cp.async.bulk.tensor.2d`. However, to keep it simple and matching the provided helper structure, I can stick to `cp.async.bulk` if done correctly, or implement the TMA tensor copy properly. Actually, the prompt provides `tma_copy_1d_g2s_fn` w
- `exp_001` final=`runtime_error`: notes suggest replacing raw 1D `cp.async.bulk` copies with TMA tensor loads (`cp.async.bulk.tensor.2d` or `.3d`) backed by explicit `mbarrier expect-tx` accounting, grid-constant `CUtensorMap` descriptors, and swizzled WGMMA-compatible shared-memory layouts. Let's look at the task: Inputs: Q, K, V of shape [B, H, S, D] where B=4, H=48, D=128, S=var (4096 in eval). Outputs: O [B, H, S, D] bf16, LSE [B, H, S] fp32. We need to implement a flash att
- `exp_002` final=`numerical_error`: se `cp.async.bulk.tensor.2d` (TMA), `mbarrier.arrive.expect_tx`, correct WGMMA descriptors, and follow the collected repair notes. Key changes needed based on the prompt: 1. Use `CUtensorMap` with `__grid_constant__` qualifier passed by value to the kernel. 2. Create TMA descriptors on the host using `cuTensorMapEncodeTiled`. 3. Replace `cp_async_bulk_g2s` with `tma_load_2d_fn` using `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::compl

### softmax_lse_reduction
- `exp_000` final=`timeout`: eq` is undefined. It should be `blocks_per_head_seq`. 2. `FLT_MAX` is undefined. Need `#include <float.h>`. Additionally, looking at the code logic, there are major issues: - The softmax accumulation and update logic inside the loop is incomplete/broken (`// Due to complexity, we fall back...`, `// Simplified softmax update...`). I need to implement a correct online softmax update using shared memory or registers, suitable for a 128-thread wa
- `exp_001` final=`runtime_error`: (128 threads). Actually, D=128 means K dimension for QK^T is 128. Wait, Q is [B,H,S,D], K is [B,H,S,D]. Attention computes P = Q @ K^T / sqrt(D). Shape of P is [B,H,S,S]. Then O = softmax(P) @ V. Shape of O is [B,H,S,D]. So we compute blocks of P of size BLOCK_M x BLOCK_N (128x128). For each block, we load a tile of Q (BLOCK_M x D), a tile of K (BLOCK_N x D), and a tile of V (BLOCK_N x D). Wait, D=128. BLOCK_M=128, BLOCK_N=128. Q tile: 128x12
- `exp_002` final=`numerical_error`: emory arrays: `__nv_bfloat16 smem_Q[HEAD_DIM][BLOCK_M];` `__nv_bfloat16 smem_K[HEAD_DIM][BLOCK_N];` `__nv_bfloat16 smem_V[HEAD_DIM][BLOCK_N];` `float smem_S[BLOCK_M][BLOCK_N];` // Softmax scores / intermediate O accumulation isn't needed here; we accumulate O directly or use shared mem for softmax online. Actually, online softmax accumulation into `O_raw` shared memory: `float smem_O[BLOCK_M][HEAD_DIM];` Barriers: `uint64_t bar_Q, bar_K, bar_V

### indexing_block_mapping
- `exp_000` final=`timeout`: CK_M * BLOCK_D; __nv_bfloat16* sV = sK + BLOCK_N * BLOCK_D; float* sO = reinterpret_cast<float*>(sV + BLOCK_N * BLOCK_D); int tid = threadIdx.x; int bid = blockIdx.x; int blocks_per_head_seq = (S + BLOCK_M - 1) / BLOCK_M; int total_bid = bid % (H * blocks_per_head_seq); int b = bid / (H * blocks_per_head_seq); int h = total_bid / blocks_per_head_seq; int bm = (total_bid % blocks_per_head_seq) * BLOCK
- `exp_001` final=`runtime_error`: kernel(const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K, const __grid_constant__ CUtensorMap tma_V, ...)` Inside kernel: Block x handles `bh = blockIdx.x % (B*H)`, `bm_tile = blockIdx.x / (B*H)`. `bm = bm_tile * BLOCK_M`. TMA coords for Q: `{bh, bm, 0}`. Loop over D in 2 chunks: `{bh, bm, 0}` and `{bh, bm, 64}`. Same for K/V but column tile `bn` changes. Wait, K and V are loaded per `bn` iteration. `bn` goes
- `exp_002` final=`numerical_error`: b, h)` might be too few. B=4, H=48 -> 192 blocks. That's fine for modern GPUs. Grid: `(B*H, num_m_blocks, 1)`. `num_m_blocks = (S + BLOCK_M - 1) / BLOCK_M`. Block: `(128, 1, 1)`. `blockIdx.x` maps to `bh_idx`. `blockIdx.y` maps to `m_block_idx`. Wait, the original used `blockIdx.y` for `m_block`. I'll keep that. `int bh_idx = blockIdx.x; int b = bh_idx / H; int h = bh_idx % H;` `int m_block = blockIdx.y * BLOCK_M;` `int n_limit = S;` Coordinate

### compile_symbol_patch
- `exp_000` final=`timeout`: The user wants to fix a CUDA kernel implementing Multi-Head Attention (MHA) for Hopper architecture. The previous attempt had compilation errors: 1. `blocks_per_seq` is undefined. It should be `blocks_per_head_seq`. 2. `FLT_MAX` is undefined. Need `#include <float.h>`. Additionally, looking at the code logic, there are major issues: - The softmax accumulation and update logic inside the loop is incomplete/broken (`// Due to complexit
- `exp_001` final=`runtime_error`: e's `shared::cluster`. The note says `cp.async.bulk.tensor.2d.shared::cluster.global...`. I'll follow note 1 exactly: `shared::cluster`. Let's assemble carefully. Check types and includes. Need `<cuda.h>` for `cuTensorMapEncodeTiled`. Define helper functions as inline/device. Make sure `stride_h` and `stride_s` aren't strictly needed if we use TMA with BH coordinate. But we still need them for O and LSE output indexing. Output O and LSE are w
- `exp_002` final=`numerical_error`: _idx * S + m_block + row)`. Correct. - Includes: `<cuda.h>` is necessary for `CUtensorMap`. - All good. Output matches. Proceeds. [Done] ` The error is `TVMFFIEnvGetStream` is undefined. I need to check how to get the stream in TVM-FFI. Usually, it's accessed via an FFI function or passed as a parameter, but here the signature of `run` doesn't include a stream parameter. Wait, TVM-FFI typically provides a way to get the current stream or we

### performance_timeout
- `exp_000` final=`timeout`: o distribute. Simpler: Put `s_m` and `s_l` in shared memory. `float* s_m = sO + BLOCK_M*BLOCK_D; float* s_l = s_m + BLOCK_M;` This saves 256 registers! Crucial for avoiding spills/timeouts. I will modify shared memory layout to include `s_m` and `s_l`. Updated layout: `sQ`, `sK`, `sV`, `sO`, `s_m`, `s_l`. Access `s_m[i]` and `s_l[i]` directly. This will drastically reduce register pressure. Done. Proceeds. [Final Check of the Code Structure]
- `exp_001` final=`runtime_error`: ich TVM/CUDA usually does for such kernels). I'll proceed. Wait, the reference says `ptxas info : Used 255 registers`. Register pressure is high. Maybe reducing spills would help performance, but correctness first. One more thing: `tma_load_3d_fn` coordinates order. `cp.async.bulk.tensor.3d ... [%0], [%1, {%3, %4, %5}], [%2]` Coords are `c0, c1, c2`. `c0` maps to `dim0`, `c1` to `dim1`, `c2` to `dim2`. My `globalDim` is `{BH, S, D}`. So `c0` is
- `exp_002` final=`numerical_error`: r wants to fix a CUDA kernel for Multi-Head Attention (MHA) forward pass. The current implementation uses `cp.async.bulk.shared::cta.global` and basic barriers, which is causing a timeout and failing an instruction check (`cp.async.bulk.tensor`). I need to refactor it to use `cp.async.bulk.tensor.2d` (TMA), `mbarrier.arrive.expect_tx`, correct WGMMA descriptors, and follow the collected repair notes. Key changes needed based on the prompt: 1.

### minimal_change_fixation
- `exp_000` final=`timeout`: one go is hard. I'll stick to a simpler, robust approach that compiles and gives correct results, then optimize if needed. But the prompt expects me to fix the *given* kernel with minimal changes, addressing the compile errors and fixing the broken softmax logic. Let's look at the broken parts: ```cpp // Compute row_max and row_sum for softmax #pragma unroll for(int i=0; i<BLOCK_M; i++) { s_row_max[i] = -FLT_MAX; s_row_sum[i]
- `exp_001` final=`runtime_error`: ost, and pass base pointers for each block? No, TMA descriptor contains globalAddress. We would need to launch with different descriptors or compute offsets. Given the constraint "minimal change" but also "must use cp.async.bulk.tensor", I will follow the reference closely. I will create TMA descriptors for the full tensor flattened to [B*H, S, D]. In kernel, block (bx) computes `bh = bx % (B*H)`, `bm_idx = bx / (B*H)`. `b = bh / H`, `h = bh % H`. Ba
- `exp_002` final=`numerical_error`: layout in SMEM without worrying about complex swizzle mapping in declarations, I can use `alignas(256) uint8_t Q_smem[BLOCK_M * HEAD_DIM * 2];` and compute pointers carefully, OR just fix the descriptor to match the C++ array layout. But C++ array `Q_smem[128][128]` has row stride 256 bytes. If I pass `lbo=256, sbo=256?` No. Actually, the easiest fix for numerical correctness is to ensure the TMA/WGMMA layout matches exactly. Let's look at sta

### uncertainty_or_guessing
- `exp_000` final=`timeout`: Attention-style MHA kernel using `__shared__` arrays manually loaded/stored or use `cp.async.bulk` correctly, but given the constraints and complexity of TMA setup in this format, maybe manual async copy (`cp.async.bulk.commit_group`) or standard shared memory loading is safer? Wait, the reference material heavily emphasizes TMA and WGMMA. I will implement a working Hopper MHA using WGMMA and shared memory. Actually, writing a fully optimiz
- `exp_001` final=`runtime_error`: * 2 So `globalStrides = {S * D * 2, D * 2}`. My code had `{dim1 * dim2 * 2, dim2 * 2}` which matches `{S*D*2, D*2}`. That seems correct. Why does it fail with "invalid argument"? Maybe `boxDim` constraints? "Elements in boxDim must be non-zero, less than or equal to 256. These dimensions must 16 byte-aligned". My boxDim: `{1, 128, 64}`. All <= 256. Aligned? Box dimensions are in elements. For BF16, 16-byte alignment means elements must be m
- `exp_002` final=`numerical_error`: ray `V[HEAD_DIM][BLOCK_N]` means `V[d][c] = base + (d * BLOCK_N + c) * 2`. So `V[0][c], V[1][c], ...` are spaced by `BLOCK_N * 2 = 256` bytes. This is fine, no bank conflicts. But maybe we want `V` layout `[BLOCK_N][HEAD_DIM]` so that `V[c][d]` is contiguous? Then `smem.V[c][d]` would be contiguous. If `V` is `[BLOCK_N][HEAD_DIM]`, then TMA load needs to match. TMA loads row-major tensor `[BHS, D]` into shared memory. If we want `[BLOCK_N][H

## Interpretation
- Qwen generally recognized the same high-level primitive family as the notes: TMA tensor maps, mbarriers, WGMMA descriptors, softmax/LSE reductions, and launch shared-memory configuration.
- The failures cluster around integrating those primitives correctly. The reasoning repeatedly debates descriptor layout constants, mbarrier parity/transaction counts, tensor-map coordinates, WGMMA fragment mapping, and softmax/LSE/indexing details.
- Many trajectories show local compile-fix behavior: Qwen identifies one concrete compiler error or typo and then declares the kernel likely correct, even when the remaining feedback is timeout, runtime, or numerical failure.
- The final Qwen kernels are not close copies of Gemini repairs. Median line similarity to Gemini is low, and feature mismatches show that even when both use the same broad primitives, they often differ in the precise instruction variant, shared-memory layout, register/shared-memory structure, or launch setup.
- This supports the earlier diagnosis: summarized notes were enough to nudge Qwen toward the right vocabulary and APIs, but not enough to reproduce the exact working repair structure.

## Generated Artifacts
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-pair-notes-fixit/analysis/qwen_vs_gemini_kernel_comparison.csv`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-pair-notes-fixit/analysis/qwen_vs_gemini_diff_stats.csv`
