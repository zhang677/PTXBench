"""Baseline CUDA solutions for TRT-LLM Tier 3 kernels.

Each solution adapts the TRT-LLM kernel logic into a standalone CUDA file
with TVM-FFI bindings (destination-passing style).

Source: https://github.com/NVIDIA/TensorRT-LLM (Apache-2.0)
"""

from flashinfer_bench.data import Definition, Solution

from accrl.utils.solution_utils import build_solution

# ---------------------------------------------------------------------------
# Shared CUDA helpers
# ---------------------------------------------------------------------------

_TVM_FFI_INCLUDES = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
"""

# ---------------------------------------------------------------------------
# 1. Causal Attention Mask  (bool)
# ---------------------------------------------------------------------------

_CAUSAL_ATTENTION_MASK_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM buildDecoderAttentionMaskKernel.
// Generates lower-triangular causal mask: output[b, i, j] = (j <= i).
// Grid: (B, ceil(S*S / 256)), Block: 256.

__global__ void causal_mask_kernel(
    int8_t* __restrict__ output,  // [B, S, S] stored as int8
    int B, int S)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int64_t total = (int64_t)S * S;
    for (int64_t idx = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.y * blockDim.x)
    {
        int i = (int)(idx / S);
        int j = (int)(idx % S);
        output[(int64_t)b * S * S + idx] = (j <= i) ? 1 : 0;
    }
}

// TVM-FFI entry: run(input, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output)
{
    int B = input.size(0);
    int S = input.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    int block = 256;
    int64_t total = (int64_t)S * S;
    int gridy = (int)((total + block - 1) / block);

    dim3 grid(B, gridy);
    causal_mask_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<int8_t*>(output.data_ptr()),
        B, S);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 2. Ban Repeat N-gram  (f32, i32 -> f32)
# ---------------------------------------------------------------------------

_BAN_REPEAT_NGRAM_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM banRepeatNgram kernel.
// Scans output_ids for trigram matches and bans completing tokens with -inf.
// One block per batch row, block=256. NGRAM=3.

__global__ void ban_repeat_ngram_kernel(
    const float* __restrict__ logits,      // [B, V]
    const int* __restrict__ output_ids,    // [B, L]
    float* __restrict__ output,            // [B, V]
    int B, int V, int L)
{
    const int NGRAM = 3;
    int b = blockIdx.x;
    if (b >= B) return;

    const float* in_row = logits + (int64_t)b * V;
    float* out_row = output + (int64_t)b * V;
    const int* ids = output_ids + (int64_t)b * L;

    // Phase 1: copy logits to output
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        out_row[i] = in_row[i];
    }
    __syncthreads();

    if (L < NGRAM) return;

    // Suffix = last (NGRAM-1) tokens
    int suffix0 = ids[L - 2];
    int suffix1 = ids[L - 1];

    // Phase 2: scan windows, ban matching tokens
    int num_windows = L - NGRAM + 1;
    for (int w = threadIdx.x; w < num_windows; w += blockDim.x) {
        // Check if prefix of window matches suffix
        if (ids[w] == suffix0 && ids[w + 1] == suffix1) {
            int banned_token = ids[w + 2];
            if (banned_token >= 0 && banned_token < V) {
                out_row[banned_token] = -INFINITY;
            }
        }
    }
}

// TVM-FFI entry: run(logits, output_ids, output)
void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView output_ids,
         tvm::ffi::TensorView output)
{
    int B = logits.size(0);
    int V = logits.size(1);
    int L = output_ids.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(logits.device().device_type, logits.device().device_id));

    dim3 grid(B);
    int block = 256;

    ban_repeat_ngram_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<const int*>(output_ids.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        B, V, L);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 3. Group RMSNorm  (bf16 -> bf16, float32 compute)
# ---------------------------------------------------------------------------

_GROUP_RMSNORM_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM generalT5LayerNorm (rmsnormKernels.cu).
// Single rmsnorm_kernel reused for both inputs.
// One block per row, block=256. Two-pass: sum-of-squares then normalize.

__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,   // [N, D]
    const __nv_bfloat16* __restrict__ weight,  // [D]
    __nv_bfloat16* __restrict__ output,         // [N, D]
    int N, int D)
{
    const float eps = 1e-6f;
    int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* in_row = input + (int64_t)row * D;
    __nv_bfloat16* out_row = output + (int64_t)row * D;

    // Pass 1: compute sum of squares (warp-level reduction + shared mem)
    __shared__ float s_partial[8];  // up to 8 warps (256/32)

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __bfloat162float(in_row[i]);
        local_sum += v * v;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        s_partial[warp_id] = local_sum;
    __syncthreads();

    // Cross-warp reduction (first warp only)
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            s_partial[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(s_partial[0] / (float)D + eps);

    // Pass 2: normalize with weight
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __bfloat162float(in_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16_rn(v * rms_inv * w);
    }
}

// TVM-FFI entry: run(input1, input2, weight1, weight2, output1, output2)
void run(tvm::ffi::TensorView input1, tvm::ffi::TensorView input2,
         tvm::ffi::TensorView weight1, tvm::ffi::TensorView weight2,
         tvm::ffi::TensorView output1, tvm::ffi::TensorView output2)
{
    int N = input1.size(0);
    int D1 = input1.size(1);
    int D2 = input2.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input1.device().device_type, input1.device().device_id));

    int block = 256;

    // Launch for input1
    rmsnorm_kernel<<<N, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input1.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output1.data_ptr()),
        N, D1);

    // Launch for input2
    rmsnorm_kernel<<<N, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input2.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight2.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output2.data_ptr()),
        N, D2);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 4. Relative Attention Bias  (f32)
# ---------------------------------------------------------------------------

_RELATIVE_ATTENTION_BIAS_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM computeAttentionBias (relativeAttentionBiasKernel.cu).
// T5-style bidirectional relative position bucketing.
// One block per head, block=256. Threads stride over S*S elements.

__device__ int relative_position_bucket(
    int rel_pos, int num_buckets, int max_distance)
{
    // Bidirectional: half buckets for positive, half for negative
    int half = num_buckets / 2;
    int max_exact = half / 2;

    int is_neg = (rel_pos < 0) ? 1 : 0;
    int abs_pos = (rel_pos < 0) ? -rel_pos : rel_pos;

    int bucket;
    if (abs_pos < max_exact) {
        bucket = abs_pos;
    } else {
        // Log-scaled for large distances
        float frac = logf((float)abs_pos / (float)max_exact)
                   / logf((float)max_distance / (float)max_exact);
        bucket = max_exact + (int)(frac * (float)(half - max_exact));
        if (bucket > half - 1)
            bucket = half - 1;
    }

    // Positive direction gets offset by half
    if (!is_neg)
        bucket += half;

    return bucket;
}

__global__ void relative_bias_kernel(
    const float* __restrict__ bias_table,  // [H, num_buckets]
    float* __restrict__ output,            // [H, S, S]
    int H, int S, int num_buckets, int max_distance)
{
    int h = blockIdx.x;
    if (h >= H) return;

    const float* table_row = bias_table + (int64_t)h * num_buckets;
    float* out_head = output + (int64_t)h * S * S;

    int64_t total = (int64_t)S * S;
    for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int qi = (int)(idx / S);
        int ki = (int)(idx % S);
        int rel_pos = ki - qi;
        int bucket = relative_position_bucket(rel_pos, num_buckets, max_distance);
        out_head[idx] = table_row[bucket];
    }
}

// TVM-FFI entry: run(bias_table, positions, output)
void run(tvm::ffi::TensorView bias_table, tvm::ffi::TensorView positions,
         tvm::ffi::TensorView output)
{
    int H = bias_table.size(0);
    int num_buckets = bias_table.size(1);
    int S = positions.size(0);
    int max_distance = 128;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(bias_table.device().device_type,
                           bias_table.device().device_id));

    dim3 grid(H);
    int block = 256;

    relative_bias_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(bias_table.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        H, S, num_buckets, max_distance);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 5. Block FP8 Quantization  (bf16 -> fp8_e4m3)
# ---------------------------------------------------------------------------

_BLOCK_FP8_QUANT_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM mxfp quantization kernel.
// Per-32-element block scaling: bf16 -> fp8_e4m3.
// One block per row. Each thread handles one 32-element block.

__global__ void block_fp8_quant_kernel(
    const __nv_bfloat16* __restrict__ input,  // [R, C]
    __nv_fp8_storage_t* __restrict__ output,  // [R, C] as uint8
    int R, int C)
{
    const int BLOCK_SIZE = 32;
    const float FP8_MAX = 448.0f;

    int row = blockIdx.x;
    if (row >= R) return;

    int num_blocks = C / BLOCK_SIZE;
    const __nv_bfloat16* in_row = input + (int64_t)row * C;
    __nv_fp8_storage_t* out_row = output + (int64_t)row * C;

    for (int blk = threadIdx.x; blk < num_blocks; blk += blockDim.x) {
        int base = blk * BLOCK_SIZE;
        float vals[32];

        // Load bf16 -> f32
        for (int i = 0; i < BLOCK_SIZE; i++)
            vals[i] = __bfloat162float(in_row[base + i]);

        // Find absmax
        float absmax = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float a = fabsf(vals[i]);
            if (a > absmax) absmax = a;
        }

        // Scale and quantize
        float scale = fmaxf(absmax, 1e-12f) / FP8_MAX;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float scaled = vals[i] / scale;
            out_row[base + i] = __nv_cvt_float_to_fp8(
                scaled, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// TVM-FFI entry: run(input, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output)
{
    int R = input.size(0);
    int C = input.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    dim3 grid(R);
    int num_blocks_per_row = C / 32;
    int block = min(num_blocks_per_row, 256);
    block = max(block, 1);

    block_fp8_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
        R, C);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 6. Fused ReLU² + FP8 Quantization  (bf16 -> fp8_e4m3)
# ---------------------------------------------------------------------------

_FUSED_RELU2_FP8_QUANT_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM mxfp quantization with fused ReLU² activation.
// Per-32-element block scaling: bf16 -> relu² -> fp8_e4m3.
// One block per row. Each thread handles one 32-element block.

__global__ void fused_relu2_fp8_quant_kernel(
    const __nv_bfloat16* __restrict__ input,  // [R, C]
    __nv_fp8_storage_t* __restrict__ output,  // [R, C] as uint8
    int R, int C)
{
    const int BLOCK_SIZE = 32;
    const float FP8_MAX = 448.0f;

    int row = blockIdx.x;
    if (row >= R) return;

    int num_blocks = C / BLOCK_SIZE;
    const __nv_bfloat16* in_row = input + (int64_t)row * C;
    __nv_fp8_storage_t* out_row = output + (int64_t)row * C;

    for (int blk = threadIdx.x; blk < num_blocks; blk += blockDim.x) {
        int base = blk * BLOCK_SIZE;
        float vals[32];

        // Load bf16 -> f32, apply ReLU²
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float v = __bfloat162float(in_row[base + i]);
            v = fmaxf(v, 0.0f);
            vals[i] = v * v;
        }

        // Find absmax
        float absmax = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float a = fabsf(vals[i]);
            if (a > absmax) absmax = a;
        }

        // Scale and quantize
        float scale = fmaxf(absmax, 1e-12f) / FP8_MAX;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float scaled = vals[i] / scale;
            out_row[base + i] = __nv_cvt_float_to_fp8(
                scaled, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// TVM-FFI entry: run(input, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output)
{
    int R = input.size(0);
    int C = input.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    dim3 grid(R);
    int num_blocks_per_row = C / 32;
    int block = min(num_blocks_per_row, 256);
    block = max(block, 1);

    fused_relu2_fp8_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
        R, C);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)


# ---------------------------------------------------------------------------
# Solution builder functions
# ---------------------------------------------------------------------------


def make_causal_attention_mask_solution(defn: Definition) -> Solution:
    return build_solution(
        _CAUSAL_ATTENTION_MASK_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_ban_repeat_ngram_solution(defn: Definition) -> Solution:
    return build_solution(
        _BAN_REPEAT_NGRAM_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_group_rmsnorm_solution(defn: Definition) -> Solution:
    return build_solution(
        _GROUP_RMSNORM_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_relative_attention_bias_solution(defn: Definition) -> Solution:
    return build_solution(
        _RELATIVE_ATTENTION_BIAS_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_block_fp8_quant_solution(defn: Definition) -> Solution:
    return build_solution(
        _BLOCK_FP8_QUANT_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_fused_relu2_fp8_quant_solution(defn: Definition) -> Solution:
    return build_solution(
        _FUSED_RELU2_FP8_QUANT_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


# Map definition name -> solution maker
SOLUTION_MAKERS: dict[str, callable] = {
    "trtllm_causal_attention_mask": make_causal_attention_mask_solution,
    "trtllm_ban_repeat_ngram": make_ban_repeat_ngram_solution,
    "trtllm_group_rmsnorm": make_group_rmsnorm_solution,
    "trtllm_relative_attention_bias": make_relative_attention_bias_solution,
    "trtllm_block_fp8_quant": make_block_fp8_quant_solution,
    "trtllm_fused_relu2_fp8_quant": make_fused_relu2_fp8_quant_solution,
}
