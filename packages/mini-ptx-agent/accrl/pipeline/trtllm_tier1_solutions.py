"""Baseline CUDA solutions for TRT-LLM Tier 1 kernels.

Each solution adapts the TRT-LLM kernel logic into a standalone CUDA file
with TVM-FFI bindings (destination-passing style).

Source: https://github.com/NVIDIA/TensorRT-LLM (Apache-2.0)
"""

from flashinfer_bench.data import Definition, Solution

from accrl.utils.solution_utils import build_solution

# ---------------------------------------------------------------------------
# Shared CUDA helpers (inlined into each kernel source)
# ---------------------------------------------------------------------------

_REDUCE_HELPERS = r"""
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float _brs[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) _brs[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? _brs[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float _brm[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) _brm[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? _brm[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}
"""

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
# 1. LayerNorm  (bf16)
# ---------------------------------------------------------------------------

_LAYERNORM_SOURCE = (
    _TVM_FFI_INCLUDES
    + _REDUCE_HELPERS
    + r"""
// Adapted from TRT-LLM generalLayerNorm kernel (layernormKernels.cu).
// Simplified: bf16 only, no quantization, two-pass (mean then variance).
// One CTA per token row.

__global__ void layernorm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    int num_tokens,
    int hidden_dim)
{
    const float EPS = 1e-6f;
    int row = blockIdx.x;
    if (row >= num_tokens) return;

    const __nv_bfloat16* x = input + (int64_t)row * hidden_dim;
    __nv_bfloat16* y = output + (int64_t)row * hidden_dim;

    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        local_sum += __bfloat162float(x[i]);
    }
    float mean = block_reduce_sum(local_sum);
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = mean / hidden_dim;
    __syncthreads();
    mean = s_mean;

    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float diff = __bfloat162float(x[i]) - mean;
        local_var += diff * diff;
    }
    float variance = block_reduce_sum(local_var);
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) s_inv_std = rsqrtf(variance / hidden_dim + EPS);
    __syncthreads();
    float inv_std = s_inv_std;

    // Pass 3: normalize
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        float g = __bfloat162float(gamma[i]);
        float b = __bfloat162float(beta[i]);
        float normed = g * (val - mean) * inv_std + b;
        y[i] = __float2bfloat16_rn(normed);
    }
}

// TVM-FFI entry: run(input, gamma, beta, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView gamma,
         tvm::ffi::TensorView beta, tvm::ffi::TensorView output)
{
    int num_tokens = input.size(0);
    int hidden_dim = input.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    int block = min(hidden_dim, 1024);
    block = 32 * ((block + 31) / 32);  // round up to warp multiple
    dim3 grid(num_tokens);

    layernorm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        num_tokens, hidden_dim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 2. Per-token FP8 quantization  (bf16 -> fp8_e4m3)
# ---------------------------------------------------------------------------

_PER_TOKEN_QUANT_SOURCE = (
    _TVM_FFI_INCLUDES
    + _REDUCE_HELPERS
    + r"""
// Adapted from TRT-LLM perTokenQuantization kernel (quantization.cuh).
// Simplified: bf16 input -> fp8_e4m3 output, per-row absmax scaling.
// One CTA per row.

__global__ void per_token_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales,
    int num_tokens,
    int hidden_dim)
{
    const float FP8_MAX = 448.0f;
    int row = blockIdx.x;
    if (row >= num_tokens) return;

    const __nv_bfloat16* x = input + (int64_t)row * hidden_dim;
    __nv_fp8_e4m3* y = output + (int64_t)row * hidden_dim;

    // Pass 1: find absmax
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float v = fabsf(__bfloat162float(x[i]));
        local_max = fmaxf(local_max, v);
    }
    float row_max = block_reduce_max(local_max);

    // Broadcast via shared memory
    __shared__ float s_row_max;
    if (threadIdx.x == 0) {
        s_row_max = fmaxf(row_max, 1e-12f);
        scales[row] = s_row_max / FP8_MAX;
    }
    __syncthreads();
    float amax = s_row_max;

    // Pass 2: quantize
    float scale_factor = FP8_MAX / amax;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float v = __bfloat162float(x[i]) * scale_factor;
        y[i] = __nv_fp8_e4m3(v);
    }
}

// TVM-FFI entry: run(input, output, scales)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
         tvm::ffi::TensorView scales)
{
    int num_tokens = input.size(0);
    int hidden_dim = input.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    int block = min(hidden_dim, 512);
    block = 32 * ((block + 31) / 32);
    dim3 grid(num_tokens);

    per_token_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        reinterpret_cast<float*>(scales.data_ptr()),
        num_tokens, hidden_dim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 3. MoE routing  (f32 -> f32, i32)
# ---------------------------------------------------------------------------

_MOE_ROUTING_SOURCE = (
    _TVM_FFI_INCLUDES
    + _REDUCE_HELPERS
    + r"""
// Adapted from TRT-LLM customMoeRoutingKernel (customMoeRoutingKernels.cu).
// Simplified: float32 input, softmax-before-topk, arbitrary top_k.
// One warp per token (since num_experts <= 128, fits in warp with multiple
// elements per lane).

__global__ void moe_routing_kernel(
    const float* __restrict__ routing_logits,
    float* __restrict__ expert_weights,
    int* __restrict__ expert_indices,
    int num_tokens,
    int num_experts,
    int top_k)
{
    // Each warp processes one token
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_global >= num_tokens) return;

    const float* logits = routing_logits + (int64_t)warp_global * num_experts;
    float* weights_out = expert_weights + (int64_t)warp_global * top_k;
    int* indices_out = expert_indices + (int64_t)warp_global * top_k;

    // Load logits into registers (max 128 experts, 4 per lane for 32 lanes)
    const int MAX_PER_LANE = 4;  // supports up to 128 experts
    float scores[MAX_PER_LANE];
    int ids[MAX_PER_LANE];
    int elems_per_lane = (num_experts + 31) / 32;

    for (int i = 0; i < MAX_PER_LANE; i++) {
        int idx = i * 32 + lane;
        if (idx < num_experts) {
            scores[i] = logits[idx];
            ids[i] = idx;
        } else {
            scores[i] = -INFINITY;
            ids[i] = -1;
        }
    }

    // Softmax: find max
    float local_max = -INFINITY;
    for (int i = 0; i < elems_per_lane; i++)
        local_max = fmaxf(local_max, scores[i]);
    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));

    // Softmax: exp and sum
    float local_sum = 0.0f;
    for (int i = 0; i < elems_per_lane; i++) {
        if (i * 32 + lane < num_experts) {
            scores[i] = expf(scores[i] - local_max);
            local_sum += scores[i];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

    // Softmax: normalize
    float inv_sum = 1.0f / local_sum;
    for (int i = 0; i < elems_per_lane; i++) {
        if (i * 32 + lane < num_experts)
            scores[i] *= inv_sum;
    }

    // Top-k selection: iterative max-finding (top_k is small: 2 or 8)
    // Use shared memory for the warp's results
    extern __shared__ char smem[];
    int warp_in_block = threadIdx.x / 32;
    float* warp_topk_vals = reinterpret_cast<float*>(smem) + warp_in_block * 16;
    int* warp_topk_ids = reinterpret_cast<int*>(smem) + (blockDim.x / 32) * 16 + warp_in_block * 16;

    for (int k = 0; k < top_k; k++) {
        // Find max in this lane
        float lane_max = -INFINITY;
        int lane_max_idx = -1;
        for (int i = 0; i < elems_per_lane; i++) {
            if (i * 32 + lane < num_experts && scores[i] > lane_max) {
                lane_max = scores[i];
                lane_max_idx = i * 32 + lane;
            }
        }

        // Warp reduce to find global max
        float warp_max = lane_max;
        int warp_max_lane = lane;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, warp_max, offset);
            int other_lane = __shfl_xor_sync(0xffffffff, warp_max_lane, offset);
            if (other_val > warp_max) {
                warp_max = other_val;
                warp_max_lane = other_lane;
            }
        }

        // The winning lane broadcasts its index
        int winner_idx = __shfl_sync(0xffffffff, lane_max_idx, warp_max_lane);

        if (lane == 0) {
            warp_topk_vals[k] = warp_max;
            warp_topk_ids[k] = winner_idx;
        }

        // Zero out the selected expert so it won't be picked again
        for (int i = 0; i < elems_per_lane; i++) {
            if (i * 32 + lane == winner_idx)
                scores[i] = -INFINITY;
        }
    }

    // Renormalize selected weights
    if (lane == 0) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) sum += warp_topk_vals[k];
        float inv = 1.0f / sum;
        for (int k = 0; k < top_k; k++) {
            weights_out[k] = warp_topk_vals[k] * inv;
            indices_out[k] = warp_topk_ids[k];
        }
    }
}

// TVM-FFI entry: run(routing_logits, expert_weights, expert_indices)
void run(tvm::ffi::TensorView routing_logits,
         tvm::ffi::TensorView expert_weights,
         tvm::ffi::TensorView expert_indices)
{
    int num_tokens = routing_logits.size(0);
    int num_experts = routing_logits.size(1);
    int top_k = expert_weights.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(routing_logits.device().device_type,
                           routing_logits.device().device_id));

    // One warp per token
    int warps_per_block = 8;
    int block = warps_per_block * 32;
    int grid = (num_tokens + warps_per_block - 1) / warps_per_block;
    // Shared memory: 16 floats + 16 ints per warp for topk scratch
    size_t smem = warps_per_block * 16 * (sizeof(float) + sizeof(int));

    moe_routing_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<const float*>(routing_logits.data_ptr()),
        reinterpret_cast<float*>(expert_weights.data_ptr()),
        reinterpret_cast<int*>(expert_indices.data_ptr()),
        num_tokens, num_experts, top_k);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 4. Per-channel scale  (bf16)
# ---------------------------------------------------------------------------

_PER_CHANNEL_SCALE_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM apply_per_channel_scale kernel (preQuantScaleKernel.cu).
// Simplified: bf16 input/output, f32 scale, straightforward multiply.

__global__ void per_channel_scale_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    int num_tokens,
    int hidden_dim)
{
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    int row = blockIdx.x;
    if (row >= num_tokens || col >= hidden_dim) return;

    int64_t idx = (int64_t)row * hidden_dim + col;
    float val = __bfloat162float(input[idx]) * scale[col];
    output[idx] = __float2bfloat16_rn(val);
}

// TVM-FFI entry: run(input, scale, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView scale,
         tvm::ffi::TensorView output)
{
    int num_tokens = input.size(0);
    int hidden_dim = input.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    int block_x = 256;
    dim3 block(block_x);
    dim3 grid(num_tokens, (hidden_dim + block_x - 1) / block_x);

    per_channel_scale_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const float*>(scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        num_tokens, hidden_dim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 5. Fused QK-Norm + RoPE  (bf16)
# ---------------------------------------------------------------------------

_FUSED_QKNORM_ROPE_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM fusedQKNormRopeKernel (fusedQKNormRopeKernel.cu).
// Simplified: bf16, separate Q/K tensors (not combined QKV), neox-style RoPE.
// One warp per (token, head) pair.

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
}

// Process one head: RMSNorm + RoPE, write to output
__device__ void process_head(
    const __nv_bfloat16* __restrict__ head_in,
    __nv_bfloat16* __restrict__ head_out,
    const __nv_bfloat16* __restrict__ norm_weight,
    int head_dim,
    float position,
    int lane)
{
    const float EPS = 1e-6f;
    const float THETA = 10000.0f;
    int half_dim = head_dim / 2;

    // Each thread handles head_dim/32 elements
    int elems_per_thread = (head_dim + 31) / 32;

    // Pass 1: Load and compute sum of squares for RMSNorm
    float sum_sq = 0.0f;
    for (int e = 0; e < elems_per_thread; e++) {
        int dim = lane * elems_per_thread + e;
        if (dim < head_dim) {
            float v = __bfloat162float(head_in[dim]);
            sum_sq += v * v;
        }
    }
    sum_sq = warp_sum(sum_sq);
    float rms_rcp = rsqrtf(sum_sq / (float)head_dim + EPS);

    // Pass 2: Normalize and apply RoPE
    for (int e = 0; e < elems_per_thread; e++) {
        int dim = lane * elems_per_thread + e;
        if (dim < head_dim) {
            // RMSNorm
            float val = __bfloat162float(head_in[dim]) * rms_rcp;
            float w = __bfloat162float(norm_weight[dim]);
            val *= w;

            // RoPE (neox-style: first half pairs with second half)
            float result;
            if (dim < half_dim) {
                // x1 position: result = x1 * cos - x2 * sin
                float freq = 1.0f / powf(THETA, (float)dim / (float)half_dim);
                float angle = position * freq;
                float cos_a, sin_a;
                __sincosf(angle, &sin_a, &cos_a);

                // Need x2 = head_in[dim + half_dim] normalized
                float x2_raw = __bfloat162float(head_in[dim + half_dim]);
                float x2 = x2_raw * rms_rcp * __bfloat162float(norm_weight[dim + half_dim]);
                result = val * cos_a - x2 * sin_a;
            } else {
                // x2 position: result = x1 * sin + x2 * cos
                int pair_dim = dim - half_dim;
                float freq = 1.0f / powf(THETA, (float)pair_dim / (float)half_dim);
                float angle = position * freq;
                float cos_a, sin_a;
                __sincosf(angle, &sin_a, &cos_a);

                // Need x1 = head_in[pair_dim] normalized
                float x1_raw = __bfloat162float(head_in[pair_dim]);
                float x1 = x1_raw * rms_rcp * __bfloat162float(norm_weight[pair_dim]);
                result = x1 * sin_a + val * cos_a;
            }

            head_out[dim] = __float2bfloat16_rn(result);
        }
    }
}

__global__ void fused_qknorm_rope_kernel(
    const __nv_bfloat16* __restrict__ query,     // [T, Qh, D]
    const __nv_bfloat16* __restrict__ key,       // [T, Kh, D]
    const __nv_bfloat16* __restrict__ q_weight,  // [D]
    const __nv_bfloat16* __restrict__ k_weight,  // [D]
    const int* __restrict__ positions,            // [T]
    __nv_bfloat16* __restrict__ query_out,        // [T, Qh, D]
    __nv_bfloat16* __restrict__ key_out,          // [T, Kh, D]
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    int global_warp = blockIdx.x * warps_per_block + warp_id;
    int total_qk_heads = num_q_heads + num_kv_heads;

    int token_idx = global_warp / total_qk_heads;
    int head_idx = global_warp % total_qk_heads;

    if (token_idx >= num_tokens) return;

    float pos = (float)positions[token_idx];

    if (head_idx < num_q_heads) {
        // Q head
        int64_t offset = ((int64_t)token_idx * num_q_heads + head_idx) * head_dim;
        process_head(query + offset, query_out + offset, q_weight, head_dim, pos, lane);
    } else {
        // K head
        int kh = head_idx - num_q_heads;
        int64_t offset = ((int64_t)token_idx * num_kv_heads + kh) * head_dim;
        process_head(key + offset, key_out + offset, k_weight, head_dim, pos, lane);
    }
}

// TVM-FFI entry: run(query, key, q_norm_weight, k_norm_weight, positions,
//                     query_out, key_out)
void run(tvm::ffi::TensorView query, tvm::ffi::TensorView key,
         tvm::ffi::TensorView q_norm_weight, tvm::ffi::TensorView k_norm_weight,
         tvm::ffi::TensorView positions,
         tvm::ffi::TensorView query_out, tvm::ffi::TensorView key_out)
{
    int num_tokens = query.size(0);
    int num_q_heads = query.size(1);
    int head_dim = query.size(2);
    int num_kv_heads = key.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(query.device().device_type, query.device().device_id));

    int total_qk_heads = num_q_heads + num_kv_heads;
    int total_warps = num_tokens * total_qk_heads;
    int warps_per_block = 8;
    int block_size = warps_per_block * 32;
    int grid_size = (total_warps + warps_per_block - 1) / warps_per_block;

    fused_qknorm_rope_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_weight.data_ptr()),
        reinterpret_cast<const int*>(positions.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(query_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(key_out.data_ptr()),
        num_tokens, num_q_heads, num_kv_heads, head_dim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)


# ---------------------------------------------------------------------------
# Solution builder functions
# ---------------------------------------------------------------------------


def make_layernorm_solution(defn: Definition) -> Solution:
    return build_solution(
        _LAYERNORM_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_per_token_quant_solution(defn: Definition) -> Solution:
    return build_solution(
        _PER_TOKEN_QUANT_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_moe_routing_solution(defn: Definition) -> Solution:
    return build_solution(
        _MOE_ROUTING_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_per_channel_scale_solution(defn: Definition) -> Solution:
    return build_solution(
        _PER_CHANNEL_SCALE_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_fused_qknorm_rope_solution(defn: Definition) -> Solution:
    return build_solution(
        _FUSED_QKNORM_ROPE_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


# Map definition name -> solution maker
SOLUTION_MAKERS: dict[str, callable] = {
    "trtllm_layernorm": make_layernorm_solution,
    "trtllm_per_token_quant_fp8": make_per_token_quant_solution,
    "trtllm_moe_routing_topk2": make_moe_routing_solution,
    "trtllm_moe_routing_topk8": make_moe_routing_solution,
    "trtllm_per_channel_scale": make_per_channel_scale_solution,
    "trtllm_fused_qknorm_rope": make_fused_qknorm_rope_solution,
}
