"""Baseline CUDA solutions for TRT-LLM Tier 2 kernels.

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
# 1. Causal Conv1D + SiLU  (bf16)
# ---------------------------------------------------------------------------

_CAUSAL_CONV1D_SILU_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM causalConv1d kernel (causalConv1d.cu, Dao-AILab).
// Depthwise causal 1D convolution with SiLU activation.
// One block per (batch, channel) pair; threads stride over sequence positions.

__global__ void causal_conv1d_silu_kernel(
    const __nv_bfloat16* __restrict__ input,   // [B, L, D]
    const __nv_bfloat16* __restrict__ weight,  // [K, D]
    const __nv_bfloat16* __restrict__ bias,    // [D]
    __nv_bfloat16* __restrict__ output,         // [B, L, D]
    int B, int L, int D, int K)
{
    int b = blockIdx.x;
    int d = blockIdx.y;
    if (b >= B || d >= D) return;

    // Load convolution weights for this channel into registers
    float w[4];  // K is const 4
    for (int i = 0; i < K && i < 4; i++)
        w[i] = __bfloat162float(weight[i * D + d]);
    float bias_val = __bfloat162float(bias[d]);

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float acc = bias_val;
        // Causal conv: left-padded by K-1, so src_pos = l - (K-1) + k
        for (int k = 0; k < K; k++) {
            int src_pos = l - (K - 1) + k;
            if (src_pos >= 0) {
                acc += w[k] * __bfloat162float(
                    input[((int64_t)b * L + src_pos) * D + d]);
            }
        }
        // SiLU activation: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + expf(-acc));
        output[((int64_t)b * L + l) * D + d] =
            __float2bfloat16_rn(acc * sigmoid_val);
    }
}

// TVM-FFI entry: run(input, weight, bias, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView bias, tvm::ffi::TensorView output)
{
    int B = input.size(0);
    int L = input.size(1);
    int D = input.size(2);
    int K = weight.size(0);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    dim3 grid(B, D);
    int block = min(L, 256);
    block = 32 * ((block + 31) / 32);  // round up to warp multiple

    causal_conv1d_silu_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        B, L, D, K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 2 & 3. TopK Last Dim  (f32 -> f32, i32)
#   Shared kernel for k=32 and k=128 variants.
# ---------------------------------------------------------------------------

_TOPK_LAST_DIM_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM topkLastDim.cu (AIR TopK radix selection).
// Baseline: iterative max-finding with workspace invalidation.
// One block per batch row.

__global__ void topk_kernel(
    float* __restrict__ workspace,   // [B, N] — mutable copy of input
    float* __restrict__ values,      // [B, k]
    int* __restrict__ indices,       // [B, k]
    int N, int k)
{
    int b = blockIdx.x;
    float* work = workspace + (int64_t)b * N;
    float* vals_out = values + (int64_t)b * k;
    int* idxs_out = indices + (int64_t)b * k;

    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];
    __shared__ int s_kill;

    for (int sel = 0; sel < k; sel++) {
        // Each thread finds max in its strided chunk
        float local_max = -INFINITY;
        int local_idx = -1;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float v = work[i];
            if (v > local_max || (v == local_max && i < local_idx)) {
                local_max = v;
                local_idx = i;
            }
        }

        s_vals[threadIdx.x] = local_max;
        s_idxs[threadIdx.x] = local_idx;
        __syncthreads();

        // Tree reduction: find global max (ties broken by smallest index)
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                float other_v = s_vals[threadIdx.x + stride];
                int other_i = s_idxs[threadIdx.x + stride];
                float cur_v = s_vals[threadIdx.x];
                int cur_i = s_idxs[threadIdx.x];
                if (other_v > cur_v ||
                    (other_v == cur_v && other_i < cur_i)) {
                    s_vals[threadIdx.x] = other_v;
                    s_idxs[threadIdx.x] = other_i;
                }
            }
            __syncthreads();
        }

        // Thread 0 writes result and broadcasts kill index
        if (threadIdx.x == 0) {
            vals_out[sel] = s_vals[0];
            idxs_out[sel] = s_idxs[0];
            s_kill = s_idxs[0];
        }
        __syncthreads();

        // Invalidate selected element so it won't be picked again
        int kill_idx = s_kill;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            if (i == kill_idx) work[i] = -INFINITY;
        }
        __syncthreads();
    }
}

// TVM-FFI entry: run(input, values, indices)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView values,
         tvm::ffi::TensorView indices)
{
    int B = input.size(0);
    int N = input.size(1);
    int k = values.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    // Allocate workspace: mutable copy of input
    size_t ws_bytes = (int64_t)B * N * sizeof(float);
    float* workspace;
    cudaMalloc(&workspace, ws_bytes);
    cudaMemcpyAsync(workspace,
                    reinterpret_cast<const float*>(input.data_ptr()),
                    ws_bytes, cudaMemcpyDeviceToDevice, stream);

    dim3 grid(B);
    int block = 256;

    topk_kernel<<<grid, block, 0, stream>>>(
        workspace,
        reinterpret_cast<float*>(values.data_ptr()),
        reinterpret_cast<int*>(indices.data_ptr()),
        N, k);

    cudaStreamSynchronize(stream);
    cudaFree(workspace);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 4. Cumulative Sum  (f32)
# ---------------------------------------------------------------------------

_CUMSUM_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
#include <cub/cub.cuh>

// Adapted from TRT-LLM cumsumLastDim.cu.
// CUB BlockLoad + BlockScan (InclusiveSum) + BlockStore, tiled.
// One block per batch row.

template <int TPB, int ITP>
__global__ void cumsum_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int N)
{
    typedef cub::BlockLoad<float, TPB, ITP,
                           cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    typedef cub::BlockScan<float, TPB,
                           cub::BLOCK_SCAN_WARP_SCANS> BlockScanT;
    typedef cub::BlockStore<float, TPB, ITP,
                            cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockScanT::TempStorage scan;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    int row = blockIdx.x;
    const float* in_row = input + (int64_t)row * N;
    float* out_row = output + (int64_t)row * N;

    float aggregate = 0.0f;
    const int tile_size = TPB * ITP;

    for (int offset = 0; offset < N; offset += tile_size) {
        float data[ITP];
        int valid = min(tile_size, N - offset);

        BlockLoadT(temp_storage.load).Load(in_row + offset, data, valid, 0.0f);
        __syncthreads();

        // Add carry-in from previous tiles
        if (threadIdx.x == 0)
            data[0] += aggregate;

        float block_agg;
        BlockScanT(temp_storage.scan).InclusiveSum(data, data, block_agg);
        __syncthreads();

        aggregate = block_agg;

        BlockStoreT(temp_storage.store).Store(out_row + offset, data, valid);
        __syncthreads();
    }
}

// TVM-FFI entry: run(input, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output)
{
    int B = input.size(0);
    int N = input.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    dim3 grid(B);

    // Dispatch config matching TRT-LLM
    if (N <= 64) {
        cumsum_kernel<32, 1><<<grid, 32, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(output.data_ptr()), N);
    } else if (N < 512) {
        cumsum_kernel<64, 2><<<grid, 64, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(output.data_ptr()), N);
    } else {
        cumsum_kernel<256, 8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<float*>(output.data_ptr()), N);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 5. Embedding Lookup  (i32 -> bf16)
# ---------------------------------------------------------------------------

_EMBEDDING_LOOKUP_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM lookup_kernel (lookupKernels.cu).
// Embedding gather with per-token scaling.
// Grid-stride loop over total elements (num_tokens * embed_dim).

__global__ void embedding_lookup_kernel(
    const int* __restrict__ indices,             // [T]
    const __nv_bfloat16* __restrict__ weight,    // [V, E]
    const float* __restrict__ per_token_scale,   // [V]
    __nv_bfloat16* __restrict__ output,           // [T, E]
    int num_tokens, int vocab_size, int embed_dim)
{
    int64_t total = (int64_t)num_tokens * embed_dim;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)blockDim.x * gridDim.x)
    {
        int token = (int)(idx / embed_dim);
        int col = (int)(idx % embed_dim);

        // abs(index) % vocab_size — matches PyTorch reference
        int raw_idx = indices[token];
        int abs_val = (raw_idx < 0) ? -raw_idx : raw_idx;
        int word_index = ((abs_val % vocab_size) + vocab_size) % vocab_size;

        float val = __bfloat162float(
            weight[(int64_t)word_index * embed_dim + col]);
        val *= per_token_scale[word_index];
        output[idx] = __float2bfloat16_rn(val);
    }
}

// TVM-FFI entry: run(indices, weight, per_token_scale, output)
void run(tvm::ffi::TensorView indices, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView per_token_scale, tvm::ffi::TensorView output)
{
    int num_tokens = indices.size(0);
    int vocab_size = weight.size(0);
    int embed_dim = weight.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(indices.device().device_type,
                           indices.device().device_id));

    int block = min(embed_dim, 512);
    int grid = min(num_tokens, 65536);

    embedding_lookup_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const int*>(indices.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const float*>(per_token_scale.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        num_tokens, vocab_size, embed_dim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 6. Penalty Application  (f32)
# ---------------------------------------------------------------------------

_PENALTY_APPLICATION_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM batchApplyPenalty (penaltyKernels.cu).
// Temperature + repetition + presence + frequency penalties on logits.
// One block per batch row, grid-stride over vocab.

__global__ void penalty_kernel(
    const float* __restrict__ logits,        // [B, V]
    const int* __restrict__ token_counts,    // [B, V]
    float* __restrict__ output,              // [B, V]
    int vocab_size)
{
    const float INV_TEMPERATURE = 1.0f;  // 1/T where T = 1.0
    const float REPETITION = 1.2f;
    const float PRESENCE = 0.6f;
    const float FREQUENCY = 0.5f;

    int b = blockIdx.x;
    const float* in_row = logits + (int64_t)b * vocab_size;
    const int* counts_row = token_counts + (int64_t)b * vocab_size;
    float* out_row = output + (int64_t)b * vocab_size;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float logit = in_row[i] * INV_TEMPERATURE;
        int count_raw = counts_row[i];
        int count = (count_raw > 0) ? count_raw : 0;

        if (count > 0) {
            // Repetition penalty (multiplicative, sign-dependent)
            if (logit < 0.0f)
                logit *= REPETITION;
            else
                logit /= REPETITION;
            // Presence penalty (subtractive, binary)
            logit -= PRESENCE;
            // Frequency penalty (subtractive, proportional to count)
            logit -= FREQUENCY * (float)count;
        }

        out_row[i] = logit;
    }
}

// TVM-FFI entry: run(logits, token_counts, output)
void run(tvm::ffi::TensorView logits, tvm::ffi::TensorView token_counts,
         tvm::ffi::TensorView output)
{
    int B = logits.size(0);
    int V = logits.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(logits.device().device_type,
                           logits.device().device_id));

    dim3 grid(B);
    int block = 512;

    penalty_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(logits.data_ptr()),
        reinterpret_cast<const int*>(token_counts.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        V);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)


# ---------------------------------------------------------------------------
# Solution builder functions
# ---------------------------------------------------------------------------


def make_causal_conv1d_silu_solution(defn: Definition) -> Solution:
    return build_solution(
        _CAUSAL_CONV1D_SILU_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_topk_last_dim_solution(defn: Definition) -> Solution:
    return build_solution(
        _TOPK_LAST_DIM_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_cumsum_solution(defn: Definition) -> Solution:
    return build_solution(
        _CUMSUM_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_embedding_lookup_solution(defn: Definition) -> Solution:
    return build_solution(
        _EMBEDDING_LOOKUP_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_penalty_application_solution(defn: Definition) -> Solution:
    return build_solution(
        _PENALTY_APPLICATION_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


# Map definition name -> solution maker
SOLUTION_MAKERS: dict[str, callable] = {
    "trtllm_causal_conv1d_silu": make_causal_conv1d_silu_solution,
    "trtllm_topk_last_dim_k32": make_topk_last_dim_solution,
    "trtllm_topk_last_dim_k128": make_topk_last_dim_solution,
    "trtllm_cumsum": make_cumsum_solution,
    "trtllm_embedding_lookup": make_embedding_lookup_solution,
    "trtllm_penalty_application": make_penalty_application_solution,
}
