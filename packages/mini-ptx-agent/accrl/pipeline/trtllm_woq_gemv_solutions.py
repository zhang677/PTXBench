"""Baseline CUDA solutions for TRT-LLM Weight-Only GEMV kernels.

Each solution implements a simplified weight-only quantized GEMV kernel
with TVM-FFI bindings (destination-passing style). One block per (m_row, n_col)
pair, threads cooperatively reduce over K with warp shuffle + shared mem.

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
#include <cuda_runtime.h>
#include <math.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
"""

# ---------------------------------------------------------------------------
# 1. W8A16 Per-channel GEMV
# ---------------------------------------------------------------------------

_W8A16_PERCHANNEL_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Weight-only INT8 GEMV with per-channel scale.
// Grid: (M, N), Block: 256. Each block computes one output element.
// Threads stride over K, warp shuffle + shared mem reduction.

__global__ void w8a16_perchannel_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [M, K]
    const int8_t* __restrict__ weight,             // [N, K]
    const __nv_bfloat16* __restrict__ scales,      // [N]
    __nv_bfloat16* __restrict__ output,            // [M, N]
    int M, int N, int K)
{
    int row = blockIdx.x;  // m index
    int col = blockIdx.y;  // n index
    if (row >= M || col >= N) return;

    float scale = __bfloat162float(scales[col]);

    const __nv_bfloat16* act_row = activation + (int64_t)row * K;
    const int8_t* wt_row = weight + (int64_t)col * K;

    // Thread-local accumulation over K
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float a = __bfloat162float(act_row[k]);
        float w = (float)wt_row[k] * scale;
        acc += a * w;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    // Cross-warp reduction via shared memory
    __shared__ float s_partial[8];  // up to 8 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        s_partial[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            output[(int64_t)row * N + col] = __float2bfloat16_rn(val);
    }
}

// TVM-FFI entry: run(activation, weight, scales, output)
void run(tvm::ffi::TensorView activation, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView scales, tvm::ffi::TensorView output)
{
    int M = activation.size(0);
    int K = activation.size(1);
    int N = weight.size(0);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(activation.device().device_type,
                           activation.device().device_id));

    dim3 grid(M, N);
    int block = 256;
    w8a16_perchannel_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
        reinterpret_cast<const int8_t*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N, K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 2. W8A16 Groupwise GEMV
# ---------------------------------------------------------------------------

_W8A16_GROUPWISE_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Weight-only INT8 GEMV with groupwise dequantization (group_size=128).
// Grid: (M, N), Block: 256. Threads stride over K.

__global__ void w8a16_groupwise_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [M, K]
    const int8_t* __restrict__ weight,             // [N, K]
    const __nv_bfloat16* __restrict__ scales,      // [num_groups, N]
    const __nv_bfloat16* __restrict__ zeros,       // [num_groups, N]
    __nv_bfloat16* __restrict__ output,            // [M, N]
    int M, int N, int K)
{
    const int GROUP_SIZE = 128;
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row >= M || col >= N) return;

    const __nv_bfloat16* act_row = activation + (int64_t)row * K;
    const int8_t* wt_row = weight + (int64_t)col * K;

    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int g = k / GROUP_SIZE;
        float scale = __bfloat162float(scales[(int64_t)g * N + col]);
        float zero = __bfloat162float(zeros[(int64_t)g * N + col]);
        float a = __bfloat162float(act_row[k]);
        float w = (float)wt_row[k] * scale + zero;
        acc += a * w;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ float s_partial[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        s_partial[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            output[(int64_t)row * N + col] = __float2bfloat16_rn(val);
    }
}

// TVM-FFI entry: run(activation, weight, scales, zeros, output)
void run(tvm::ffi::TensorView activation, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView scales, tvm::ffi::TensorView zeros,
         tvm::ffi::TensorView output)
{
    int M = activation.size(0);
    int K = activation.size(1);
    int N = weight.size(0);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(activation.device().device_type,
                           activation.device().device_id));

    dim3 grid(M, N);
    int block = 256;
    w8a16_groupwise_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
        reinterpret_cast<const int8_t*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(zeros.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N, K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 3. W4A16 Per-channel GEMV
# ---------------------------------------------------------------------------

_W4A16_PERCHANNEL_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Weight-only INT4 packed GEMV with per-channel scale.
// Each int8 byte = 2 INT4 values: lo = (byte & 0xF) - 8, hi = ((byte >> 4) & 0xF) - 8
// Grid: (M, N), Block: 256. Threads stride over K_packed, 2 FMAs per byte.

__global__ void w4a16_perchannel_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [M, K]
    const int8_t* __restrict__ weight,             // [N, K_packed]
    const __nv_bfloat16* __restrict__ scales,      // [N]
    __nv_bfloat16* __restrict__ output,            // [M, N]
    int M, int N, int K, int K_packed)
{
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row >= M || col >= N) return;

    float scale = __bfloat162float(scales[col]);
    const __nv_bfloat16* act_row = activation + (int64_t)row * K;
    const int8_t* wt_row = weight + (int64_t)col * K_packed;

    float acc = 0.0f;
    for (int kp = threadIdx.x; kp < K_packed; kp += blockDim.x) {
        int byte_val = (int)(unsigned char)wt_row[kp];
        float w_lo = (float)((byte_val & 0x0F) - 8) * scale;
        float w_hi = (float)(((byte_val >> 4) & 0x0F) - 8) * scale;

        int k_base = kp * 2;
        float a_lo = __bfloat162float(act_row[k_base]);
        float a_hi = __bfloat162float(act_row[k_base + 1]);
        acc += a_lo * w_lo + a_hi * w_hi;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ float s_partial[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        s_partial[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            output[(int64_t)row * N + col] = __float2bfloat16_rn(val);
    }
}

// TVM-FFI entry: run(activation, weight, scales, output)
void run(tvm::ffi::TensorView activation, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView scales, tvm::ffi::TensorView output)
{
    int M = activation.size(0);
    int K = activation.size(1);
    int N = weight.size(0);
    int K_packed = weight.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(activation.device().device_type,
                           activation.device().device_id));

    dim3 grid(M, N);
    int block = 256;
    w4a16_perchannel_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
        reinterpret_cast<const int8_t*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N, K, K_packed);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 4. W4A16 Groupwise GEMV
# ---------------------------------------------------------------------------

_W4A16_GROUPWISE_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Weight-only INT4 packed GEMV with groupwise dequantization (group_size=128).
// Each int8 byte = 2 INT4 values. Per-group scale and zero point.
// Grid: (M, N), Block: 256.

__global__ void w4a16_groupwise_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [M, K]
    const int8_t* __restrict__ weight,             // [N, K_packed]
    const __nv_bfloat16* __restrict__ scales,      // [num_groups, N]
    const __nv_bfloat16* __restrict__ zeros,       // [num_groups, N]
    __nv_bfloat16* __restrict__ output,            // [M, N]
    int M, int N, int K, int K_packed)
{
    const int GROUP_SIZE = 128;
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row >= M || col >= N) return;

    const __nv_bfloat16* act_row = activation + (int64_t)row * K;
    const int8_t* wt_row = weight + (int64_t)col * K_packed;

    float acc = 0.0f;
    for (int kp = threadIdx.x; kp < K_packed; kp += blockDim.x) {
        int k_base = kp * 2;  // logical K index for lo nibble

        int byte_val = (int)(unsigned char)wt_row[kp];
        float w_lo_int = (float)((byte_val & 0x0F) - 8);
        float w_hi_int = (float)(((byte_val >> 4) & 0x0F) - 8);

        // Group index for lo and hi elements
        int g_lo = k_base / GROUP_SIZE;
        int g_hi = (k_base + 1) / GROUP_SIZE;

        float s_lo = __bfloat162float(scales[(int64_t)g_lo * N + col]);
        float z_lo = __bfloat162float(zeros[(int64_t)g_lo * N + col]);
        float s_hi = __bfloat162float(scales[(int64_t)g_hi * N + col]);
        float z_hi = __bfloat162float(zeros[(int64_t)g_hi * N + col]);

        float w_lo = w_lo_int * s_lo + z_lo;
        float w_hi = w_hi_int * s_hi + z_hi;

        float a_lo = __bfloat162float(act_row[k_base]);
        float a_hi = __bfloat162float(act_row[k_base + 1]);
        acc += a_lo * w_lo + a_hi * w_hi;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ float s_partial[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        s_partial[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane_id == 0)
            output[(int64_t)row * N + col] = __float2bfloat16_rn(val);
    }
}

// TVM-FFI entry: run(activation, weight, scales, zeros, output)
void run(tvm::ffi::TensorView activation, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView scales, tvm::ffi::TensorView zeros,
         tvm::ffi::TensorView output)
{
    int M = activation.size(0);
    int K = activation.size(1);
    int N = weight.size(0);
    int K_packed = weight.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(activation.device().device_type,
                           activation.device().device_id));

    dim3 grid(M, N);
    int block = 256;
    w4a16_groupwise_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
        reinterpret_cast<const int8_t*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(zeros.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N, K, K_packed);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)


# ---------------------------------------------------------------------------
# Solution builder functions
# ---------------------------------------------------------------------------


def make_w8a16_perchannel_solution(defn: Definition) -> Solution:
    return build_solution(
        _W8A16_PERCHANNEL_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_w8a16_groupwise_solution(defn: Definition) -> Solution:
    return build_solution(
        _W8A16_GROUPWISE_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_w4a16_perchannel_solution(defn: Definition) -> Solution:
    return build_solution(
        _W4A16_PERCHANNEL_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_w4a16_groupwise_solution(defn: Definition) -> Solution:
    return build_solution(
        _W4A16_GROUPWISE_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


# Map definition name -> solution maker
SOLUTION_MAKERS: dict[str, callable] = {
    "trtllm_woq_gemv_w8a16_perchannel": make_w8a16_perchannel_solution,
    "trtllm_woq_gemv_w8a16_groupwise": make_w8a16_groupwise_solution,
    "trtllm_woq_gemv_w4a16_perchannel": make_w4a16_perchannel_solution,
    "trtllm_woq_gemv_w4a16_groupwise": make_w4a16_groupwise_solution,
}
