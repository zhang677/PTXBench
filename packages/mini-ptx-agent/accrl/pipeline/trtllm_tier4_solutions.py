"""Baseline CUDA solutions for TRT-LLM Tier 4 kernels.

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
# 1. Selective Scan (Mamba SSM)
# ---------------------------------------------------------------------------

_SELECTIVE_SCAN_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM selectiveScan kernel.
// Mamba S6 selective scan with softplus delta discretization and D skip.
// One block per (batch, dim_chunk). Sequential over time steps.

__device__ float softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__global__ void selective_scan_kernel(
    const float* __restrict__ x,          // [B, L, D]
    const float* __restrict__ delta,      // [B, L, D]
    const float* __restrict__ A,          // [D, N]
    const float* __restrict__ B_mat,      // [B, L, N]
    const float* __restrict__ C_mat,      // [B, L, N]
    const float* __restrict__ D_param,    // [D]
    float* __restrict__ output,           // [B, L, D]
    int B_sz, int L, int D, int N)
{
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B_sz || d >= D) return;

    // Local state vector for this (b, d)
    float h[16];  // N=16 (state_dim is const 16)
    for (int n = 0; n < N; n++) h[n] = 0.0f;

    float d_skip = D_param[d];

    for (int t = 0; t < L; t++) {
        int64_t idx_bld = (int64_t)b * L * D + (int64_t)t * D + d;
        int64_t idx_bln = (int64_t)b * L * N + (int64_t)t * N;

        float x_t = x[idx_bld];
        float dt = softplus(delta[idx_bld]);

        float y_t = 0.0f;
        for (int n = 0; n < N; n++) {
            float a_val = A[(int64_t)d * N + n];
            float dA = expf(dt * a_val);
            float dB = dt * B_mat[idx_bln + n];
            h[n] = dA * h[n] + dB * x_t;
            y_t += h[n] * C_mat[idx_bln + n];
        }
        y_t += d_skip * x_t;

        output[idx_bld] = y_t;
    }
}

// TVM-FFI entry: run(x, delta, A, B, C, D_param, output)
void run(tvm::ffi::TensorView x, tvm::ffi::TensorView delta,
         tvm::ffi::TensorView A, tvm::ffi::TensorView B_mat,
         tvm::ffi::TensorView C_mat, tvm::ffi::TensorView D_param,
         tvm::ffi::TensorView output)
{
    int B_sz = x.size(0);
    int L = x.size(1);
    int D = x.size(2);
    int N = A.size(1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

    int block = 128;
    dim3 grid(B_sz, (D + block - 1) / block);

    selective_scan_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(x.data_ptr()),
        reinterpret_cast<const float*>(delta.data_ptr()),
        reinterpret_cast<const float*>(A.data_ptr()),
        reinterpret_cast<const float*>(B_mat.data_ptr()),
        reinterpret_cast<const float*>(C_mat.data_ptr()),
        reinterpret_cast<const float*>(D_param.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        B_sz, L, D, N);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 2. LRU Recurrence (RecurrentGemma)
# ---------------------------------------------------------------------------

_LRU_RECURRENCE_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM lruKernel.
// Linear Recurrent Unit with sigmoid gates, softplus decay, GELU output gating.
// One block per (batch, dim_chunk). Sequential over time steps.

__device__ float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float device_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ float device_gelu(float x) {
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float k = 0.7978845608f;  // sqrt(2/pi)
    float cdf = 0.5f * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__global__ void lru_recurrence_kernel(
    const float* __restrict__ x,          // [B, L, D]
    const float* __restrict__ param_a,    // [D]
    const float* __restrict__ gate_x,     // [B, L, D]
    const float* __restrict__ gate_a,     // [B, L, D]
    const float* __restrict__ y_param,    // [B, L, D]
    float* __restrict__ output,           // [B, L, D]
    int B_sz, int L, int D)
{
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B_sz || d >= D) return;

    float a = -device_softplus(param_a[d]);  // negative for stability
    float h = 0.0f;

    for (int t = 0; t < L; t++) {
        int64_t idx = (int64_t)b * L * D + (int64_t)t * D + d;

        float gx = device_sigmoid(gate_x[idx]);
        float ga = device_sigmoid(gate_a[idx]);
        float decay = expf(a * ga);

        h = decay * h + gx * x[idx];

        float out = h * device_gelu(y_param[idx]);
        output[idx] = out;
    }
}

// TVM-FFI entry: run(x, param_a, gate_x, gate_a, y_param, output)
void run(tvm::ffi::TensorView x, tvm::ffi::TensorView param_a,
         tvm::ffi::TensorView gate_x, tvm::ffi::TensorView gate_a,
         tvm::ffi::TensorView y_param, tvm::ffi::TensorView output)
{
    int B_sz = x.size(0);
    int L = x.size(1);
    int D = x.size(2);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

    int block = 128;
    dim3 grid(B_sz, (D + block - 1) / block);

    lru_recurrence_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(x.data_ptr()),
        reinterpret_cast<const float*>(param_a.data_ptr()),
        reinterpret_cast<const float*>(gate_x.data_ptr()),
        reinterpret_cast<const float*>(gate_a.data_ptr()),
        reinterpret_cast<const float*>(y_param.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        B_sz, L, D);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 3. Sage Attention Quant (4D block FP8)
# ---------------------------------------------------------------------------

_SAGE_ATTENTION_QUANT_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM sageAttentionKernels.
// Block-wise FP8 quantization for 4D attention tensors [B,H,S,D].
// Blocks along seq dim (block_size=64). Per-block absmax scaling.
// Grid: (B*H, S/BLOCK). Block: min(BLOCK*D/4, 256) threads.

__global__ void sage_attention_quant_kernel(
    const __nv_bfloat16* __restrict__ input,  // [B, H, S, D]
    __nv_fp8_storage_t* __restrict__ output,  // [B, H, S, D]
    int B, int H, int S, int D)
{
    const int BLOCK = 64;
    const float FP8_MAX = 448.0f;

    int bh = blockIdx.x;
    int blk_idx = blockIdx.y;
    if (bh >= B * H) return;

    int64_t head_offset = (int64_t)bh * S * D;
    int seq_start = blk_idx * BLOCK;
    int64_t block_elems = (int64_t)BLOCK * D;

    // Find absmax over the block
    __shared__ float s_partial[8];  // up to 8 warps

    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < block_elems; i += blockDim.x) {
        int s_off = (int)(i / D);
        int d_off = (int)(i % D);
        int64_t global_idx = head_offset + (int64_t)(seq_start + s_off) * D + d_off;
        float v = fabsf(__bfloat162float(input[global_idx]));
        if (v > local_max) local_max = v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0)
        s_partial[warp_id] = local_max;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane_id == 0)
            s_partial[0] = val;
    }
    __syncthreads();

    float absmax = fmaxf(s_partial[0], 1e-12f);
    float scale = absmax / FP8_MAX;

    // Quantize
    for (int64_t i = threadIdx.x; i < block_elems; i += blockDim.x) {
        int s_off = (int)(i / D);
        int d_off = (int)(i % D);
        int64_t global_idx = head_offset + (int64_t)(seq_start + s_off) * D + d_off;
        float v = __bfloat162float(input[global_idx]);
        float scaled = v / scale;
        output[global_idx] = __nv_cvt_float_to_fp8(
            scaled, __NV_SATFINITE, __NV_E4M3);
    }
}

// TVM-FFI entry: run(input, output)
void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output)
{
    int B = input.size(0);
    int H = input.size(1);
    int S = input.size(2);
    int D = input.size(3);
    const int BLOCK = 64;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

    dim3 grid(B * H, S / BLOCK);
    int block = min((int)(BLOCK * D / 4), 256);
    block = max(block, 32);

    sage_attention_quant_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_storage_t*>(output.data_ptr()),
        B, H, S, D);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 4. Ring Attention Recovery (online softmax merging)
# ---------------------------------------------------------------------------

_RING_ATTENTION_RECOVERY_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM recoverFromRingAtten kernel.
// Online softmax merging of two partial attention outputs.
// Grid: (B*H, S). Each thread handles one (b,h,s) position across D.

__global__ void ring_attention_recovery_kernel(
    const float* __restrict__ accu_out,   // [B, H, S, D]
    const float* __restrict__ new_out,    // [B, H, S, D]
    const float* __restrict__ accu_max,   // [B, H, S]
    const float* __restrict__ accu_sum,   // [B, H, S]
    const float* __restrict__ new_max,    // [B, H, S]
    const float* __restrict__ new_sum,    // [B, H, S]
    float* __restrict__ output,           // [B, H, S, D]
    float* __restrict__ out_max,          // [B, H, S]
    float* __restrict__ out_sum,          // [B, H, S]
    int B, int H, int S, int D)
{
    int bh = blockIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (bh >= B * H || s >= S) return;

    int64_t stat_idx = (int64_t)bh * S + s;
    float am = accu_max[stat_idx];
    float as_ = accu_sum[stat_idx];
    float nm = new_max[stat_idx];
    float ns = new_sum[stat_idx];

    // Global max
    float m = fmaxf(am, nm);
    // Rescale factors
    float exp_accu = expf(am - m);
    float exp_new = expf(nm - m);
    // Global sum
    float total_sum = exp_accu * as_ + exp_new * ns;
    float denom = fmaxf(total_sum, 1e-12f);
    float inv_sum = 1.0f / denom;

    // Weights for each partial output
    float w_accu = exp_accu * as_ * inv_sum;
    float w_new = exp_new * ns * inv_sum;

    int64_t out_offset = (int64_t)bh * S * D + (int64_t)s * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float a = accu_out[out_offset + d];
        float n = new_out[out_offset + d];
        output[out_offset + d] = w_accu * a + w_new * n;
    }

    // Write stats (only one thread per s position)
    if (threadIdx.x == 0) {
        out_max[stat_idx] = m;
        out_sum[stat_idx] = total_sum;
    }
}

// TVM-FFI entry: run(accu_out, new_out, accu_max, accu_sum, new_max, new_sum,
//                     output, out_max, out_sum)
void run(tvm::ffi::TensorView accu_out, tvm::ffi::TensorView new_out,
         tvm::ffi::TensorView accu_max, tvm::ffi::TensorView accu_sum,
         tvm::ffi::TensorView new_max, tvm::ffi::TensorView new_sum,
         tvm::ffi::TensorView output, tvm::ffi::TensorView out_max,
         tvm::ffi::TensorView out_sum)
{
    int B = accu_out.size(0);
    int H = accu_out.size(1);
    int S = accu_out.size(2);
    int D = accu_out.size(3);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(accu_out.device().device_type,
                           accu_out.device().device_id));

    // blockDim.x handles D, blockDim.y handles S positions per block
    int tx = min(D, 128);
    int ty = 1;
    dim3 block(tx, ty);
    dim3 grid(B * H, (S + ty - 1) / ty);

    ring_attention_recovery_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(accu_out.data_ptr()),
        reinterpret_cast<const float*>(new_out.data_ptr()),
        reinterpret_cast<const float*>(accu_max.data_ptr()),
        reinterpret_cast<const float*>(accu_sum.data_ptr()),
        reinterpret_cast<const float*>(new_max.data_ptr()),
        reinterpret_cast<const float*>(new_sum.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        reinterpret_cast<float*>(out_max.data_ptr()),
        reinterpret_cast<float*>(out_sum.data_ptr()),
        B, H, S, D);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)

# ---------------------------------------------------------------------------
# 5. Beam Score Update (broadcast add)
# ---------------------------------------------------------------------------

_BEAM_SCORE_UPDATE_SOURCE = (
    _TVM_FFI_INCLUDES
    + r"""
// Adapted from TRT-LLM beamSearchKernels.
// Broadcast add: output[b,bw,v] = log_probs[b,bw,v] + cum_log_probs[b,bw].
// Grid: (B, BW). Block: 256.

__global__ void beam_score_update_kernel(
    const float* __restrict__ log_probs,      // [B, BW, V]
    const float* __restrict__ cum_log_probs,   // [B, BW]
    float* __restrict__ output,               // [B, BW, V]
    int B, int BW, int V)
{
    int b = blockIdx.x;
    int bw = blockIdx.y;
    if (b >= B || bw >= BW) return;

    float cum = cum_log_probs[(int64_t)b * BW + bw];
    int64_t row_offset = ((int64_t)b * BW + bw) * V;

    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        output[row_offset + v] = log_probs[row_offset + v] + cum;
    }
}

// TVM-FFI entry: run(log_probs, cum_log_probs, output)
void run(tvm::ffi::TensorView log_probs, tvm::ffi::TensorView cum_log_probs,
         tvm::ffi::TensorView output)
{
    int B = log_probs.size(0);
    int BW = log_probs.size(1);
    int V = log_probs.size(2);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(log_probs.device().device_type,
                           log_probs.device().device_id));

    dim3 grid(B, BW);
    int block = 256;

    beam_score_update_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(log_probs.data_ptr()),
        reinterpret_cast<const float*>(cum_log_probs.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        B, BW, V);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
)


# ---------------------------------------------------------------------------
# Solution builder functions
# ---------------------------------------------------------------------------


def make_selective_scan_solution(defn: Definition) -> Solution:
    return build_solution(
        _SELECTIVE_SCAN_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_lru_recurrence_solution(defn: Definition) -> Solution:
    return build_solution(
        _LRU_RECURRENCE_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_sage_attention_quant_solution(defn: Definition) -> Solution:
    return build_solution(
        _SAGE_ATTENTION_QUANT_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_ring_attention_recovery_solution(defn: Definition) -> Solution:
    return build_solution(
        _RING_ATTENTION_RECOVERY_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


def make_beam_score_update_solution(defn: Definition) -> Solution:
    return build_solution(
        _BEAM_SCORE_UPDATE_SOURCE,
        defn.name,
        language="cuda",
        binding="tvm-ffi",
    )


# Map definition name -> solution maker
SOLUTION_MAKERS: dict[str, callable] = {
    "trtllm_selective_scan": make_selective_scan_solution,
    "trtllm_lru_recurrence": make_lru_recurrence_solution,
    "trtllm_sage_attention_quant": make_sage_attention_quant_solution,
    "trtllm_ring_attention_recovery": make_ring_attention_recovery_solution,
    "trtllm_beam_score_update": make_beam_score_update_solution,
}
