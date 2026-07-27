#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace mha_bwd_impl {

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S, int d)
{
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int NTID = 256;

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    int stride_s = d;
    int stride_bh = S * d;
    int64_t bh_off = ((int64_t)b * H + h) * stride_bh;

    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* O_bh = O_fwd + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + b * H * S + h * S;

    __nv_bfloat16* dQ_ptr = dQ_out + bh_off;
    __nv_bfloat16* dK_ptr = dK_out + bh_off;
    __nv_bfloat16* dV_ptr = dV_out + bh_off;

    float inv_sqrt_d = rsqrtf(static_cast<float>(d));
    int tid = threadIdx.x;

    // Shared memory layout (bf16 units):
    //   Q_s:  [0, BM*d)             -> Q tile [BM][d]
    //   dO_s: [BM*d, 2*BM*d)        -> dO tile [BM][d]
    //   K_s:  [2*BM*d, 2*BM*d+BN*d)-> K tile [BN][d] (Pass1) or [BN][BK] (Pass2)
    //   V_s:  [2*BM*d+BN*d, ... )   -> V tile
    //   float area starts after all bf16
    size_t off_K = 2ULL * BM * d;
    size_t off_V = off_K + BN * d;
    size_t off_float_bytes = (off_V + BN * d) * sizeof(__nv_bfloat16);

    extern __shared__ char smem_raw[];

    __nv_bfloat16* Q_s  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* dO_s = Q_s + BM * d;
    __nv_bfloat16* K_s  = Q_s + off_K;
    __nv_bfloat16* V_s  = Q_s + off_V;
    float* rowsum_s     = reinterpret_cast<float*>(smem_raw + off_float_bytes);
    float* delta_s      = rowsum_s + BM;
    float* P_s          = delta_s + BM * BN;

    // ===== PASS 1: Compute dQ =====
    for (int qs = 0; qs < S; qs += BM) {
        int qb = qs + BM < S ? BM : S - qs;

        // Load Q[qs:qs+qb, :] and dO[qs:qs+qb, :]
        for (int i = tid; i < qb * d; i += NTID) {
            int r = i / d, c = i % d;
            Q_s[i]  = Q_bh[(qs + r) * stride_s + c];
            dO_s[i] = dO_bh[(qs + r) * stride_s + c];
        }
        __syncthreads();

        // Compute rowsum[qr] = dot(dO[qr,:], O[qr,:]) -- one thread per row
        int tpr = NTID / BM;
        int mr = tid / tpr;
        if (tid % tpr == 0 && mr < qb) {
            float rs = 0.0f;
            const __nv_bfloat16* dO_r = dO_bh + (qs + mr) * stride_s;
            const __nv_bfloat16* O_r  = O_bh  + (qs + mr) * stride_s;
            for (int k = 0; k < d; k += 4) {
                rs += __bfloat162float(dO_r[k])     * __bfloat162float(O_r[k]);
                rs += __bfloat162float(dO_r[k + 1]) * __bfloat162float(O_r[k + 1]);
                rs += __bfloat162float(dO_r[k + 2]) * __bfloat162float(O_r[k + 2]);
                rs += __bfloat162float(dO_r[k + 3]) * __bfloat162float(O_r[k + 3]);
            }
            rowsum_s[mr] = rs;
        }
        __syncthreads();

        // Initialize per-thread dQ accumulator
        int dq_tot = qb * d;
        int dq_ept = dq_tot / NTID;
        int dq_rmd = dq_tot % NTID;
        int dq_st  = tid * dq_ept + (tid < dq_rmd ? tid : dq_rmd);
        int dq_cnt = dq_ept + (tid < dq_rmd ? 1 : 0);

        float dq_loc[16];
        for (int i = 0; i < dq_cnt; i++) dq_loc[i] = 0.0f;

        // Loop over key tiles
        for (int ks = 0; ks < S; ks += BN) {
            int kb = ks + BN < S ? BN : S - ks;

            // Load FULL K[ks:ks+kb, :] and V[ks:ks+kb, :]
            for (int i = tid; i < kb * d; i += NTID) {
                int r = i / d, c = i % d;
                K_s[i] = K_bh[(ks + r) * stride_s + c];
                V_s[i] = V_bh[(ks + r) * stride_s + c];
            }
            __syncthreads();

            // Compute S = Q@K^T/sqrt(d) and D = dO@V^T via BK-chunk accumulation
            int stot = qb * kb;
            int se   = stot / NTID;
            int sr   = stot % NTID;
            int ss   = tid * se + (tid < sr ? tid : sr);
            int sc   = se + (tid < sr ? 1 : 0);

            float s_loc[5] = {}, d_loc[5] = {};

            for (int bk = 0; bk < d; bk += BK) {
                int ba = bk + BK < d ? BK : d - bk;

                // After sync below, K_s/V_s may be overwritten by next bk iteration load
                // But we load K_s/V_s ONCE at the top of k_tile, so no reload needed here.
                // Read Q_s, dO_s, K_s, V_s from shared memory (all stable)

                for (int ei = 0; ei < sc; ei++) {
                    int li = ss + ei;
                    int qi = li / kb;
                    int kj = li % kb;
                    if (qi >= qb) continue;
                    float sv = 0.0f, dv = 0.0f;
                    for (int kk = 0; kk < ba; kk++) {
                        sv += __bfloat162float(Q_s[qi * d + bk + kk]) *
                              __bfloat162float(K_s[kj * d + bk + kk]);
                        dv += __bfloat162float(dO_s[qi * d + bk + kk]) *
                              __bfloat162float(V_s[kj * d + bk + kk]);
                    }
                    s_loc[ei] += sv * inv_sqrt_d;
                    d_loc[ei] += dv;
                }
            }

            // All threads finished BK loop -- write P and delta to smem
            for (int ei = 0; ei < sc; ei++) {
                int li = ss + ei;
                int qi = li / kb;
                int kj = li % kb;
                if (qi >= qb) continue;
                float p  = expf(s_loc[ei] - L_bh[qs + qi]);
                float dl = p * (d_loc[ei] - rowsum_s[qi]);
                delta_s[li] = dl;
                P_s[li]     = p;
            }
            __syncthreads();

            // Compute dQ contribution: delta[qb][kb] @ K[ks:ks+kb, :]
            // dQ[qr][dc] += sum_{kj} delta[qr][kj] * K[kj][dc]
            for (int bk = 0; bk < d; bk += BK) {
                int ba = bk + BK < d ? BK : d - bk;
                for (int ei = 0; ei < dq_cnt; ei++) {
                    int li = dq_st + ei;
                    int qr = li / d;
                    int dc = li % d;
                    if (qr >= qb || dc < bk || dc >= bk + ba) continue;
                    int kl = dc - bk;
                    float acc = 0.0f;
                    for (int kj = 0; kj < kb; kj++) {
                        acc += delta_s[qr * kb + kj] *
                               __bfloat162float(K_s[kj * d + bk + kl]);
                    }
                    dq_loc[ei] += acc;
                }
            }
            __syncthreads(); // Ensure all threads before next k_tile loads K/V
        }

        // Write dQ to global memory
        for (int ei = 0; ei < dq_cnt; ei++) {
            int li = dq_st + ei;
            int qr = li / d;
            int dc = li % d;
            if (qr < qb) {
                dQ_ptr[(qs + qr) * stride_s + dc] = __float2bfloat16(dq_loc[ei]);
            }
        }
    }

    // ===== PASS 2: Compute dK and dV =====
    for (int ks = 0; ks < S; ks += BN) {
        int kb = ks + BN < S ? BN : S - ks;

        // Initialize dK and dV accumulators for this key tile
        int kv_tot = kb * d;
        int kv_ept = kv_tot / NTID;
        int kv_rmd = kv_tot % NTID;
        int kv_st  = tid * kv_ept + (tid < kv_rmd ? tid : kv_rmd);
        int kv_cnt = kv_ept + (tid < kv_rmd ? 1 : 0);

        float dk_loc[16] = {}, dv_loc[16] = {};

        for (int qs = 0; qs < S; qs += BM) {
            int qb = qs + BM < S ? BM : S - qs;

            // Load Q[qs:qs+qb, :] and dO[qs:qs+qb, :]
            for (int i = tid; i < qb * d; i += NTID) {
                int r = i / d, c = i % d;
                Q_s[i]  = Q_bh[(qs + r) * stride_s + c];
                dO_s[i] = dO_bh[(qs + r) * stride_s + c];
            }
            __syncthreads();

            // Compute rowsum
            int tpr = NTID / BM;
            int mr = tid / tpr;
            if (tid % tpr == 0 && mr < qb) {
                float rs = 0.0f;
                const __nv_bfloat16* dO_r = dO_bh + (qs + mr) * stride_s;
                const __nv_bfloat16* O_r  = O_bh  + (qs + mr) * stride_s;
                for (int k = 0; k < d; k += 4) {
                    rs += __bfloat162float(dO_r[k])     * __bfloat162float(O_r[k]);
                    rs += __bfloat162float(dO_r[k + 1]) * __bfloat162float(O_r[k + 1]);
                    rs += __bfloat162float(dO_r[k + 2]) * __bfloat162float(O_r[k + 2]);
                    rs += __bfloat162float(dO_r[k + 3]) * __bfloat162float(O_r[k + 3]);
                }
                rowsum_s[mr] = rs;
            }
            __syncthreads();

            // Compute S and D via BK-chunk accumulation
            int stot = qb * kb;
            int se   = stot / NTID;
            int sr   = stot % NTID;
            int ss   = tid * se + (tid < sr ? tid : sr);
            int sc   = se + (tid < sr ? 1 : 0);

            float s_loc[5] = {}, d_loc[5] = {};

            for (int bk = 0; bk < d; bk += BK) {
                int ba = bk + BK < d ? BK : d - bk;

                // Load K/V chunks [kb][BK]
                for (int i = tid; i < kb * ba; i += NTID) {
                    int r = i / ba, c = i % ba;
                    K_s[r * BK + c] = K_bh[(ks + r) * stride_s + bk + c];
                    V_s[r * BK + c] = V_bh[(ks + r) * stride_s + bk + c];
                }
                __syncthreads();

                for (int ei = 0; ei < sc; ei++) {
                    int li = ss + ei;
                    int qi = li / kb;
                    int kj = li % kb;
                    if (qi >= qb) continue;
                    float sv = 0.0f, dv = 0.0f;
                    for (int kk = 0; kk < ba; kk++) {
                        sv += __bfloat162float(Q_s[qi * d + bk + kk]) *
                              __bfloat162float(K_s[kj * BK + kk]);
                        dv += __bfloat162float(dO_s[qi * d + bk + kk]) *
                              __bfloat162float(V_s[kj * BK + kk]);
                    }
                    s_loc[ei] += sv * inv_sqrt_d;
                    d_loc[ei] += dv;
                }
                __syncthreads();
            }

            // Write P and delta
            for (int ei = 0; ei < sc; ei++) {
                int li = ss + ei;
                int qi = li / kb;
                int kj = li % kb;
                if (qi >= qb) continue;
                float p  = expf(s_loc[ei] - L_bh[qs + qi]);
                float dl = p * (d_loc[ei] - rowsum_s[qi]);
                delta_s[li] = dl;
                P_s[li]     = p;
            }
            __syncthreads();

            // Compute dK and dV contributions
            // dK[kj][dc] += sum_qi delta[qi][kj] * Q[qi][dc]
            // dV[kj][dc] += sum_qi P[qi][kj]     * dO[qi][dc]
            for (int bk = 0; bk < d; bk += BK) {
                int ba = bk + BK < d ? BK : d - bk;
                for (int ei = 0; ei < kv_cnt; ei++) {
                    int li = kv_st + ei;
                    int kj = li / d;
                    int dc = li % d;
                    if (kj >= kb || dc < bk || dc >= bk + ba) continue;
                    int kl = dc - bk;
                    float dk = 0.0f, dv = 0.0f;
                    for (int qi = 0; qi < qb; qi++) {
                        dk += delta_s[qi * kb + kj] *
                              __bfloat162float(Q_s[qi * d + bk + kl]);
                        dv += P_s[qi * kb + kj] *
                              __bfloat162float(dO_s[qi * d + bk + kl]);
                    }
                    dk_loc[ei] += dk;
                    dv_loc[ei] += dv;
                }
            }
            __syncthreads(); // Sync before next q_tile overwrites Q_s/dO_s
        }

        // Write dK and dV to global memory
        for (int ei = 0; ei < kv_cnt; ei++) {
            int li = kv_st + ei;
            int kj = li / d;
            int dc = li % d;
            if (kj < kb) {
                dK_ptr[(ks + kj) * stride_s + dc] = __float2bfloat16(dk_loc[ei]);
                dV_ptr[(ks + kj) * stride_s + dc] = __float2bfloat16(dv_loc[ei]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int NTID = 256;

    // Shared memory: Q_s[BM][d] + dO_s[BM][d] + K_s[BN][d] + V_s[BN][d] + float area
    size_t smem_bf16 = (size_t)(2 * BM * d + 2 * BN * d) * sizeof(__nv_bfloat16);
    size_t smem_float = (size_t)BM * sizeof(float) + 2ULL * (size_t)BM * BN * sizeof(float);
    size_t smem_bytes = smem_bf16 + smem_float;

    int64_t grid = B * H;
    dim3 block(NTID);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Zero outputs
    size_t out_bytes = (size_t)(B * H * S * d) * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dK.data_ptr(), 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV.data_ptr(), 0, out_bytes, stream));

    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        (int)B, (int)H, (int)S, (int)d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl