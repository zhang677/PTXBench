#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
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

#define CU_CHECK(call) \
    do { \
        CUresult _err = (call); \
        if (_err != CUDA_SUCCESS) { \
            fprintf(stderr, "CU Error %d at %s:%d\n", _err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace mha_bwd_impl {

__device__ __forceinline__ void get_d_coord_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int chunk = col / 8;
    int chunk_swizzled = chunk ^ (row % 8);
    return row * 64 + chunk_swizzled * 8 + (col % 8);
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_fence_fn() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_fn() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void wgmma_wait_fn() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ static inline uint64_t make_wgmma_desc(void* ptr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle_mode) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t desc = 0;
    desc |= (addr & 0x3FFFF) >> 4;
    desc |= ((uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16);
    desc |= ((uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32);
    // Base offset for 1024-byte alignment
    uint32_t base_offset = (addr >> 7) & 7;
    desc |= ((uint64_t)base_offset << 49);
    desc |= ((uint64_t)swizzle_mode << 62);
    return desc;
}

template<int trans_a, int trans_b>
__device__ __forceinline__ void wgmma_m64n64k16_ss_fn(float* c, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,%34,%35;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
          "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
          "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
          "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
        : "l"(da), "l"(db), "n"(trans_a), "n"(trans_b));
}

template<int trans_a, int trans_b>
__device__ __forceinline__ void wgmma_64x64_64_ss(float* acc, uint64_t desc_a, uint64_t desc_b) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint64_t da = desc_a;
        uint64_t db = desc_b;
        
        if constexpr (trans_a == 0) da += (k * 32) >> 4;
        else                        da += (k * 2048) >> 4;
        
        if constexpr (trans_b == 0) db += (k * 32) >> 4;
        else                        db += (k * 2048) >> 4;
        
        wgmma_m64n64k16_ss_fn<trans_a, trans_b>(acc, da, db);
    }
}

__device__ __forceinline__ void store_acc_global_n64_bf16(
    __nv_bfloat16* C, float* ac, int bm, int bn, int M, int N, int tid, int stride) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * stride + gn] = __float2bfloat16(ac[r]);
    }
}

struct __align__(128) SharedStorage {
    __nv_bfloat16 Q_0[64*64];
    __nv_bfloat16 Q_1[64*64];
    __nv_bfloat16 dO_0[64*64];
    __nv_bfloat16 dO_1[64*64];
    __nv_bfloat16 O_0[64*64];
    __nv_bfloat16 O_1[64*64];
    __nv_bfloat16 K_0[64*64];
    __nv_bfloat16 K_1[64*64];
    __nv_bfloat16 V_0[64*64];
    __nv_bfloat16 V_1[64*64];
    __nv_bfloat16 dS[64*64];
    __nv_bfloat16 P[64*64];
    float D[64];
    float L[64];
    uint64_t bar_Q[1];
    uint64_t bar_K[1];
};

__global__ __launch_bounds__(128, 1)
void mha_bwd_kernel_1(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const float* __restrict__ L_gmem,
    __nv_bfloat16* __restrict__ dQ_out,
    int S, int d, int H, float inv_sqrt_d)
{
    setmaxnreg_inc_sync_fn<248>();
    
    int q_idx = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    
    extern __shared__ __align__(128) char smem_raw[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_raw);
    
    int phase_K = 0;
    
    if (tid == 0) {
        init_smem_barrier_fn(&smem->bar_Q[0], 1);
        init_smem_barrier_fn(&smem->bar_K[0], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->bar_Q[0], 6 * 8192);
        tma_load_4d_fn(&tma_Q, &smem->bar_Q[0], smem->Q_0, 0, q_idx * 64, h, b);
        tma_load_4d_fn(&tma_Q, &smem->bar_Q[0], smem->Q_1, 64, q_idx * 64, h, b);
        tma_load_4d_fn(&tma_dO, &smem->bar_Q[0], smem->dO_0, 0, q_idx * 64, h, b);
        tma_load_4d_fn(&tma_dO, &smem->bar_Q[0], smem->dO_1, 64, q_idx * 64, h, b);
        tma_load_4d_fn(&tma_O, &smem->bar_Q[0], smem->O_0, 0, q_idx * 64, h, b);
        tma_load_4d_fn(&tma_O, &smem->bar_Q[0], smem->O_1, 64, q_idx * 64, h, b);
    }
    mbarrier_wait_fn(&smem->bar_Q[0], 0);
    
    if (tid < 64) smem->L[tid] = L_gmem[b * H * S + h * S + q_idx * 64 + tid];
    
    float d_val = 0.0f;
    int row = tid / 2;
    int col_grp = tid % 2;
    __nv_bfloat16* dO_ptr = (col_grp == 0) ? smem->dO_0 : smem->dO_1;
    __nv_bfloat16* O_ptr  = (col_grp == 0) ? smem->O_0  : smem->O_1;
    for (int c = 0; c < 64; ++c) {
        int idx = swizzle_128B(row, c);
        d_val += __bfloat162float(dO_ptr[idx]) * __bfloat162float(O_ptr[idx]);
    }
    d_val += __shfl_xor_sync(0xffffffff, d_val, 1);
    if (col_grp == 0) smem->D[row] = d_val;
    __syncthreads();
    
    uint64_t desc_Q_0 = make_wgmma_desc(smem->Q_0, 1, 1024, 1);
    uint64_t desc_Q_1 = make_wgmma_desc(smem->Q_1, 1, 1024, 1);
    uint64_t desc_dO_0 = make_wgmma_desc(smem->dO_0, 1, 1024, 1);
    uint64_t desc_dO_1 = make_wgmma_desc(smem->dO_1, 1, 1024, 1);
    uint64_t desc_K_0 = make_wgmma_desc(smem->K_0, 1, 1024, 1);
    uint64_t desc_K_1 = make_wgmma_desc(smem->K_1, 1, 1024, 1);
    uint64_t desc_V_0 = make_wgmma_desc(smem->V_0, 1, 1024, 1);
    uint64_t desc_V_1 = make_wgmma_desc(smem->V_1, 1, 1024, 1);
    uint64_t desc_dS = make_wgmma_desc(smem->dS, 1, 1024, 1);
    uint64_t desc_K_0_MN = make_wgmma_desc(smem->K_0, 8192, 1024, 1);
    uint64_t desc_K_1_MN = make_wgmma_desc(smem->K_1, 8192, 1024, 1);
    
    float dQ_0_acc[32], dQ_1_acc[32];
    for (int i=0; i<32; ++i) { dQ_0_acc[i] = 0; dQ_1_acc[i] = 0; }
    
    for (int k_idx = 0; k_idx < S / 64; ++k_idx) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->bar_K[0], 4 * 8192);
            tma_load_4d_fn(&tma_K, &smem->bar_K[0], smem->K_0, 0, k_idx * 64, h, b);
            tma_load_4d_fn(&tma_K, &smem->bar_K[0], smem->K_1, 64, k_idx * 64, h, b);
            tma_load_4d_fn(&tma_V, &smem->bar_K[0], smem->V_0, 0, k_idx * 64, h, b);
            tma_load_4d_fn(&tma_V, &smem->bar_K[0], smem->V_1, 64, k_idx * 64, h, b);
        }
        mbarrier_wait_fn(&smem->bar_K[0], phase_K);
        phase_K ^= 1;
        
        fence_proxy_async_fn();
        wgmma_fence_fn();
        
        float S_acc[32];
        for (int i=0; i<32; ++i) S_acc[i] = 0;
        wgmma_64x64_64_ss<0, 0>(S_acc, desc_Q_0, desc_K_0);
        wgmma_64x64_64_ss<0, 0>(S_acc, desc_Q_1, desc_K_1);
        
        float dP_acc[32];
        for (int i=0; i<32; ++i) dP_acc[i] = 0;
        wgmma_64x64_64_ss<0, 0>(dP_acc, desc_dO_0, desc_V_0);
        wgmma_64x64_64_ss<0, 0>(dP_acc, desc_dO_1, desc_V_1);
        
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        
        for (int r = 0; r < 32; ++r) {
            int row_q, col_k;
            get_d_coord_(tid, r, row_q, col_k);
            float l = smem->L[row_q];
            float d_ = smem->D[row_q];
            float p = expf(S_acc[r] * inv_sqrt_d - l);
            float ds = p * (dP_acc[r] - d_) * inv_sqrt_d;
            int idx = swizzle_128B(row_q, col_k);
            smem->dS[idx] = __float2bfloat16(ds);
        }
        __syncthreads();
        fence_proxy_async_fn();
        wgmma_fence_fn();
        
        wgmma_64x64_64_ss<0, 1>(dQ_0_acc, desc_dS, desc_K_0_MN);
        wgmma_64x64_64_ss<0, 1>(dQ_1_acc, desc_dS, desc_K_1_MN);
        
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        __syncthreads();
    }
    
    store_acc_global_n64_bf16(dQ_out + b*H*S*d + h*S*d, dQ_0_acc, q_idx*64, 0, S, d, tid, d);
    store_acc_global_n64_bf16(dQ_out + b*H*S*d + h*S*d, dQ_1_acc, q_idx*64, 64, S, d, tid, d);
}

__global__ __launch_bounds__(128, 1)
void mha_bwd_kernel_2(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const float* __restrict__ L_gmem,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S, int d, int H, float inv_sqrt_d)
{
    setmaxnreg_inc_sync_fn<248>();
    
    int k_idx = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    
    extern __shared__ __align__(128) char smem_raw[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_raw);
    
    int phase_Q = 0;
    
    if (tid == 0) {
        init_smem_barrier_fn(&smem->bar_Q[0], 1);
        init_smem_barrier_fn(&smem->bar_K[0], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->bar_K[0], 4 * 8192);
        tma_load_4d_fn(&tma_K, &smem->bar_K[0], smem->K_0, 0, k_idx * 64, h, b);
        tma_load_4d_fn(&tma_K, &smem->bar_K[0], smem->K_1, 64, k_idx * 64, h, b);
        tma_load_4d_fn(&tma_V, &smem->bar_K[0], smem->V_0, 0, k_idx * 64, h, b);
        tma_load_4d_fn(&tma_V, &smem->bar_K[0], smem->V_1, 64, k_idx * 64, h, b);
    }
    mbarrier_wait_fn(&smem->bar_K[0], 0);
    __syncthreads();
    
    uint64_t desc_K_0 = make_wgmma_desc(smem->K_0, 1, 1024, 1);
    uint64_t desc_K_1 = make_wgmma_desc(smem->K_1, 1, 1024, 1);
    uint64_t desc_V_0 = make_wgmma_desc(smem->V_0, 1, 1024, 1);
    uint64_t desc_V_1 = make_wgmma_desc(smem->V_1, 1, 1024, 1);
    uint64_t desc_Q_0 = make_wgmma_desc(smem->Q_0, 1, 1024, 1);
    uint64_t desc_Q_1 = make_wgmma_desc(smem->Q_1, 1, 1024, 1);
    uint64_t desc_dO_0 = make_wgmma_desc(smem->dO_0, 1, 1024, 1);
    uint64_t desc_dO_1 = make_wgmma_desc(smem->dO_1, 1, 1024, 1);
    uint64_t desc_dS = make_wgmma_desc(smem->dS, 1, 1024, 1);
    uint64_t desc_P = make_wgmma_desc(smem->P, 1, 1024, 1);
    uint64_t desc_Q_0_MN = make_wgmma_desc(smem->Q_0, 8192, 1024, 1);
    uint64_t desc_Q_1_MN = make_wgmma_desc(smem->Q_1, 8192, 1024, 1);
    uint64_t desc_dO_0_MN = make_wgmma_desc(smem->dO_0, 8192, 1024, 1);
    uint64_t desc_dO_1_MN = make_wgmma_desc(smem->dO_1, 8192, 1024, 1);
    
    float dK_0_acc[32], dK_1_acc[32];
    float dV_0_acc[32], dV_1_acc[32];
    for (int i=0; i<32; ++i) { 
        dK_0_acc[i] = 0; dK_1_acc[i] = 0; 
        dV_0_acc[i] = 0; dV_1_acc[i] = 0;
    }
    
    for (int q_idx = 0; q_idx < S / 64; ++q_idx) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->bar_Q[0], 6 * 8192);
            tma_load_4d_fn(&tma_Q, &smem->bar_Q[0], smem->Q_0, 0, q_idx * 64, h, b);
            tma_load_4d_fn(&tma_Q, &smem->bar_Q[0], smem->Q_1, 64, q_idx * 64, h, b);
            tma_load_4d_fn(&tma_dO, &smem->bar_Q[0], smem->dO_0, 0, q_idx * 64, h, b);
            tma_load_4d_fn(&tma_dO, &smem->bar_Q[0], smem->dO_1, 64, q_idx * 64, h, b);
            tma_load_4d_fn(&tma_O, &smem->bar_Q[0], smem->O_0, 0, q_idx * 64, h, b);
            tma_load_4d_fn(&tma_O, &smem->bar_Q[0], smem->O_1, 64, q_idx * 64, h, b);
        }
        mbarrier_wait_fn(&smem->bar_Q[0], phase_Q);
        phase_Q ^= 1;
        
        if (tid < 64) smem->L[tid] = L_gmem[b * H * S + h * S + q_idx * 64 + tid];
        
        float d_val = 0.0f;
        int row = tid / 2;
        int col_grp = tid % 2;
        __nv_bfloat16* dO_ptr = (col_grp == 0) ? smem->dO_0 : smem->dO_1;
        __nv_bfloat16* O_ptr  = (col_grp == 0) ? smem->O_0  : smem->O_1;
        for (int c = 0; c < 64; ++c) {
            int idx = swizzle_128B(row, c);
            d_val += __bfloat162float(dO_ptr[idx]) * __bfloat162float(O_ptr[idx]);
        }
        d_val += __shfl_xor_sync(0xffffffff, d_val, 1);
        if (col_grp == 0) smem->D[row] = d_val;
        __syncthreads();
        
        fence_proxy_async_fn();
        wgmma_fence_fn();
        
        float S_acc[32];
        for (int i=0; i<32; ++i) S_acc[i] = 0;
        wgmma_64x64_64_ss<0, 0>(S_acc, desc_K_0, desc_Q_0);
        wgmma_64x64_64_ss<0, 0>(S_acc, desc_K_1, desc_Q_1);
        
        float dP_acc[32];
        for (int i=0; i<32; ++i) dP_acc[i] = 0;
        wgmma_64x64_64_ss<0, 0>(dP_acc, desc_V_0, desc_dO_0);
        wgmma_64x64_64_ss<0, 0>(dP_acc, desc_V_1, desc_dO_1);
        
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        
        for (int r = 0; r < 32; ++r) {
            int row_k, col_q;
            get_d_coord_(tid, r, row_k, col_q);
            float l = smem->L[col_q];
            float d_ = smem->D[col_q];
            float p = expf(S_acc[r] * inv_sqrt_d - l);
            float ds = p * (dP_acc[r] - d_) * inv_sqrt_d;
            int idx = swizzle_128B(row_k, col_q);
            smem->dS[idx] = __float2bfloat16(ds);
            smem->P[idx]  = __float2bfloat16(p);
        }
        __syncthreads();
        fence_proxy_async_fn();
        wgmma_fence_fn();
        
        wgmma_64x64_64_ss<0, 1>(dK_0_acc, desc_dS, desc_Q_0_MN);
        wgmma_64x64_64_ss<0, 1>(dK_1_acc, desc_dS, desc_Q_1_MN);
        
        wgmma_64x64_64_ss<0, 1>(dV_0_acc, desc_P, desc_dO_0_MN);
        wgmma_64x64_64_ss<0, 1>(dV_1_acc, desc_P, desc_dO_1_MN);
        
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        __syncthreads();
    }
    
    store_acc_global_n64_bf16(dK_out + b*H*S*d + h*S*d, dK_0_acc, k_idx*64, 0, S, d, tid, d);
    store_acc_global_n64_bf16(dK_out + b*H*S*d + h*S*d, dK_1_acc, k_idx*64, 64, S, d, tid, d);
    store_acc_global_n64_bf16(dV_out + b*H*S*d + h*S*d, dV_0_acc, k_idx*64, 0, S, d, tid, d);
    store_acc_global_n64_bf16(dV_out + b*H*S*d + h*S*d, dV_1_acc, k_idx*64, 64, S, d, tid, d);
}

CUresult create_tma_4d_descriptor_2B(
    CUtensorMap* d_desc, void* globalAddress,
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1,
    CUtensorMapSwizzle swizzle) 
{
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {
        dim0 * 2,
        dim0 * dim1 * 2,
        dim0 * dim1 * dim2 * 2
    };
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d_desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    CUtensorMap tma_Q, tma_dO, tma_O, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), d, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dO, dO.data_ptr(), d, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), d, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), d, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), d, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));

    int grid = S / 64;
    dim3 grid_dim(grid, H, B);
    dim3 block_dim(128);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    mha_bwd_kernel_1<<<grid_dim, block_dim, sizeof(SharedStorage), stream>>>(
        tma_Q, tma_dO, tma_O, tma_K, tma_V,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, d, H, 1.0f / sqrtf((float)d));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel_2, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    mha_bwd_kernel_2<<<grid_dim, block_dim, sizeof(SharedStorage), stream>>>(
        tma_Q, tma_dO, tma_O, tma_K, tma_V,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, d, H, 1.0f / sqrtf((float)d));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl