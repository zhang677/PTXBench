#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_cuda_mha {

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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
    uint64_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t desc = 0;
    desc |= (addr & 0x3FFFF) >> 4; 
    desc |= ((uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16);  
    desc |= ((uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32);  
    uint64_t base_offset = (addr >> 7) & 0x7;
    desc |= (base_offset << 49);
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

__device__ __forceinline__ void get_d_coord_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

__global__ void __launch_bounds__(128)
kernel_mha_bwd_dQ(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_gmem,
    int S, int H, int B)
{
    int bm = blockIdx.x;
    int bh_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)(smem_pool + 0);
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 8192);
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem_pool + 16384);
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem_pool + 24576);
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem_pool + 40960);
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem_pool + 49152);
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem_pool + 57344);
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_pool + 73728);
    __nv_bfloat16* smem_dS = (__nv_bfloat16*)(smem_pool + 81920);
    float* smem_D = (float*)(smem_pool + 90112);
    float* smem_L = (float*)(smem_pool + 90368);
    uint64_t* mbar = (uint64_t*)(smem_pool + 90624);

    float scale = 1.0f / (8.0f * 1.41421356f); 

    if (tid == 0) {
        init_smem_barrier_fn(mbar, 128);
    }
    __syncthreads();

    int phase = 0;
    
    // Pre-load L values for the block
    if (tid < 64) {
        int global_row = bm * 64 + tid;
        smem_L[tid] = (global_row < S) ? L[bh_idx * S + global_row] : 0.0f;
    }
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 32768); // 4 loads of 8192 bytes
        tma_load_2d_fn(&tma_Q, mbar, smem_Q0, 0, bh_idx * S + bm * 64);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q1, 64, bh_idx * S + bm * 64);
        tma_load_2d_fn(&tma_dO, mbar, smem_dO0, 0, bh_idx * S + bm * 64);
        tma_load_2d_fn(&tma_dO, mbar, smem_dO1, 64, bh_idx * S + bm * 64);
        tma_load_2d_fn(&tma_O, mbar, smem_O0, 0, bh_idx * S + bm * 64);
        tma_load_2d_fn(&tma_O, mbar, smem_O1, 64, bh_idx * S + bm * 64);
    } else {
        mbarrier_arrive_fn(mbar);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    __syncthreads();

    // Calculate Dot Product D = sum(O * dO)
    if (tid < 64) {
        float sum = 0;
        for (int c = 0; c < 64; c++) {
            int chunk_x = c / 8;
            int offset = c % 8;
            int swizzled_chunk_x = (tid % 8) ^ chunk_x;
            int swizzled_c = swizzled_chunk_x * 8 + offset;
            
            float o0 = __bfloat162float(smem_O0[tid * 64 + swizzled_c]);
            float do0 = __bfloat162float(smem_dO0[tid * 64 + swizzled_c]);
            sum += o0 * do0;
            
            float o1 = __bfloat162float(smem_O1[tid * 64 + swizzled_c]);
            float do1 = __bfloat162float(smem_dO1[tid * 64 + swizzled_c]);
            sum += o1 * do1;
        }
        smem_D[tid] = sum;
    }
    __syncthreads();

    float acc_dQ0[32];
    float acc_dQ1[32];
    for(int i = 0; i < 32; i++) { acc_dQ0[i] = 0.0f; acc_dQ1[i] = 0.0f; }

    for (int j = 0; j < S; j += 64) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 32768); // 4 loads of 8192 bytes
            tma_load_2d_fn(&tma_K, mbar, smem_K0, 0, bh_idx * S + j);
            tma_load_2d_fn(&tma_K, mbar, smem_K1, 64, bh_idx * S + j);
            tma_load_2d_fn(&tma_V, mbar, smem_V0, 0, bh_idx * S + j);
            tma_load_2d_fn(&tma_V, mbar, smem_V1, 64, bh_idx * S + j);
        } else {
            mbarrier_arrive_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();

        fence_proxy_async_fn();
        wgmma_fence_fn();

        float acc_S[32];
        float acc_dP[32];
        for(int i = 0; i < 32; i++) { acc_S[i] = 0.0f; acc_dP[i] = 0.0f; }

        // Q @ K^T AND dO @ V^T
        for (int k = 0; k < 4; k++) {
            uint64_t desc_Q0 = make_wgmma_desc(smem_Q0 + k * 16, 1, 1024, 1);
            uint64_t desc_K0 = make_wgmma_desc(smem_K0 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_S, desc_Q0, desc_K0);
            
            uint64_t desc_Q1 = make_wgmma_desc(smem_Q1 + k * 16, 1, 1024, 1);
            uint64_t desc_K1 = make_wgmma_desc(smem_K1 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_S, desc_Q1, desc_K1);
            
            uint64_t desc_dO0 = make_wgmma_desc(smem_dO0 + k * 16, 1, 1024, 1);
            uint64_t desc_V0  = make_wgmma_desc(smem_V0 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_dP, desc_dO0, desc_V0);
            
            uint64_t desc_dO1 = make_wgmma_desc(smem_dO1 + k * 16, 1, 1024, 1);
            uint64_t desc_V1  = make_wgmma_desc(smem_V1 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_dP, desc_dO1, desc_V1);
        }
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        
        __syncthreads();
        
        float acc_dS[32];
        for (int r = 0; r < 32; r++) {
            int lm, ln;
            get_d_coord_(tid, r, lm, ln);
            int global_row = bm * 64 + lm;
            int global_col = j + ln;
            
            float p_val = 0.0f;
            if (global_row < S && global_col < S) {
                float s_val = acc_S[r] * scale - smem_L[lm];
                p_val = fast_exp2f_fn(s_val * 1.44269504f); // Convert natural exp to base-2 exp
            }
            
            float dp_val = acc_dP[r];
            float ds_val = p_val * (dp_val - smem_D[lm]) * scale;
            acc_dS[r] = ds_val;
            
            // Write dS to SMEM normally
            int chunk_x = ln / 8;
            int offset = ln % 8;
            int swizzled_chunk_x = (lm % 8) ^ chunk_x;
            int swizzled_ln = swizzled_chunk_x * 8 + offset;
            smem_dS[lm * 64 + swizzled_ln] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        wgmma_fence_fn();
        
        // dQ += dS @ K
        for (int k = 0; k < 4; k++) {
            uint64_t desc_dS = make_wgmma_desc(smem_dS + k * 16, 1, 1024, 1); 
            
            uint64_t desc_K0 = make_wgmma_desc(smem_K0 + k * 1024, 8192, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 1>(acc_dQ0, desc_dS, desc_K0);
            
            uint64_t desc_K1 = make_wgmma_desc(smem_K1 + k * 1024, 8192, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 1>(acc_dQ1, desc_dS, desc_K1);
        }
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        
        __syncthreads();
    }

    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        int gm = bm * 64 + lm;
        if (gm < S) {
            int idx = bh_idx * S * 128 + gm * 128 + ln;
            dQ_gmem[idx] = __float2bfloat16(acc_dQ0[r]);
            dQ_gmem[bh_idx * S * 128 + gm * 128 + 64 + ln] = __float2bfloat16(acc_dQ1[r]);
        }
    }
}

__global__ void __launch_bounds__(128)
kernel_mha_bwd_dK_dV(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK_gmem,
    __nv_bfloat16* __restrict__ dV_gmem,
    int S, int H, int B)
{
    int bn = blockIdx.x; 
    int bh_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem_pool + 0);
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem_pool + 8192);
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_pool + 16384);
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_pool + 24576);
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 40960);
    __nv_bfloat16* smem_dO0 = (__nv_bfloat16*)(smem_pool + 49152);
    __nv_bfloat16* smem_dO1 = (__nv_bfloat16*)(smem_pool + 57344);
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem_pool + 73728);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 81920);
    __nv_bfloat16* smem_dS = (__nv_bfloat16*)(smem_pool + 90112);
    float* smem_D = (float*)(smem_pool + 98304);
    float* smem_L = (float*)(smem_pool + 98560);
    uint64_t* mbar = (uint64_t*)(smem_pool + 98816);

    float scale = 1.0f / (8.0f * 1.41421356f); 

    if (tid == 0) {
        init_smem_barrier_fn(mbar, 128);
    }
    __syncthreads();

    int phase = 0;
    
    // Pre-load invariant K, V tiles
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 16384); // 4 loads of 8192 bytes
        tma_load_2d_fn(&tma_K, mbar, smem_K0, 0, bh_idx * S + bn * 64);
        tma_load_2d_fn(&tma_K, mbar, smem_K1, 64, bh_idx * S + bn * 64);
        tma_load_2d_fn(&tma_V, mbar, smem_V0, 0, bh_idx * S + bn * 64);
        tma_load_2d_fn(&tma_V, mbar, smem_V1, 64, bh_idx * S + bn * 64);
    } else {
        mbarrier_arrive_fn(mbar);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    __syncthreads();

    float acc_dK0[32];
    float acc_dK1[32];
    float acc_dV0[32];
    float acc_dV1[32];
    for(int i = 0; i < 32; i++) { 
        acc_dK0[i] = 0.0f; acc_dK1[i] = 0.0f;
        acc_dV0[i] = 0.0f; acc_dV1[i] = 0.0f;
    }

    for (int i = 0; i < S; i += 64) {
        if (tid < 64) {
            int global_row = i + tid;
            smem_L[tid] = (global_row < S) ? L[bh_idx * S + global_row] : 0.0f;
        }
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 49152); // 6 loads of 8192 bytes
            tma_load_2d_fn(&tma_Q, mbar, smem_Q0, 0, bh_idx * S + i);
            tma_load_2d_fn(&tma_Q, mbar, smem_Q1, 64, bh_idx * S + i);
            tma_load_2d_fn(&tma_dO, mbar, smem_dO0, 0, bh_idx * S + i);
            tma_load_2d_fn(&tma_dO, mbar, smem_dO1, 64, bh_idx * S + i);
            tma_load_2d_fn(&tma_O, mbar, smem_O0, 0, bh_idx * S + i);
            tma_load_2d_fn(&tma_O, mbar, smem_O1, 64, bh_idx * S + i);
        } else {
            mbarrier_arrive_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();

        // Calculate Dot Product D = sum(O * dO)
        if (tid < 64) {
            float sum = 0;
            for (int c = 0; c < 64; c++) {
                int chunk_x = c / 8;
                int offset = c % 8;
                int swizzled_chunk_x = (tid % 8) ^ chunk_x;
                int swizzled_c = swizzled_chunk_x * 8 + offset;
                
                float o0 = __bfloat162float(smem_O0[tid * 64 + swizzled_c]);
                float do0 = __bfloat162float(smem_dO0[tid * 64 + swizzled_c]);
                sum += o0 * do0;
                
                float o1 = __bfloat162float(smem_O1[tid * 64 + swizzled_c]);
                float do1 = __bfloat162float(smem_dO1[tid * 64 + swizzled_c]);
                sum += o1 * do1;
            }
            smem_D[tid] = sum;
        }
        __syncthreads();

        fence_proxy_async_fn();
        wgmma_fence_fn();

        float acc_S[32];
        float acc_dP[32];
        for(int r = 0; r < 32; r++) { acc_S[r] = 0.0f; acc_dP[r] = 0.0f; }

        // Q @ K^T AND dO @ V^T
        for (int k = 0; k < 4; k++) {
            uint64_t desc_Q0 = make_wgmma_desc(smem_Q0 + k * 16, 1, 1024, 1);
            uint64_t desc_K0 = make_wgmma_desc(smem_K0 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_S, desc_Q0, desc_K0);
            
            uint64_t desc_Q1 = make_wgmma_desc(smem_Q1 + k * 16, 1, 1024, 1);
            uint64_t desc_K1 = make_wgmma_desc(smem_K1 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_S, desc_Q1, desc_K1);
            
            uint64_t desc_dO0 = make_wgmma_desc(smem_dO0 + k * 16, 1, 1024, 1);
            uint64_t desc_V0  = make_wgmma_desc(smem_V0 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_dP, desc_dO0, desc_V0);
            
            uint64_t desc_dO1 = make_wgmma_desc(smem_dO1 + k * 16, 1, 1024, 1);
            uint64_t desc_V1  = make_wgmma_desc(smem_V1 + k * 16, 1, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 0>(acc_dP, desc_dO1, desc_V1);
        }
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        
        __syncthreads();
        
        // Write P^T AND dS^T to SMEM efficiently matching meta logical matrix layouts directly mapped without overhead translation loops
        for (int r = 0; r < 32; r++) {
            int lm, ln;
            get_d_coord_(tid, r, lm, ln);
            int global_row = i + lm;
            int global_col = bn * 64 + ln;
            
            float p_val = 0.0f;
            if (global_row < S && global_col < S) {
                float s_val = acc_S[r] * scale - smem_L[lm];
                p_val = fast_exp2f_fn(s_val * 1.44269504f);
            }
            
            float dp_val = acc_dP[r];
            float ds_val = p_val * (dp_val - smem_D[lm]) * scale;
            
            // Write P^T to SMEM (Row=ln, Col=lm)
            int p_row = ln;
            int p_col = lm;
            int p_chunk_x = p_col / 8;
            int p_offset = p_col % 8;
            int p_swizzled_chunk_x = (p_row % 8) ^ p_chunk_x;
            int p_swizzled_c = p_swizzled_chunk_x * 8 + p_offset;
            smem_P[p_row * 64 + p_swizzled_c] = __float2bfloat16(p_val);
            
            // Write dS^T to SMEM (Row=ln, Col=lm)
            int ds_row = ln;
            int ds_col = lm;
            int ds_chunk_x = ds_col / 8;
            int ds_offset = ds_col % 8;
            int ds_swizzled_chunk_x = (ds_row % 8) ^ ds_chunk_x;
            int ds_swizzled_c = ds_swizzled_chunk_x * 8 + ds_offset;
            smem_dS[ds_row * 64 + ds_swizzled_c] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        wgmma_fence_fn();
        
        // dV += P^T @ dO AND dK += dS^T @ Q 
        for (int k = 0; k < 4; k++) {
            uint64_t desc_P = make_wgmma_desc(smem_P + k * 16, 1, 1024, 1);
            uint64_t desc_dO0 = make_wgmma_desc(smem_dO0 + k * 1024, 8192, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 1>(acc_dV0, desc_P, desc_dO0);
            
            uint64_t desc_dO1 = make_wgmma_desc(smem_dO1 + k * 1024, 8192, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 1>(acc_dV1, desc_P, desc_dO1);
            
            uint64_t desc_dS = make_wgmma_desc(smem_dS + k * 16, 1, 1024, 1);
            uint64_t desc_Q0 = make_wgmma_desc(smem_Q0 + k * 1024, 8192, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 1>(acc_dK0, desc_dS, desc_Q0);
            
            uint64_t desc_Q1 = make_wgmma_desc(smem_Q1 + k * 1024, 8192, 1024, 1);
            wgmma_m64n64k16_ss_fn<0, 1>(acc_dK1, desc_dS, desc_Q1);
        }
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        
        __syncthreads();
    }

    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        int gm = bn * 64 + lm;
        if (gm < S) {
            int idx = bh_idx * S * 128 + gm * 128 + ln;
            dK_gmem[idx] = __float2bfloat16(acc_dK0[r]);
            dK_gmem[bh_idx * S * 128 + gm * 128 + 64 + ln] = __float2bfloat16(acc_dK1[r]);
            
            dV_gmem[idx] = __float2bfloat16(acc_dV0[r]);
            dV_gmem[bh_idx * S * 128 + gm * 128 + 64 + ln] = __float2bfloat16(acc_dV1[r]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int grid_dQ = (S + 63) / 64;
    dim3 block_dQ(128);
    cudaLaunchConfig_t config_dQ = {};
    config_dQ.gridDim = dim3(grid_dQ, B * H);
    config_dQ.blockDim = block_dQ;
    config_dQ.dynamicSmemBytes = 98304; // 96KB
    config_dQ.stream = stream;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_mha_bwd_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));

    CUDA_CHECK(cudaLaunchKernelEx(&config_dQ, kernel_mha_bwd_dQ, tma_Q, tma_K, tma_V, tma_O, tma_dO, static_cast<const float*>(L.data_ptr()), static_cast<__nv_bfloat16*>(dQ.data_ptr()), S, H, B));

    int grid_dK_dV = (S + 63) / 64;
    dim3 block_dK_dV(128);
    cudaLaunchConfig_t config_dK_dV = {};
    config_dK_dV.gridDim = dim3(grid_dK_dV, B * H);
    config_dK_dV.blockDim = block_dK_dV;
    config_dK_dV.dynamicSmemBytes = 102400; // 100KB
    config_dK_dV.stream = stream;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_mha_bwd_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config_dK_dV, kernel_mha_bwd_dK_dV, tma_Q, tma_K, tma_V, tma_O, tma_dO, static_cast<const float*>(L.data_ptr()), static_cast<__nv_bfloat16*>(dK.data_ptr()), static_cast<__nv_bfloat16*>(dV.data_ptr()), S, H, B));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_cuda_mha::run);

}