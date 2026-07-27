#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

// -------------------------------------------------------------------------
// Helper functions
// -------------------------------------------------------------------------

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_local_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo, uint32_t k_offset_bytes) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + k_offset_bytes;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, 
    CUtensorMapFloatOOBfill oobFill) 
{
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        dataType,
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

// -------------------------------------------------------------------------
// Kernel
// -------------------------------------------------------------------------

constexpr int STAGE_COUNT = 3;

__global__ void gemm_n7168_k5120_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    int rank = cluster_rank_fn(); 
    int block_m = blockIdx.x / 2;
    int block_n = blockIdx.y;
    
    __shared__ __align__(1024) uint8_t smem_A[STAGE_COUNT][16384];
    __shared__ __align__(1024) uint8_t smem_B[STAGE_COUNT][16384];
    __shared__ uint64_t mbar_load[STAGE_COUNT];
    __shared__ uint64_t mbar_mma[STAGE_COUNT];
    __shared__ uint32_t tmem_addr_smem;
    
    if (threadIdx.x == 0) {
        for(int i = 0; i < STAGE_COUNT; ++i) {
            init_smem_barrier_fn(&mbar_load[i], 1);
            init_smem_barrier_fn(&mbar_mma[i], 1);
        }
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();
    
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr_smem, 64); // 64 cols per CTA -> 128 cols combined
    }
    __syncthreads();
    uint32_t tmem_addr = tmem_addr_smem;
    
    int num_steps = K / 64;
    
    for (int step = 0; step < num_steps; ++step) {
        int s_load = step % STAGE_COUNT;

        // 1. Wait for mbar_mma if we are going to overwrite this stage
        if (step >= STAGE_COUNT && threadIdx.x == 0) {
            mbarrier_wait_fn(&mbar_mma[s_load], (step / STAGE_COUNT - 1) & 1);
        }
        __syncthreads();

        // 2. Issue TMA load (each CTA loads full A and its N-half of B)
        if (threadIdx.x == 0) {
            uint32_t tx_bytes = 16384 + 8192;
            mbarrier_arrive_and_expect_tx_fn(&mbar_load[s_load], tx_bytes);
            tma_load_2d_fn(&tma_A, &mbar_load[s_load], smem_A[s_load], step * 64, block_m * 128);
            tma_load_2d_fn(&tma_B, &mbar_load[s_load], smem_B[s_load], step * 64, block_n * 128 + rank * 64);
        }

        // 3. Issue MMA for the PREVIOUS stage to hide memory latency
        if (step >= 1) {
            int s_mma = (step - 1) % STAGE_COUNT;
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&mbar_load[s_mma], ((step - 1) / STAGE_COUNT) & 1);

                uint32_t idesc = make_instr_desc_fn(128, 64);
                for (int k_step = 0; k_step < 4; ++k_step) {
                    uint32_t k_off = k_step * 32;
                    uint64_t cur_desc_A = make_smem_desc_sm100_fn(smem_A[s_mma], 1024, k_off);
                    uint64_t cur_desc_B = make_smem_desc_sm100_fn(smem_B[s_mma], 1024, k_off);
                    uint32_t accum = (step - 1 > 0 || k_step > 0) ? 1 : 0;
                    umma_f16_cg2_fn(tmem_addr, cur_desc_A, cur_desc_B, idesc, accum);
                }
                umma_commit_local_fn(&mbar_mma[s_mma]);
            }
        }
    }

    // Drain the last MMA stage
    if (num_steps > 0) {
        int s_mma = (num_steps - 1) % STAGE_COUNT;
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&mbar_load[s_mma], ((num_steps - 1) / STAGE_COUNT) & 1);

            uint32_t idesc = make_instr_desc_fn(128, 64);
            for (int k_step = 0; k_step < 4; ++k_step) {
                uint32_t k_off = k_step * 32;
                uint64_t cur_desc_A = make_smem_desc_sm100_fn(smem_A[s_mma], 1024, k_off);
                uint64_t cur_desc_B = make_smem_desc_sm100_fn(smem_B[s_mma], 1024, k_off);
                uint32_t accum = (num_steps - 1 > 0 || k_step > 0) ? 1 : 0;
                umma_f16_cg2_fn(tmem_addr, cur_desc_A, cur_desc_B, idesc, accum);
            }
            umma_commit_local_fn(&mbar_mma[s_mma]);
        }

        // Wait for all outstanding MMAs to completely finish
        if (threadIdx.x == 0) {
            for (int i = 0; i < min(STAGE_COUNT, num_steps); ++i) {
                int step_to_wait = num_steps - 1 - i;
                int s_w = step_to_wait % STAGE_COUNT;
                mbarrier_wait_fn(&mbar_mma[s_w], (step_to_wait / STAGE_COUNT) & 1);
            }
        }
    }
    
    // Cluster sync ensures CTA 1 only proceeds when CTA 0's MMAs are completely finished
    cluster_sync_fn();
    
    // Epilogue Phase
    if (num_steps > 0) {
        __nv_bfloat16* smem_out = (__nv_bfloat16*)smem_A[0];
        
        // Load collectively from local TMEM (64 columns) using the full warpgroup
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_addr + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            uint32_t base = threadIdx.x * 64 + col;
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
        }
        __syncthreads();
        
        uint32_t warp_id = threadIdx.x / 32;
        uint32_t lane_id = threadIdx.x % 32;
        
        // Coalesced vectorized write-back to Global C Matrix
        for (uint32_t step = 0; step < 32; ++step) {
            uint32_t row = step * 4 + warp_id;
            if (row >= 128) continue;
            uint32_t global_row = block_m * 128 + row;
            uint32_t col_start = lane_id * 2; 
            uint32_t global_col = block_n * 128 + rank * 64 + col_start;
            
            if (global_row < M && global_col + 1 < N) {
                uint32_t data = *reinterpret_cast<uint32_t*>(&smem_out[row * 64 + col_start]);
                *reinterpret_cast<uint32_t*>(C + (uint64_t)global_row * N + global_col) = data;
            }
        }
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 64);
    }
}

// -------------------------------------------------------------------------
// Host Runner
// -------------------------------------------------------------------------

namespace tvm_ffi_gemm {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id)); 
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_A, A_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_B, B_ptr, K, N, 64, 64,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int grid_x = (M + 127) / 128 * 2;
    int grid_y = (N + 127) / 128;
    
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;       
    config.blockDim = block;
    config.dynamicSmemBytes = 0; 
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_n7168_k5120_kernel, tma_A, tma_B, C_ptr, M, N, K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

} // namespace tvm_ffi_gemm