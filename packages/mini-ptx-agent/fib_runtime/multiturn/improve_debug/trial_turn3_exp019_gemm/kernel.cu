#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

namespace tvm_ffi_gemm {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
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
    desc |= ((uint64_t)swizzle_mode << 62); 
    uint64_t base_offset = (addr >> 7) & 0x7;
    desc |= (base_offset << 49);
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
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

__global__ void __launch_bounds__(128, 1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M,
    int N
) {
    int bm = blockIdx.x;
    int bn = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_A = (__nv_bfloat16*)smem_pool;          
    __nv_bfloat16* smem_B = (__nv_bfloat16*)(smem_pool + 8192); 
    uint64_t* mbar = (uint64_t*)(smem_pool + 16384);

    if (tid == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
    }

    // Initial Stage 0 Load
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        tma_load_2d_fn(&tma_A, &mbar[0], smem_A, 0, bm * 64);
        tma_load_2d_fn(&tma_B, &mbar[0], smem_B, 0, bn * 64);
    }

    // Pipelined WGMMA Loop (Process chunks of 64 K)
    for (int k = 0; k < 80; k++) {
        int stage = k % 2;
        mbarrier_wait_fn(&mbar[stage], k % 2); 
        
        fence_proxy_async_fn();
        __syncwarp();
        wgmma_fence_fn();
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Both Descriptors Interpret Contiguous Inner Dimension Correctly
            uint64_t dA = make_wgmma_desc(smem_A + i * 16, 1, 1024, 1);
            uint64_t dB = make_wgmma_desc(smem_B + i * 16, 1, 1024, 1); 
            
            wgmma_m64n64k16_ss_fn<0, 0>(acc, dA, dB);
        }
        
        wgmma_commit_fn();
        wgmma_wait_fn<0>();
        __syncthreads();
        
        // Async Reload Next Stage Safely Post Consumer Flush
        if (k + 1 < 80) {
            int next_stage = (stage + 1) % 2;
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[next_stage], 16384);
                tma_load_2d_fn(&tma_A, &mbar[next_stage], smem_A, (k + 1) * 64, bm * 64);
                tma_load_2d_fn(&tma_B, &mbar[next_stage], smem_B, (k + 1) * 64, bn * 64);
            }
        }
    }
    
    // Direct Register-to-Gmem Writeout Mapping Expander 
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        
        int gm = bm * 64 + lm;
        int gn = bn * 64 + ln;
        
        if (gm < M && gn < N) {
            C[(int64_t)gm * N + gn] = __float2bfloat16(acc[r]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M_val = A.size(0);
    int64_t N_val = B.size(0);
    
    const __nv_bfloat16* a_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* b_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    CUresult res;
    
    // Align inner contiguous dimensions perfectly matching physical host row major linearization
    res = create_tma_2d_descriptor_2B(&tma_A, (void*)a_ptr, 5120, M_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA A error\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_B, (void*)b_ptr, 5120, N_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA B error\n"); exit(1); }

    dim3 grid((M_val + 63) / 64, N_val / 64);
    dim3 block(128);
    int smem_size = 16400; 

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    gemm_kernel<<<grid, block, smem_size, stream>>>(tma_A, tma_B, c_ptr, M_val, N_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

} // namespace tvm_ffi_gemm
