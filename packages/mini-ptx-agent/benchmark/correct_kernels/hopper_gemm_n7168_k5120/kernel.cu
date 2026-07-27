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

namespace gemm_cuda {

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
    uint32_t parity = phase & 1;
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(parity));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
           "r"(c0), "r"(c1) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ static inline uint64_t make_wgmma_desc(void* ptr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle_mode) {
    uint64_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t desc = 0;
    desc |= (addr >> 4) & 0x3FFF;
    desc |= ((uint64_t)(lbo_bytes >> 4) << 16);
    desc |= ((uint64_t)(sbo_bytes >> 4) << 32);
    desc |= ((uint64_t)swizzle_mode << 62);
    return desc;
}

__device__ __forceinline__ void wgmma_fence_operand_array_fn(float* a, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a[i]) :: "memory");
    }
}

__device__ __forceinline__ void wgmma_fence_fn() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_fn() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_fn() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n256k16_fn(float* c, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        "%128,%129,p,1,1,0,0;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]),"+f"(c[32]),"+f"(c[33]),"+f"(c[34]),"+f"(c[35]),"+f"(c[36]),"+f"(c[37]),"+f"(c[38]),"+f"(c[39]),"+f"(c[40]),"+f"(c[41]),"+f"(c[42]),"+f"(c[43]),"+f"(c[44]),"+f"(c[45]),"+f"(c[46]),"+f"(c[47]),"+f"(c[48]),"+f"(c[49]),"+f"(c[50]),"+f"(c[51]),"+f"(c[52]),"+f"(c[53]),"+f"(c[54]),"+f"(c[55]),"+f"(c[56]),"+f"(c[57]),"+f"(c[58]),"+f"(c[59]),"+f"(c[60]),"+f"(c[61]),"+f"(c[62]),"+f"(c[63]),"+f"(c[64]),"+f"(c[65]),"+f"(c[66]),"+f"(c[67]),"+f"(c[68]),"+f"(c[69]),"+f"(c[70]),"+f"(c[71]),"+f"(c[72]),"+f"(c[73]),"+f"(c[74]),"+f"(c[75]),"+f"(c[76]),"+f"(c[77]),"+f"(c[78]),"+f"(c[79]),"+f"(c[80]),"+f"(c[81]),"+f"(c[82]),"+f"(c[83]),"+f"(c[84]),"+f"(c[85]),"+f"(c[86]),"+f"(c[87]),"+f"(c[88]),"+f"(c[89]),"+f"(c[90]),"+f"(c[91]),"+f"(c[92]),"+f"(c[93]),"+f"(c[94]),"+f"(c[95]),"+f"(c[96]),"+f"(c[97]),"+f"(c[98]),"+f"(c[99]),"+f"(c[100]),"+f"(c[101]),"+f"(c[102]),"+f"(c[103]),"+f"(c[104]),"+f"(c[105]),"+f"(c[106]),"+f"(c[107]),"+f"(c[108]),"+f"(c[109]),"+f"(c[110]),"+f"(c[111]),"+f"(c[112]),"+f"(c[113]),"+f"(c[114]),"+f"(c[115]),"+f"(c[116]),"+f"(c[117]),"+f"(c[118]),"+f"(c[119]),"+f"(c[120]),"+f"(c[121]),"+f"(c[122]),"+f"(c[123]),"+f"(c[124]),"+f"(c[125]),"+f"(c[126]),"+f"(c[127])
        : "l"(da), "l"(db));
}

__device__ __forceinline__ void store_acc_smem_bf16_n256_swizzled_fn(
    __nv_bfloat16* sC, float* ac, int ltid, int row_offset) {
    int warp = ltid >> 5;
    int lane_id = ltid & 31;
    int row0 = row_offset + warp * 16 + (lane_id >> 2);
    int row1 = row0 + 8;
    int col_base = (lane_id & 3) * 2;
    int swizzle_mask = row0 % 8;
    
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int chunk = i; 
        int chunk_swizzled = chunk ^ swizzle_mask;
        int col_swizzled = chunk_swizzled * 8 + col_base;
        
        __nv_bfloat162 val0 = __floats2bfloat162_rn(ac[i * 4 + 0], ac[i * 4 + 1]);
        *( (__nv_bfloat162*)&sC[row0 * 256 + col_swizzled] ) = val0;
        
        __nv_bfloat162 val1 = __floats2bfloat162_rn(ac[i * 4 + 2], ac[i * 4 + 3]);
        *( (__nv_bfloat162*)&sC[row1 * 256 + col_swizzled] ) = val1;
    }
}

union SharedUnion {
    struct {
        alignas(8) uint64_t mbar_A[4];
        alignas(8) uint64_t mbar_B[4];
        alignas(128) __nv_bfloat16 A[4][128 * 64];
        alignas(128) __nv_bfloat16 B[4][256 * 64];
    } in;
    struct {
        alignas(128) __nv_bfloat16 C[128 * 256];
    } out;
};

extern __shared__ __align__(128) uint8_t smem_buf[];

__global__ void __launch_bounds__(256, 1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    setmaxnreg_inc_sync_fn<240>();

    SharedUnion& smem = *(SharedUnion*)smem_buf;

    int tid = threadIdx.x;
    int bm = blockIdx.x * 128;
    int bn = blockIdx.y * 256;

    if (tid == 0) {
        for (int i = 0; i < 4; i++) {
            init_smem_barrier_fn(&smem.in.mbar_A[i], 1);
            init_smem_barrier_fn(&smem.in.mbar_B[i], 1);
        }
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    int K_ITERS = K / 64;

    for (int i = 0; i < 3 && i < K_ITERS; i++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.in.mbar_A[i], 128 * 64 * 2);
            tma_load_2d_fn(&tma_A, &smem.in.mbar_A[i], smem.in.A[i], i * 64, bm);

            mbarrier_arrive_and_expect_tx_fn(&smem.in.mbar_B[i], 256 * 64 * 2);
            tma_load_2d_fn(&tma_B, &smem.in.mbar_B[i], smem.in.B[i], i * 64, bn);
        }
    }

    float acc[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        acc[i] = 0.0f;
    }

    int wg_idx = tid / 128;

    for (int step = 0; step < K_ITERS; step++) {
        int stage = step % 4;
        
        mbarrier_wait_fn(&smem.in.mbar_A[stage], step / 4);
        mbarrier_wait_fn(&smem.in.mbar_B[stage], step / 4);
        
        fence_async_shared_fn();
        __syncthreads();
        
        wgmma_fence_operand_array_fn(acc, 128);
        wgmma_fence_fn();

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            void* ptr_A = smem.in.A[stage];
            void* ptr_B = smem.in.B[stage];
            
            ptr_A = (uint8_t*)ptr_A + wg_idx * 8192 + k * 32;
            ptr_B = (uint8_t*)ptr_B + k * 32;
            
            uint64_t desc_A = make_wgmma_desc(ptr_A, 0, 1024, 1);
            uint64_t desc_B = make_wgmma_desc(ptr_B, 0, 1024, 1);
            
            wgmma_m64n256k16_fn(acc, desc_A, desc_B);
        }
        wgmma_commit_fn();
        
        int load_step = step + 3;
        if (load_step < K_ITERS) {
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
            __syncthreads();

            if (tid == 0) {
                int next_stage = load_step % 4;
                mbarrier_arrive_and_expect_tx_fn(&smem.in.mbar_A[next_stage], 128 * 64 * 2);
                tma_load_2d_fn(&tma_A, &smem.in.mbar_A[next_stage], smem.in.A[next_stage], load_step * 64, bm);

                mbarrier_arrive_and_expect_tx_fn(&smem.in.mbar_B[next_stage], 256 * 64 * 2);
                tma_load_2d_fn(&tma_B, &smem.in.mbar_B[next_stage], smem.in.B[next_stage], load_step * 64, bn);
            }
        }
    }
    
    wgmma_wait_fn();
    __syncthreads();

    int ltid = tid % 128;
    store_acc_smem_bf16_n256_swizzled_fn(smem.out.C, acc, ltid, wg_idx * 64);
    
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = tid + i * 256;
        int row = idx / 32;
        int col_f4 = idx % 32;
        int col = col_f4 * 8; 
        
        int g_row = bm + row;
        int g_col = bn + col;
        
        if (g_row < M && g_col < N) {
            int chunk_swizzled = col_f4 ^ (row % 8);
            int col_swizzled = chunk_swizzled * 8;
            
            float4 val = *(float4*)&smem.out.C[row * 256 + col_swizzled];
            *(float4*)&C[(int64_t)g_row * N + g_col] = val;
        }
    }
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    if (M == 0 || N == 0 || K == 0) return;

    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    
    create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    create_tma_2d_descriptor_2B(&tma_B, b_ptr, K, N, 64, 256, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int64_t threads = 256;
    
    int grid_x = (M + 127) / 128;
    int grid_y = (N + 255) / 256;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x, grid_y, 1);
    config.blockDim = dim3(threads, 1, 1);
    
    int smem_size = sizeof(gemm_cuda::SharedUnion);
    config.dynamicSmemBytes = smem_size;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(
        gemm_cuda::gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
        
    CUDA_CHECK(cudaFuncSetAttribute(
        gemm_cuda::gemm_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_cuda::gemm_kernel, tma_A, tma_B, c_ptr, M, N, K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

} // namespace gemm_cuda