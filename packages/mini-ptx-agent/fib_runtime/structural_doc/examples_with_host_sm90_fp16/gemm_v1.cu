#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using TmaDescriptor = CUtensorMap;

#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                cudaGetErrorString(_e), __FILE__, __LINE__);         \
        exit(1);                                                     \
    }                                                                \
} while(0)

#define CU_CHECK(call) do {                                          \
    CUresult _e = (call);                                            \
    if (_e != CUDA_SUCCESS) {                                        \
        const char* _s = nullptr;                                    \
        cuGetErrorString(_e, &_s);                                   \
        fprintf(stderr, "CUDA driver error %s at %s:%d\n",          \
                _s ? _s : "unknown", __FILE__, __LINE__);            \
        exit(1);                                                     \
    }                                                                \
} while(0)

static void create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle,
    CUtensorMapFloatOOBfill oobFill) {
  uint64_t globalDim[2]     = {gmem_inner_dim, gmem_outer_dim};
  uint64_t globalStrides[1] = {gmem_inner_dim * 2};
  uint32_t boxDim[2]        = {smem_inner_dim, smem_outer_dim};
  uint32_t elementStrides[2]= {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      d, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, globalAddress,
      globalDim, globalStrides, boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, oobFill));
}

namespace tvm_ffi_gemm_v1_kernel {

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}


__device__ __forceinline__ void wgmma_fence_acc64_fn(float* a, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a[i]) :: "memory");
    }
}


__device__ __forceinline__ void wgmma_fence_fn() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}


__device__ __forceinline__ void wgmma_m64n64k16_fn_(float* c, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,0,0;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
        : "l"(da), "l"(db));
}


__device__ __forceinline__ void wgmma_commit_fn() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}


__device__ __forceinline__ void wgmma_wait_fn() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}


__device__ __forceinline__ void get_coord_64x64_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}
__device__ __forceinline__ void store_acc64_global_f32_fn(
    half* C, float* ac, int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord_64x64_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = __float2half(ac[r]);
    }
}


__global__ __launch_bounds__(128, 1) void gemm_v1_kernel(half* A, half* B, half* C, int32_t M, int32_t N, int32_t K) {
    __shared__ half sA[4096];
    __shared__ half sB[4096];
    int32_t tid = threadIdx.x;
    int32_t bm = (blockIdx.y * 64);
    int32_t bn = (blockIdx.x * 64);
    float acc64[32];
    for(int _i=0; _i<32; _i++) acc64[_i]=0.0f;
    uint32_t base_desc_a = ((uint32_t)__cvta_generic_to_shared(sA) >> 4);
    uint32_t base_desc_b = ((uint32_t)__cvta_generic_to_shared(sB) >> 4);
    int32_t k_base = 0;
    while ((k_base < K)) {
        int32_t i = tid;
        while ((i < 4096)) {
            int32_t m_local = (i / 64);
            int32_t k_local = (i % 64);
            int32_t gm = (bm + m_local);
            int32_t gk = (k_base + k_local);
            int32_t k_swizzled = (k_local ^ ((m_local & 7) * 8));
            int32_t smem_idx = ((m_local * 64) + k_swizzled);
            if (((gm < M) && (gk < K))) {
                sA[smem_idx] = A[((gm * K) + gk)];
            }
            else {
                sA[smem_idx] = static_cast<half>(0);
            }
            i = (i + 128);
        }
        int32_t j = tid;
        while ((j < 4096)) {
            int32_t kb = (j / 64);
            int32_t nb = (j % 64);
            int32_t gkb = (k_base + kb);
            int32_t gnb = (bn + nb);
            int32_t ks = (kb ^ ((nb & 7) * 8));
            int32_t si = ((nb * 64) + ks);
            if (((gkb < K) && (gnb < N))) {
                sB[si] = B[((gnb * K) + gkb)];
            }
            else {
                sB[si] = static_cast<half>(0);
            }
            j = (j + 128);
        }
        __syncthreads();
        fence_proxy_async_fn();
        __syncwarp();
        uint64_t da;
        uint64_t db;
        #pragma unroll
        for (int32_t ki = 0; ki < 64; ki += 16) {
            wgmma_fence_acc64_fn(acc64, 32);
            wgmma_fence_fn();
            da = (static_cast<uint64_t>(((base_desc_a + ((ki * 2) / 16)) & 16383)) | 4611686293305294848);
            db = (static_cast<uint64_t>(((base_desc_b + ((ki * 2) / 16)) & 16383)) | 4611686293305294848);
            wgmma_m64n64k16_fn_(acc64, da, db);
            wgmma_commit_fn();
            wgmma_fence_acc64_fn(acc64, 32);
            wgmma_wait_fn();
        }
        k_base = (k_base + 64);
    }
    store_acc64_global_f32_fn(C, acc64, bm, bn, M, N, tid);
}




void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(0));
  half* a = static_cast<half*>(A.data_ptr());
  half* b = static_cast<half*>(B.data_ptr());
  half* c = static_cast<half*>(C.data_ptr());
  dim3 grid((64 - 1 + N) / 64, (64 - 1 + M) / 64, 1);
  dim3 block(128, 1, 1);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  gemm_v1_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_v1_kernel::run);

}  // namespace tvm_ffi_gemm_v1_kernel
