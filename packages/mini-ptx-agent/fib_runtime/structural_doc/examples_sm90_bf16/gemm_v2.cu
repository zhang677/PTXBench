/*start of generated code*/
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>

using TmaDescriptor = CUtensorMap;




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


__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
           "r"(c0), "r"(c1) : "memory");
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
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
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
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord_64x64_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}


extern "C" {
__global__ __launch_bounds__(128, 1) void gemm_v2_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, float* C, int32_t M, int32_t N, int32_t K) {
    __shared__ __nv_bfloat16 sA[4096];
    __shared__ __nv_bfloat16 sB[4096];
    __shared__ uint64_t mbar[1];
    int32_t tid = threadIdx.x;
    int32_t bm = (blockIdx.y * 64);
    int32_t bn = (blockIdx.x * 64);
    float acc64[32];
    for(int _i=0; _i<32; _i++) acc64[_i]=0.0f;
    uint32_t base_desc_a = ((uint32_t)__cvta_generic_to_shared(sA) >> 4);
    uint32_t base_desc_b = ((uint32_t)__cvta_generic_to_shared(sB) >> 4);
    if ((tid == 0)) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    int32_t kb = 0;
    while ((kb < K)) {
        if ((tid == 0)) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 16384);
            tma_load_2d_fn(&dA, mbar, sA, kb, bm);
            tma_load_2d_fn(&dB, mbar, sB, kb, bn);
        }
        int32_t phase = ((kb / 64) & 1);
        mbarrier_wait_fn(mbar, phase);
        __syncwarp();
        wgmma_fence_acc64_fn(acc64, 32);
        wgmma_fence_fn();
        uint64_t da;
        uint64_t db;
        #pragma unroll
        for (int32_t ki = 0; ki < 64; ki += 16) {
            da = (static_cast<uint64_t>(((base_desc_a + ((ki * 2) / 16)) & 16383)) | 4611686293305294848);
            db = (static_cast<uint64_t>(((base_desc_b + ((ki * 2) / 16)) & 16383)) | 4611686293305294848);
            wgmma_m64n64k16_fn_(acc64, da, db);
        }
        wgmma_commit_fn();
        wgmma_fence_acc64_fn(acc64, 32);
        wgmma_wait_fn();
        kb = (kb + 64);
    }
    store_acc64_global_f32_fn(C, acc64, bm, bn, M, N, tid);
}


}
