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


__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
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


__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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
__global__ __launch_bounds__(256, 1) void gemm_v3_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, float* C, int32_t M, int32_t N, int32_t K) {
    extern __shared__ __align__(1024) uint8_t __INTERNAL_DYN_SHMEM__[];
    /* align "dynamic_shared" to 1024 */;
    __nv_bfloat16* A_smem[3];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 0); /*size = 8192 bytes*/;
    A_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 8192); /*size = 8192 bytes*/;
    A_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 16384); /*size = 8192 bytes*/;
    __nv_bfloat16* B_smem[3];
    B_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 24576); /*size = 8192 bytes*/;
    B_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 32768); /*size = 8192 bytes*/;
    B_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 40960); /*size = 8192 bytes*/;
    uint64_t* full_barriers[3];
    full_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 49152); /*size = 8 bytes*/;
    full_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 49160); /*size = 8 bytes*/;
    full_barriers[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 49168); /*size = 8 bytes*/;
    uint64_t* empty_barriers[3];
    empty_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 49176); /*size = 8 bytes*/;
    empty_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 49184); /*size = 8 bytes*/;
    empty_barriers[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 49192); /*size = 8 bytes*/;
    int32_t tid = threadIdx.x;
    int32_t wg = (tid / 128);
    int32_t ltid = (tid % 128);
    int32_t lane = (tid % 32);
    int32_t bm = (blockIdx.y * 64);
    int32_t bn = (blockIdx.x * 64);
    int32_t nk = (((K + 64) - 1) / 64);
    if ((tid == 0)) {
        #pragma unroll
        for (int32_t s = 0; s < 3; s += 1) {
            init_smem_barrier_fn(full_barriers[s], 1);
            init_smem_barrier_fn(empty_barriers[s], 4);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();
    int32_t c_stage;
    int32_t c_k;
    uint32_t base_desc_a;
    uint64_t db;
    int32_t p_phase;
    int32_t p_stage;
    uint32_t base_desc_b;
    int32_t k_off;
    uint64_t da;
    uint32_t b_s_off;
    int32_t c_phase;
    uint32_t a_s_off;
    int32_t p_k;
    if ((wg == 0)) {
        if ((tid == 0)) {
            p_stage = 0;
            p_phase = 0;
            p_k = 0;
            while ((p_k < nk)) {
                if ((p_k >= 3)) {
                    mbarrier_wait_fn(empty_barriers[p_stage], (p_phase ^ 1));
                }
                mbarrier_arrive_and_expect_tx_fn(full_barriers[p_stage], 16384);
                tma_load_2d_fn(&dA, full_barriers[p_stage], A_smem[p_stage], (p_k * 64), bm);
                tma_load_2d_fn(&dB, full_barriers[p_stage], B_smem[p_stage], (p_k * 64), bn);
                p_stage = (p_stage + 1);
                if ((p_stage == 3)) {
                    p_stage = 0;
                    p_phase = (p_phase ^ 1);
                }
                p_k = (p_k + 1);
            }
        }
    }
    else {
        float acc64[32];
        for(int _i=0; _i<32; _i++) acc64[_i]=0.0f;
        base_desc_a = ((uint32_t)__cvta_generic_to_shared(A_smem[0]) >> 4);
        base_desc_b = ((uint32_t)__cvta_generic_to_shared(B_smem[0]) >> 4);
        c_stage = 0;
        c_phase = 0;
        c_k = 0;
        while ((c_k < nk)) {
            mbarrier_wait_fn(full_barriers[c_stage], c_phase);
            __syncwarp();
            wgmma_fence_acc64_fn(acc64, 32);
            wgmma_fence_fn();
            a_s_off = (static_cast<uint32_t>(c_stage) * 512);
            b_s_off = (static_cast<uint32_t>(c_stage) * 512);
            #pragma unroll
            for (int32_t ki = 0; ki < 64; ki += 16) {
                k_off = ((ki * 2) / 16);
                da = (static_cast<uint64_t>((((base_desc_a + a_s_off) + k_off) & 16383)) | 4611686293305294848);
                db = (static_cast<uint64_t>((((base_desc_b + b_s_off) + k_off) & 16383)) | 4611686293305294848);
                wgmma_m64n64k16_fn_(acc64, da, db);
            }
            wgmma_commit_fn();
            wgmma_fence_acc64_fn(acc64, 32);
            wgmma_wait_fn();
            if ((lane == 0)) {
                mbarrier_arrive_fn(empty_barriers[c_stage]);
            }
            __syncwarp();
            c_stage = (c_stage + 1);
            if ((c_stage == 3)) {
                c_stage = 0;
                c_phase = (c_phase ^ 1);
            }
            c_k = (c_k + 1);
        }
        store_acc64_global_f32_fn(C, acc64, bm, bn, M, N, ltid);
    }
}


}
