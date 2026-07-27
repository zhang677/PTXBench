/*start of generated code*/
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>

using TmaDescriptor = CUtensorMap;




__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}


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


__device__ __forceinline__ void warpgroup_reg_dealloc_48_fn() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;\n");
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


__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
           "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}


__device__ __forceinline__ void warpgroup_reg_alloc_232_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
}


__device__ __forceinline__ void wgmma_fence_4acc_fn(float* a0, float* a1, float* a2, float* a3, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a0[i]), "+f"(a1[i]), "+f"(a2[i]), "+f"(a3[i]) :: "memory");
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


__device__ __forceinline__ void mbarrier_arrive_remote_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t smem_addr = __cvta_generic_to_shared(&bar[0]);
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote_addr) : "memory");
}


__device__ __forceinline__ void get_coord_4ac_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}
__device__ __forceinline__ void store_4acc_f32_fn(
    float* C, float* a00, float* a01, float* a10, float* a11,
    int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord_4ac_(tid, r, lm, ln);
        if (bm + lm < M && bn + ln < N)
            C[(int64_t)(bm + lm) * N + bn + ln] = a00[r];
        if (bm + lm < M && bn + 64 + ln < N)
            C[(int64_t)(bm + lm) * N + bn + 64 + ln] = a01[r];
        if (bm + 64 + lm < M && bn + ln < N)
            C[(int64_t)(bm + 64 + lm) * N + bn + ln] = a10[r];
        if (bm + 64 + lm < M && bn + 64 + ln < N)
            C[(int64_t)(bm + 64 + lm) * N + bn + 64 + ln] = a11[r];
    }
}


extern "C" {
__global__ __launch_bounds__(256, 1) void gemm_v5_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, float* C, int32_t M, int32_t N, int32_t K) {
    extern __shared__ __align__(1024) uint8_t __INTERNAL_DYN_SHMEM__[];
    /* align "dynamic_shared" to 1024 */;
    __nv_bfloat16* A_smem[2];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 0); /*size = 16384 bytes*/;
    A_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 16384); /*size = 16384 bytes*/;
    __nv_bfloat16* B_smem[2];
    B_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 32768); /*size = 16384 bytes*/;
    B_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 49152); /*size = 16384 bytes*/;
    uint64_t* full_barriers[2];
    full_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 65536); /*size = 8 bytes*/;
    full_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 65544); /*size = 8 bytes*/;
    uint64_t* empty_barriers[2];
    empty_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 65552); /*size = 8 bytes*/;
    empty_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 65560); /*size = 8 bytes*/;
    int32_t tid = threadIdx.x;
    int32_t wg = (tid / 128);
    int32_t ltid = (tid % 128);
    int32_t lane = (tid % 32);
    uint32_t cta = cluster_rank_fn();
    int32_t n_tile = (blockIdx.x / 2);
    int32_t bn = (n_tile * 128);
    int32_t cluster_m = (blockIdx.y * 256);
    int32_t bm = (cluster_m + (static_cast<int32_t>(cta) * 128));
    int32_t nk = (((K + 64) - 1) / 64);
    if ((tid == 0)) {
        #pragma unroll
        for (int32_t s = 0; s < 2; s += 1) {
            init_smem_barrier_fn(full_barriers[s], 1);
            init_smem_barrier_fn(empty_barriers[s], 8);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();
    int32_t c_stage;
    int32_t c_k;
    uint32_t base_desc_a;
    uint64_t da1;
    int32_t p_phase;
    uint64_t da0;
    int32_t p_stage;
    uint32_t base_desc_b;
    int32_t k_off;
    uint64_t db0;
    uint64_t db1;
    uint32_t b_s_off;
    int32_t c_phase;
    uint32_t a_s_off;
    int32_t p_k;
    if ((wg == 0)) {
        warpgroup_reg_dealloc_48_fn();
        if ((tid == 0)) {
            p_stage = 0;
            p_phase = 0;
            p_k = 0;
            while ((p_k < nk)) {
                if ((p_k >= 2)) {
                    mbarrier_wait_fn(empty_barriers[p_stage], (p_phase ^ 1));
                }
                #pragma unroll
                for (int32_t ma = 0; ma < 2; ma += 1) {
                    tma_load_2d_fn(&dA, full_barriers[p_stage], (A_smem[p_stage] + ((ma * 64) * 64)), (p_k * 64), (bm + (ma * 64)));
                }
                if ((cta == 0)) {
                    #pragma unroll
                    for (int32_t nb = 0; nb < 2; nb += 1) {
                        tma_load_multicast_2d_fn(&dB, full_barriers[p_stage], (B_smem[p_stage] + ((nb * 64) * 64)), (p_k * 64), (bn + (nb * 64)), 3);
                    }
                }
                mbarrier_arrive_and_expect_tx_fn(full_barriers[p_stage], 32768);
                p_stage = (p_stage + 1);
                if ((p_stage == 2)) {
                    p_stage = 0;
                    p_phase = (p_phase ^ 1);
                }
                p_k = (p_k + 1);
            }
        }
    }
    else {
        warpgroup_reg_alloc_232_fn();
        float acc_00[32];
        float acc_01[32];
        float acc_10[32];
        float acc_11[32];
        for(int _i=0;_i<32;_i++){acc_00[_i]=0.0f;acc_01[_i]=0.0f;acc_10[_i]=0.0f;acc_11[_i]=0.0f;};
        base_desc_a = ((uint32_t)__cvta_generic_to_shared(A_smem[0]) >> 4);
        base_desc_b = ((uint32_t)__cvta_generic_to_shared(B_smem[0]) >> 4);
        c_stage = 0;
        c_phase = 0;
        c_k = 0;
        while ((c_k < nk)) {
            mbarrier_wait_fn(full_barriers[c_stage], c_phase);
            __syncwarp();
            wgmma_fence_4acc_fn(acc_00, acc_01, acc_10, acc_11, 32);
            wgmma_fence_fn();
            a_s_off = (static_cast<uint32_t>(c_stage) * 1024);
            b_s_off = (static_cast<uint32_t>(c_stage) * 1024);
            #pragma unroll
            for (int32_t ki = 0; ki < 64; ki += 16) {
                k_off = ((ki * 2) / 16);
                da0 = (static_cast<uint64_t>((((base_desc_a + a_s_off) + k_off) & 16383)) | 4611686293305294848);
                db0 = (static_cast<uint64_t>((((base_desc_b + b_s_off) + k_off) & 16383)) | 4611686293305294848);
                da1 = (static_cast<uint64_t>(((((base_desc_a + a_s_off) + 512) + k_off) & 16383)) | 4611686293305294848);
                db1 = (static_cast<uint64_t>(((((base_desc_b + b_s_off) + 512) + k_off) & 16383)) | 4611686293305294848);
                wgmma_m64n64k16_fn_(acc_00, da0, db0);
                wgmma_m64n64k16_fn_(acc_01, da0, db1);
                wgmma_m64n64k16_fn_(acc_10, da1, db0);
                wgmma_m64n64k16_fn_(acc_11, da1, db1);
            }
            wgmma_commit_fn();
            wgmma_fence_4acc_fn(acc_00, acc_01, acc_10, acc_11, 32);
            wgmma_wait_fn();
            if ((lane < 2)) {
                mbarrier_arrive_remote_fn(empty_barriers[c_stage], lane);
            }
            __syncwarp();
            c_stage = (c_stage + 1);
            if ((c_stage == 2)) {
                c_stage = 0;
                c_phase = (c_phase ^ 1);
            }
            c_k = (c_k + 1);
        }
        store_4acc_f32_fn(C, acc_00, acc_01, acc_10, acc_11, bm, bn, M, N, ltid);
    }
}


}
