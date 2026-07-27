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


__device__ __forceinline__ uint32_t elect_one_fn() {
    uint32_t pred;
    asm volatile(
        "{\n.reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n}\n" : "=r"(pred));
    return pred;
}


__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}


__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}


__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}


__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
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


__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
           "r"(c0), "r"(c1) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}


__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}


__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}


__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}


#include <cuda_bf16.h>
__device__ __forceinline__ void tmem_store_bf16_row_fn(
    __nv_bfloat16* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    uint32_t m_idx = m_base + tid;
    if (m_idx >= M) return;
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        uint32_t nc = n_base + col;
        __nv_bfloat16* out = D + (uint64_t)m_idx * N + nc;
        if (nc     < N) out[0] = __float2bfloat16(f0);
        if (nc + 1 < N) out[1] = __float2bfloat16(f1);
        if (nc + 2 < N) out[2] = __float2bfloat16(f2);
        if (nc + 3 < N) out[3] = __float2bfloat16(f3);
    }
}


__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}


extern "C" {
__global__ __launch_bounds__(128, 1) void gemm_level1_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, __nv_bfloat16* D, uint32_t M, uint32_t N, uint32_t K, uint32_t num_n_blocks) {
    extern __shared__ __align__(1024) uint8_t __INTERNAL_DYN_SHMEM__[];
    /* align "dynamic_shared" to 1024 */;
    __nv_bfloat16* A_smem = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 0); /*size = 16384 bytes*/;
    __nv_bfloat16* B_smem = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 16384); /*size = 8192 bytes*/;
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 24576); /*size = 8 bytes*/;
    uint64_t* empty_bar = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 24584); /*size = 8 bytes*/;
    uint64_t* tmem_bar = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 24592); /*size = 8 bytes*/;
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(__INTERNAL_DYN_SHMEM__ + 24600); /*size = 4 bytes*/;
    uint64_t* _pad = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 24604); /*size = 960 bytes*/;
    int32_t tid = threadIdx.x;
    int32_t warp_idx = (tid / 32);
    uint32_t cta = cluster_rank_fn();
    uint32_t cluster_id = (blockIdx.x / 2);
    uint32_t m_block = (cluster_id / num_n_blocks);
    uint32_t n_block = (cluster_id % num_n_blocks);
    uint32_t el0;
    if ((warp_idx == 0)) {
        el0 = elect_one_fn();
        if ((el0 == 1)) {
            prefetch_tma_descriptor_fn(&dA);
            prefetch_tma_descriptor_fn(&dB);
        }
    }
    uint32_t el1;
    if ((warp_idx == 1)) {
        el1 = elect_one_fn();
        if ((el1 == 1)) {
            init_smem_barrier_fn(full_bar, 1);
            init_smem_barrier_fn(empty_bar, 1);
            init_smem_barrier_fn(tmem_bar, 1);
            fence_smem_barrier_init_fn();
        }
    }
    if ((warp_idx == 2)) {
        tmem_alloc_fn(tmem_addr, 128);
    }
    cluster_sync_fn();
    uint32_t base_a = ((uint32_t)__cvta_generic_to_shared(A_smem) >> 4);
    uint32_t base_b = ((uint32_t)__cvta_generic_to_shared(B_smem) >> 4);
    int32_t is_leader = (static_cast<int32_t>(cta) == 0);
    uint32_t num_k_blocks = (((K + 64) - 1) / 64);
    int32_t m_coord = static_cast<int32_t>((m_block * 128));
    int32_t n_coord = static_cast<int32_t>(((n_block * 128) + (cta * 64)));
    uint32_t phase = static_cast<uint32_t>(0);
    uint32_t first_umma = static_cast<uint32_t>(1);
    uint32_t kb = static_cast<uint32_t>(0);
    while ((kb < num_k_blocks)) {
        mbarrier_wait_fn(empty_bar, (phase ^ 1));
        int32_t k_coord;
        uint32_t el_tma;
        if ((warp_idx == 0)) {
            el_tma = elect_one_fn();
            if ((el_tma == 1)) {
                k_coord = static_cast<int32_t>((kb * 64));
                tma_load_2d_fn(&dA, full_bar, A_smem, k_coord, m_coord);
                tma_load_2d_fn(&dB, full_bar, B_smem, k_coord, n_coord);
                mbarrier_arrive_and_expect_tx_fn(full_bar, 24576);
            }
        }
        mbarrier_wait_fn(full_bar, phase);
        cluster_sync_fn();
        int32_t k_off;
        uint64_t ad;
        uint32_t el_umma;
        uint64_t bd;
        uint32_t acc_flag;
        if ((is_leader == 1)) {
            if ((warp_idx == 1)) {
                el_umma = elect_one_fn();
                if ((el_umma == 1)) {
                    tcgen05_fence_after_fn();
                    #pragma unroll
                    for (int32_t ki = 0; ki < 64; ki += 16) {
                        k_off = ((ki * 2) / 16);
                        ad = ((static_cast<uint64_t>((base_a + k_off)) & 16383) | 4611756662049472512);
                        bd = ((static_cast<uint64_t>((base_b + k_off)) & 16383) | 4611756662049472512);
                        if ((ki == 0)) {
                            acc_flag = (static_cast<uint32_t>(1) - first_umma);
                            umma_f16_cg2_fn(0, ad, bd, 270533776, acc_flag);
                        }
                        else {
                            umma_f16_cg2_fn(0, ad, bd, 270533776, static_cast<uint32_t>(1));
                        }
                    }
                    umma_commit_2sm_fn(empty_bar);
                    if ((kb == (num_k_blocks - 1))) {
                        umma_commit_2sm_fn(tmem_bar);
                    }
                }
            }
        }
        first_umma = static_cast<uint32_t>(0);
        phase = (phase ^ 1);
        kb = (kb + 1);
    }
    mbarrier_wait_fn(tmem_bar, 0);
    tcgen05_fence_after_fn();
    uint32_t m_base = (m_block * 128);
    uint32_t n_base = (n_block * 128);
    tmem_store_bf16_row_fn(D, static_cast<uint32_t>(tid), M, N, m_base, n_base, 128);
    cluster_sync_fn();
    if ((warp_idx == 2)) {
        tmem_dealloc_fn(0, 128);
    }
}


}
