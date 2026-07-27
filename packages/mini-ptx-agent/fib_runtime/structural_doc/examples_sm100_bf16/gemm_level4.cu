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


__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
}


__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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
__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    // Phase 1: TMEM -> SMEM
    // Each of the 128 threads owns one row (tid 0..127 -> row 0..127). Load 4
    // FP32 cols at a time from TMEM, convert to BF16, write to smem_out[tid*BN+col].
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    // Phase 2: SMEM -> Global (coalesced vectorized 8-byte writes)
    // REQUIRES 4 warps: each step processes 4 rows (one per warp). row = step*4 + warp_id.
    // Each warp: 32 lanes write 4 BF16 (8 bytes) each, covering 32*4=128 cols per row.
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
}


__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}


extern "C" {
__global__ __launch_bounds__(128, 1) void gemm_level4_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, __nv_bfloat16* D, uint32_t M, uint32_t N, uint32_t K, uint32_t num_n_blocks) {
    extern __shared__ __align__(1024) uint8_t __INTERNAL_DYN_SHMEM__[];
    /* align "dynamic_shared" to 1024 */;
    __nv_bfloat16* A_smem[8];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 0); /*size = 16384 bytes*/;
    A_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 16384); /*size = 16384 bytes*/;
    A_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 32768); /*size = 16384 bytes*/;
    A_smem[3] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 49152); /*size = 16384 bytes*/;
    A_smem[4] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 65536); /*size = 16384 bytes*/;
    A_smem[5] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 81920); /*size = 16384 bytes*/;
    A_smem[6] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 98304); /*size = 16384 bytes*/;
    A_smem[7] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 114688); /*size = 16384 bytes*/;
    __nv_bfloat16* B_smem[8];
    B_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 131072); /*size = 8192 bytes*/;
    B_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 139264); /*size = 8192 bytes*/;
    B_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 147456); /*size = 8192 bytes*/;
    B_smem[3] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 155648); /*size = 8192 bytes*/;
    B_smem[4] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 163840); /*size = 8192 bytes*/;
    B_smem[5] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 172032); /*size = 8192 bytes*/;
    B_smem[6] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 180224); /*size = 8192 bytes*/;
    B_smem[7] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 188416); /*size = 8192 bytes*/;
    uint64_t* full_bars[8];
    full_bars[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196608); /*size = 8 bytes*/;
    full_bars[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196616); /*size = 8 bytes*/;
    full_bars[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196624); /*size = 8 bytes*/;
    full_bars[3] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196632); /*size = 8 bytes*/;
    full_bars[4] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196640); /*size = 8 bytes*/;
    full_bars[5] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196648); /*size = 8 bytes*/;
    full_bars[6] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196656); /*size = 8 bytes*/;
    full_bars[7] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196664); /*size = 8 bytes*/;
    uint64_t* empty_bars[8];
    empty_bars[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196672); /*size = 8 bytes*/;
    empty_bars[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196680); /*size = 8 bytes*/;
    empty_bars[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196688); /*size = 8 bytes*/;
    empty_bars[3] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196696); /*size = 8 bytes*/;
    empty_bars[4] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196704); /*size = 8 bytes*/;
    empty_bars[5] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196712); /*size = 8 bytes*/;
    empty_bars[6] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196720); /*size = 8 bytes*/;
    empty_bars[7] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196728); /*size = 8 bytes*/;
    uint64_t* tmem_bar = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196736); /*size = 8 bytes*/;
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(__INTERNAL_DYN_SHMEM__ + 196744); /*size = 4 bytes*/;
    uint64_t* _pad = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 196748); /*size = 864 bytes*/;
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
            #pragma unroll
            for (int32_t s = 0; s < 8; s += 1) {
                init_smem_barrier_fn(full_bars[s], 2);
                init_smem_barrier_fn(empty_bars[s], 1);
            }
            init_smem_barrier_fn(tmem_bar, 1);
            fence_smem_barrier_init_fn();
        }
    }
    if ((warp_idx == 2)) {
        tmem_alloc_fn(tmem_addr, 128);
    }
    cluster_sync_fn();
    uint32_t base_a = ((uint32_t)__cvta_generic_to_shared(A_smem[0]) >> 4);
    uint32_t base_b = ((uint32_t)__cvta_generic_to_shared(B_smem[0]) >> 4);
    uint32_t num_k_blocks = (((K + 64) - 1) / 64);
    int32_t m_coord = static_cast<int32_t>((m_block * 128));
    int32_t n_coord = static_cast<int32_t>(((n_block * 128) + (cta * 64)));
    int32_t is_leader = (static_cast<int32_t>(cta) == 0);
    uint32_t tma_phase;
    int32_t tma_kc;
    uint32_t tma_kb;
    uint32_t el_tma;
    int32_t tma_stage;
    if ((warp_idx == 0)) {
        el_tma = elect_one_fn();
        if ((el_tma == 1)) {
            tma_stage = 0;
            tma_phase = static_cast<uint32_t>(0);
            tma_kb = static_cast<uint32_t>(0);
            while ((tma_kb < num_k_blocks)) {
                mbarrier_wait_fn(empty_bars[tma_stage], (tma_phase ^ 1));
                if ((is_leader == 1)) {
                    mbarrier_arrive_and_expect_tx_fn(full_bars[tma_stage], 24576);
                }
                if ((is_leader == 0)) {
                    mbarrier_arrive_expect_tx_cluster_fn(full_bars[tma_stage], 24576, 0);
                }
                tma_kc = static_cast<int32_t>((tma_kb * 64));
                tma_load_2d_cg2_fn(&dA, full_bars[tma_stage], A_smem[tma_stage], tma_kc, m_coord);
                tma_load_2d_cg2_fn(&dB, full_bars[tma_stage], B_smem[tma_stage], tma_kc, n_coord);
                tma_stage = (tma_stage + 1);
                if ((tma_stage == 8)) {
                    tma_stage = 0;
                    tma_phase = (tma_phase ^ 1);
                }
                tma_kb = (tma_kb + 1);
            }
        }
    }
    uint32_t b_s_off;
    uint32_t mma_phase;
    uint32_t mma_kb;
    uint32_t a_s_off;
    int32_t k_off;
    uint32_t el_mma;
    uint64_t bd;
    uint32_t acc_flag;
    uint32_t first_umma;
    uint64_t ad;
    int32_t mma_stage;
    if ((warp_idx == 1)) {
        if ((is_leader == 1)) {
            el_mma = elect_one_fn();
            if ((el_mma == 1)) {
                mma_stage = 0;
                mma_phase = static_cast<uint32_t>(0);
                first_umma = static_cast<uint32_t>(1);
                mma_kb = static_cast<uint32_t>(0);
                while ((mma_kb < num_k_blocks)) {
                    mbarrier_wait_fn(full_bars[mma_stage], mma_phase);
                    tcgen05_fence_after_fn();
                    a_s_off = (static_cast<uint32_t>(mma_stage) * 1024);
                    b_s_off = (static_cast<uint32_t>(mma_stage) * 512);
                    #pragma unroll
                    for (int32_t ki = 0; ki < 64; ki += 16) {
                        k_off = ((ki * 2) / 16);
                        ad = ((static_cast<uint64_t>(((base_a + a_s_off) + k_off)) & 16383) | 4611756662049472512);
                        bd = ((static_cast<uint64_t>(((base_b + b_s_off) + k_off)) & 16383) | 4611756662049472512);
                        if ((ki == 0)) {
                            acc_flag = (static_cast<uint32_t>(1) - first_umma);
                            umma_f16_cg2_fn(0, ad, bd, 270533776, acc_flag);
                        }
                        else {
                            umma_f16_cg2_fn(0, ad, bd, 270533776, static_cast<uint32_t>(1));
                        }
                    }
                    umma_commit_2sm_fn(empty_bars[mma_stage]);
                    if ((mma_kb == (num_k_blocks - 1))) {
                        umma_commit_2sm_fn(tmem_bar);
                    }
                    first_umma = static_cast<uint32_t>(0);
                    mma_stage = (mma_stage + 1);
                    if ((mma_stage == 8)) {
                        mma_stage = 0;
                        mma_phase = (mma_phase ^ 1);
                    }
                    mma_kb = (mma_kb + 1);
                }
            }
        }
    }
    mbarrier_wait_fn(tmem_bar, 0);
    tcgen05_fence_after_fn();
    if ((is_leader == 1)) {
        tmem_epilogue_coalesced_4w_fn(D, A_smem[0], M, N, m_block, n_block, 128, 128);
    }
    cluster_sync_fn();
    if ((warp_idx == 2)) {
        tmem_dealloc_fn(0, 128);
    }
}


}
