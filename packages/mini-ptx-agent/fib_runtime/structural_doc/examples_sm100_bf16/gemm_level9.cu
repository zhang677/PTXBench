/*start of generated code*/
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>

using TmaDescriptor = CUtensorMap;




__device__ __forceinline__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}


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


__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a));
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


__device__ __forceinline__ void tma_store_wait_n_fn(int count) {
    // cp.async.bulk.wait_group requires immediate operand, use switch for common values
    switch (count) {
        case 0: asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); break;
        case 1: asm volatile("cp.async.bulk.wait_group 1;\n" ::: "memory"); break;
        case 2: asm volatile("cp.async.bulk.wait_group 2;\n" ::: "memory"); break;
        case 3: asm volatile("cp.async.bulk.wait_group 3;\n" ::: "memory"); break;
        case 4: asm volatile("cp.async.bulk.wait_group 4;\n" ::: "memory"); break;
        case 5: asm volatile("cp.async.bulk.wait_group 5;\n" ::: "memory"); break;
        case 6: asm volatile("cp.async.bulk.wait_group 6;\n" ::: "memory"); break;
        case 7: asm volatile("cp.async.bulk.wait_group 7;\n" ::: "memory"); break;
        default: asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); break;
    }
}


__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(count));
}


__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}


__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}


#include <cuda_bf16.h>
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


__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}


__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}


__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}


__device__ __forceinline__ void tma_store_2d_sm100_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(sa) : "memory");
}


__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}


__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}


__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}


extern "C" {
__global__ __launch_bounds__(256, 1) void gemm_level9_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, const __grid_constant__ TmaDescriptor dD, uint32_t M, uint32_t N, uint32_t K, uint32_t num_ctas) {
    extern __shared__ __align__(1024) uint8_t __INTERNAL_DYN_SHMEM__[];
    /* align "dynamic_shared" to 1024 */;
    __nv_bfloat16* cd_smem_0 = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 0); /*size = 16384 bytes*/;
    __nv_bfloat16* cd_smem_1 = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 16384); /*size = 16384 bytes*/;
    __nv_bfloat16* A_smem[4];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 32768); /*size = 32768 bytes*/;
    A_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 65536); /*size = 32768 bytes*/;
    A_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 98304); /*size = 32768 bytes*/;
    A_smem[3] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 131072); /*size = 32768 bytes*/;
    __nv_bfloat16* B_smem[4];
    B_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 163840); /*size = 16384 bytes*/;
    B_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 180224); /*size = 16384 bytes*/;
    B_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 196608); /*size = 16384 bytes*/;
    B_smem[3] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 212992); /*size = 16384 bytes*/;
    uint64_t* full_bars[4];
    full_bars[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229376); /*size = 8 bytes*/;
    full_bars[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229384); /*size = 8 bytes*/;
    full_bars[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229392); /*size = 8 bytes*/;
    full_bars[3] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229400); /*size = 8 bytes*/;
    uint64_t* empty_bars[4];
    empty_bars[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229408); /*size = 8 bytes*/;
    empty_bars[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229416); /*size = 8 bytes*/;
    empty_bars[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229424); /*size = 8 bytes*/;
    empty_bars[3] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229432); /*size = 8 bytes*/;
    uint64_t* tmem_full_bars[1];
    tmem_full_bars[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229440); /*size = 8 bytes*/;
    uint64_t* tmem_empty_bars[1];
    tmem_empty_bars[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229448); /*size = 8 bytes*/;
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(__INTERNAL_DYN_SHMEM__ + 229456); /*size = 4 bytes*/;
    uint64_t* _pad = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 229460); /*size = 400 bytes*/;
    int32_t tid = threadIdx.x;
    int32_t warp_idx = (tid / 32);
    int32_t lane_idx = get_lane_idx();
    uint32_t cta = cluster_rank_fn();
    int32_t is_leader = (static_cast<int32_t>(cta) == 0);
    uint32_t cta_id = blockIdx.x;
    uint32_t num_m_blocks = (((M + 256) - 1) / 256);
    uint32_t num_n_blocks = (((N + 256) - 1) / 256);
    uint32_t num_tiles = (num_m_blocks * num_n_blocks);
    uint32_t num_k_blocks = (((K + 64) - 1) / 64);
    uint32_t el0;
    if ((warp_idx == 0)) {
        el0 = elect_one_fn();
        if ((el0 == 1)) {
            prefetch_tma_descriptor_fn(&dA);
            prefetch_tma_descriptor_fn(&dB);
            prefetch_tma_descriptor_fn(&dD);
        }
    }
    uint32_t el1;
    if ((warp_idx == 1)) {
        el1 = elect_one_fn();
        if ((el1 == 1)) {
            #pragma unroll
            for (int32_t s = 0; s < 4; s += 1) {
                init_smem_barrier_fn(full_bars[s], 2);
                init_smem_barrier_fn(empty_bars[s], 1);
            }
            #pragma unroll
            for (int32_t e = 0; e < 1; e += 1) {
                init_smem_barrier_fn(tmem_full_bars[e], 1);
                init_smem_barrier_fn(tmem_empty_bars[e], 256);
            }
            fence_smem_barrier_init_fn();
        }
    }
    if ((warp_idx == 2)) {
        tmem_alloc_fn(tmem_addr, 512);
    }
    cluster_sync_fn();
    uint32_t base_a0 = ((uint32_t)__cvta_generic_to_shared(A_smem[0]) >> 4);
    uint32_t base_b0 = ((uint32_t)__cvta_generic_to_shared(B_smem[0]) >> 4);
    uint32_t cd_base_0 = (uint32_t)__cvta_generic_to_shared(cd_smem_0);
    uint32_t cd_base_1 = (uint32_t)__cvta_generic_to_shared(cd_smem_1);
    int32_t n_coord;
    uint32_t tpg_t;
    uint32_t fm_t;
    uint32_t gi_t;
    uint32_t mg_t;
    uint32_t mb_t;
    int32_t tma_kc;
    uint32_t tma_phase;
    uint32_t ig_t;
    int32_t m_coord;
    uint32_t tile_tma;
    uint32_t nb_t;
    uint32_t tma_kb;
    uint32_t el_tma;
    int32_t tma_stage;
    if ((warp_idx == 0)) {
        el_tma = elect_one_fn();
        if ((el_tma == 1)) {
            tma_stage = 0;
            tma_phase = static_cast<uint32_t>(0);
            tile_tma = cta_id;
            while ((tile_tma < num_tiles)) {
                tpg_t = (num_n_blocks * 16);
                gi_t = (tile_tma / tpg_t);
                fm_t = (gi_t * 16);
                mg_t = static_cast<uint32_t>(min(static_cast<int32_t>(16), static_cast<int32_t>((num_m_blocks - fm_t))));
                ig_t = (tile_tma % tpg_t);
                mb_t = (fm_t + (ig_t % mg_t));
                nb_t = (ig_t / mg_t);
                m_coord = static_cast<int32_t>((mb_t * 256));
                n_coord = static_cast<int32_t>(((nb_t * 256) + (cta * 128)));
                tma_kb = static_cast<uint32_t>(0);
                while ((tma_kb < num_k_blocks)) {
                    mbarrier_wait_fn(empty_bars[tma_stage], (tma_phase ^ 1));
                    if ((is_leader == 1)) {
                        mbarrier_arrive_and_expect_tx_fn(full_bars[tma_stage], 98304);
                    }
                    if ((is_leader == 0)) {
                        mbarrier_arrive_cluster_fn(full_bars[tma_stage], 0);
                    }
                    tma_kc = static_cast<int32_t>((tma_kb * 64));
                    tma_load_2d_cg2_fn(&dA, full_bars[tma_stage], A_smem[tma_stage], tma_kc, m_coord);
                    tma_load_2d_cg2_fn(&dB, full_bars[tma_stage], B_smem[tma_stage], tma_kc, n_coord);
                    tma_stage = (tma_stage + 1);
                    if ((tma_stage == 4)) {
                        tma_stage = 0;
                        tma_phase = (tma_phase ^ 1);
                    }
                    tma_kb = (tma_kb + 1);
                }
                tile_tma = (tile_tma + num_ctas);
            }
        }
    }
    uint32_t mma_kb;
    int32_t k_off;
    int32_t last_idx;
    uint32_t el_umma;
    uint64_t bd;
    uint32_t acc_flag;
    uint32_t tile_mma;
    uint64_t ad;
    uint32_t a_lo_base;
    int32_t epi_idx;
    uint32_t el_commit;
    int32_t mma_stage;
    uint32_t cur_b_lo;
    uint32_t a_stride_16;
    uint32_t mma_phase;
    uint32_t tmem_off;
    uint32_t my_a_lo;
    uint32_t my_b_lo;
    uint32_t cur_a_lo;
    uint32_t b_lo_base;
    uint32_t b_stride_16;
    uint32_t last_ph;
    uint32_t epi_phase;
    int32_t tile_mma_iter;
    uint32_t b_lo;
    uint32_t a_lo;
    if ((warp_idx == 1)) {
        if ((is_leader == 1)) {
            a_lo_base = (base_a0 & 16383);
            b_lo_base = (base_b0 & 16383);
            a_stride_16 = static_cast<uint32_t>(2048);
            b_stride_16 = static_cast<uint32_t>(1024);
            my_a_lo = (a_lo_base + (static_cast<uint32_t>(lane_idx) * a_stride_16));
            my_b_lo = (b_lo_base + (static_cast<uint32_t>(lane_idx) * b_stride_16));
            mma_stage = 0;
            mma_phase = static_cast<uint32_t>(0);
            tile_mma = cta_id;
            tile_mma_iter = 0;
            while ((tile_mma < num_tiles)) {
                epi_idx = (tile_mma_iter % 1);
                epi_phase = (static_cast<uint32_t>((tile_mma_iter / 1)) & 1);
                mbarrier_wait_fn(tmem_empty_bars[epi_idx], (epi_phase ^ 1));
                tcgen05_fence_after_fn();
                mma_kb = static_cast<uint32_t>(0);
                while ((mma_kb < num_k_blocks)) {
                    mbarrier_wait_fn(full_bars[mma_stage], mma_phase);
                    tcgen05_fence_after_fn();
                    cur_a_lo = __shfl_sync(4294967295, my_a_lo, mma_stage);
                    cur_b_lo = __shfl_sync(4294967295, my_b_lo, mma_stage);
                    el_umma = elect_one_fn();
                    if ((el_umma == 1)) {
                        #pragma unroll
                        for (int32_t ki = 0; ki < 64; ki += 16) {
                            k_off = ((ki * 2) / 16);
                            b_lo = (cur_b_lo + k_off);
                            bd = ((static_cast<uint64_t>(b_lo) & 16383) | 4611756662049472512);
                            #pragma unroll
                            for (int32_t w = 0; w < 2; w += 1) {
                                a_lo = ((cur_a_lo + (w * 1024)) + k_off);
                                ad = ((static_cast<uint64_t>(a_lo) & 16383) | 4611756662049472512);
                                tmem_off = static_cast<uint32_t>((((epi_idx * 2) * 256) + (w * 256)));
                                if ((ki == 0)) {
                                    acc_flag = static_cast<uint32_t>((mma_kb > 0));
                                    umma_f16_cg2_fn(tmem_off, ad, bd, 272630928, acc_flag);
                                }
                                else {
                                    umma_f16_cg2_fn(tmem_off, ad, bd, 272630928, static_cast<uint32_t>(1));
                                }
                            }
                        }
                    }
                    el_commit = elect_one_fn();
                    if ((el_commit == 1)) {
                        umma_commit_2sm_fn(empty_bars[mma_stage]);
                        if ((mma_kb == (num_k_blocks - 1))) {
                            umma_commit_2sm_fn(tmem_full_bars[epi_idx]);
                        }
                    }
                    mma_stage = (mma_stage + 1);
                    if ((mma_stage == 4)) {
                        mma_stage = 0;
                        mma_phase = (mma_phase ^ 1);
                    }
                    mma_kb = (mma_kb + 1);
                }
                tile_mma_iter = (tile_mma_iter + 1);
                tile_mma = (tile_mma + num_ctas);
            }
            if ((tile_mma_iter > 0)) {
                last_idx = ((tile_mma_iter - 1) % 1);
                last_ph = (static_cast<uint32_t>(((tile_mma_iter - 1) / 1)) & 1);
                mbarrier_wait_fn(tmem_empty_bars[last_idx], last_ph);
            }
        }
    }
    uint32_t tpg_e;
    uint32_t r6;
    uint32_t r5;
    uint32_t fm_e;
    uint32_t el_sw;
    uint32_t mg_e;
    int32_t n_idx;
    uint32_t el_final;
    uint32_t swz_col;
    uint32_t epi_phase_e;
    uint32_t p1;
    int32_t m_idx;
    uint32_t nb_e;
    uint32_t p2;
    uint32_t gi_e;
    uint32_t r7;
    uint32_t p3;
    uint32_t _cd_base;
    int32_t tile_epi_iter;
    uint32_t r1;
    uint32_t r3;
    uint32_t el_st;
    int32_t _stage_idx;
    int32_t epi_idx_e;
    uint32_t smem_addr;
    uint32_t ig_e;
    uint32_t tile_epi;
    uint32_t r4;
    uint32_t r0;
    int32_t local_tid;
    int32_t epi_warp;
    uint32_t r2;
    uint32_t tmem_col;
    uint32_t mb_e;
    uint32_t p0;
    if ((warp_idx >= 4)) {
        local_tid = (tid - 128);
        epi_warp = (local_tid / 32);
        tile_epi = cta_id;
        tile_epi_iter = 0;
        while ((tile_epi < num_tiles)) {
            tpg_e = (num_n_blocks * 16);
            gi_e = (tile_epi / tpg_e);
            fm_e = (gi_e * 16);
            mg_e = static_cast<uint32_t>(min(static_cast<int32_t>(16), static_cast<int32_t>((num_m_blocks - fm_e))));
            ig_e = (tile_epi % tpg_e);
            mb_e = (fm_e + (ig_e % mg_e));
            nb_e = (ig_e / mg_e);
            epi_idx_e = (tile_epi_iter % 1);
            epi_phase_e = (static_cast<uint32_t>((tile_epi_iter / 1)) & 1);
            mbarrier_wait_fn(tmem_full_bars[epi_idx_e], epi_phase_e);
            tcgen05_fence_after_fn();
            #pragma unroll
            for (int32_t w = 0; w < 2; w += 1) {
                #pragma unroll
                for (int32_t s = 0; s < 4; s += 1) {
                    _stage_idx = (((w * 4) + s) % 2);
                    _cd_base = ((_stage_idx == 0) ? cd_base_0 : cd_base_1);
                    if ((epi_warp == 0)) {
                        el_sw = elect_one_fn();
                        if ((el_sw == 1)) {
                            tma_store_wait_n_fn(1);
                        }
                    }
                    named_barrier_sync_fn(1, 128);
                    #pragma unroll
                    for (int32_t i = 0; i < 8; i += 1) {
                        tmem_col = static_cast<uint32_t>((((((epi_idx_e * 2) * 256) + (w * 256)) + (s * 64)) + (i * 8)));
                        r0 = static_cast<uint32_t>(0);
                        r1 = static_cast<uint32_t>(0);
                        r2 = static_cast<uint32_t>(0);
                        r3 = static_cast<uint32_t>(0);
                        r4 = static_cast<uint32_t>(0);
                        r5 = static_cast<uint32_t>(0);
                        r6 = static_cast<uint32_t>(0);
                        r7 = static_cast<uint32_t>(0);
                        tmem_load_8x_fn(tmem_col, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
                        tmem_load_fence_fn();
                        p0 = pack_bf16_fn(r0, r1);
                        p1 = pack_bf16_fn(r2, r3);
                        p2 = pack_bf16_fn(r4, r5);
                        p3 = pack_bf16_fn(r6, r7);
                        swz_col = (static_cast<uint32_t>(i) ^ (static_cast<uint32_t>(local_tid) % 8));
                        smem_addr = ((_cd_base + (static_cast<uint32_t>(local_tid) * 128)) + (swz_col * 16));
                        st_shared_128_fn(smem_addr, p0, p1, p2, p3);
                    }
                    if (((w == 1) && (s == 3))) {
                        tcgen05_fence_before_fn();
                        mbarrier_arrive_cluster_fn(tmem_empty_bars[epi_idx_e], 0);
                    }
                    __syncwarp();
                    tma_store_fence_fn();
                    named_barrier_sync_fn(1, 128);
                    if ((epi_warp == 0)) {
                        el_st = elect_one_fn();
                        if ((el_st == 1)) {
                            n_idx = static_cast<int32_t>(((nb_e * 256) + (s * 64)));
                            m_idx = static_cast<int32_t>(((mb_e * 256) + (w * 128)));
                            if ((_stage_idx == 0)) {
                                tma_store_2d_sm100_fn(&dD, cd_smem_0, n_idx, m_idx);
                            }
                            else {
                                tma_store_2d_sm100_fn(&dD, cd_smem_1, n_idx, m_idx);
                            }
                            tma_store_commit_fn();
                        }
                    }
                }
            }
            tile_epi_iter = (tile_epi_iter + 1);
            tile_epi = (tile_epi + num_ctas);
        }
        if ((epi_warp == 0)) {
            el_final = elect_one_fn();
            if ((el_final == 1)) {
                tma_store_wait_fn();
            }
        }
    }
    __syncthreads();
    cluster_sync_fn();
    if ((warp_idx == 2)) {
        tmem_dealloc_fn(0, 512);
    }
}


}
