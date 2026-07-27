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


__device__ __forceinline__ void warpgroup_reg_dealloc_40_fn() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
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


__device__ __forceinline__ void warpgroup_reg_alloc_224_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 224;\n");
}


__device__ __forceinline__ void wgmma_fence_fn() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
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


__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}


__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(count));
}


__device__ __forceinline__ void stsm_x2_fn_(__nv_bfloat162 v0, __nv_bfloat162 v1, void* p) {
    uint32_t s0=*reinterpret_cast<uint32_t*>(&v0);
    uint32_t s1=*reinterpret_cast<uint32_t*>(&v1);
    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(s0), "r"(s1));
}

__device__ __forceinline__ void store_accum_swizzle_fn(
    __nv_bfloat16* sD, float* ac, int wi, int li, int mo) {
    #pragma unroll
    for (int i=0;i<32;i++) {
        int ao=i/8, iao=i%8;
        int row=iao/8+li, col=iao;
        col^=row%8;
        uint8_t* sp=reinterpret_cast<uint8_t*>(sD)+
            wi*(16*128)+mo*128+ao*128*128+row*128+col*16;
        __nv_bfloat162 v0=__floats2bfloat162_rn(ac[i*4],ac[i*4+1]);
        __nv_bfloat162 v1=__floats2bfloat162_rn(ac[i*4+2],ac[i*4+3]);
        stsm_x2_fn_(v0,v1,sp);
    }
}


__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}


__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "r"(c0), "r"(c1) : "memory");
}


__device__ __forceinline__ void tma_store_arrive_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}


extern "C" {
__global__ __launch_bounds__(384, 1) void gemm_v10_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, const __grid_constant__ TmaDescriptor dC, __nv_bfloat16* C, int32_t M, int32_t N, int32_t K, int32_t num_tiles, int32_t num_clusters) {
    extern __shared__ __align__(1024) uint8_t __INTERNAL_DYN_SHMEM__[];
    /* align "dynamic_shared" to 1024 */;
    __nv_bfloat16* D_smem = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 0); /*size = 65536 bytes*/;
    __nv_bfloat16* A_smem[3];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 65536); /*size = 16384 bytes*/;
    A_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 81920); /*size = 16384 bytes*/;
    A_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 98304); /*size = 16384 bytes*/;
    __nv_bfloat16* B_smem[3];
    B_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 114688); /*size = 32768 bytes*/;
    B_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 147456); /*size = 32768 bytes*/;
    B_smem[2] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 180224); /*size = 32768 bytes*/;
    uint64_t* full_barriers[3];
    full_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 212992); /*size = 8 bytes*/;
    full_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 213000); /*size = 8 bytes*/;
    full_barriers[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 213008); /*size = 8 bytes*/;
    uint64_t* empty_barriers[3];
    empty_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 213016); /*size = 8 bytes*/;
    empty_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 213024); /*size = 8 bytes*/;
    empty_barriers[2] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 213032); /*size = 8 bytes*/;
    int32_t tid = threadIdx.x;
    int32_t wg = (tid / 128);
    int32_t ltid = (tid % 128);
    int32_t lane = (tid % 32);
    uint32_t cta = cluster_rank_fn();
    int32_t num_n_tiles = (((N + 256) - 1) / 256);
    int32_t num_m_clusters = (((M + 256) - 1) / 256);
    int32_t nk = (((K + 64) - 1) / 64);
    int32_t cluster_id = (blockIdx.x / 2);
    int32_t b_half_off = (static_cast<int32_t>(cta) * 128);
    int32_t warp_in_wg = (ltid / 32);
    if ((tid == 0)) {
        #pragma unroll
        for (int32_t s = 0; s < 3; s += 1) {
            init_smem_barrier_fn(full_barriers[s], 1);
            init_smem_barrier_fn(empty_barriers[s], 16);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();
    int32_t m_wg_off;
    uint32_t base_desc_b;
    int32_t first_pid_m_p;
    int32_t c_tile;
    int32_t p_tile;
    int32_t c_stage;
    int32_t tile_n_c;
    int32_t p_phase;
    int32_t p_bm;
    int32_t c_bn;
    uint64_t da;
    uint32_t b_s_off;
    int32_t c_phase;
    int32_t first_pid_m_c;
    int32_t p_stage;
    int32_t math_wg_idx;
    int32_t group_id_c;
    int32_t num_pid_in_group_c;
    int32_t group_size_m_p;
    int32_t tile_n_p;
    int32_t tile_m_p;
    int32_t k_off;
    int32_t pk;
    uint32_t a_m_off;
    int32_t p_bn;
    uint32_t base_desc_a;
    uint64_t db;
    int32_t c_bm;
    int32_t tile_m_c;
    int32_t num_pid_in_group_p;
    int32_t p_giter;
    int32_t c_first;
    int32_t group_size_m_c;
    int32_t ck;
    uint32_t a_s_off;
    int32_t group_id_p;
    if ((wg == 0)) {
        warpgroup_reg_dealloc_40_fn();
        if ((tid == 0)) {
            p_stage = 0;
            p_phase = 0;
            p_giter = 0;
            p_tile = cluster_id;
            while ((p_tile < num_tiles)) {
                num_pid_in_group_p = (8 * num_n_tiles);
                group_id_p = (p_tile / num_pid_in_group_p);
                first_pid_m_p = (group_id_p * 8);
                group_size_m_p = min(8, (num_m_clusters - first_pid_m_p));
                tile_m_p = (first_pid_m_p + (p_tile % group_size_m_p));
                tile_n_p = ((p_tile % num_pid_in_group_p) / group_size_m_p);
                p_bn = (tile_n_p * 256);
                p_bm = ((tile_m_p * 256) + (static_cast<int32_t>(cta) * 128));
                pk = 0;
                while ((pk < nk)) {
                    if ((p_giter >= 3)) {
                        mbarrier_wait_fn(empty_barriers[p_stage], (p_phase ^ 1));
                    }
                    tma_load_2d_fn(&dA, full_barriers[p_stage], A_smem[p_stage], (pk * 64), p_bm);
                    tma_load_multicast_2d_fn(&dB, full_barriers[p_stage], (B_smem[p_stage] + (b_half_off * 64)), (pk * 64), (p_bn + b_half_off), 3);
                    mbarrier_arrive_and_expect_tx_fn(full_barriers[p_stage], 49152);
                    p_stage = (p_stage + 1);
                    if ((p_stage == 3)) {
                        p_stage = 0;
                        p_phase = (p_phase ^ 1);
                    }
                    p_giter = (p_giter + 1);
                    pk = (pk + 1);
                }
                p_tile = (p_tile + num_clusters);
            }
        }
    }
    else {
        warpgroup_reg_alloc_224_fn();
        a_m_off = (static_cast<uint32_t>((wg - 1)) * 512);
        m_wg_off = ((wg - 1) * 64);
        math_wg_idx = (wg - 1);
        base_desc_a = ((uint32_t)__cvta_generic_to_shared(A_smem[0]) >> 4);
        base_desc_b = ((uint32_t)__cvta_generic_to_shared(B_smem[0]) >> 4);
        c_stage = 0;
        c_phase = 0;
        c_first = 1;
        c_tile = cluster_id;
        while ((c_tile < num_tiles)) {
            num_pid_in_group_c = (8 * num_n_tiles);
            group_id_c = (c_tile / num_pid_in_group_c);
            first_pid_m_c = (group_id_c * 8);
            group_size_m_c = min(8, (num_m_clusters - first_pid_m_c));
            tile_m_c = (first_pid_m_c + (c_tile % group_size_m_c));
            tile_n_c = ((c_tile % num_pid_in_group_c) / group_size_m_c);
            c_bn = (tile_n_c * 256);
            c_bm = ((tile_m_c * 256) + (static_cast<int32_t>(cta) * 128));
            float acc[128];
for(int _i=0; _i<128; _i++) acc[_i]=0.0f;
            ck = 0;
            while ((ck < nk)) {
                mbarrier_wait_fn(full_barriers[c_stage], c_phase);
                __syncwarp();
                wgmma_fence_fn();
                a_s_off = (static_cast<uint32_t>(c_stage) * 1024);
                b_s_off = (static_cast<uint32_t>(c_stage) * 2048);
                #pragma unroll
                for (int32_t ki = 0; ki < 64; ki += 16) {
                    k_off = ((ki * 2) / 16);
                    da = (static_cast<uint64_t>(((((base_desc_a + a_s_off) + a_m_off) + k_off) & 16383)) | 4611686293305344000);
                    db = (static_cast<uint64_t>((((base_desc_b + b_s_off) + k_off) & 16383)) | 4611686293305344000);
                    wgmma_m64n256k16_fn(acc, da, db);
                }
                wgmma_commit_fn();
                wgmma_wait_fn();
                if ((lane < 2)) {
                    mbarrier_arrive_remote_fn(empty_barriers[c_stage], lane);
                }
                __syncwarp();
                c_stage = (c_stage + 1);
                if ((c_stage == 3)) {
                    c_stage = 0;
                    c_phase = (c_phase ^ 1);
                }
                ck = (ck + 1);
            }
            if ((c_first == 0)) {
                if ((math_wg_idx == 0)) {
                    if ((ltid == 0)) {
                        tma_store_wait_fn();
                    }
                }
                named_barrier_sync_fn(0, 256);
            }
            c_first = 0;
            store_accum_swizzle_fn(D_smem, acc, warp_in_wg, lane, m_wg_off);
            tma_store_fence_fn();
            named_barrier_sync_fn(0, 256);
            if ((math_wg_idx == 0)) {
                if ((ltid == 0)) {
                    if ((c_bm < M)) {
                        if ((c_bn < N)) {
                            #pragma unroll
                            for (int32_t t = 0; t < 4; t += 1) {
                                tma_store_2d_fn(&dC, (D_smem + (t * 8192)), (c_bn + (t * 64)), c_bm);
                            }
                        }
                    }
                    tma_store_arrive_fn();
                }
            }
            c_tile = (c_tile + num_clusters);
        }
        if ((math_wg_idx == 0)) {
            if ((ltid == 0)) {
                tma_store_wait_fn();
            }
        }
    }
    cluster_sync_fn();
}


}
