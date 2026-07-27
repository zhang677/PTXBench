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


__device__ __forceinline__ void warpgroup_reg_dealloc_40_fn() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
}


__device__ __forceinline__ void warpgroup_reg_alloc_232_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
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


__device__ __forceinline__ void get_coord_n256_fn(int ltid, int r, int& row, int& col) {
    int chunk = r / 32;
    int local_reg = r % 32;
    int t0 = ltid % 4, t1 = (ltid / 4) % 8, t2 = ltid / 32;
    int r0 = local_reg % 2, r1 = (local_reg / 2) % 2, r2 = local_reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64;
    col = chunk * 64 + (lin / 64);
}
__device__ __forceinline__ void store_acc_global_n256_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 128; r++) {
        int lm, ln;
        get_coord_n256_fn(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}


extern "C" {
__global__ __launch_bounds__(384, 1) void gemm_v7_kernel(const __grid_constant__ TmaDescriptor dA, const __grid_constant__ TmaDescriptor dB, float* C, int32_t M, int32_t N, int32_t K, int32_t num_tiles, int32_t num_clusters) {
    extern __shared__ __align__(1024) uint8_t __INTERNAL_DYN_SHMEM__[];
    /* align "dynamic_shared" to 1024 */;
    __nv_bfloat16* A_smem[2];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 0); /*size = 16384 bytes*/;
    A_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 16384); /*size = 16384 bytes*/;
    __nv_bfloat16* B_smem[2];
    B_smem[0] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 32768); /*size = 32768 bytes*/;
    B_smem[1] = reinterpret_cast<__nv_bfloat16*>(__INTERNAL_DYN_SHMEM__ + 65536); /*size = 32768 bytes*/;
    uint64_t* full_barriers[2];
    full_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 98304); /*size = 8 bytes*/;
    full_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 98312); /*size = 8 bytes*/;
    uint64_t* empty_barriers[2];
    empty_barriers[0] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 98320); /*size = 8 bytes*/;
    empty_barriers[1] = reinterpret_cast<uint64_t*>(__INTERNAL_DYN_SHMEM__ + 98328); /*size = 8 bytes*/;
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
    if ((wg == 0)) {
        warpgroup_reg_dealloc_40_fn();
    }
    else {
        warpgroup_reg_alloc_232_fn();
    }
    int32_t p_tile = cluster_id;
    while ((p_tile < num_tiles)) {
        if ((tid == 0)) {
            #pragma unroll
            for (int32_t s = 0; s < 2; s += 1) {
                init_smem_barrier_fn(full_barriers[s], 1);
                init_smem_barrier_fn(empty_barriers[s], 16);
            }
            fence_smem_barrier_init_fn();
        }
        cluster_sync_fn();
        int32_t num_pid_in_group = (8 * num_n_tiles);
        int32_t group_id = (p_tile / num_pid_in_group);
        int32_t first_pid_m = (group_id * 8);
        int32_t group_size_m = min(8, (num_m_clusters - first_pid_m));
        int32_t tile_m = (first_pid_m + (p_tile % group_size_m));
        int32_t tile_n = ((p_tile % num_pid_in_group) / group_size_m);
        int32_t t_bn = (tile_n * 256);
        int32_t cluster_m = (tile_m * 256);
        int32_t t_bm = (cluster_m + (static_cast<int32_t>(cta) * 128));
        int32_t c_stage;
        int32_t m_wg_off;
        uint32_t base_desc_a;
        int32_t c_k;
        uint64_t db;
        int32_t p_phase;
        int32_t p_stage;
        uint32_t base_desc_b;
        int32_t k_off;
        uint64_t da;
        uint32_t b_s_off;
        int32_t c_phase;
        uint32_t a_m_off;
        uint32_t a_s_off;
        int32_t p_k;
        if ((wg == 0)) {
            if ((tid == 0)) {
                p_stage = 0;
                p_phase = 0;
                p_k = 0;
                while ((p_k < nk)) {
                    if ((p_k >= 2)) {
                        mbarrier_wait_fn(empty_barriers[p_stage], (p_phase ^ 1));
                    }
                    tma_load_2d_fn(&dA, full_barriers[p_stage], A_smem[p_stage], (p_k * 64), t_bm);
                    tma_load_multicast_2d_fn(&dB, full_barriers[p_stage], (B_smem[p_stage] + (b_half_off * 64)), (p_k * 64), (t_bn + b_half_off), 3);
                    mbarrier_arrive_and_expect_tx_fn(full_barriers[p_stage], 49152);
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
            float acc[128];
for(int _i=0; _i<128; _i++) acc[_i]=0.0f;
            a_m_off = (static_cast<uint32_t>((wg - 1)) * 512);
            m_wg_off = ((wg - 1) * 64);
            base_desc_a = ((uint32_t)__cvta_generic_to_shared(A_smem[0]) >> 4);
            base_desc_b = ((uint32_t)__cvta_generic_to_shared(B_smem[0]) >> 4);
            c_stage = 0;
            c_phase = 0;
            c_k = 0;
            while ((c_k < nk)) {
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
                if ((c_stage == 2)) {
                    c_stage = 0;
                    c_phase = (c_phase ^ 1);
                }
                c_k = (c_k + 1);
            }
            store_acc_global_n256_fn(C, acc, (t_bm + m_wg_off), t_bn, M, N, ltid);
        }
        cluster_sync_fn();
        p_tile = (p_tile + num_clusters);
    }
}


}
