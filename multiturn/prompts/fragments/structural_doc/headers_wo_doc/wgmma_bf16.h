__device__ __forceinline__ void wgmma_fence_fn() {
    // WGMMA fence synchronization. Ensure memory operations complete before WGMMA reads
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_fn() {
    // WGMMA commit group. Group WGMMA operations for collective waiting
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void wgmma_wait_fn() {
    // WGMMA wait group. Block until N groups remain pending
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void wgmma_fence_operand_fn(float& r) {
    // Prevent compiler from optimizing away accumulator dependency.
    asm volatile("" : "+f"(r) :: "memory");
}

__device__ __forceinline__ void wgmma_fence_operand_array_fn(float* a, int n) {
    // Fence all registers in an accumulator array.
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a[i]) :: "memory");
    }
}

/*

    wgmma.mma_async.sync.aligned.shape.dtype.bf16.bf16  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;
    wgmma.mma_async.sync.aligned.shape.dtype.bf16.bf16  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;
    .shape   = {.m64n8k16, .m64n16k16, .m64n24k16, .m64n32k16,
                .m64n40k16, .m64n48k16, .m64n56k16, .m64n64k16,
                .m64n72k16, .m64n80k16, .m64n88k16, .m64n96k16,
                .m64n104k16, .m64n112k16, .m64n120k16, .m64n128k16,
                .m64n136k16, .m64n144k16, .m64n152k16, .m64n160k16,
                .m64n168k16, .m64n176k16, .m64n184k16, .m64n192k16,
                .m64n200k16, .m64n208k16, .m64n216k16, .m64n224k16,
                .m64n232k16, .m64n240k16, .m64n248k16, .m64n256k16};
    .dtype   = {.f16, .f32};

    The 5 immediate operands:
      scale-d     = 0  — Overwrite mode (D = A×B); 1 — Accumulate mode (D = A×B + D)
      imm-scale-a = 1  — No scaling; The valid values of imm-scale-a and imm-scale-b are -1 and 1
      imm-scale-b = 1  — No scaling
      imm-trans-a = 0  — A is K-major (would be 1 for MN-major)
      imm-trans-b = 0  — B is K-major (would be 1 for MN-major)
*/

/*
wgmma descriptor bit layout:
  [13:0]   — matrix-descriptor-encode(Matrix start address)
  [29:16]  — matrix-descriptor-encode(LBO (Leading Byte Offset))
  [45:32]  — matrix-descriptor-encode(SBO (Stride Byte Offset))
  [63:62]  — Swizzle mode: 0=NONE, 1=128B, 2=64B, 3=32B // WGMMA requires CU_TENSOR_MAP_SWIZZLE_128B for TMA and swizzle_mode=1 for the descriptor to match.
  where matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4

  Below uses N as example, M is the same.
  K-Major descriptor under 128B swizzled layouts
  ```
  ATOM_MMODE_DIM = BLOCK_M;
  ATOM_KMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
  SBO = 8 * 128 = 1024; // 8 spans along M-mode (core matrix)
  LBO = 1 # Not used, assumed to be 1.

  ```
  MN-Major descriptor under 128B swizzled layouts
  ```
  ATOM_KMODE_DIM = BLOCK_K;
  ATOM_MMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
  SBO = 128 * 8 = 1024; // 8 spans along K-mode (core matrix)
  LBO = (ATOM_KMODE_DIM / 8) * SBO; // Remember each element in a row is 128bits (16B) when calculating LBO ("offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows."), so LBO is essentially 128B x (# of spans lined up along K-mode). In other words, SBO x (# of core matrices lined up along K-mode)
  ```

  K-Major descriptor under non-swizzled layout
  ```
  ATOM_MMODE_DIM = BLOCK_M;
  ATOM_KMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
  SBO = 8 * 16 = 128; // 8 spans along M-mode
  LBO = (ATOM_MMODE_DIM / 8) * SBO; // BLOCK_M spans along M-mode. Each 8 spans compose a core matrix; each core matrix has SBO bytes. There are ATOM_MMODE_DIM / 8U core matrices
  ```

  MN-Major descriptor under non-swizzled layouts
  ATOM_KMODE_DIM = BLOCK_K;
  ATOM_MMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
  LBO = 16 * 8 = 128; // 8 spans along K-mode
  SBO = (ATOM_KMODE_DIM / 8) * LBO; // Switch LBO and SBO compared with the swizzle layouts

  wgmma just takes in the atom described by {ATOM_KMODE_DIM, ATOM_MMODE_DIM} (K-major) or {ATOM_MMODE_DIM, ATOM_KMODE_DIM} (MN-major), and does not loop over these dimensions.
  Therefore, MN-Major descriptor under 128B swizzled layouts works for m/n64 at the largest. m/n>64 requires iterating with ptr offsets
*/

__device__ static inline uint64_t make_wgmma_desc(void* ptr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle_mode) {
    uint64_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t desc = 0;
    desc |= (addr & 0x3FFFF) >> 4; // [13:0] address
    desc |= ((uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16);  // [29:16] LBO
    desc |= ((uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32);  // [45:32] SBO
    desc |= ((uint64_t)swizzle_mode << 62); // [63:62] swizzle
    return desc;
}

// An example usage pattern for wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
/*
Each thread holds 128 accumulators
float acc[128];
for(int _i=0; _i<128; _i++) acc[_i]=0.0f; 
*/
__device__ __forceinline__ void wgmma_m64n256k16_fn(float* c, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127},"
        "%128,%129,p,1,1,0,0;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]),"+f"(c[32]),"+f"(c[33]),"+f"(c[34]),"+f"(c[35]),"+f"(c[36]),"+f"(c[37]),"+f"(c[38]),"+f"(c[39]),"+f"(c[40]),"+f"(c[41]),"+f"(c[42]),"+f"(c[43]),"+f"(c[44]),"+f"(c[45]),"+f"(c[46]),"+f"(c[47]),"+f"(c[48]),"+f"(c[49]),"+f"(c[50]),"+f"(c[51]),"+f"(c[52]),"+f"(c[53]),"+f"(c[54]),"+f"(c[55]),"+f"(c[56]),"+f"(c[57]),"+f"(c[58]),"+f"(c[59]),"+f"(c[60]),"+f"(c[61]),"+f"(c[62]),"+f"(c[63]),"+f"(c[64]),"+f"(c[65]),"+f"(c[66]),"+f"(c[67]),"+f"(c[68]),"+f"(c[69]),"+f"(c[70]),"+f"(c[71]),"+f"(c[72]),"+f"(c[73]),"+f"(c[74]),"+f"(c[75]),"+f"(c[76]),"+f"(c[77]),"+f"(c[78]),"+f"(c[79]),"+f"(c[80]),"+f"(c[81]),"+f"(c[82]),"+f"(c[83]),"+f"(c[84]),"+f"(c[85]),"+f"(c[86]),"+f"(c[87]),"+f"(c[88]),"+f"(c[89]),"+f"(c[90]),"+f"(c[91]),"+f"(c[92]),"+f"(c[93]),"+f"(c[94]),"+f"(c[95]),"+f"(c[96]),"+f"(c[97]),"+f"(c[98]),"+f"(c[99]),"+f"(c[100]),"+f"(c[101]),"+f"(c[102]),"+f"(c[103]),"+f"(c[104]),"+f"(c[105]),"+f"(c[106]),"+f"(c[107]),"+f"(c[108]),"+f"(c[109]),"+f"(c[110]),"+f"(c[111]),"+f"(c[112]),"+f"(c[113]),"+f"(c[114]),"+f"(c[115]),"+f"(c[116]),"+f"(c[117]),"+f"(c[118]),"+f"(c[119]),"+f"(c[120]),"+f"(c[121]),"+f"(c[122]),"+f"(c[123]),"+f"(c[124]),"+f"(c[125]),"+f"(c[126]),"+f"(c[127])
        : "l"(da), "l"(db));
}

// An example usage pattern for wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16. Each thread holds 32 accumulators
__device__ __forceinline__ void wgmma_m64n64k16_fn_(float* c, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,0,0;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
        : "l"(da), "l"(db));
}

__device__ __forceinline__ void get_coord_n256_(int ltid, int r, int& row, int& col) {
    // ltid: local thread id, r: register id
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
    // Store WGMMA accumulator to global memory for n256.
    #pragma unroll
    for (int r = 0; r < 128; r++) {
        int lm, ln;
        get_coord_n256_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}

__device__ __forceinline__ void get_coord_n64_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}
__device__ __forceinline__ void store_acc_global_n64_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    // Store WGMMA accumulator to global memory for n64.
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord_n64_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}

/*
# ==============================================================================
# Multi-accumulator support for 2x2 tiling (128x128 with 4x m64n64k16)
# ==============================================================================
*/

/*
// Each thread holds 4 x 32 accumulators in a 2x2 layout:
float acc_00[32];
float acc_01[32];
float acc_10[32];
float acc_11[32];
for(int _i=0;_i<32;_i++){acc_00[_i]=0.0f;acc_01[_i]=0.0f;acc_10[_i]=0.0f;acc_11[_i]=0.0f;};
*/

__device__ __forceinline__ void wgmma_fence_4acc_fn(float* a0, float* a1, float* a2, float* a3, int n) {
    // This forces the compiler to treat each accumulator register as live at this point — it cannot reorder, eliminate, or coalesce the registers across this barrier
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a0[i]), "+f"(a1[i]), "+f"(a2[i]), "+f"(a3[i]) :: "memory");
    }
}

__device__ __forceinline__ void get_coord_4ac_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}

// Store 4 m64n64 accumulators (2x2 layout) to global float32.
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

// Store WGMMA accumulator to SMEM for n256.
__device__ __forceinline__ void store_acc_smem_bf16_n256_fn(
    __nv_bfloat16* sC, float* ac, int ltid, int row_offset) {
    int warp = ltid >> 5;
    int lane_id = ltid & 31;
    int row0 = row_offset + warp * 16 + (lane_id >> 2);
    int row1 = row0 + 8;
    int col_base = (lane_id & 3) * 2;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int col = col_base + i * 8;
        sC[row0 * 256 + col + 0] = __float2bfloat16(ac[i * 4 + 0]);
        sC[row0 * 256 + col + 1] = __float2bfloat16(ac[i * 4 + 1]);
        sC[row1 * 256 + col + 0] = __float2bfloat16(ac[i * 4 + 2]);
        sC[row1 * 256 + col + 1] = __float2bfloat16(ac[i * 4 + 3]);
    }
}

__device__ __forceinline__ void stsm_x2_fn_(
    __nv_bfloat162 v0, __nv_bfloat162 v1, void* p) {
    // stmatrix is efficient on both Ampere and Hopper
    uint32_t s0 = *reinterpret_cast<uint32_t*>(&v0);
    uint32_t s1 = *reinterpret_cast<uint32_t*>(&v1);
    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(s0), "r"(s1));
}

// Store WGMMA accumulator to SMEM with 128B swizzle for n256.
/*
int32_t tid = threadIdx.x;
int32_t wg = (tid / 128);
int32_t ltid = (tid % 128);
int32_t lane = (tid % 32);
int32_t warp_in_wg = (ltid / 32);
*/
__device__ __forceinline__ void store_accum_n256_swizzle_fn(
    __nv_bfloat16* D_smem, float* acc, int warp_in_wg, int lane, int m_offset) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int ao = i / 8, iao = i % 8;
        int row = iao / 8 + lane, col = iao;
        col ^= row % 8;
        uint8_t* sp = reinterpret_cast<uint8_t*>(D_smem) +
            warp_in_wg * (16 * 128) + m_offset * 128 + ao * 128 * 128 + row * 128 + col * 16;
        __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[i * 4], acc[i * 4 + 1]);
        __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[i * 4 + 2], acc[i * 4 + 3]);
        stsm_x2_fn_(v0, v1, sp);
    }
}