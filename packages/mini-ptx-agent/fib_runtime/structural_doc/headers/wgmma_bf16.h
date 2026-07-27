/*
    The warpgroup level matrix multiply and accumulate operation has either of the following forms,
    where matrix `D` is called accumulator:

    - D = A * B + D
    - D = A * B, where the input from accumulator D is disabled.

    The `wgmma` instructions perform warpgroup level matrix multiply-and-accumulate operation by
    having all threads in a warpgroup collectively perform the following actions:

    1. Load matrices A, B and D into registers or into shared memory.
    2. Perform the following `fence` operations:
        - `wgmma.fence` operations to indicate that the register/shared-memory across the warpgroup have been written into.
        - `fence.proxy.async` operation to make the generic proxy operations visible to the async proxy.
    3. Issue the asynchronous matrix multiply and accumulate operations using the `wgmma.mma_async` operation on the input matrices. The `wgmma.mma_async` operation is performed in the async proxy.
    4. Create a wgmma-group and commit all the prior outstanding `wgmma.mma_async` operations into the group, by using `wgmma.commit_group` operation.
    5. Wait for the completion of the required wgmma-group.
    6. Once the wgmma-group completes, all the `wgmma.mma_async` operations have been performed and completed.

    All the wgmma instructions have to be executed in a warpgroup (128 threads), not just a warp (32 threads).
*/

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
    .shape   = {.m64n{N}k16 where N={8, 16, 32, ..., 256}};
    .dtype   = {.f16, .f32};

    The 5 immediate operands:
      scale-d     = 0  — Overwrite mode (D = A×B); 1 — Accumulate mode (D = A×B + D)
      imm-scale-a = 1  — No scaling; The valid values of imm-scale-a and imm-scale-b are -1 and 1
      imm-scale-b = 1  — No scaling
      imm-trans-a = 0  — core matrix A is K-major (would be 1 for MN-major)
      imm-trans-b = 0  — core matrix B is K-major (would be 1 for MN-major)
    
    Note that imm-trans-a applies only when A comes from shared memory. For register-sourced A in wgmma.m64n{N}k16,
    the per-thread A fragment follows the same 64x16 warpgroup-distributed logical coordinate pattern as the
    accumulator tile of wgmma.m64n16k16, with each pair of bf16 A elements packed into one 32-bit register.
    Refer to `get_d_coord_` below for the register-fragment coordinate mapping.

    Note that tensor core and tma NEVER do transpose! trans-a and trans-b tell the tensor core how to interpret the core matrices in SMEM.
    Such information is used to decide the correct LBO and SBO, and doesn't imply whether the original tensor is transposed or not.
*/

/*
wgmma descriptor bit layout:
[13:0]   — matrix-descriptor-encode(Matrix start address)
[29:16]  — matrix-descriptor-encode(LBO (Leading Byte Offset))
[45:32]  — matrix-descriptor-encode(SBO (Stride Byte Offset))
[51:49]  - Matrix base offset. This is valid for all swizzling modes except the no-swizzle mode.
[63:62]  — Swizzle mode: 0=NONE, 1=128B, 2=64B, 3=32B // WGMMA requires CU_TENSOR_MAP_SWIZZLE_128B for TMA and swizzle_mode=1 for the descriptor to match.
where matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4

The value of base offset is 0 when the repeating pattern of the specified swizzling mode starts as per the below table:

| Swizzling mode | Starting address of the repeating pattern |
|---|---|
| 128-Byte swizzle | 1024-Byte boundary |
| 64-Byte swizzle | 512-Byte boundary |
| 32-Byte swizzle | 256-Byte boundary |

Otherwise, the base offset must be a non-zero value, computed using the following formula: base offset = (pattern start addr >> 0x7) & 0x7
This is used to resolve SMEM alignment problems in case SMEM addresses are not aligned to the byte boundary of the repeating pattern for the swizzle mode.

Below uses N as example, M is the same.
K-Major descriptor under 128B swizzled layouts
```
BOX_MMODE_DIM = BLOCK_M;
BOX_KMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
SBO = 8 * 128 = 1024; // 8 spans along M-mode (core matrix)
LBO = 1 # Not used, assumed to be 1.

```
MN-Major descriptor under 128B swizzled layouts
```
BOX_KMODE_DIM = BLOCK_K;
BOX_MMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
SBO = 128 * 8 = 1024; // 8 spans along K-mode (core matrix)
LBO = (BOX_KMODE_DIM / 8) * SBO; // Remember each element in a row is 128bits (16B) when calculating LBO ("offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows."), so LBO is essentially 128B x (# of spans lined up along K-mode). In other words, SBO x (# of core matrices lined up along K-mode)
```

K-Major descriptor under non-swizzled layout
```
BOX_MMODE_DIM = BLOCK_M;
BOX_KMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
SBO = 8 * 16 = 128; // 8 spans along M-mode
LBO = (BOX_MMODE_DIM / 8) * SBO; // BLOCK_M spans along M-mode. Each 8 spans compose a core matrix; each core matrix has SBO bytes. There are BOX_MMODE_DIM / 8U core matrices
```

MN-Major descriptor under non-swizzled layouts
```
BOX_KMODE_DIM = BLOCK_K;
BOX_MMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
LBO = 16 * 8 = 128; // 8 spans along K-mode
SBO = (BOX_KMODE_DIM / 8) * LBO; // Switch LBO and SBO compared with the swizzle layouts
```

wgmma just takes matrices described by m64n{N}k16, and does not loop over these dimensions in boxes.
Therefore, MN-Major descriptor under 128B swizzled layouts works for m/n64 at the largest. m/n>64 requires iterating with ptr offsets.
Given swizzle mode, matrix start address, and the matrix base offset, wgmma can figure out the ID of the starting chunk. 
Therefore, a K-major 128B swizzle tile contains eight 16B chunks, and each k16 wgmma consumes only two chunks; 
the user only needs to provide the descriptor whose matrix start address points to the desired logical K slice, 
while keeping the swizzle mode and matrix base offset consistent with how the data was laid out in shared memory.
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

/*
Shared-A/shared-B 64n64k16 WGMMA. For other shapes, change the number of output registers
Computes D = A * B + D for logical shapes:
A: 64x16 bf16, B: 16x64 bf16, D: 64x64 f32.
Each of the 128 warpgroup threads owns 32 f32 accumulator registers.
float acc[32];
for(int _i=0; _i<32; _i++) acc[_i]=0.0f; 
*/
template<int trans_a, int trans_b>
__device__ __forceinline__ void wgmma_m64n64k16_ss_fn(float* c, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,%34,%35;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
          "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
          "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
          "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
        : "l"(da), "l"(db), "n"(trans_a), "n"(trans_b));
}

/*
Register-A/shared-B 64n64k16 WGMMA. For other shapes, change the number of output registers
A is provided as a warpgroup-distributed register fragment.
Each thread supplies 4 packed b32 registers, containing 8 bf16 A elements.
B is provided through a WGMMA shared-memory descriptor.
Each thread owns 32 f32 accumulator registers for D.
*/
template<int trans_b>
__device__ __forceinline__ void wgmma_m64n64k16_rs_fn_(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1, %37;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "n"(trans_b));
}

/*
wgmma accumulator D's layout can be viewed as a hierachical tiling. Tile shape's unit is an element. Tiles are layed out in row-major and there are 64 rows because m=64.
Layer 0: 16 rows x 8 cols, indexed by [t2, r2]
Layer 1: 8 rows x 2 cols, indexed by [r1, t0]
Layer 2: 1 row x 1 col, indexed by [t1, r0]
Layer i-1 is composed by Layer i tiles layed out in row-major
*/
__device__ __forceinline__ void get_d_coord_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}

__device__ __forceinline__ void store_acc_global_n256_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    // Store WGMMA accumulator to global memory for n256.
    #pragma unroll
    for (int r = 0; r < 128; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}

__device__ __forceinline__ void store_acc_global_n64_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    // Store WGMMA accumulator to global memory for n64.
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
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

// Store 4 m64n64 accumulators (2x2 layout) to global float32.
__device__ __forceinline__ void store_4acc_f32_fn(
    float* C, float* a00, float* a01, float* a10, float* a11,
    int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
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

/*
```
stmatrix
```

Collectively store one or more matrices to shared memory.

Syntax

```
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;
.shape  = {.m8n8};
.num    = {.x1, .x2, .x4};
.ss     = {.shared{::cta}};
.type   = {.b16, .b8};
```

Description
Collectively store one or more matrices across all threads in a warp to the location indicated by
the address operand `p` , in `.shared` state space. If no state space is provided, generic
addressing is used, such that the address in `p` points into `.shared` space. If the generic
address doesn't fall in `.shared` state space, then the behavior is undefined.

The `.shape` qualifier indicates the dimensions of the matrices being loaded. Each matrix element
holds 16-bit or 8-bit data as indicated by the `.type` qualifier.

The values `.x1` , `.x2` and `.x4` for `.num` indicate one, two or four matrices
respectively.

The mandatory `.sync` qualifier indicates that `stmatrix` causes the executing thread to wait
until all threads in the warp execute the same `stmatrix` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same `stmatrix` instruction. In conditionally executed code, an `stmatrix` instruction should only be
used if it is known that all threads in the warp evaluate the condition identically, otherwise the behavior is undefined.

The behavior of `stmatrix` is undefined if all threads do not use the same qualifiers, or if any thread in the warp has exited.

The source operand `r` is a brace-enclosed vector expression consisting of 1, 2, or 4 32-bit
registers as per the value of `.num` . Each component of the vector expression holds a fragment
from the corresponding matrix.

Consecutive instances of row need not be stored contiguously in memory. The eight addresses required
for each matrix are provided by eight threads, depending upon the value of `.num` as shown in the following table. Each address corresponds to the start of a matrix row. Addresses addr0-addr7
correspond to the rows of the first matrix, addresses addr8-addr15 correspond to the rows of the second matrix, and so on.

| ``` .num ```   | Threads 0-7   | Threads 8-15   | Threads 16-23   | Threads 24-31   |
|----------------|---------------|----------------|-----------------|-----------------|
| ``` .x1 ```    | addr0-addr7   | -              | -               | -               |
| ``` .x2 ```    | addr0-addr7   | addr8-addr15   | -               | -               |
| ``` .x4 ```    | addr0-addr7   | addr8-addr15   | addr16-addr23   | addr24-addr31   |

When storing 8x8 matrices, a group of four consecutive threads stores 16 bytes. The matrix addresses
must be naturally aligned accordingly.

Each thread in a warp stores fragments of a row, with thread 0 storing the first fragment from its
register
`r` , and so on. A group of four threads stores an entire row of the matrix as shown in [Figure 107](#mma-stmatrix-fragments) .

> ```
> stmatrix fragment layout for one 8x8 matrix with 16-bit elements:
> 
>   Each thread Tn stores register r to shared memory:
>     Row = n / 4
>     Cols = ((n%4)*2) to ((n%4)*2+1)
> 
>   A group of 4 consecutive threads stores an entire row.
>   When .num = .x2, second matrix stored from next source register.
> ```
When `.num` = `.x2` , the elements of the second matrix are storedd from the next source register
in each thread as per the layout in above table. Similarly, when
`.num` = `.x4` , elements of the third and fourth matrices are stored from the subsequent source registers in each thread.
*/
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
