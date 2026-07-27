/*

Dynamic [Tensor Memory](#tensor-memory) allocation management instructions

tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32  [dst], nCols;
tcgen05.dealloc.cta_group.sync.aligned.b32              taddr, nCols;

`tcgen05.alloc` is a blocking instruction which dynamically allocates the specified number of columns in the [Tensor Memory](#tensor-memory) and writes
the address of the allocated Tensor Memory into shared memory at the location specified by address operand dst. The `tcgen05.alloc` blocks if the
requested amount of Tensor Memory is not available and unblocks as soon as the requested amount of Tensor Memory becomes available for allocation.

`tcgen05.dealloc` is a potentially blocking instruction which deallocates the Tensor Memory specified by the Tensor Memory address `taddr` . The operand `taddr` must point to a previous Tensor Memory allocation.

If `.cta_group::2` is specified,
- issuing warp and [peer CTA](#tcgen05-peer-cta) warp must synchronize Tensor Memory accesses before attempting to collectively deallocate the Tensor Memory , and
- `tcgen05.dealloc` may block to collectively performs the deallocation with the other peer CTA's warp.

All of the Tensor Memory that was allocated using `tcgen05.alloc` instruction in a kernel must be explicitly deallocated using `tcgen05.dealloc` before the kernel exits.

The unsigned 32-bit operand `nCols` specify the number of columns to be allocated or de-allocated. The unit of allocation and de-allocation is 32 columns and all of lanes
per column. The number of columns must be a power of 2. The operand `nCols` must be within the range [32, 512]. The number of columns allocated should not increase between
any two allocations in the execution order within the CTA. Operand `nCols` must be power of 2.

Qualifier `.cta_group` specifies the number of CTAs involved in the allocation and de-allocation operation. When `.cta_group::2` is specified, one warp from each of the peer CTAs must collectively perform the allocation and de-allocation. Refer to the
[Issue Granularity](#tcgen05-issue-granularity) section.

When `.cta_group::2` is specified, the issuing warp must make sure that peer CTA is launched and its warps eventually participate in collective operations.

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group` qualifier.

The mandatory `.sync` qualifier indicates that the instruction causes the executing thread
to wait until all threads in the warp execute the same instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the
same instruction. In conditionally executed code, the instruction should only be used if it
is known that all threads in the warp evaluate the condition identically, otherwise behavior
is undefined.

The behavior of the instruction is undefined if all the threads in the warp do not use the
same values of `nCols` , or if any thread in the warp has exited.

*/

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

/*
Asynchronous collective load from tensor memory into registers.

tcgen05.ld.sync.aligned.shape1.num{.pack}.b32    r, [taddr];
.shape1 = { .16x64b, .16x128b, .16x256b, .32x32b }
.num    = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }
.pack   = { .pack::16b }

Instruction `tcgen05.ld` asynchronously loads data from the Tensor Memory at the location specified by the 32-bit address operand `taddr` into the destination register `r`, collectively across all threads of the warps.

All the threads in the warp must specify the same value of `taddr` , which must be the
base address of the collective load operation. Otherwise, the behavior is undefined.

The `.shape` qualifier and the `.num` qualifier together determines the total dimension of the data which is loaded from the Tensor Memory. 
The `.shape` qualifier indicates the base dimension of data to be accessed as described in the [Data Movement Shape](#tcgen05-data-movement-shape). 
The `.num` qualifier indicates the repeat factor on the base dimension resulting in the total dimension of the data that is accessed.

The destination operand `r` is a brace-enclosed vector expression consisting of one or more 32-bit
registers as per the value of `.shape` and `.num` .

The mandatory `.sync` qualifier indicates that `tcgen05.ld` causes the executing thread
to wait until all threads in the warp execute the same `tcgen05.ld` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the
same `tcgen05.ld` instruction. In conditionally executed code, a `tcgen05.ld` instruction
should only be used if it is known that all threads in the warp evaluate the condition
identically, otherwise behavior is undefined.

The behavior of `tcgen05.ld` is undefined if all threads do not use the same values of `taddr` ,
or if any thread in the warp has exited.

*/

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    // Load 4×32b from TMEM column
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    // Load 8×32b from TMEM column
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

/*

Waits for the completion of all prior asynchronous `tcgen05.ld` / `tcgen05.st` instructions.

```
tcgen05.wait_operation.sync.aligned;
.wait_operation = { .wait::ld, .wait::st }
```

Instruction `tcgen05.wait::ld` causes the executing thread to block until all prior `tcgen05.ld` operations issued by the executing thread have completed.
*/

__device__ __forceinline__ void tmem_load_fence_fn() {
    // Fence for TMEM loads (tcgen05.wait::ld).
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

/*

Specialized fence for the asynchronous tcgen05 operations.

```
tcgen05.fence::before_thread_sync ;
tcgen05.fence::after_thread_sync  ;
```

The instruction `tcgen05.fence::before_thread_sync` orders all the prior asynchronous `tcgen05` operations with respect to the subsequent `tcgen05` and the execution
ordering operations.

The instruction `tcgen05.fence::after_thread_sync` orders all the subsequent asynchronous `tcgen05` operations with respect to the prior `tcgen05` and the execution ordering
operations.

The `tcgen05.fence::*` instructions compose with execution ordering instructions across a thread scope and provide ordering between `tcgen05` instructions across the same scope.

The `tcgen05.fence::before_thread_sync` instructions behave as code motion fence for prior `tcgen05` instructions as they cannot be hoisted across. The `tcgen05.fence::after_thread_sync` instructions behave as code motion fence for subsequent `tcgen05` instructions as they cannot
be hoisted across.

*/


__device__ __forceinline__ void tcgen05_fence_after_fn() {
    // tcgen05 fence after thread sync.
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    // tcgen05 fence before thread sync.
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

/*

Perform the 5 th generation of matrix multiply and accumulate operation.

```
tcgen05.mma.cta_group.kind   [d-tmem],  a-desc,  b-desc, idesc,
                             { disable-output-lane }, enable-input-d {, scale-input-d};
tcgen05.mma.cta_group.kind   [d-tmem], [a-tmem], b-desc, idesc,
                             { disable-output-lane }, enable-input-d {, scale-input-d};
.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }
.cta_group = { .cta_group::1, .cta_group::2 }
```

Instruction `tcgen05.mma` is an asynchronous instruction which initiates an *MxNxK* matrix
multiply and accumulate operation, `D = A*B+D` where the `A` matrix is *MxK* , the `B` matrix is *KxN* , and the `D` matrix is *MxN* .

The operation of the form `D = A*B` is issued when the input predicate argument `enable-input-d` is false.

The 32-bit register operand `idesc` is the instruction descriptor as described
in [Instruction descriptor](make_instr_desc_fn) , specifies
the shapes, exact types, sparsity and other details of the input matrices,
output matrix and the matrix multiply and accumulate operation.

The qualifier `.cta_group::2` specifies that the matrix multiply and accumulate operation is performed on the
Tensor Memory of the executing thread's CTA and its peer CTA.

The instruction `tcgen05.mma` has single thread semantics, unlike the collective
instructions `mma.sync` or `wgmma.mma_async` . So, a single thread issuing the `tcgen05.mma` will result in the initiation of the whole matrix multiply and
accumulate operation. Refer to the section [Issue Granularity](#tcgen05-issue-granularity).

The qualifier `.kind` specifies the general kind of the element types of the multiplicand
matrices. The exact types of the elements of the input and output matrices for each MMA-kind
are specified in the Instruction descriptor.

The address operand `d-tmem` specifies the address of the destination and the accumulation
matrix `D` in the Tensor Memory . The address operand `a-tmem` specifies the address of the matrix `A` in the Tensor Memory.

The 64-bit register operand `a-desc` and `b-desc` are the matrix descriptors which
represent the matrices `A` and `B` in shared memory respectively. The format of the
matrix descriptor is described in [Matrix Descriptors](make_smem_desc_sm100_fn) .

*/

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    // SM100 UMMA instruction: cta_group::2, kind::f16 (BF16×BF16→FP32 or FP16xFP16→FP32). accum=0 clears accumulator, accum=1 accumulates.
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

/*

Makes the mbarrier object track the completion of all prior async-tcgen05 operations initiated
by the executing thread.

```
tcgen05.commit.cta_group.completion_mechanism{.shared::cluster}{.multicast}.b64 [mbar] {, ctaMask};
.completion_mechanism = { .mbarrier::arrive::one }
.cta_group            = { .cta_group::1, .cta_group::2 }
.multicast            = { .multicast::cluster }
```

The instruction `tcgen05.commit` is an asynchronous instruction which makes the mbarrier object,
specified by the address operand `mbar` , track the completion of all the prior asynchronous `tcgen05` operations, as listed in [mbarrier based completion mechanism](#tcgen05-memory-consistency-model-mbarrier-completion) ,
initiated by the executing thread. Upon the completion of the tracked asynchronous `tcgen05` operations, the signal specified by the `.completion_mechanism` is triggered by the system on the mbarrier object.

This instruction accesses its `mbarrier` operand using generic-proxy.

The instruction `tcgen05.commit.cta_group::2` tracks for the completion of all prior asynchronous `tcgen05` operations with `.cta_group::2` issued by the current thread.

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group` qualifier.

The qualifier `.mbarrier::arrive::one` indicates that upon the completion of the prior
asynchronous `tcgen05` operation issued by the current thread, an arrive-on operation, with
the count argument of 1, is signaled on the mbarrier object. The scope of the arrive-on operation
is the cluster scope.

The optional qualifier `.multicast::cluster` allows signaling on the mbarrier objects of multiple
CTAs in the cluster. Operand `ctaMask` specifies the CTAs in the cluster such that each bit
position in the 16-bit `ctaMask` operand corresponds to the `%cluster_ctarank` of the destination
CTA. The mbarrier signal is multicast to the same offset as `mbar` in the shared memory of each destination CTA.

*/

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    // UMMA commit with cta_group::2 multicast barrier arrive.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    // cta_group::2 TMA 2D load. Applies PEER_BIT_MASK to barrier address.
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    // Cluster-scoped arrive_expect_tx: signals a barrier in target_cta's shared memory.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
}

__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta) {
    // Arrive at a barrier on a remote CTA in the cluster.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a));
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    // Build SM100 UMMA instruction descriptor (FP16×FP16→FP32, K-major).
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (0u << 7);           // a_format = FP16
    d |= (0u << 10);          // b_format = FP16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    // Build SM100 UMMA SMEM descriptor (version=1, SWIZZLE_128B).
    // For K-major, SBO = 8 * TILE_K * sizeof(element)
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t pack_half_fn(uint32_t fp32_a, uint32_t fp32_b) {
    // Pack two FP32 values (as uint32 bit patterns) into one uint32 of two FP16.
    __half a = __float2half(__uint_as_float(fp32_a));
    __half b = __float2half(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}


__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    // 128-bit vectorized store to shared memory (st.shared.v4.b32).
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void tma_store_2d_sm100_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    // SM100 TMA 2D store from shared to global memory.
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(sa) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_store_half_row_fn(
    __half* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    // Read TMEM columns, convert FP32->FP16, store to global D. Each thread reads its own row (m_base + tid) for BN columns starting at n_base.
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
        __half* out = D + (uint64_t)m_idx * N + nc;
        if (nc     < N) out[0] = __float2half(f0);
        if (nc + 1 < N) out[1] = __float2half(f1);
        if (nc + 2 < N) out[2] = __float2half(f2);
        if (nc + 3 < N) out[3] = __float2half(f3);
    }
}

/*
Coalesced epilogue: TMEM -> SMEM staging -> vectorized coalesced global writes.

Phase 1: Each of 128 threads loads its row from TMEM (FP32), converts to FP16,
writes to shared memory. Phase 2: Coalesced 8-byte stores to global D.

REQUIRES blockDim.x == 128 (exactly 4 warps). The row mapping in Phase 2
uses "row = step*4 + warp_id"; other thread counts will produce wrong output.

Args:
    D: Global output pointer (FP16).
    smem_out: BM*BN FP16 buffer in shared memory (caller must allocate).
    M, N: Full matrix dimensions.
    m_block, n_block: Block start indices.
    bm, bn: Block tile sizes (BM, BN).
*/

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __half* D, __half* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    // Phase 1: TMEM -> SMEM
    // Each of the 128 threads owns one row (tid 0..127 -> row 0..127). Load 4
    // FP32 cols at a time from TMEM, convert to FP16, write to smem_out[tid*BN+col].
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2half(__uint_as_float(r0));
        smem_out[base + 1] = __float2half(__uint_as_float(r1));
        smem_out[base + 2] = __float2half(__uint_as_float(r2));
        smem_out[base + 3] = __float2half(__uint_as_float(r3));
    }
    __syncthreads();
    // Phase 2: SMEM -> Global (coalesced vectorized 8-byte writes)
    // REQUIRES 4 warps: each step processes 4 rows (one per warp). row = step*4 + warp_id.
    // Each warp: 32 lanes write 4 FP16 (8 bytes) each, covering 32*4=128 cols per row.
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