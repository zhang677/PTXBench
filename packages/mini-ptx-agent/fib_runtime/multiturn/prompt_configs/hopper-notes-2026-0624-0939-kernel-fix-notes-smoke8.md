Below is the hardware parameter of NVIDIA H100
| Component | Value |
|-----------|---------------|
| SM Count | 132 |
| Peak FP8 | ~1978 TFLOPS |
| Peak BF16 | ~989 TFLOPS |
| Cluster Size | Up to 16 SMs |
| Max Registers / CTA | 65536 |
| Max Registers / Thread | 255 |
| Max CTA size (# of threads) | 1024 |

| Level | Capacity | Latency | Bandwidth |
|-------|----|------|---------|
| Global (HBM) | 80 GB | ~500 cycles | 3.35 TB/s |
| L2 Cache | 50 MB | ~100 cycles | ~12TB/s |
| Shared Memory | 256 KB / SM (228 KB usable with 28KB L1 Cache) | ~30 cycles | 128 B/cycle/SM |
| Registers | 256 KB / SM  | 1 cycle | Unlimited (matched with compute) |

/*
    // Get the SM ID of the current thread
*/
__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

/*
    elect.sync elects one predicated active leader thread from among a set of threads specified by membermask. 
    The predicate destination p is set to True for the leader thread, and False for all other threads.
    Election of a leader thread happens deterministically, i.e. the same leader thread is elected for the same membermask every time.
*/
__device__ __forceinline__ bool elect_one_sync_fn() {
    // Elect a single thread in the warp. Returns 1 for elected, 0 for others.
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

/*

    A warpgroup is a set of four contiguous warps (128 threads) such that the warp-rank of the first warp is a multiple of 4.
    warp-rank of a warp is defined as: (%tid.x + %tid.y * %ntid.x  + %tid.z * %ntid.x * %ntid.y) / 32
    setmaxnreg provides a hint to the system to update the maximum number of per-thread registers owned by the executing warp to the value specified by the imm-reg-count operand.
    Qualifier .dec is used to release extra registers such that the absolute per-thread maximum register count is reduced from its current value to imm-reg-count. 
    Qualifier .inc is used to request additional registers such that the absolute per-thread maximum register count is increased from its current value to imm-reg-count.

    A pool of available registers is maintained per-CTA. Register adjustments requested by the setmaxnreg instructions are handled by supplying extra registers from this pool to the requesting warp or by releasing extra registers from the requesting warp to this pool, depending upon the value of the .action qualifier.

    The setmaxnreg.inc instruction blocks the execution until enough registers are available in the CTA’s register pool. After the instruction setmaxnreg.inc obtains new registers from the CTA pool, the initial contents of the new registers are undefined. The new registers must be initialized before they are used.

    The same setmaxnreg instruction must be executed by all warps in a warpgroup. After executing a setmaxnreg instruction, all warps in the warpgroup must synchronize explicitly before executing subsequent setmaxnreg instructions. If a setmaxnreg instruction is not executed by all warps in the warpgroup, then the behavior is undefined.

    Operand imm-reg-count is an integer constant. The value of imm-reg-count must be in the range 24 to 256 (both inclusive) and must be a multiple of 8.

    Changes to the register file of the warp always happen at the tail-end of the register file.

    The setmaxnreg instruction requires that the kernel has been launched with a valid value of maximum number of per-thread registers specified via the appropriate compilation via the appropriate compile-time option or the appropriate performance tuning directive. Otherwise, the setmaxnreg instruction may have no effect.

    When qualifier .dec is specified, the maximum number of per-thread registers owned by the warp prior to the execution of setmaxnreg instruction should be greater than or equal to the imm-reg-count. Otherwise, the behaviour is undefined.

    When qualifier .inc is specified, the maximum number of per-thread registers owned by the warp prior to the execution of setmaxnreg instruction should be less than or equal to the imm-reg-count. Otherwise, the behaviour is undefined.

    The mandatory .sync qualifier indicates that setmaxnreg instruction causes the executing thread to wait until all threads in the warp execute the same setmaxnreg instruction before resuming execution.

    The mandatory .aligned qualifier indicates that all threads in the warpgroup must execute the same setmaxnreg instruction. In conditionally executed code, setmaxnreg instruction should only be used if it is known that all threads in warpgroup evaluate the condition identically, otherwise the behavior is undefined.
*/

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_dec_sync_fn() {
    // Decrease the maximum number of registers for the warpgroup to NUM_REGS
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}


__device__ __forceinline__ float fast_exp2f_fn(float x) {
    // Find the base-2 exponential of a value. ln2 = 0.6931471805599453
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    // 128-bit vectorized store to shared memory (st.shared.v4.b32).
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    // Pack two FP32 values (as uint32 bit patterns) into one uint32 of two BF16.
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

/*
Cluster is a group of CTAs that run concurrently or in parallel and can synchronize and communicate with each other via shared memory. The executing CTA has to make sure that the shared memory of the peer CTA exists before communicating with it via shared memory and the peer CTA hasn’t exited before completing the shared memory operation.

Threads within the different CTAs in a cluster can synchronize and communicate with each other via shared memory. Cluster-wide barriers can be used to synchronize all the threads within the cluster. Each CTA in a cluster has a unique CTA identifier within its cluster (cluster_ctaid). Each cluster of CTAs has 1D, 2D or 3D shape specified by the parameter cluster_nctaid. Each CTA in the cluster also has a unique CTA identifier (cluster_ctarank) across all dimensions. The total number of CTAs across all the dimensions in the cluster is specified by cluster_nctarank. Threads may read and use these values through predefined, read-only special registers %cluster_ctaid, %cluster_nctaid, %cluster_ctarank, %cluster_nctarank.
*/

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// Cluster synchronization
/*
    Host-Side Cluster Launch

    ```cpp
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;       // grid.x MUST be divisible by cluster_size
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel_fn, arg0, arg1, ...);
    // The 2nd arg (kernel_fn) must be the kernel function pointer directly (for template deduction). Do NOT cast it to other types
    ```
    Note: Grid not divisible by cluster size — causes `cudaErrorInvalidConfiguration`.
*/

/*
    Performs barrier synchronization and communication within a cluster.
*/

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_arrive_fn() {
    asm volatile("barrier.cluster.arrive;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_wait_fn() {
    asm volatile("barrier.cluster.wait;\n" ::: "memory");
}


// MBarrier operations

/*  
    mbarrier.init initializes the mbarrier object at the location specified by the address operand [addr] with the unsigned 32-bit integer count. 
    Initialization of the mbarrier object involves:
    Initializing the current phase to 0.
    Initializing the expected arrival count to count.
    Initializing the pending arrival count to count.
    Initializing the tx-count to 0.
*/
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {

    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

/*
    fence.op_restrict.release.cluster;
    The fence instruction establishes an ordering between memory accesses requested by this thread. When .op_restrict is .mbarrier_init, the synchronizing effect of the fence only applies to the prior mbarrier.init operations executed by the same thread on mbarrier objects in .shared::cta state space. 
*/

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

/*
    mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64           state, [addr]{, count};
    mbarrier.arrive{.sem.scope}{.shared::cluster}.b64         _, [addr] {,count}
    mbarrier.arrive.expect_tx{.sem.scope}{.shared{::cta}}.b64 state, [addr], txCount;
    mbarrier.arrive.expect_tx{.sem.scope}{.shared::cluster}.b64   _, [addr], txCount;
    mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64  state, [addr], count;

    .sem   = { .release, .relaxed }
    .scope = { .cta, .cluster }

    A thread executing mbarrier.arrive performs an [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation on the mbarrier object at the location specified by the address operand [addr].
    The optional qualifier .expect_tx specifies that an [expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation) operation is performed prior to the arrive-on operation. The 32-bit unsigned integer operand txCount specifies the expectCount argument to the expect-tx operation. When both qualifiers .arrive and .expect_tx are specified, then the count argument of the arrive-on operation is assumed to be 1.
    mbarrier.arrive operation on an mbarrier object located in .shared::cta returns an opaque 64-bit register capturing the phase of the mbarrier object prior to the arrive-on operation in the destination operand state. Contents of the state operand are implementation specific. Optionally, sink symbol '_' can be used for the state argument.
    mbarrier.arrive operation on an mbarrier object located in .shared::cluster but not in .shared::cta cannot return a value. Sink symbol ‘_’ is mandatory for the destination operand for such cases.
    If the .sem qualifier is absent, .release is assumed by default. The .relaxed qualifier does not provide any memory ordering semantics and visibility guarantees.

*/

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
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
    // Arrive at a barrier on a remote CTA (target_cta) in the cluster.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta)); // mapa maps the address of the shared variable in the target CTA.
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a));
}

/*
    The mbarrier.try_wait operation tests for the completion of the current or the immediately preceding phase of an mbarrier object at the location specified by the operand [addr].
    mbarrier.try_wait is a potentially blocking instruction which tests for the completion of the phase. If the phase is not complete, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes OR before the phase completes following a system-dependent time limit.
    mbarrier.try_wait test for completion of the phase indicated by the 32-bit unsigned integer operand phaseParity, which is the integer parity of either the current phase or the immediately preceding phase of the mbarrier object.
    The .parity variant of the instructions test for the completion of the phase indicated by the operand phaseParity, which is the integer parity of either the current phase or the immediately preceding phase of the mbarrier object. An even phase has integer parity 0 and an odd phase has integer parity of 1. So the valid values of phaseParity operand are 0 and 1.
    try_wait operation is valid only for :
    the current incomplete phase, for which waitComplete returns False.
    the immediately preceding phase, for which waitComplete returns True.
    The following ordering of memory operations hold for the executing thread when mbarrier.try_wait having acquire semantics returns True :
    All memory accesses (except async operations) requested prior, in program order, to mbarrier.arrive having release semantics during the completed phase by the participating threads of the CTA are performed and are visible to the executing thread.
    All cp.async operations requested prior, in program order, to cp.async.mbarrier.arrive during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread.
    All cp.async.bulk asynchronous operations using the same mbarrier object requested prior, in program order, to mbarrier.arrive having release semantics during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread.
    All memory accesses requested after the mbarrier.try_wait, in program order, are not performed and not visible to memory accesses performed prior to mbarrier.arrive having release semantics, in program order, by other threads participating in the mbarrier.
*/
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

/*
    Accessing the same memory location across multiple proxies needs a cross-proxy fence.
*/
__device__ __forceinline__ void fence_proxy_async_fn() {
    // Generic async proxy fence. Required after manual shared memory writes (generic proxy) before WGMMA consumption (asynchronous "async" proxy)
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    // Async proxy fence on shared memory (fence.proxy.async.shared::cta).
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/*
    barrier{.cta}.sync{.aligned}      a{, b};
    barrier{.cta}.arrive{.aligned}    a, b;
    Performs barrier synchronization and communication within a CTA. Each CTA instance has sixteen barriers numbered 0..15.
    barrier{.cta} instructions can be used by the threads within the CTA for synchronization and communication.

    Operands a and b have type .u32. Source operand a specifies a logical barrier resource as an immediate constant or register with value 0 through 15. Operand b specifies the number of threads participating in the barrier. If no thread count is specified, all threads in the CTA participate in the barrier. When specifying a thread count, the value must be a multiple of the warp size. Note that a non-zero thread count is required for barrier{.cta}.arrive.

    Depending on operand b, either specified number of threads (in multiple of warp size) or all threads in the CTA participate in barrier{.cta} instruction. The barrier{.cta} instructions signal the arrival of the executing threads at the named barrier.

    barrier{.cta} instruction causes executing thread to wait for all non-exited threads from its warp and marks warps’ arrival at barrier. In addition to signaling its arrival at the barrier, the barrier{.cta}.sync instruction causes executing thread to wait for non-exited threads of all other warps participating in the barrier to arrive. barrier{.cta}.arrive does not cause executing thread to wait for threads of other participating warps.

    When a barrier completes, the waiting threads are restarted without delay, and the barrier is reinitialized so that it can be immediately reused.

    The barrier{.cta}.sync or or barrier{.cta}.arrive instruction guarantees that when the barrier completes, prior memory accesses requested by this thread are performed relative to all threads participating in the barrier. The barrier{.cta}.sync instruction further guarantees that no new memory access is requested by this thread before the barrier completes.

    A memory read (e.g., by ld or atom) has been performed when the value read has been transmitted from memory and cannot be modified by another thread participating in the barrier. A memory write (e.g., by st, red or atom) has been performed when the value written has become visible to other threads participating in the barrier, that is, when the previous value can no longer be read.

    Instruction barrier{.cta} has optional .aligned modifier. When specified, it indicates that all threads in CTA will execute the same barrier{.cta} instruction. In conditionally executed code, an aligned barrier{.cta} instruction should only be used if it is known that all threads in CTA evaluate the condition identically, otherwise behavior is undefined.

    Different warps may execute different forms of the barrier{.cta} instruction using the same barrier name and thread count. One example mixes barrier{.cta}.sync and barrier{.cta}.arrive to implement producer/consumer models. The producer threads execute barrier{.cta}.arrive to announce their arrival at the barrier and continue execution without delay to produce the next value, while the consumer threads execute the barrier{.cta}.sync to wait for a resource to be produced. The roles are then reversed, using a different barrier, where the producer threads execute a barrier{.cta}.sync to wait for a resource to consumed, while the consumer threads announce that the resource has been consumed with barrier{.cta}.arrive. Care must be taken to keep a warp from executing more barrier{.cta} instructions than intended (barrier{.cta}.arrive followed by any other barrier{.cta} instruction to the same barrier) prior to the reset of the barrier.
*/
__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
     // Named barrier for math warpgroup sync (before TMA store). Can be used for sync threads within a warpgroup that has `count` number of threads.
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void named_barrier_arrive_fn(int bar_id, int count) {
    asm volatile("barrier.arrive.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

// Grid-level barrier using atomic operations
__device__ unsigned int __grid_sync_count = 0;
__device__ volatile int __grid_sync_sense = 0;

__device__ __forceinline__ void grid_sync_fn() {
    __syncthreads();
    __threadfence();
    // Only (0,0,0) thread of each block participates in the grid barrier
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned int num_blocks = gridDim.x * gridDim.y * gridDim.z;
        // Atomically increment the arrival counter
        unsigned int arrived = atomicAdd(&__grid_sync_count, 1);
        if (arrived == num_blocks - 1) {
            // Last block: reset counter and flip sense
            __grid_sync_count = 0;
            __threadfence();
            __grid_sync_sense ^= 1;
        } else {
            // Wait for the sense to flip
            int expected = __grid_sync_sense ^ 1;
            while (__grid_sync_sense != expected) {
                // Spin-wait
            }
        }
    }
    __syncthreads();
}

/*

    // ---- Shared-Memory Bank Swizzling ----
    Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle. When loading and storing shared memory, bank conflicts arise if the same bank is used multiple times within a transaction, resulting in reduced bandwidth.
    The swizzle patterns define the mapping of the 16-byte chunks along the swizzle width to subgroups of four banks.
    The tables define the mapping of the 16-byte chunks along the 128 bytes to eight subgroups of four banks.

    In the below examples, positions with the same x lie in the same bank
    CU_TENSOR_MAP_SWIZZLE_NONE
    ```
    __shared__ int4 smem[8][8];
    smem[y][x] <-> smem[y][x]
    ```
    CU_TENSOR_MAP_SWIZZLE_128B
    ```
    __shared__ __align__(1024) int4 smem[8][8];
    smem[y][x] <-> smem[y][(y % 8) ^ x]
    ```
    CU_TENSOR_MAP_SWIZZLE_64B
    ``` 
    __shared__ __align__(512) int4 smem[4][8];
    smem[y][x] <-> smem[y][(y % 4) ^ x]
    ```
    CU_TENSOR_MAP_SWIZZLE_32B
    ```
    __shared__ __align__(256) int4 smem[2][8];
    smem[y][x] <-> smem[y][(y % 2) ^ x]
    ```

    // ---- Create TMA descriptors ----
    CUresult cuTensorMapEncodeTiled ( CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
    enum CUtensorMapDataType: Tensor map data type
    Values:
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    
    enum CUtensorMapFloatOOBfill: Tensor map out-of-bounds fill type
    Values:
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA

    enum CUtensorMapL2promotion: Tensor map L2 promotion type
    Values:
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B

    enum CUtensorMapSwizzle: Tensor map swizzling mode of shared memory banks
    Values:
    CU_TENSOR_MAP_SWIZZLE_NONE = 0
    CU_TENSOR_MAP_SWIZZLE_32B,        // Swizzle 16B chunks within 32B  span
    CU_TENSOR_MAP_SWIZZLE_64B,        // Swizzle 16B chunks within 64B  span
    CU_TENSOR_MAP_SWIZZLE_128B,       // Swizzle 16B chunks within 128B span
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,         // Swizzle 32B chunks within 128B span
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B, // Swizzle 32B chunks within 128B span, additionally swap lower 8B with upper 8B within each 16B for every alternate row
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,         // Swizzle 64B chunks within 128B span

    The box loaded by TMA will be used by mma in terms of atoms. Therefore, TMA setup is also related with the LBO and SBO of the mma instructions that will consume the data
    One core matrix is a matrix with a MN-mode and a K-mode. The chunks lay out along the major mode, and the spans lay out along the minor mode. 
    8 spans along the minor mode form a "core matrix". Each core matrix has a strided direction and a contiguous (leading) direction, 
    such that its length is 8 in the strided direction and 16 bytes in the contiguous direction. An Atom is composed of core matrices along the major mode.
    
    | Swizzling Mode	| Major-ness	| Atom Layout: MN-mode x K-mode |
    |----------------|-------------------|---------------------------|
    | 128B| MN| 128B × 8|
    | 128B| K  | 8 × 128B|
    | 64B | MN| 64B × 8 |
    | 64B | K  | 8 × 64B |
    | 32B | MN| 32B × 8 |
    | 32B | K  | 8 × 32B |
    | None| MN| 16B × 8 |
    | None| K  | 8 × 16B |

    Below uses M as example, N is the same.
    K-Major descriptor under 128B swizzled layouts. Just replace 128 with 64 or 32 for other swizzling mode
    ```
    constexpr uint32_t BOX_MMODE_DIM = BLOCK_M;
    constexpr uint32_t BOX_KMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
    constexpr uint32_t SBO = 8 * 128U; // 8 spans along M-mode (the strided dimension)
    // LBO is not used in the all swizzled case because wgmma's K-mode has 32B, which equals the smallest swizzling bytes

    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        K, M,
        BOX_KMODE_DIM, BOX_MMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```

    K-Major descriptor under non-swizzled layout:
    ```
    constexpr uint32_t BOX_MMODE_DIM = BLOCK_M;
    constexpr uint32_t BOX_KMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
    constexpr uint32_t SBO = 8 * 16U; // 8 spans along M-mode
    constexpr uint32_t LBO = BOX_MMODE_DIM * 16U; // Each core matrix has 16 bytes in the leading dimension
    
    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        K, M,
        BOX_KMODE_DIM, BOX_MMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```
    For M-Major the situation can be little bit more confusing because LBO and SBO mean visually different things for Swizzle vs no Swizzle case.
    MN-Major descriptor under 128B swizzled layouts. Just replace 128 with 64 or 32 for other swizzling mode
    ```
    constexpr uint32_t BOX_KMODE_DIM = BLOCK_K;
    constexpr uint32_t BOX_MMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
    constexpr uint32_t SBO = 8 * 128U; // 8 spans along K-mode
    constexpr uint32_t LBO = BOX_KMODE_DIM * 128U // The spans lay out along the minor mode. LBO jumps to the next groups of spans
    create_tma_2d_descriptor_2B(    
        &desc,
        ptr,
        M, K,
        BOX_MMODE_DIM, BOX_KMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```

    MN-Major descriptor under non-swizzled layouts
    ```
    constexpr uint32_t BOX_KMODE_DIM = BLOCK_K;
    constexpr uint32_t BOX_MMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
    constexpr uint32_t LBO = 8 * 16U; // 8 spans along K-mode
    constexpr uint32_t SBO = BOX_KMODE_DIM * 16U; // Switch LBO and SBO compared with the swizzle layouts

    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        M, K,
        BOX_MMODE_DIM, BOX_KMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```
    

    // The descriptor lives on the host stack and is passed by value. The __grid_constant__ attribute tells the compiler to place it in constant memory accessible to TMA hardware. Passing a plain struct by value WITHOUT __grid_constant__ causes illegal memory access. 
    // ---- Kernel signature ----
    __global__ void cta_gemm_kernel( 
        const __grid_constant__ CUtensorMap tma_A, ...)
    {
        // Use &tma_A directly — it's in grid-constant memory, accessible to TMA        
        tma_load_2d_fn(&tma_A, &mbar[s], smem_dst, coord0, coord1);    
        ...
    }

    // ---- Host setup ----      
    void run(tvm::ffi::TensorView A, ...) {     
        ...
        // 1. Create TMA descriptors for A
        CUtensorMap dA;
        __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
        create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, BOX_KMODE_DIM, BOX_MMODE_DIM, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        ...
        // 2. CTA Launch — pass by value, no cudaMalloc needed
        cta_gemm_kernel<<<grid, block, 0, stream>>>(dA, ...);
    }
    Note: `create_tma_2d_descriptor_2B` returns `CUresult`, not `cudaError_t`. Use a separate check macro.
*/
CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        dataType,
        2, // tensorRank
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, // BE CAREFUL: inner_box_bytes ≤ swizzle_size (for bf16 and CU_TENSOR_MAP_SWIZZLE_128B boxDim[0] ≤ 64). boxDim array specifies number of elements to be traversed along each of the tensorRank dimensions. Elements in boxDim must be non-zero, less than or equal to 256. These dimensions must 16 byte-aligned
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    // Prefetch TMA descriptor.
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

/*
    cp.async.bulk.tensor is a non-blocking instruction which initiates an asynchronous copy operation of tensor data from the location in .src state space to the location in the .dst state space.

    // global -> shared::cta
    cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
         [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}

    .dst =       { .shared::cta }
    .src =       { .global }
    .dim =       { .1d, .2d, .3d, .4d, .5d }
    .completion_mechanism = { .mbarrier::complete_tx::bytes }
    .cta_group = { .cta_group::1, .cta_group::2 } // Default is .cta_group::1
    .load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 } // Default is .tile
    .level::cache_hint =    { .L2::cache_hint }


    // global -> shared::cluster
    cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
         [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}
         {, ctaMask} {, cache-policy}

    .dst =       { .shared::cluster }
    .src =       { .global }
    .dim =       { .1d, .2d, .3d, .4d, .5d }
    .completion_mechanism = { .mbarrier::complete_tx::bytes }
    .cta_group = { .cta_group::1, .cta_group::2 } // Default is .cta_group::1
    .load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 } // Default is .tile
    .level::cache_hint =    { .L2::cache_hint }
    .multicast = { .multicast::cluster  }

    The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation. The tensorMap is accessed in tensormap proxy. 
    The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory. 
    The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk.tensor variant uses mbarrier based completion mechanism. Upon the completion of the asynchronous copy operation, the complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.
    The optional qualifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit ctaMask operand corresponds to the %cluster_ctarank of the destination CTA. The source data is multicast to the same offset as dstMem in the shared memory of each destination CTA. 
    When .cta_group is specified as .cta_group::1, the mbarrier signal is also multicasted to the same offset as mbar in the shared memory of the destination CTA.
    When .cta_group::1 is specified, the mbarrier object mbar that is specified must be in the shared memory of the same CTA as the shared memory destination dstMem.
*/

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    // c0 and c1 are tensorCoords; globalDim={dim0, dim1} → coords={coord_in_dim0, coord_in_dim1}.
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


/*
    // shared::cta -> global
    cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
         [tensorMap, tensorCoords], [srcMem] {, cache-policy}

    .dst =       { .global }
    .src =       { .shared::cta }
    .dim =       { .1d, .2d, .3d, .4d, .5d }
    .completion_mechanism = { .bulk_group }
    .load_mode = { .tile, .tile::scatter4, .im2col_no_offs }
    .level::cache_hint =    { .L2::cache_hint }
*/

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    // TMA store accesses shared memory across async and generic proxies
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}


/*
    cp.async.bulk.commit_group instruction creates a new per-thread bulk async-group and batches all prior cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions satisfying the following conditions into the new bulk async-group:
    - The prior cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions use bulk_group based completion mechanism, and
    - They are initiated by the executing thread but not committed to any bulk async-group.

    If there are no uncommitted cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions then cp.async.bulk.commit_group results in an empty bulk async-group.

    An executing thread can wait for the completion of all cp{.reduce}.async.bulk{.prefetch}{.tensor} operations in a bulk async-group using cp.async.bulk.wait_group.

    There is no memory ordering guarantee provided between any two cp{.reduce}.async.bulk{.prefetch}{.tensor} operations within the same bulk async-group.
*/

__device__ __forceinline__ void tma_store_commit_fn() {
    // TMA store arrive/commit.
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

/*
    cp.async.bulk.wait_group instruction will cause the executing thread to wait until only N or fewer of the most recent bulk async-groups are pending and all the prior bulk async-groups committed by the executing threads are complete. For example, when N is 0, the executing thread waits on all the prior bulk async-groups to complete. Operand N is an integer constant.

    By default, cp.async.bulk.wait_group instruction will cause the executing thread to wait until completion of all the bulk async operations in the specified bulk async-group. A bulk async operation includes the following:

    - Optionally, reading from the tensormap.
    - Reading from the source locations.
    - Writing to their respective destination locations.
    - Writes being made visible to the executing thread.
*/

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    // cp.async.bulk.wait_group requires immediate operand, use switch for common values
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

/*
    // global -> shared::cluster
    cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}
  [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}

    .dst =       { .shared::cluster }
    .src =       { .global }
    .completion_mechanism = { .mbarrier::complete_tx::bytes }
    .level::cache_hint =    { .L2::cache_hint }
    .multicast = { .multicast::cluster }

    // shared::cta -> global
    cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}
  [dstMem], [srcMem], size {, cache-policy} {, byteMask}

    .dst =       { .global }
    .src =       { .shared::cta }
    .completion_mechanism = { .bulk_group }
    .level::cache_hint =    { .L2::cache_hint }

    cp.async.bulk is a non-blocking instruction which initiates an asynchronous bulk-copy operation from the location specified by source address operand srcMem to the location specified by destination address operand dstMem.
    The 32-bit operand size specifies the amount of memory to be copied, in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The memory range [dstMem, dstMem + size - 1] must not overflow the destination memory space and the memory range [srcMem, srcMem + size - 1] must not overflow the source memory space. Otherwise, the behavior is undefined. The addresses dstMem and srcMem must be aligned to 16 bytes.
    When the destination of the copy is .shared::cta the destination address has to be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined.
    The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk variant uses mbarrier based completion mechanism. The complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.
    The modifier .bulk_group specifies that the cp.async.bulk variant uses bulk [async-group](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group) based completion mechanism.
    The copy operation in cp.async.bulk is treated as a weak memory operation and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope. The copy operation is performed in the async proxy.
*/

__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
}

__device__ __forceinline__ void tma_copy_1d_s2g_fn(void const* smem, void* gmem, int32_t bytes) {
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :: "l"(gmem), "r"(smem_int), "r"(bytes) : "memory");
}

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




# CUDA Instruction Usage Notes

Use these notes as an operator-agnostic cookbook for CUDA/PTX/API instruction usage.
Do not treat the presence of an instruction as optimization advice. Use a variant only when its shape, layout, address-space, and synchronization contract match the current kernel.
Task names are source metadata only; the instruction contract is the lesson.

## Summary

Repairs consistently replaced raw global/shared bulk-copy patterns with exact Hopper TMA tensor-copy forms tied to shared-memory mbarriers, then made WGMMA consume those tiles only after the required proxy and group-synchronization fences. The strongest recurring contracts are: CUtensorMap descriptors must be host-encoded with exact rank/dim/stride/box/swizzle semantics and passed by value as const __grid_constant__ kernel parameters; cp.async.bulk.tensor operands are ordered as [dst shared], [tensorMap, {coords}], [mbarrier]; mbarrier.init/fence/expect_tx/wait parity sequencing must match complete_tx byte totals; WGMMA descriptors must encode the actual shared-layout address/LBO/SBO/swizzle, sometimes including base-offset bits for swizzled addresses; and wgmma.wait_group requires an immediate operand after commit_group. Launch-side fixes also opt in exact dynamic shared-memory bytes before large-smem kernels.

## Instruction Cookbook

Each section documents how to use one primitive legally. Preserve operand order, address-space qualifiers, immediates, descriptor assumptions, synchronization order, and shape/layout constraints.

### 1. cuTensorMapEncodeTiled + const __grid_constant__ CUtensorMap kernel parameter contract

Metadata: source_count=8; tags=tensormap, grid-constant, host-api

#### Variant 1

Shape/context: BF16 rank-2 flattened layout [outer, inner=D] consumed as TMA coordinates {inner,row}; shared tile uses 128B swizzle with boxDim={64,128}

Example completeness: complete

Minimal correct pattern:

```cpp
CUtensorMap tma;
cuuint64_t globalDim[2] = {128, (cuuint64_t)(B * H * S)};
cuuint64_t globalStrides[1] = {128 * 2};
cuuint32_t boxDim[2] = {64, 128};
cuuint32_t elementStrides[2] = {1, 1};
CU_CHECK(cuTensorMapEncodeTiled(&tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, q_ptr, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
__global__ void k(const __grid_constant__ CUtensorMap tma_Q) { }
```

Wrong patterns:

```cpp
cuuint64_t globalDim[2] = {(cuuint64_t)(B * H * S), 128};
```

```cpp
cuuint64_t globalStrides[1] = {(cuuint64_t)(B * H * S) * 2};
```

```cpp
cuuint32_t boxDim[2] = {128, 64};
```

```cpp
__global__ void k(CUtensorMap tma_Q) { }
```

Operand contract:
- For flattened contiguous BF16 rows, globalDim must be {inner_dim_D, outer_dim_rows}, not reversed.
- globalStrides entries are in bytes; for BF16 rank-2 examples, globalStrides[0] = inner_dim * 2.
- boxDim entries are element counts, not bytes.
- In these 128B-swizzled examples, boxDim={64,128} is paired with WGMMA-compatible shared layouts.
- Descriptor passed by value to a kernel must be declared const __grid_constant__ CUtensorMap.

Required sequence:
- Populate globalDim/globalStrides/boxDim/elementStrides consistently with the flattened tensor layout.
- Call cuTensorMapEncodeTiled and check the returned CUresult.
- Launch the kernel with the CUtensorMap passed by value as a const __grid_constant__ parameter.
- Use the address of that kernel parameter in cp.async.bulk.tensor inline PTX.

Diagnostics:
- Illegal memory access or undefined TMA behavior when descriptor placement or dimension ordering is wrong.
- Wrong numerical tiles when globalDim/globalStrides/boxDim do not match the issued TMA coordinates.

Do not do:
- Do not treat flattened [outer, inner] storage as globalDim={outer, inner} when device code issues coordinates as {inner, outer}.
- Do not compute globalStrides in elements.
- Do not pass a plain by-value CUtensorMap kernel parameter without __grid_constant__.

#### Variant 2

Shape/context: BF16 rank-2 flattened layout [outer, inner=D] consumed by 2D TMA with 128B swizzle and smaller shared tile boxDim={64,64}

Example completeness: complete

Minimal correct pattern:

```cpp
CUtensorMap tma;
CU_CHECK(cuTensorMapEncodeTiled(
    &tma,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    2,
    Q_ptr,
    (const cuuint64_t[2]){128, global_S},
    (const cuuint64_t[1]){128 * 2},
    (const cuuint32_t[2]){64, 64},
    (const cuuint32_t[2]){1, 1},
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
__global__ void k(const __grid_constant__ CUtensorMap tma_Q, uint64_t* bar, __nv_bfloat16* sQ, int bh_row) { }
```

Wrong patterns:

```cpp
CUtensorMap* d_tma; cudaMalloc(&d_tma, sizeof(CUtensorMap));
```

```cpp
cuTensorMapEncodeTiled(d_tma, ..., (const cuuint64_t[1]){256}, ...);
```

```cpp
__global__ void k(CUtensorMap tma_Q, uint64_t* bar, __nv_bfloat16* sQ, int bh_row) { }
```

Operand contract:
- cuTensorMapEncodeTiled returns CUresult and must be checked as a driver-API result.
- For BF16 rank-2 flattened storage, globalStrides[0] is bytes: gmem_inner_dim * 2.
- Descriptor object must be in TMA-visible memory; these repairs use by-value const __grid_constant__ kernel parameters.

Required sequence:
- Encode the descriptor on host with rank=2 and exact dim/stride/box/swizzle arguments.
- Launch kernel with descriptor passed by value as const __grid_constant__ CUtensorMap.
- Use &tma_Q in cp.async.bulk.tensor inline PTX.

Diagnostics:
- Illegal memory access or undefined behavior when the descriptor is not accessible from TMA-visible memory.
- Mismatch from using CUDA runtime-style checking on a CUresult-returning API.

Do not do:
- Do not place the descriptor only in a cudaMalloc allocation and then treat that as the working contract shown here.
- Do not omit __grid_constant__ on the by-value kernel parameter.

#### Variant 3

Shape/context: BF16 rank-3 logical layout [D, S, B*H] consumed by cp.async.bulk.tensor.3d with coordinates {c0,c1,c2}; swizzle depends on consumer layout

Example completeness: complete

Minimal correct pattern:

```cpp
CUresult create_tma_3d_descriptor(CUtensorMap* d, void* gAddr,
                                 uint64_t dim0, uint64_t dim1, uint64_t dim2,
                                 uint32_t box0, uint32_t box1, uint32_t box2,
                                 CUtensorMapSwizzle swizzle) {
  cuuint64_t globalDim[3] = {dim0, dim1, dim2};
  cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
  cuuint32_t boxDim[3] = {box0, box1, box2};
  cuuint32_t elemStrides[3] = {1, 1, 1};
  return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gAddr, globalDim, globalStrides, boxDim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
```

Wrong patterns:

```cpp
cuuint64_t globalDim[2] = {dim0, dim1};
```

```cpp
return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gAddr, globalDim, globalStrides, boxDim, elemStrides, ...);
```

Operand contract:
- Rank argument to cuTensorMapEncodeTiled must match descriptor rank, coordinate-vector rank, and lengths of globalDim/globalStrides/boxDim.
- For BF16 rank-3, globalStrides are bytes: stride for dim1 is dim0 * 2 and stride for dim2 is dim0 * dim1 * 2.
- boxDim entries are element counts, not bytes.
- Swizzle mode must match the shared-memory layout consumed later.

Required sequence:
- Populate rank-3 globalDim/globalStrides/boxDim/elementStrides consistently with [D,S,BH] access.
- Call cuTensorMapEncodeTiled and check the CUresult.
- Pass the descriptor by value to a kernel as const __grid_constant__ CUtensorMap.

Diagnostics:
- Wrong data slice selection when a logical outer dimension is omitted from the descriptor.
- Descriptor creation failure or incorrect TMA interpretation when rank or strides are inconsistent.

Do not do:
- Do not encode a [D,S,BH] access pattern with only a rank-2 descriptor if device code varies the omitted dimension.
- Do not mix a swizzled consumer with a descriptor created for a different swizzle.

### 2. cp.async.bulk.tensor.{2d,3d}.shared::{cta|cluster}.global.mbarrier::complete_tx::bytes

Metadata: source_count=8; tags=tma, ptx, async-copy

#### Variant 1

Shape/context: BF16 2D tensor-map load from flattened [B*H*S, D] into 128B-swizzled shared tiles using coordinates {d_offset,row_offset}; shared::cluster destination form

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
         "l"((uint64_t)d),
         "r"((uint32_t)__cvta_generic_to_shared(bar)),
         "r"(c0), "r"(c1)
      : "memory");
}
```

Wrong patterns:

```cpp
asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
```

Operand contract:
- Operand order is exactly [dst shared], [tensorMap, {coord0, coord1}], [mbarrier].
- Destination and mbarrier operands are shared-memory addresses converted with __cvta_generic_to_shared and passed with "r".
- tensorMap operand is a 64-bit generic address to a CUtensorMap in param/const/global space and is passed with "l".
- Coordinates are tensor coordinates in descriptor dimension order, not byte offsets.
- This working variant uses .shared::cluster.global even when the destination is CTA shared memory.

Required sequence:
- Create and pass a valid rank-2 CUtensorMap descriptor.
- Initialize and arm a shared mbarrier for the expected byte count.
- Issue one or more cp.async.bulk.tensor.2d operations tied to that barrier.
- Wait on the barrier before consuming the shared tile.

Diagnostics:
- Evaluator/profiler warning requesting cp.async.bulk.tensor usage.
- Timeouts or incorrect layout when raw pointer bulk-copy is substituted for tensor-map loads.

Do not do:
- Do not substitute cp.async.bulk.shared::cta.global on a raw pointer when the consumer expects tensor-map swizzle/layout semantics.
- Do not treat c0/c1 as byte offsets.

#### Variant 2

Shape/context: BF16 2D tensor-map load from flattened [B*H*S, D] into shared tiles using coordinates {d_offset,row_offset}; shared::cta destination form

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
         "l"((uint64_t)d),
         "r"((uint32_t)__cvta_generic_to_shared(bar)),
         "r"(c0), "r"(c1) : "memory");
}
```

Wrong patterns:

```cpp
asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(s_ptr), "l"(gmem), "r"(bytes), "r"(m_ptr) : "memory");
```

Operand contract:
- Operand order is [dstMem], [tensorMap, {coord0, coord1}], [mbarrier].
- Coordinate-vector rank must match descriptor rank exactly.
- The mbarrier operand is a shared-memory address converted with __cvta_generic_to_shared.
- For .mbarrier::complete_tx::bytes, completion units are bytes.

Required sequence:
- Initialize and publish the mbarrier.
- Call mbarrier.arrive.expect_tx with the total byte count for the phase.
- Issue one or more cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes instructions.
- Wait with mbarrier.try_wait.parity before reading the destination.

Diagnostics:
- Deadlock or undefined behavior if descriptor rank/layout does not match the device-side coordinate usage.

Do not do:
- Do not issue a lower-rank TMA instruction than the descriptor/access pattern requires.

#### Variant 3

Shape/context: BF16 rank-3 logical layout [D, S, B*H] loaded into shared memory with cp.async.bulk.tensor.3d; shared::cluster destination form

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
      :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
         "l"((uint64_t)d),
         "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
         "r"(c0), "r"(c1), "r"(c2) : "memory");
}
```

Wrong patterns:

```cpp
asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
```

Operand contract:
- Operand order is [dst shared], [tensorMap, {coord0, coord1, coord2}], [mbarrier].
- Coordinates are logical tensor coordinates, not byte offsets.
- tensorMap operand must be a generic address to a CUtensorMap object in parameter/constant/global memory.
- This working variant uses .shared::cluster destination with cluster size 1.

Required sequence:
- Build a legal rank-3 CUtensorMap descriptor on host and pass it by value as __grid_constant__.
- Initialize shared mbarrier storage.
- Issue mbarrier.arrive.expect_tx.shared::cta.b64 with the total byte count for the tracked 3D loads.
- Issue one or more cp.async.bulk.tensor.3d operations using that barrier.
- Wait for completion before reading shared memory.
- Execute fence.proxy.async before WGMMA or other async-proxy consumers.

Diagnostics:
- Evaluator warning that cp.async.bulk.tensor was not exercised.
- Timeouts from replacing tensor-map loads with generic bulk copies.

Do not do:
- Do not substitute cp.async.bulk.shared::cta.global for tiled rank-3 tensor loads.
- Do not treat c0/c1/c2 as flat byte offsets.

#### Variant 4

Shape/context: BF16 rank-3 logical layout [D, S, B*H] loaded into shared memory with cp.async.bulk.tensor.3d; shared::cta destination form

Example completeness: partial

Missing details:
- Host-side kernel parameter declaration for const __grid_constant__ CUtensorMap
- Full tensor-map creation and bounds requirements for all box dimensions

Partial pattern:

```cpp
__device__ __forceinline__ void tma_load_3d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
         "l"((uint64_t)d),
         "r"(c0), "r"(c1), "r"(c2),
         "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}
```

Wrong patterns:

```cpp
cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes ... // used with a descriptor/access pattern that needs rank 3
```

Operand contract:
- Coordinate-vector rank must match tensor-map rank exactly: 3 coordinates for a 3D descriptor.
- Destination operand is a shared-memory address converted with __cvta_generic_to_shared.
- tensorMap operand is passed as a 64-bit generic address with "l"((uint64_t)d).
- mbarrier operand is a shared-memory address converted with __cvta_generic_to_shared.

Required sequence:
- Initialize mbarrier with mbarrier.init.shared.b64.
- Publish initialization with fence.mbarrier_init.release.cluster.
- Call mbarrier.arrive.expect_tx.shared::cta.b64 with the total byte count.
- Issue cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes operations.
- Wait with mbarrier.try_wait.parity.shared.b64 before consuming the tile.

Diagnostics:
- Incorrect data if descriptor rank/layout omits indexed logical dimensions.

Do not do:
- Do not describe a multi-slice tensor with a lower-rank descriptor and then vary omitted dimensions outside the descriptor contract.

### 3. mbarrier.init.shared.b64 + fence.mbarrier_init.release.cluster + mbarrier.arrive.expect_tx.shared::{cta|none}.b64 + mbarrier.try_wait.parity.shared.b64

Metadata: source_count=8; tags=mbarrier, synchronization, tma-completion

#### Variant 1

Shape/context: Shared .b64 CTA-local mbarrier tracking one TMA phase whose completion units are bytes; explicit .shared::cta expect_tx form

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait_phase(uint64_t* bar, uint32_t phase) {
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}
```

Wrong patterns:

```cpp
mbarrier.arrive.shared.b64 _, [%0];
```

```cpp
mbarrier.try_wait.parity.shared.b64 %p, [%0], %1; // without prior init/fence
```

```cpp
mbarrier_wait_fn(bar, 0); // after issuing TMA loads without prior expect_tx
```

Operand contract:
- mbarrier object must live in shared memory, be .b64, and be 8-byte aligned.
- Only mbarrier.init is legal before initialization; any other mbarrier op on uninitialized storage is undefined.
- arrive.expect_tx both performs one arrive-on and increments tx-count by tx_bytes for .complete_tx::bytes operations.
- txCount must equal the total completed bytes of the tracked async operations for the phase.
- try_wait.parity phase operand must be parity 0 or 1 of the current or immediately preceding phase.

Required sequence:
- One thread executes mbarrier.init.shared.b64.
- Execute fence.mbarrier_init.release.cluster.
- Synchronize threads before other threads touch the barrier object.
- Before issuing tracked async copies, call mbarrier.arrive.expect_tx.shared::cta.b64 with exact byte total.
- Issue the cp.async.bulk.tensor operations bound to that barrier.
- Poll mbarrier.try_wait.parity.shared.b64 until it succeeds.
- Toggle parity only when reusing the barrier for the next phase.

Diagnostics:
- Deadlock or timeout when tx-count mismatch prevents wait completion.
- Undefined behavior when waiting on or arriving at an uninitialized barrier.

Do not do:
- Do not emulate expect_tx with a plain mbarrier.arrive after issuing TMA.
- Do not hardcode parity 0 across a multi-phase barrier reuse loop.
- Do not set txCount smaller or larger than the sum of bytes reported by tracked complete_tx operations.

#### Variant 2

Shape/context: Shared .b64 barrier tracking TMA completion in examples that use mbarrier.arrive.expect_tx.shared.b64 spelling rather than explicit ::cta

Example completeness: complete

Minimal correct pattern:

```cpp
if (threadIdx.x == 0) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)), "r"(1));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__syncthreads();
if (threadIdx.x == 0) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)), "r"(32768) : "memory");
}
asm volatile("{\n.reg .pred P;\nWAIT_%=:\n mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n @!P bra WAIT_%=;\n}" :: "r"((uint32_t)__cvta_generic_to_shared(&bar)), "r"(0));
```

Wrong patterns:

```cpp
asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes ..."); asm volatile("mbarrier.arrive.shared.b64 _, [%0];");
```

Operand contract:
- Shared-memory barrier storage is still .b64, 8-byte aligned, and addressed via __cvta_generic_to_shared.
- For .complete_tx::bytes async operations, expect_tx units are bytes.
- The init count in these examples is 1 because one thread performs the arrive.expect_tx.

Required sequence:
- mbarrier.init.shared.b64
- fence.mbarrier_init.release.cluster
- __syncthreads() before first use by other threads
- mbarrier.arrive.expect_tx.shared.b64
- tracked async operations
- mbarrier.try_wait.parity.shared.b64

Diagnostics:
- Undefined behavior from using mbarrier.arrive plus async copies without matching expect_tx accounting.

Do not do:
- Do not pair cp.async.bulk.tensor completion with plain mbarrier.arrive when byte completion tracking is required.

### 4. fence.proxy.async{.shared::cta} + wgmma.fence.sync.aligned + wgmma.commit_group.sync.aligned + wgmma.wait_group.sync.aligned

Metadata: source_count=8; tags=wgmma, proxy-fence, group-sync

#### Variant 1

Shape/context: Shared tiles produced by TMA completion or generic shared writes, then consumed by WGMMA through the async proxy; generic fence spelling without explicit shared::cta qualifier

Example completeness: complete

Minimal correct pattern:

```cpp
mbarrier_wait_fn(bar_q, 0);
asm volatile("fence.proxy.async;" ::: "memory");
asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
wgmma_m64n64k16_ss_fn<0, 0>(acc, da, db);
asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
```

Wrong patterns:

```cpp
mbarrier_wait_fn(bar_q, 0); wgmma_m64n64k16_ss_fn<0, 0>(acc, da, db);
```

```cpp
wgmma_m64n64k16_ss_fn<0, 0>(acc, da, db); asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); float x = acc[0];
```

```cpp
wgmma_m64n64k16_ss_fn<1, 1>(acc00, da_top, db_left); asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); asm volatile("fence.proxy.async;" ::: "memory");
```

Operand contract:
- fence.proxy.async has no operands and is the cross-proxy fence from generic-proxy shared writes to async-proxy consumers.
- wgmma.fence.sync.aligned must execute before issuing WGMMA after operand production.
- wgmma.wait_group.sync.aligned takes an immediate constant operand, not a register.
- All 128 threads of the participating warpgroup must execute the same WGMMA fence/issue/commit/wait sequence.

Required sequence:
- Wait for TMA completion if shared data came from cp.async.bulk.tensor.
- Execute fence.proxy.async before WGMMA reads the shared tile.
- Synchronize participating threads as required by the producer pattern.
- Execute wgmma.fence.sync.aligned.
- Issue one or more wgmma.mma_async instructions.
- Execute wgmma.commit_group.sync.aligned.
- Execute wgmma.wait_group.sync.aligned N before consuming accumulators.

Diagnostics:
- Silent wrong answers from stale shared-memory visibility across proxy domains.
- Data hazards from consuming accumulator results before committed WGMMA groups complete.

Do not do:
- Do not assume mbarrier completion or __syncthreads alone replaces fence.proxy.async.
- Do not place fence.proxy.async after WGMMA issue.
- Do not read accumulator registers after mma_async without matching commit+wait.

#### Variant 2

Shape/context: Shared memory written through generic proxy inside a CTA, then consumed by WGMMA; explicit fence.proxy.async.shared::cta spelling plus accumulator liveness fencing

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ __forceinline__ void wgmma_fence_operand(float* a, int n) {
  #pragma unroll
  for (int i = 0; i < n; i++) asm volatile("" : "+f"(a[i]) :: "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
```

Wrong patterns:

```cpp
for (int r = 0; r < 32; r++) acc_frag[r] = 0.0f; wgmma_m64n64k16_ss<1, 1>(acc_frag, da, db);
```

Operand contract:
- Use fence.proxy.async.shared::cta when the relevant shared-memory writes were generic-proxy accesses in the CTA.
- Accumulator liveness fencing in these examples uses asm volatile("" : "+f"(a[i]) :: "memory") on each FP accumulator register.
- wgmma.fence.sync.aligned remains required before WGMMA issue.

Required sequence:
- Produce or store shared operands.
- Fence accumulator operands if inline PTX consumes/modifies them.
- Execute fence.proxy.async.shared::cta.
- Execute wgmma.fence.sync.aligned.
- Issue wgmma.mma_async, then commit and wait.

Diagnostics:
- Silent wrong answers from stale shared-memory data in the async proxy.
- Compiler reordering or register-materialization hazards around inline-asm accumulator arrays.

Do not do:
- Do not skip the async-proxy fence when shared operands were written through the generic proxy.
- Do not assume accumulator arrays remain correctly materialized without explicit operand fences around inline asm.

### 5. WGMMA shared-memory descriptor encoding (make_wgmma_desc / make_desc)

Metadata: source_count=8; tags=wgmma-descriptor, shared-layout, swizzle

#### Variant 1

Shape/context: 128B-swizzled BF16 shared-memory operands where descriptor must include swizzle base-offset bits derived from the shared address

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ static inline uint64_t make_wgmma_desc(const void* ptr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle_mode) {
  uint64_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
  uint64_t desc = 0;
  desc |= (addr & 0x3FFFF) >> 4;
  desc |= ((uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16);
  desc |= ((uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32);
  uint32_t base_offset = (addr >> 7) & 0x7;
  desc |= ((uint64_t)base_offset << 49);
  desc |= ((uint64_t)swizzle_mode << 62);
  return desc;
}
```

Wrong patterns:

```cpp
desc |= ((uint64_t)swizzle_mode << 62); return desc; // omits base_offset for swizzled layout
```

```cpp
uint64_t da = make_wgmma_desc(pa, 2048, 1024, 1);
```

Operand contract:
- Descriptor address field stores shared address bits as (addr & 0x3FFFF) >> 4.
- LBO and SBO fields store byte offsets encoded in 16-byte units: (value & 0x3FFFF) >> 4.
- For swizzled modes, bits [51:49] must encode matrix base offset = (addr >> 7) & 0x7 when the swizzle pattern start must be represented.
- Bits [63:62] select swizzle mode; 1 means 128B swizzle.
- LBO/SBO must match the actual shared-memory core-matrix layout consumed by WGMMA, not the original logical tensor layout.

Required sequence:
- Lay out or load the shared tile using the same swizzle convention expected by WGMMA.
- Build the descriptor from the exact shared pointer and correct LBO/SBO/swizzle/base-offset values.
- Use the descriptor with matching WGMMA trans immediates after proxy and WGMMA fences.

Diagnostics:
- Silent wrong results when swizzle/base-offset/stride fields do not match the shared tile.
- Corruption when base_offset is omitted for swizzled shared addresses.

Do not do:
- Do not omit base_offset bits for swizzled shared-memory descriptors unless alignment makes them unnecessary for that exact address.
- Do not cargo-cult LBO values like 2048 from a different layout.

#### Variant 2

Shape/context: 128B-swizzled BF16 shared-memory operands using repaired K-major 64x64 tiles where valid descriptors use LBO=1, SBO=1024, swizzle_mode=1

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ static inline uint64_t make_desc(void* ptr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle_mode) {
  uint64_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
  uint64_t desc = 0;
  desc |= (addr & 0x3FFFF) >> 4;
  desc |= ((uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16);
  desc |= ((uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32);
  desc |= ((uint64_t)swizzle_mode << 62);
  return desc;
}
uint64_t dA = make_desc(sQ0, 1, 1024, 1);
uint64_t dB = make_desc(sK0, 1, 1024, 1);
```

Wrong patterns:

```cpp
uint64_t dA = make_desc(sQ0, 1024, 1, 0);
```

```cpp
uint64_t da00 = make_wgmma_desc(&sQ[k_off_bytes], 128, 1024, 1);
```

Operand contract:
- Address/LBO/SBO fields are still encoded as shared address >> 4 and byte offsets >> 4.
- For the repaired 128B-swizzled K-major BF16 tiles in these examples, valid descriptors use swizzle_mode=1, LBO=1, SBO=1024.
- The descriptor base pointer must point at the exact shared-memory K-slice/core-matrix start.

Required sequence:
- Populate shared memory in the exact swizzled layout that LBO=1 and SBO=1024 describe.
- Construct descriptors from the exact shared tile base pointer.
- Issue WGMMA with matching transpose immediates.

Diagnostics:
- Wrong answers from mismatched stride or swizzle fields even when compilation succeeds.

Do not do:
- Do not reuse row-major byte strides such as BLOCK_N*2 as WGMMA SBO for a swizzled tile.
- Do not mix descriptor fields from another shared layout.

#### Variant 3

Shape/context: Mixed repaired layouts showing additional valid descriptor field combinations beyond LBO=1/SBO=1024, such as V or transposed/shared-major variants

Example completeness: partial

Missing details:
- The notes do not provide a standalone derivation of why the alternate 8192-byte LBO is correct for that exact layout.

Partial pattern:

```cpp
uint64_t da0 = make_wgmma_desc((char*)sQ0 + k * 32, 1, 1024, 1);
uint64_t db0 = make_wgmma_desc((char*)sK0 + k * 32, 1, 1024, 1);
uint64_t db1 = make_wgmma_desc((char*)sV1 + k * 2048, 8192, 1024, 1);
```

Wrong patterns:

```cpp
uint64_t da_top = make_wgmma_desc(pa, 2048, 1024, 1);
```

Operand contract:
- Descriptor fields must match the actual shared-memory operand layout and the chosen trans_a/trans_b interpretation.
- One repaired example uses a V descriptor with LBO=8192 and SBO=1024 for its shared layout, distinct from the Q/K shared-shared case.

Required sequence:
- Construct descriptor from a shared-memory pointer plus exact LBO/SBO/swizzle for that operand layout.
- Pair it with matching WGMMA trans immediates.

Diagnostics:
- Silent wrong results from descriptor stride/transpose mismatch.

Do not do:
- Do not assume one LBO/SBO pair is valid for every operand family or stage.

### 6. wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 and related warpgroup contracts

Metadata: source_count=7; tags=wgmma, ptx, warpgroup

#### Variant 1

Shape/context: Shared/shared BF16 operands described by 64-bit WGMMA descriptors; 128-thread warpgroup; 32 FP32 accumulators per thread

Example completeness: complete

Minimal correct pattern:

```cpp
template<int trans_a, int trans_b>
__device__ __forceinline__ void wgmma_m64n64k16_ss_fn(float* c, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%32,%33,p,1,1,%34,%35;\n}\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]),
        "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]),
        "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]),
        "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
      : "l"(da), "l"(db), "n"(trans_a), "n"(trans_b));
}
```

Wrong patterns:

```cpp
asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0}, %1, %2, 1, 1, 0, 0;" : "+f"(acc[0]) : "l"(dA), "l"(dB));
```

```cpp
wgmma_m64n64k16_ss_fn<1, 1>(acc, da, db); // when repaired layout requires <0,0> or <0,1>
```

Operand contract:
- Instruction destination is a tuple of exactly 32 FP32 accumulator registers per thread for m64n64k16.f32 output.
- A and B operands are 64-bit shared-memory descriptors passed with "l" constraints.
- trans_a and trans_b are immediates passed with "n" and must match how the shared core matrices are interpreted.
- All 128 threads of the warpgroup must execute the same WGMMA instruction sequence.

Required sequence:
- Prepare shared-memory descriptors matching the actual tile layout.
- Fence producer visibility and execute wgmma.fence.sync.aligned.
- Issue one or more wgmma.mma_async instructions.
- Commit and wait before consuming results.

Diagnostics:
- Incorrect results from wrong trans_a/trans_b or malformed destination tuple.
- Undefined behavior when only part of a warpgroup executes the instruction.

Do not do:
- Do not issue WGMMA from a single warp or 64-thread CTA when the issuing region requires a full warpgroup.
- Do not hardcode transpose flags from another shared layout.

#### Variant 2

Shape/context: Warpgroup participation / launch shape contract for any WGMMA-issuing region

Example completeness: partial

Missing details:
- Valid descriptor values are stubbed as zero; only participation and launch contract are illustrated.

Partial pattern:

```cpp
__global__ void kernel_using_wgmma() {
  float acc[32];
  uint64_t da = 0;
  uint64_t db = 0;
  for (int i = 0; i < 32; ++i) acc[i] = 0.0f;
  wgmma_m64n64k16_ss<1, 1>(acc, da, db);
}

void launch() {
  dim3 block(128);
  kernel_using_wgmma<<<1, block>>>();
}
```

Wrong patterns:

```cpp
dim3 block(64); kernel_using_wgmma_bad<<<1, block>>>();
```

Operand contract:
- WGMMA is warpgroup-scoped and requires all 128 threads of the warpgroup to execute the same aligned instruction stream.
- Helper mappings used around WGMMA may assume the full 128-thread warpgroup layout.

Required sequence:
- Launch with at least one complete 128-thread warpgroup for each WGMMA-issuing group.
- Ensure all participating threads reach the same wgmma.fence, mma_async, commit_group, and wait_group instructions.

Diagnostics:
- Incorrect execution model or undefined behavior when attempting WGMMA from half a warpgroup.

Do not do:
- Do not launch 64-thread CTAs for code paths that execute warpgroup MMA instructions.
- Do not predicate away only part of the warpgroup around aligned WGMMA instructions.

### 7. wgmma.wait_group.sync.aligned immediate-operand contract

Metadata: source_count=4; tags=wgmma-wait, immediate, ptxas

#### Variant 1

Shape/context: Hopper WGMMA wait on pending committed groups; valid only with a compile-time immediate operand executed uniformly by the warpgroup

Example completeness: complete

Minimal correct pattern:

```cpp
template<int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}

__device__ void use_wait() {
  wgmma_wait<0>();
}
```

Wrong patterns:

```cpp
__device__ __forceinline__ void wgmma_wait(uint32_t n) { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "r"(n) : "memory"); }
```

```cpp
uint32_t n = 0; wgmma_wait(n);
```

Operand contract:
- The wait-group operand must be an immediate constant, not a register.
- Inline asm must bind the operand with constraint "n", not "r".
- Instruction spelling is exactly wgmma.wait_group.sync.aligned <imm>.
- All participating warpgroup threads must execute the same aligned wait instruction.

Required sequence:
- Issue one or more wgmma.mma_async.sync.aligned instructions.
- Execute wgmma.commit_group.sync.aligned.
- Execute wgmma.wait_group.sync.aligned <imm> before consuming results.

Diagnostics:
- ptxas error: Arguments mismatch for instruction 'wgmma.wait_group'

Do not do:
- Do not pass a runtime variable to wgmma.wait_group.
- Do not use a register operand constraint for the wait-group count.

### 8. cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize) and large-smem launch contract

Metadata: source_count=6; tags=launch, dynamic-smem, host-api

#### Variant 1

Shape/context: Ordinary kernel launch path requesting dynamic shared memory above the default per-block limit

Example completeness: complete

Minimal correct pattern:

```cpp
size_t smem_size = 7 * 64 * 64 * sizeof(__nv_bfloat16) + 16;
CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));
attention_kernel<<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    B, H, S, tma_Q, tma_K, tma_V);
CUDA_CHECK(cudaGetLastError());
```

Wrong patterns:

```cpp
attention_kernel<<<grid, block, smem_size, stream>>>(...); // missing cudaFuncSetAttribute(..., smem_size)
```

Operand contract:
- The byte count passed to cudaFuncSetAttribute must match the dynamic shared-memory byte count used at launch.
- The first argument must be the actual kernel symbol used for launch.
- cudaGetLastError should be checked immediately after launch.

Required sequence:
- Compute exact dynamic shared-memory byte requirement.
- Call cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size).
- Launch the kernel with the same smem_size as the third execution-configuration argument.
- Check cudaGetLastError() after launch.

Diagnostics:
- Launch failure or invalid configuration when requesting large dynamic shared memory without opt-in.

Do not do:
- Do not set one shared-memory byte count in cudaFuncSetAttribute and launch with another.
- Do not omit post-launch cudaGetLastError when debugging launch configuration.

#### Variant 2

Shape/context: cudaLaunchKernelEx path for a Hopper kernel using by-value CUtensorMap parameters, cluster attributes, and large dynamic shared memory

Example completeness: partial

Missing details:
- The surrounding host definitions of grid_x, stream, O_ptr, LSE_ptr, B, H, S, D, stride_h, stride_s, and scale are omitted.

Partial pattern:

```cpp
size_t smem_bytes = 197000;
CUDA_CHECK(cudaFuncSetAttribute(mha_impl::attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
cudaLaunchConfig_t config = {};
config.gridDim = dim3(grid_x, 1, 1);
config.blockDim = dim3(128, 1, 1);
config.dynamicSmemBytes = smem_bytes;
config.stream = stream;
CUDA_CHECK(cudaLaunchKernelEx(&config, mha_impl::attention_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, B, H, S, D, stride_h, stride_s, scale));
```

Wrong patterns:

```cpp
mha_impl::attention_kernel<<<grid, dim3(128,1,1), smem_bytes, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, B, H, S, D, stride_h, stride_s, scale);
```

Operand contract:
- The dynamic shared-memory byte count passed to cudaFuncSetAttribute must match config.dynamicSmemBytes.
- cudaLaunchKernelEx takes the actual kernel function pointer as its second argument, followed by kernel parameters by value.
- ClusterDimension must be supplied through cudaLaunchAttribute; the shown working example uses a 1x1x1 cluster.

Required sequence:
- Call cudaFuncSetAttribute with the exact dynamic shared-memory byte count.
- Populate cudaLaunchConfig_t with gridDim, blockDim, dynamicSmemBytes, and stream.
- Populate cluster launch attributes.
- Launch with cudaLaunchKernelEx.

Diagnostics:
- Launch failures from exceeding default dynamic shared-memory limits or using the wrong launch form for the repaired kernel path.

Do not do:
- Do not request large dynamic shared memory at launch without the matching attribute opt-in.
- Do not cast the kernel function pointer to another type when passing it to cudaLaunchKernelEx.

### 9. Compile-time/device-code helper contracts (headers, declaration order, min overload)

Metadata: source_count=2; tags=cuda-cpp, compile-contracts, headers

#### Variant 1

Shape/context: CUDA device code using FLT_MAX and launch-index arithmetic

Example completeness: complete

Minimal correct pattern:

```cpp
#include <float.h>
int blocks_per_head_seq = (S + 63) / 64;
int total_bid = blockIdx.x;
int bm = total_bid % blocks_per_head_seq;
float m[2] = {-FLT_MAX, -FLT_MAX};
```

Wrong patterns:

```cpp
int blocks_per_head_seq = (S + BLOCK_M - 1) / BLOCK_M;
int total_bid = bid % (H * blocks_per_seq);
float m[2] = {-FLT_MAX, -FLT_MAX};
```

Operand contract:
- FLT_MAX requires including <float.h> in the CUDA translation unit.
- Identifiers used in launch-index arithmetic must be declared exactly and reused consistently.

Required sequence:
- Include <float.h> before using FLT_MAX in device code.
- Define the tile-count variable once and reuse the same identifier consistently.

Diagnostics:
- identifier "blocks_per_seq" is undefined
- identifier "FLT_MAX" is undefined

Do not do:
- Do not rely on FLT_MAX being transitively included.
- Do not mix similarly named indexing variables.

#### Variant 2

Shape/context: CUDA C++ device/kernel code using ordinary scalar min and device helper calls

Example completeness: complete

Minimal correct pattern:

```cpp
__device__ __forceinline__ float rcp_safe(float x) {
  return 1.0f / x;
}
extern "C" __global__ void k(int64_t S, float* out, float x) {
  int m_block = (int)(blockIdx.x * 128);
  int valid_m = min((int)(S - m_block), (int)128);
  if (threadIdx.x == 0) out[0] = rcp_safe(x);
}
```

Wrong patterns:

```cpp
int valid_m = min<int>((int)S - m_block, 128);
```

```cpp
extern "C" __global__ void k(float* out, float x) { if (threadIdx.x == 0) out[0] = rcp_safe(x); }
__device__ __forceinline__ float rcp_safe(float x) { return 1.0f / x; }
```

Operand contract:
- Prefer ordinary overloaded min(a, b) with both operands cast to the same scalar type in device code.
- A device helper must be declared or defined before the kernel call site.

Required sequence:
- Cast both min operands to the intended type before calling min.
- Place the device helper definition or prototype before the kernel definition.

Diagnostics:
- error: type name is not allowed
- warning #174-D: expression has no effect
- identifier "rcp_safe" is undefined

Do not do:
- Do not rely on template-argument syntax min<int>(...) in the failing nvcc contexts shown here.
- Do not define a device helper only after the kernel and expect implicit declaration.

### 10. TVMFFIEnvGetStream host API contract

Metadata: source_count=1; tags=tvm-ffi, stream, host-api

#### Variant 1

Shape/context: Host-side TVM-FFI CUDA launcher code using tvm::ffi::TensorView and runtime stream lookup

Example completeness: complete

Minimal correct pattern:

```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda_runtime.h>
void run(tvm::ffi::TensorView Q) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  cudaStream_t stream = 0;
  void* stream_ptr = TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
  if (stream_ptr) stream = static_cast<cudaStream_t>(stream_ptr);
}
```

Wrong patterns:

```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <cuda_runtime.h>
void run(tvm::ffi::TensorView Q) { void* stream_ptr = TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id); }
```

Operand contract:
- TVMFFIEnvGetStream is declared by <tvm/ffi/extra/c_env_api.h>, not by <tvm/ffi/tvm_ffi.h> alone.
- The accessor takes device_type and device_id from TensorView.device().
- The returned opaque pointer must be cast to cudaStream_t before use.

Required sequence:
- Include <tvm/ffi/extra/c_env_api.h> in the translation unit.
- Call cudaSetDevice(tensor.device().device_id) before using the stream.
- Fetch the stream pointer with TVMFFIEnvGetStream(device_type, device_id).
- Cast non-null pointer to cudaStream_t.

Diagnostics:
- identifier "TVMFFIEnvGetStream" is undefined

Do not do:
- Do not assume tvm_ffi.h alone declares the stream API.
- Do not pass raw void* directly as a kernel-launch stream.

### 11. setmaxnreg.inc.sync.aligned.u32

Metadata: source_count=1; tags=setmaxnreg, warpgroup, immediate

#### Variant 1

Shape/context: Hopper warpgroup-uniform region before a WGMMA-heavy path; immediate register count 24..256, multiple of 8

Example completeness: partial

Missing details:
- The notes do not show the matching later warpgroup synchronization or any decrement path.

Partial pattern:

```cpp
template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}
if ((threadIdx.x / 128) == 0) {
  setmaxnreg_inc_sync_fn<200>();
}
```

Wrong patterns:

```cpp
if ((threadIdx.x & 1) == 0) { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(200) : "memory"); }
```

Operand contract:
- Immediate register count must be compile-time constant, 24..256 inclusive, and a multiple of 8.
- All warps in the warpgroup must execute the same setmaxnreg instruction; divergent execution is undefined.
- The .inc form is valid only when the current per-thread max register count is less than or equal to the requested count.

Required sequence:
- Enter a warpgroup-uniform region.
- Execute setmaxnreg.inc.sync.aligned.u32 IMM.
- Synchronize warpgroup appropriately before later setmaxnreg changes.

Diagnostics:
- Undefined behavior or hangs if only a subset of the warpgroup executes the instruction.

Do not do:
- Do not place setmaxnreg inside a condition that differs across warps in the same warpgroup.
- Do not use a runtime value or a non-multiple-of-8 immediate.

## Do Not Overgeneralize

- Operator-specific attention tile semantics beyond what is needed to state exact instruction/layout contracts
- High-level optimization motivations not tied to a concrete CUDA/PTX/API contract
- Manually swizzled intermediate softmax packing as a standalone family, because the notes only provide partial layout examples and no stable end-to-end contract beyond matching WGMMA swizzle expectations


