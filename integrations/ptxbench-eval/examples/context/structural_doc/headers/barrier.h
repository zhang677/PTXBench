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