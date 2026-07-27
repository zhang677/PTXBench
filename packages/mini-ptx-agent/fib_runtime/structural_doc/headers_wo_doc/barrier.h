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
    CUDA functions:
    void __threadfence(); ensures that no writes to all memory made by the calling thread after the call to __threadfence() are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to __threadfence().
    void __threadfence_system(); ensures that all writes to all memory made by the calling thread before the call to __threadfence_system() are observed by all threads in the device, host threads, and all threads in peer devices as occurring before all writes to all memory made by the calling thread after the call to __threadfence_system().
*/

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