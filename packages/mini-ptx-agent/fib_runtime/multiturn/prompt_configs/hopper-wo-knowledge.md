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

    // The TMA setup is also related with the LBO and SBO of the mma instructions that will consume the data
    One Atom is a matrix with a MN-mode and a K-mode. A "row" is an array with the same index at MN-mode. A "column" is an array with with same index at K-mode.
    The chunks lay out along the major mode, and the spans lay out along the minor mode. 8 spans along the minor mode form a "core matrix".
    
    LBO and SBO are defined as follows for matrices whose element types are normalized to 128-bits.
    [Leading Dimension Byte Offset](LBO)

    | Major-ness   | Case | Definition|
    |--------------|------|-------------|
    | K-Major      | No-Swizzling | offset from the first column to the second columns of the 8x2 tile in the 128-bit element type normalized matrix. |
    | K-Major      | Swizzling | Not used, assumed to be 1. |
    | MN-Major     | No-Swizzling | offset from the first 8 columns to the next 8 columns. |
    | MN-Major     | Swizzling | offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows.  |

    [Stride Dimension Byte Offset](SBO)

    | Major-ness   | Case | Definition|
    |--------------|------|--------|
    | K-Major      | All | The offset from the first 8 rows to the next 8 rows.  |
    | MN-Major     | No-Swizzling | offset from the first row to the next row. |
    | MN-Major     | Swizzling | offset from the first 8 columns to the next 8 columns |


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
    K-Major descriptor under 128B swizzled layouts
    ```
    constexpr uint32_t ATOM_MMODE_DIM = BLOCK_M;
    constexpr uint32_t ATOM_KMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
    constexpr uint32_t SBO = 8U * 128U; // 8 spans along M-mode

    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        K, M,
        ATOM_KMODE_DIM, ATOM_MMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```

    K-Major descriptor under non-swizzled layout:
    ```
    constexpr uint32_t ATOM_MMODE_DIM = BLOCK_M;
    constexpr uint32_t ATOM_KMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
    constexpr uint32_t SBO = 8U * 16U; // 8 spans along M-mode
    constexpr uint32_t LBO = (ATOM_MMODE_DIM / 8U) * SBO; // BLOCK_M spans along M-mode. Each 8 spans compose a core matrix; each core matrix has SBO bytes. There are ATOM_MMODE_DIM / 8U core matrices
    
    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        K, M,
        ATOM_KMODE_DIM, ATOM_MMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```
    For M-Major the situation can be little bit more confusing because LBO and SBO mean visually different things for Swizzle vs no Swizzle case.
    MN-Major descriptor under 128B swizzled layouts
    ```
    // Swizzle 128B -> stride-1 row slabs in MN-major order
    constexpr uint32_t ATOM_KMODE_DIM = BLOCK_K;
    constexpr uint32_t ATOM_MMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
    constexpr uint32_t SBO = 128 * 8U; // 8 spans along K-mode 
    constexpr uint32_t LBO = (ATOM_KMODE_DIM / 8U) * SBO; // Remember each element in a row is 128bits (16B) now, so LBO is essentially 128B x (# of spans lined up along K-mode). In other words, SBO x (# of core matrices lined up along K-mode)

    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        M, K,
        ATOM_MMODE_DIM, ATOM_KMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```

    MN-Major descriptor under non-swizzled layouts
    ```
    constexpr uint32_t ATOM_KMODE_DIM = BLOCK_K;
    constexpr uint32_t ATOM_MMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
    constexpr uint32_t LBO = 16 * 8U; // 8 spans along K-mode
    constexpr uint32_t SBO = (ATOM_KMODE_DIM / 8U) * LBO; // Switch LBO and SBO compared with the swizzle layouts

    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        M, K,
        ATOM_MMODE_DIM, ATOM_KMODE_DIM,
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
        create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
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

__device__ __forceinline__ void tma_store_commit_fn() {
    // TMA store arrive/commit.
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    // cp.async.bulk.wait_group requires immediate operand, use switch for common values
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}


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

Below are some common errors and solutions.
# Common Causes of Incorrect Results
1. Wrong wgmma descriptor stride: Revise the tma doc

2. Swizzle mismatch between TMA and wgmma descriptor

3. Missing fence.proxy.async (manual loads before WGMMA)
```cpp
// WRONG
smem[i] = global[i];
__syncthreads();
wgmma_m64n64k16_fn_(acc, da, db);  // WGMMA may see stale data!
// CORRECT
smem[i] = global[i];
__syncthreads();
fence_proxy_async_fn()
__syncwarp();
wgmma_fence_fn();
wgmma_m64n64k16_fn_(acc, da, db);
```

4. Barrier race conditions
```cpp
// WRONG: empty arrives before math finishes
mbarrier_arrive_fn(empty_barriers[s]);
wgmma_wait_fn();  // Race!
// CORRECT
wgmma_wait_fn();
mbarrier_arrive_fn(empty_barriers[s]);
```

# Execution timeout
Hangs and Deadlocks
Cause: Barrier expected arrivals don't match actual arrivals.
Debug:
```cpp
  // Add debug prints (disable in production!)
  if (tid == 0) {
      printf("Block %d: init barrier %d with %d arrivals\n",
             blockIdx.x, s, expected_arrivals);
}
```
Common mistakes: - init_smem_barrier_fn(empty_barriers[s], 4) but only 3 consumer warps call arrive() - Cluster barriers initialized with wrong count - Not all threads reach arrive() due to early exit.

# Register Spill
Symptom: Performance drops dramatically
Causes: 1. Too many accumulators live simultaneously 2. Loop unrolling creates too many variables 3. Missing #pragma unroll causing inefficient code
Solutions: 1. Process tiles sequentially with accumulator reuse 2. Use #pragma nounroll where appropriate 3. Reduce NUM_MATH_REGS request

Debug hints end.

