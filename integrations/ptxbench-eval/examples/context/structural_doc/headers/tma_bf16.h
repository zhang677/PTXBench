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