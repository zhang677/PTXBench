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