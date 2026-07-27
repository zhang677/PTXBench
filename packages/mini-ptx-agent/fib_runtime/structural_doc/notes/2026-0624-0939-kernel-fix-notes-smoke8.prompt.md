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
