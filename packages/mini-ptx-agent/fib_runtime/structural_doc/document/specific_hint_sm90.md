# SM90 (Hopper) Kernel Hints

Common pitfalls and working patterns observed from LLM-generated CUDA kernels targeting sm_90.

---

## 1. Shared Memory Declaration Syntax

**Wrong** (appears in ~28% of generated kernels):
```cpp
extern __attribute__((shared)) alignas(128) uint8_t smem_buf[];   // COMPILE ERROR
extern __attribute__((shared)) alignas(128) char smem_buf[];      // COMPILE ERROR
```
Error: `error: attribute does not apply to any entity`

**Correct**:
```cpp
extern __shared__ __align__(128) uint8_t smem_buf[];
// or with higher alignment:
extern __shared__ __align__(1024) uint8_t smem_buf[];
```

`__shared__` is a CUDA storage-class specifier, not a GNU `__attribute__`. Use `__align__(N)` instead of C++ `alignas(N)`.

---

## 2. Debugging mbarrier Deadlocks

Mbarrier deadlocks (kernel hangs / TIMEOUT) are the most common runtime failure in Hopper TMA kernels. The barrier completes when:
```
total_arrive_count == init_count  AND  total_tx_bytes_completed == expected_tx_bytes
```

If either condition never becomes true, `mbarrier.try_wait.parity` spins forever.

### 2.1 Invariants to Check

**Invariant 1: `expect_tx` must exactly match TMA byte count.**

Each `tma_load_2d_fn` / `tma_load_multicast_2d_fn` call automatically adds its transfer size to the barrier's completed transaction count. The `mbarrier_arrive_and_expect_tx_fn` call pre-credits `tx_bytes` to the expected count. If these don't match, the barrier never completes.

```
expected_tx_bytes = sum of all TMA loads targeting this barrier on this CTA
```

For a 128x64 BF16 tile: `128 * 64 * 2 = 16384 bytes`
For a 256x64 BF16 tile: `256 * 64 * 2 = 32768 bytes`

With multicast, the TMA hardware auto-credits bytes to each receiving CTA's barrier. So a multicast load of 16384 bytes credits 16384 bytes to every CTA in the mask.

Example byte calculation (128M x 128N tile, BK=64, no multicast):
```
A tiles: 2 tiles of 64x64 = 2 * 8192 = 16384 bytes
B tiles: 2 tiles of 64x64 = 2 * 8192 = 16384 bytes
Total: 32768 bytes
```

Example byte calculation (128M x 256N tile, cluster=2, B multicast):
```
A tile: 128x64 = 16384 bytes (per CTA, no multicast)
B tile: 256x64 = 32768 bytes (multicast, auto-credited to both CTAs)
Total per CTA: 16384 + 32768 = 49152 bytes
```

**Invariant 2: `init_count` must match actual arrivals.**

For `full_barriers` (producer -> consumer): typically `init_count = 1` because only `tid == 0` calls `mbarrier_arrive_and_expect_tx_fn`.

For `empty_barriers` (consumer -> producer): count must match the number of threads that call the arrive function.

Non-cluster case (gemm_v4 pattern):
```
init_count = num_warps_in_consumer_warpgroups
           = num_consumer_wgs * 4
```
Each consumer warp's lane 0 calls `mbarrier_arrive_fn`.

Cluster case (gemm_v5+ pattern):
```
init_count = num_consumer_wgs * warps_per_wg * cluster_size
```
Each consumer warp's `lane < cluster_size` threads call `mbarrier_arrive_remote_fn`.

| Config | Consumer WGs | Warps/WG | Cluster | empty_barrier count |
|--------|-------------|----------|---------|-------------------|
| gemm_v4: 256 threads, no cluster | 1 | 4 | 1 | 4 |
| gemm_v5: 256 threads, cluster=2 | 1 | 4 | 2 | 8 |
| gemm_v6+: 384 threads, cluster=2 | 2 | 4 | 2 | 16 |

**Invariant 3: Phase parity must be consistent between producer and consumer.**

Phase starts at 0 and flips each time the stage index wraps around:
```cpp
// After advancing stage:
stage = stage + 1;
if (stage == NUM_STAGES) {
    stage = 0;
    phase = phase ^ 1;
}
```

Producer waits on empty_barrier with `phase ^ 1` (previous phase) to confirm consumers finished with the buffer:
```cpp
// Producer: wait for consumers to release buffer before overwriting
if (k >= NUM_STAGES) {
    mbarrier_wait_fn(empty_barriers[p_stage], p_phase ^ 1);
}
```

Consumer waits on full_barrier with current phase:
```cpp
// Consumer: wait for producer to fill buffer
mbarrier_wait_fn(full_barriers[c_stage], c_phase);
```

**Invariant 4: Grid dimensions must be divisible by cluster size.**

If `grid.x = 7` and `cluster_size = 2`, the hardware cannot form the last cluster. This causes `cudaErrorInvalidConfiguration` or silent hangs.

### 2.2 Debugging Checklist

When a kernel hangs (TIMEOUT):

1. **Verify tx_bytes**: Manually add up all TMA load sizes targeting each barrier. Include multicast credits. Compare against the value passed to `mbarrier_arrive_and_expect_tx_fn`.

2. **Verify empty_barrier count**: Count the actual number of `mbarrier_arrive_fn` / `mbarrier_arrive_remote_fn` calls that execute per CTA, across all consumer warpgroups and across all CTAs in the cluster (for remote arrives). This must equal `init_count`.

3. **Verify phase tracking**: Walk through the first few iterations by hand. Ensure producer and consumer phase variables stay in sync with the stage index modular wrapping.

4. **Verify grid divisibility**: `grid.x` (or whichever axis maps to cluster) must be a multiple of `cluster_size`.

5. **Check the prefill window**: The producer must not wait on `empty_barriers` during the first `NUM_STAGES` iterations (the pipeline fill phase), because no consumer has released those stages yet.

### 2.3 Working Non-Cluster Pattern (from gemm_v4)

Producer-consumer with 2-stage pipeline, 256 threads, 1 consumer warpgroup:

```cpp
// --- Initialization (tid == 0 only) ---
for (int s = 0; s < NUM_STAGES; s++) {
    init_smem_barrier_fn(full_barriers[s], 1);      // 1 = only tid 0 arrives
    init_smem_barrier_fn(empty_barriers[s], 4);      // 4 = 4 warps * lane==0
}
fence_smem_barrier_init_fn();
// __syncthreads() or cluster_sync_fn() after init

// --- Producer (wg == 0, tid == 0) ---
int p_stage = 0, p_phase = 0;
for (int k = 0; k < nk; k++) {
    if (k >= NUM_STAGES) {
        mbarrier_wait_fn(empty_barriers[p_stage], p_phase ^ 1);  // wait for consumer
    }
    mbarrier_arrive_and_expect_tx_fn(full_barriers[p_stage], tx_bytes);  // pre-credit
    tma_load_2d_fn(&dA, full_barriers[p_stage], A_smem[p_stage], ...);  // auto-credits
    tma_load_2d_fn(&dB, full_barriers[p_stage], B_smem[p_stage], ...);  // auto-credits
    // advance stage
    p_stage++;
    if (p_stage == NUM_STAGES) { p_stage = 0; p_phase ^= 1; }
}

// --- Consumer (wg == 1) ---
int c_stage = 0, c_phase = 0;
for (int k = 0; k < nk; k++) {
    mbarrier_wait_fn(full_barriers[c_stage], c_phase);  // wait for producer
    __syncwarp();
    // ... WGMMA compute ...
    wgmma_commit_fn();
    wgmma_wait_fn();
    // Signal empty — one thread per warp
    if (lane == 0) {
        mbarrier_arrive_fn(empty_barriers[c_stage]);  // 4 warps → 4 arrivals
    }
    __syncwarp();
    // advance stage
    c_stage++;
    if (c_stage == NUM_STAGES) { c_stage = 0; c_phase ^= 1; }
}
```

---

## 3. Cluster TMA Multicast Patterns

Cluster multicast allows one CTA to issue a TMA load that delivers data to shared memory of all CTAs in the cluster, reducing global memory bandwidth. This is the key technique needed to reach cuBLAS-level performance. Patterns below are extracted from working examples (gemm_v5 through gemm_v10).

### 3.1 Architecture Overview

```
Cluster (2 CTAs along blockIdx.x)
├── CTA 0 (cta = cluster_rank_fn() == 0)
│   ├── wg 0: producer (tid 0 issues TMA)
│   └── wg 1+: consumer(s) (WGMMA)
└── CTA 1 (cta = cluster_rank_fn() == 1)
    ├── wg 0: producer (tid 0 issues TMA)
    └── wg 1+: consumer(s) (WGMMA)

Grid: blockIdx.x = cluster_id * cluster_size + cta_rank
Each CTA computes a different M-tile but the same N-tile.
B (N-dimension) is shared via multicast; A (M-dimension) is loaded per-CTA.
```

### 3.2 Two Multicast Strategies

**Strategy A: One CTA issues multicast for the full shared tile (gemm_v5, gemm_v6)**

Only `cta == 0` issues the multicast B load. All CTAs receive the same data at the same smem offset.

```cpp
// Producer (tid == 0):
// A: each CTA loads its own rows (no multicast)
tma_load_2d_fn(&dA, full_barriers[stage], A_smem[stage], k_coord, bm);

// B: only CTA 0 issues multicast for all CTAs
if (cta == 0) {
    tma_load_multicast_2d_fn(&dB, full_barriers[stage], B_smem[stage],
        k_coord, bn, /*mask=*/0x3);  // 0x3 = binary 11 = both CTAs
}

// Pre-credit: A_bytes + B_bytes (multicast auto-credits B to each CTA)
mbarrier_arrive_and_expect_tx_fn(full_barriers[stage], A_bytes + B_bytes);
```

**Strategy B: Each CTA issues multicast for its half of the shared tile (gemm_v7, gemm_v8, gemm_v9, gemm_v10)**

Each CTA loads a different half of B via multicast. Both CTAs end up with the full B tile because multicast delivers to all CTAs.

```cpp
// Each CTA loads its own half of B
int b_half_off = cta * 128;  // CTA 0 → cols [0,128), CTA 1 → cols [128,256)

// Producer (tid == 0, on every CTA):
tma_load_2d_fn(&dA, full_barriers[stage], A_smem[stage], k_coord, bm);

tma_load_multicast_2d_fn(&dB, full_barriers[stage],
    B_smem[stage] + b_half_off * BK,  // different smem offset per CTA
    k_coord, bn + b_half_off,          // different global coord per CTA
    /*mask=*/0x3);

// tx_bytes: A_bytes + B_full_bytes
// Each CTA receives: own A (16384) + CTA0's B half (16384) + CTA1's B half (16384) = 49152
mbarrier_arrive_and_expect_tx_fn(full_barriers[stage], 49152);
```

Strategy B is preferred for gemm_v7+ because it balances TMA load issuing across CTAs and avoids having CTA 0 as a bottleneck.

### 3.3 Complete Cluster Multicast Barrier Protocol

```cpp
// === Initialization (tid == 0 only) ===
for (int s = 0; s < NUM_STAGES; s++) {
    init_smem_barrier_fn(full_barriers[s], 1);
    init_smem_barrier_fn(empty_barriers[s], EMPTY_COUNT);
    // EMPTY_COUNT = num_consumer_wgs * warps_per_wg * cluster_size
    // e.g., 2 consumer WGs × 4 warps × 2 CTAs = 16
}
fence_smem_barrier_init_fn();  // MUST be .release.cluster (not .cta)
// ^^^ fence.mbarrier_init.release.cluster ensures all CTAs see the init

cluster_sync_fn();  // synchronize entire cluster before starting pipeline
// ^^^ barrier.cluster.arrive + barrier.cluster.wait

// === Producer (wg == 0, tid == 0) ===
p_stage = 0; p_phase = 0;
for (int k = 0; k < nk; k++) {
    if (k >= NUM_STAGES)
        mbarrier_wait_fn(empty_barriers[p_stage], p_phase ^ 1);

    // TMA loads (per-CTA A + multicast B)
    tma_load_2d_fn(&dA, full_barriers[p_stage], A_smem[p_stage], ...);
    tma_load_multicast_2d_fn(&dB, full_barriers[p_stage], ..., mask);
    mbarrier_arrive_and_expect_tx_fn(full_barriers[p_stage], tx_bytes);

    p_stage++;
    if (p_stage == NUM_STAGES) { p_stage = 0; p_phase ^= 1; }
}

// === Consumer (wg >= 1) ===
c_stage = 0; c_phase = 0;
for (int k = 0; k < nk; k++) {
    mbarrier_wait_fn(full_barriers[c_stage], c_phase);
    __syncwarp();

    // ... WGMMA compute ...
    wgmma_commit_fn();
    wgmma_wait_fn();

    // Signal empty to ALL CTAs in the cluster via remote arrive
    if (lane < cluster_size) {
        mbarrier_arrive_remote_fn(empty_barriers[c_stage], lane);
        // lane=0 → arrive at CTA 0's barrier
        // lane=1 → arrive at CTA 1's barrier
    }
    __syncwarp();

    c_stage++;
    if (c_stage == NUM_STAGES) { c_stage = 0; c_phase ^= 1; }
}
```

### 3.4 Remote Arrive Mechanics

In the non-cluster case, consumers call `mbarrier_arrive_fn` which arrives at the local CTA's barrier. In the cluster case, consumers must signal **all** CTAs that the buffer is released:

```cpp
// Non-cluster: local arrive only
if (lane == 0) {
    mbarrier_arrive_fn(empty_barriers[stage]);
}
// → count = num_consumer_wgs * 4 (one per warp)

// Cluster: remote arrive to every CTA
if (lane < cluster_size) {
    mbarrier_arrive_remote_fn(empty_barriers[stage], /*target_cta=*/lane);
}
// → count = num_consumer_wgs * 4 * cluster_size (from all CTAs combined)
```

`mbarrier_arrive_remote_fn` uses `mapa.shared::cluster` to translate a local smem address to the target CTA's address space:
```cpp
__device__ void mbarrier_arrive_remote_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t smem_addr = __cvta_generic_to_shared(&bar[0]);
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
        :: "r"(remote_addr) : "memory");
}
```

### 3.5 Cluster Launch Setup (Host Side)

Cluster kernels must use `cudaLaunchKernelEx` instead of triple-chevron `<<<>>>`:

```cpp
cudaLaunchConfig_t config = {};
config.gridDim = grid;    // grid.x MUST be divisible by cluster_size
config.blockDim = block;
config.dynamicSmemBytes = smem_size;
config.stream = stream;

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim.x = 2;  // cluster size along x
attrs[0].val.clusterDim.y = 1;
attrs[0].val.clusterDim.z = 1;
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel_fn, arg0, arg1, ...);
```

**Common pitfalls:**
- Passing `(const void*)kernel_fn` — breaks template deduction. Pass the typed function pointer directly.
- Grid not divisible by cluster size — causes `cudaErrorInvalidConfiguration`.
- Using `cudaError_t` to capture `CUresult` — the driver API `cuTensorMapEncodeTiled` returns `CUresult`, not `cudaError_t`. Use a separate check macro.

### 3.6 Tile-to-CTA Mapping

With cluster_size=2 along x, map M-dimension to cluster and N-dimension to block:

```cpp
uint32_t cta = cluster_rank_fn();         // 0 or 1 within cluster
int cluster_id = blockIdx.x / cluster_size;
int cluster_m = tile_m * (cluster_size * BM);  // cluster covers cluster_size * BM rows
int bm = cluster_m + cta * BM;                 // each CTA gets BM rows
int bn = tile_n * BN;                          // N tile is shared across cluster
```

### 3.7 Persistent Kernel Pattern (gemm_v9, gemm_v10)

For persistent kernels that process multiple output tiles per CTA launch, the producer and consumer use separate tile iterators. The producer's global iteration counter (`p_giter`) replaces the per-tile `p_k >= NUM_STAGES` check:

```cpp
// Producer (tid == 0) — iterates over all tiles × k_steps continuously
p_stage = 0; p_phase = 0; p_giter = 0;
p_tile = cluster_id;
while (p_tile < num_tiles) {
    // ... compute tile coords ...
    for (int k = 0; k < nk; k++) {
        if (p_giter >= NUM_STAGES)
            mbarrier_wait_fn(empty_barriers[p_stage], p_phase ^ 1);
        // TMA loads ...
        mbarrier_arrive_and_expect_tx_fn(full_barriers[p_stage], tx_bytes);
        p_stage++; if (p_stage == NUM_STAGES) { p_stage = 0; p_phase ^= 1; }
        p_giter++;  // global counter never resets
    }
    p_tile += num_clusters;
}

// Consumer — iterates over tiles, resetting acc each tile
c_stage = 0; c_phase = 0;
c_tile = cluster_id;
while (c_tile < num_tiles) {
    float acc[128]; for (int i=0; i<128; i++) acc[i]=0.0f;
    for (int k = 0; k < nk; k++) {
        mbarrier_wait_fn(full_barriers[c_stage], c_phase);
        // ... WGMMA ...
        if (lane < cluster_size)
            mbarrier_arrive_remote_fn(empty_barriers[c_stage], lane);
        c_stage++; if (c_stage == NUM_STAGES) { c_stage = 0; c_phase ^= 1; }
    }
    // store output tile ...
    c_tile += num_clusters;
}
```

The key difference from non-persistent: barriers are initialized **once** at kernel start, and `p_giter` tracks the global iteration count across tiles (not reset per tile) so the producer knows when it's safe to overwrite the first buffers.

### 3.8 TMA Store for Output C (gemm_v8+)

Instead of writing C directly to global memory, stage through shared memory and use TMA store:

```cpp
// After WGMMA compute, all consumer warpgroups write to C_smem:
store_acc_smem_bf16_n256_fn(C_smem, acc, ltid, m_wg_off);

// Sync all consumer threads (named barrier or __syncthreads)
__syncthreads();  // or named_barrier_sync_fn(0, num_consumer_threads)

// Fence before TMA store
tma_store_fence_fn();  // fence.proxy.async.shared::cta

// Issue TMA stores (tid == 0 only):
if (tid == 0) {
    tma_store_2d_fn(&dC, C_smem,       bn,     bm);       // first 64 rows
    tma_store_2d_fn(&dC, C_smem+16384,  bn,     bm + 64);  // next 64 rows
    tma_store_commit_fn();   // cp.async.bulk.commit_group
    tma_store_wait_n_fn(0);  // cp.async.bulk.wait_group 0
}
__syncthreads();
```

For persistent kernels (gemm_v9, gemm_v10), overlap the TMA store wait with the next tile's compute by waiting before the store of the next tile rather than immediately after.
