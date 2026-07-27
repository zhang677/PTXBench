# SM90 Kernel Hints

## 1. Shared Memory Declaration

**Wrong**:
```cpp
extern __attribute__((shared)) alignas(128) uint8_t smem[];   // COMPILE ERROR
```
Error: `error: attribute does not apply to any entity`

**Correct**:
```cpp
extern __shared__ __align__(128) uint8_t smem[];
// or with higher alignment:
extern __shared__ __align__(1024) uint8_t smem_buf[];
```
`__shared__` is a CUDA storage-class specifier, not a GNU attribute. Use `__align__(N)` not `alignas(N)`.

## 2. Avoiding mbarrier Deadlocks

An mbarrier completes when `arrive_count == init_count` AND `tx_bytes_completed == expected_tx_bytes`. If either is wrong, `mbarrier_wait_fn` spins forever.

### Rules

**Rule 1: `expect_tx` must equal total TMA bytes arriving at this barrier on this CTA.**

Each `tma_load_2d_fn` / `tma_load_multicast_2d_fn` auto-credits its transfer size. The `mbarrier_arrive_and_expect_tx_fn` pre-credits the expected total.

Tile size examples (BF16):
- 64x64 tile = 8192 bytes
- 128x64 tile = 16384 bytes
- 256x64 tile = 32768 bytes

With multicast, TMA auto-credits bytes to every receiving CTA's barrier.

**Rule 2: `init_count` must match actual arrive calls.**

For `full_barriers`: typically `count=1` (only tid==0 calls `mbarrier_arrive_and_expect_tx_fn`).

For `empty_barriers`: count depends on how many threads arrive.
- Non-cluster: `num_consumer_wgs * warps_per_wg` (one `mbarrier_arrive_fn` per warp via `lane==0`)
- Cluster: `num_consumer_wgs * warps_per_wg * cluster_size` (one `mbarrier_arrive_remote_fn` per warp per CTA via `lane < cluster_size`)

**Rule 3: Phase parity must be consistent.**

Phase starts at 0, flips when stage wraps:
```cpp
stage++;
if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
```
Producer waits on empty with `phase ^ 1` (previous). Consumer waits on full with current `phase`.

**Rule 4: Grid must be divisible by cluster size.**

**Rule 5: Skip empty_barrier wait during pipeline fill.**

Producer must not wait on `empty_barriers` for the first `NUM_STAGES` iterations.

### An Example Non-Cluster Pattern

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

## 3. Cluster TMA Multicast

Cluster multicast lets one TMA load deliver data to all CTAs in a cluster, reducing global memory traffic. In GEMM: A (M-dimension) is loaded per-CTA; B (N-dimension) is shared via multicast.

### Setup

```
Cluster size 2 along blockIdx.x:
  cluster_id = blockIdx.x / 2
  cta = cluster_rank_fn()           // 0 or 1
  bm = tile_m_base + cta * BLOCK_M  // each CTA gets different M rows
  bn = tile_n * BLOCK_N             // same N tile shared
```

### Multicast Load Pattern

Strategy 1: Each CTA loads its own half of B via multicast (all CTAs receive all halves):

```cpp
int b_half_off = cta * (BLOCK_N / 2);

// Producer (tid==0, every CTA):
// A: each CTA loads its own rows (no multicast)
tma_load_2d_fn(&dA, full_barriers[s], A_smem[s], k*BK, bm);

// B: only CTA 0 issues multicast for all CTAs
if (cta == 0) {
    tma_load_multicast_2d_fn(&dB, full_barriers[stage], B_smem[stage],
        k_coord, bn, /*mask=*/0x3);  // 0x3 = binary 11 = both CTAs
}

// Pre-credit: A_bytes + B_bytes (multicast auto-credits B to each CTA)
mbarrier_arrive_and_expect_tx_fn(full_barriers[stage], A_bytes + B_bytes);
```

Strategy 2: Each CTA loads a different half of B via multicast. Both CTAs end up with the full B tile because multicast delivers to all CTAs.

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

### Barrier Counts for Cluster

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

### Consumer Empty Arrive (Remote)

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

### Persistent Kernel Extension

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

### TMA Store for Output

After WGMMA, stage results through shared memory and use TMA bulk copy to global:

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

For persistent kernels, overlap TMA store wait with next tile's compute by deferring `tma_store_wait_n_fn(0)` to before the next tile's store.
