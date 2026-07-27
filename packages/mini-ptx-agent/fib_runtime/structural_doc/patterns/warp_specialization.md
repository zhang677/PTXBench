# Optimization Pattern: Warp Specialization

Warp specialization partitions a CTA's warp groups into dedicated producer (TMA loads) and consumer (WGMMA compute) roles that communicate through mbarriers, overlapping memory traffic with math.

```cpp
if (tid == 0) {
    // === Initialization (tid == 0 only) ===
    for (int s = 0; s < NUM_STAGES; s++) {
        init_smem_barrier_fn(full_barriers[s], 1);
        init_smem_barrier_fn(empty_barriers[s], EMPTY_COUNT);
        // EMPTY_COUNT = num_consumer_wgs * warps_per_wg * cluster_size
        // e.g., 2 consumer WGs × 4 warps × 2 CTAs = 16
    }
    fence_smem_barrier_init_fn();
}
// __syncthreads() or cluster_sync_fn() after init

if (wg == 0) {
    setmaxnreg_dec_sync_fn<40>();
    if (tid == 0) {
        // --- Producer (wg == 0, tid == 0) ---
        int p_stage = 0, p_phase = 0;
        for (int k = 0; k < num_k_tiles; k++) {
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
    }
}
else {
    // --- Consumer (wg == 1) ---
    setmaxnreg_inc_sync_fn<232>();
    int c_stage = 0, c_phase = 0;
    for (int k = 0; k < num_k_tiles; k++) {
        mbarrier_wait_fn(full_barriers[c_stage], c_phase);  // wait for producer
        __syncwarp();
        // ... WGMMA fence and compute ...
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
}
```