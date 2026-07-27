# Optimization Pattern: Warp Specialization + Cluster Multicast

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
cluster_sync_fn();  // synchronize entire cluster before starting pipeline

if (wg == 0) {
    setmaxnreg_dec_sync_fn<48>();
    if (tid == 0) {
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
    }
}
else {
    // === Consumer (wg >= 1) ===
    c_stage = 0; c_phase = 0;
    for (int k = 0; k < nk; k++) {
        mbarrier_wait_fn(full_barriers[c_stage], c_phase);
        __syncwarp();
        wgmma_fence_fn();
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
    // Store
}
```