# Optimization Pattern: Independent TMA/Math Tile Scheduling

TMA and Math warpgroups process different tile indices

```cpp
// TMA warpgroup: runs ahead, filling pipeline
if (wg == 0 && tid == 0) {
    int stage = 0, phase = 0;
    for (int tile = cluster_id; tile < num_tiles; tile += num_clusters) {
        for (int k = 0; k < num_k; k++) {
            // Wait for empty (math consumed previous data)
            if (stage == 0 && phase > 0) {
                mbarrier_wait_fn(empty_barriers[0], phase ^ 1);
            } else if (tile > cluster_id || k >= NUM_STAGES) {
                mbarrier_wait_fn(empty_barriers[stage], phase ^ 1);
            }
            // Issue TMA
            tma_load_2d_fn(&dA, full_barriers[stage], sA[stage], k*BK, bm);
            tma_load_multicast_2d_fn(&dB, full_barriers[stage], sB[stage], k*BK, bn, 0x3);
            mbarrier_arrive_and_expect_tx_fn(full_barriers[stage], TX);
            // Advance
            stage++;
            if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
    } 
}
}

// Math warpgroup: independent tile loop
else {
    int stage = 0, phase = 0;
    for (int tile = cluster_id; tile < num_tiles; tile += num_clusters) {
        // Initialize accumulator
        for (int i = 0; i < 128; i++) acc[i] = 0;
        for (int k = 0; k < num_k; k++) {
            mbarrier_wait_fn(full_barriers[stage], phase);
            // WGMMA fence and compute
            wgmma_commit_fn();
            wgmma_wait_fn();
            // Signal empty to remote CTAs
            if (lane < CLUSTER_SIZE) {
            mbarrier_arrive_remote_fn(empty_barriers[c_stage], lane);
            }
            __syncwarp();
            stage++;
            if (stage == NUM_STAGES) { stage = 0; phase ^= 1; }
        }
        // Store result
    }
}
```