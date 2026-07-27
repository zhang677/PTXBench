# Optimization Pattern: Warp Specialization + Persisten Kernel + Cluster Multicast

```cpp
if ((wg == 0)) {
    setmaxnreg_dec_sync_fn<40>();
}
else {
    setmaxnreg_inc_sync_fn<232>();
}
cluster_id = blockIdx.x / cluster_size;
p_tile = cluster_id;
while ((p_tile < num_tiles)) {
    if (tid == 0) {
        // === Initialization (tid == 0 only) ===
        for (int s = 0; s < NUM_STAGES; s++) {
            init_smem_barrier_fn(full_barriers[s], 1);
            init_smem_barrier_fn(empty_barriers[s], EMPTY_COUNT);
            // EMPTY_COUNT = num_consumer_wgs * warps_per_wg * cluster_size
            // e.g., 2 consumer WGs × 4 warps × 2 CTAs = 16
        }
        fence_smem_barrier_init_fn();  // MUST be .release.cluster (not .cta)
        // ^^^ fence.mbarrier_init.release.cluster ensures all CTAs see the init
    }

    cluster_sync_fn();  // synchronize entire cluster before starting pipeline
    // ^^^ barrier.cluster.arrive + barrier.cluster.wait

    // === Tile index arithmetic (1D p_tile -> per-warpgroup global coords) ===
    // Host passes num_tiles = num_m_tiles * num_n_tiles; device only needs num_n_tiles.
    int tile_m    = p_tile / num_n_tiles;           // row-major tile walk
    int tile_n    = p_tile % num_n_tiles;           // (swap / and % for column-major)
    int cluster_m = tile_m * BM_CLUSTER;            // cluster-level M base
    int t_bn      = tile_n * BN;                    // N base (shared across CTAs in cluster)
    int t_bm      = cluster_m + cta * BM_PER_CTA;   // this CTA's M base (cta = cluster_rank)
    int m_wg_off  = (wg - num_producer_wgs) * BM_PER_WG;  // warpgroup's M offset within CTA
    // Producer uses t_bm, t_bn for TMA coords; consumer uses (t_bm + m_wg_off, t_bn) for store.

    if (wg == 0) {
        if (tid == 0) {
            // === Producer (wg == 0, tid == 0) ===
            p_stage = 0; p_phase = 0;
            for (int k = 0; k < nk; k++) {
                if (k >= NUM_STAGES)
                    mbarrier_wait_fn(empty_barriers[p_stage], p_phase ^ 1);

                // TMA loads (per-CTA A at t_bm + multicast B at t_bn)
                tma_load_2d_fn(&dA, full_barriers[p_stage], A_smem[p_stage], k * BK, t_bm /* + per-CTA A sub-row */);
                tma_load_multicast_2d_fn(&dB, full_barriers[p_stage], k * BK, t_bn, mask);
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
            // ... WGMMA fence and compute ...
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
    }
    cluster_sync_fn();
    p_tile = (p_tile + num_clusters);
}
```