# Optimization Pattern: Persistent Kernel
Instread of launching one block per output tile, launch once and loop. Benefits: - Reduced kernel launch overhead - Better load balancing (work stealing) - Barrier state preserved across tiles

```cpp
int cluster_id = blockIdx.x / cluster_size;
int num_clusters = gridDim.x / cluster_size;

// Each cluster loops over its assigned tiles
for (int tile = cluster_id; tile < num_tiles; tile += num_clusters) {
    // Initialize barriers for this tile
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            init_smem_barrier_fn(full_barriers[s], 1);
            init_smem_barrier_fn(empty_barriers[s], EMPTY_COUNT);
        }
    }
    cluster_sync_fn();

    // Process tile (TMA loads + WGMMA compute)

    cluster_sync_fn();
}
```

## Tile index arithmetic

The loop variable `tile` (often `p_tile`) is a flat 1D id. Decompose it into the
output-tile grid, then split the M/N coords across cluster → CTA → warpgroup.

```cpp
// Host: total tile count passed as a kernel arg.
int num_m_tiles = ceil_div(M, BM_CLUSTER);   // M per cluster
int num_n_tiles = ceil_div(N, BN);           // N per CTA (shared by cluster)
int num_tiles   = num_m_tiles * num_n_tiles;

// Device, inside the persistent loop:
int tile_m = p_tile / num_n_tiles;           // row-major over tiles
int tile_n = p_tile % num_n_tiles;

int cluster_m = tile_m * BM_CLUSTER;         // cluster-level M base
int t_bn      = tile_n * BN;                 // N base (all CTAs in cluster share)
int t_bm      = cluster_m + cta * BM_PER_CTA;// this CTA's M base
int m_wg_off  = (wg - num_producer_wgs) * BM_PER_WG;  // warpgroup's M offset within CTA

// Global coord for this warpgroup's output region:
int out_m = t_bm + m_wg_off;
int out_n = t_bn;
```

Key points:
- `num_tiles` is computed on the host and passed in; the device never needs the 2D grid shape, only `num_n_tiles` for the `/` and `%`.
- The split mirrors the launch hierarchy: **cluster** picks `(tile_m, tile_n)`, **CTA (`cta = cluster_rank`)** adds its slice of M via multicast sharing of B, **warpgroup** adds its slice of M within the CTA tile.
- Swap `/` and `%` (use `tile_m = p_tile % num_m_tiles`) for column-major tile walk — sometimes better L2 reuse of B.
- For swizzled/Hilbert-style tile orderings, remap `p_tile → (tile_m, tile_n)` through a lookup or bit-interleave before the offset math; the rest is unchanged.
