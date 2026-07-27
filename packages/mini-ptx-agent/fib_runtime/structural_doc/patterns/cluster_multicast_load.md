# Optimization Pattern: Cluster Multicast Load

**Strategy A: One CTA issues multicast for the full shared tile**

Only `cta == 0` issues the multicast B load. All CTAs receive the same data at the same smem offset.

```cpp
// Producer (tid == 0):
// A: each CTA loads its own rows (no multicast)
tma_load_2d_fn(&dA, full_barriers[stage], A_smem[stage], k_coord, bm);

// B: only CTA 0 issues multicast for all CTAs
// cta = cluster_rank_fn();
if (cta == 0) {
    tma_load_multicast_2d_fn(&dB, full_barriers[stage], B_smem[stage],
        k_coord, bn, /*mask=*/0x3);  // 0x3 = binary 11 = both CTAs
}

// Pre-credit: A_bytes + B_bytes (multicast auto-credits B to each CTA)
mbarrier_arrive_and_expect_tx_fn(full_barriers[stage], A_bytes + B_bytes);
```

**Strategy B: Each CTA issues multicast for its half of the shared tile**

Each CTA loads a different half of B via multicast. Both CTAs end up with the full B tile because multicast delivers to all CTAs.

```cpp
// Each CTA loads its own half of B
int b_half_off = cluster_rank_fn() * 128;  // BLOCK_N=256

// Producer (tid == 0, on every CTA):
tma_load_2d_fn(&dA, full_barriers[stage], A_smem[stage], k_coord, bm);

// CTA 0 loads B cols [0,128), multicast to CTA 1
// CTA 1 loads B cols [128,256), multicast to CTA 0
tma_load_multicast_2d_fn(&dB, full_barriers[stage],
    B_smem[stage] + b_half_off * BK,  // different smem offset per CTA
    k_coord, bn + b_half_off, // different global coord per CTA
    0x3); // Multicast mask: send to CTAs 0 and 1


// tx_bytes: A_bytes + B_full_bytes
// Each CTA receives: own A (16384) + CTA0's B half (16384) + CTA1's B half (16384) = 49152
mbarrier_arrive_and_expect_tx_fn(full_barriers[stage], 49152);
```

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