# Optimization Pattern: TMA Store

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
    tma_store_2d_fn(&dC, C_smem, bn, bm); // first 64 rows
    tma_store_2d_fn(&dC, C_smem+16384, bn, bm + 64);  // next 64 rows
    tma_store_commit_fn();   // cp.async.bulk.commit_group
    tma_store_wait_fn<0>();  // cp.async.bulk.wait_group 0
}
__syncthreads();
```