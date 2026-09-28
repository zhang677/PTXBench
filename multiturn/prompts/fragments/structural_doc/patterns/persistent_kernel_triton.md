# Optimization Pattern: Persistent Tile Scheduling in Triton

A standard Triton persistent kernel launches fewer program instances than output tiles and lets each
program process multiple tiles sequentially. Triton exposes this policy through the host grid and a
device loop; there is no separate `persistent=True` launch option.

Persistence does not remove a kernel launch, and a cyclic schedule is not work stealing. Its useful
effects are a bounded program count, amortized per-program setup, and reduced wave-quantization or
scheduler-tail overhead. It can also reduce available parallelism, increase register lifetimes, and
make one slow program responsible for several tiles, so retain a full-grid baseline.

## Grid and cyclic tile ownership

Start with at most one program per SM, capped by the number of output tiles:

```python
def run(a, b, c):
    torch.cuda.set_device(a.device)
    M, K = a.shape
    N = c.shape[1]
    block_m = 128
    block_n = 128
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(num_sms, num_tiles),)

    _persistent_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
```

The caller supplies `c`; the persistent wrapper must not allocate or replace the output. Querying the
device and tensor metadata in `run` is host-side launch logic, not GPU computation.

Inside the kernel, stride the flat tile loop by the actual number of launched programs:

```python
start_tile = tl.program_id(0)
tile_stride = tl.num_programs(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_tiles = num_pid_m * num_pid_n

for tile_id in tl.range(start_tile, num_tiles, tile_stride):
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    # Reinitialize all tile-local accumulators and loop-carried state here.
    # Load, compute, and store exactly tile (pid_m, pid_n).
```

This cyclic assignment is race-free when every flat `tile_id` maps to one output tile and only that
program stores it. Do not use `tl.program_id(0)` as the output tile after entering the loop, and do
not advance to the next tile before the current tile's store is logically complete.

## Combining persistence with other Triton controls

- L2 grouping composes with persistence by remapping each loop `tile_id` to `(pid_m, pid_n)` using
  the grouped-order formula. Apply the remapping on every iteration.
- Pointer-tensor kernels must reconstruct offsets and masks for every tile. Tensor-descriptor kernels
  must update descriptor offsets for every tile; do not retain offsets from the previous iteration.
- `num_stages` and `tl.range(..., num_stages=...)` control software pipelining, not persistent tile
  ownership. More stages can make a persistent program too large to sustain the intended occupancy.
- For the restricted Hopper warp-specialized path, use device-created descriptors, keep
  `flatten=False`, and place `warp_specialize=WARP_SPECIALIZE` on the simple persistent matmul loop as
  described in the Hopper contract. Always retain `WARP_SPECIALIZE=False` as a baseline.

## Tuning and validation

- One program per SM is a starting point, not a universal optimum. Compare the full grid and small
  multiples or fractions of the SM count when resource use and workload size justify them.
- Include small tile domains. When `num_tiles` is already smaller than or close to the SM count,
  persistence may add loop overhead without improving utilization.
- Reinitialize accumulators, online-reduction state, masks, and output coordinates for every tile.
  Only values intentionally invariant across tiles should remain live across iterations.
- Verify that all `num_tiles` are covered exactly once for non-divisible M and N, then measure latency,
  registers, spills, shared memory, and achieved occupancy against the full-grid kernel.
