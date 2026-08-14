# Optimization Pattern: L2-Aware Grouped Tile Ordering in Triton

Standard Triton exposes tile ordering directly through `tl.program_id` arithmetic. It does not
provide an L2-cache residency API; grouped ordering changes which output tile each program computes
so programs launched near one another are more likely to reuse an operand already resident in L2.
The GPU scheduler may still reorder programs, so treat the mapping as a locality hint and measure it.

## Grouped one-dimensional mapping

Launch a one-dimensional grid containing one program per output tile. Convert its flat program ID
into `(pid_m, pid_n)` in groups of `GROUP_M` M tiles:

```python
@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n
```

Use it in the kernel before constructing pointer tensors or descriptor offsets:

```python
tile_id = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
pid_m, pid_n = _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M,
)

offset_m = pid_m * BLOCK_M
offset_n = pid_n * BLOCK_N
```

The matching host grid is:

```python
grid = (
    triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
)
```

For a row-major output-tile grid, `GROUP_M=1` recovers ordinary row-major traversal. A larger group
visits several M tiles at the same N position before advancing N, so those programs can reuse the
same B tile. The final M group can be smaller than `GROUP_M`; computing `group_size_m` is required to
keep the mapping bijective and in bounds.

## Persistent-kernel composition

In a persistent kernel, apply the same mapping to the loop's `tile_id`, not only to the initial
`tl.program_id(0)`:

```python
start_tile = tl.program_id(0)
tile_stride = tl.num_programs(0)
num_tiles = num_pid_m * num_pid_n

for tile_id in tl.range(start_tile, num_tiles, tile_stride):
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    # Compute and store exactly output tile (pid_m, pid_n).
```

Do not group a full-grid launch and then accidentally use the original flat ID for a descriptor
offset, output mask, or store. Every address calculation for the tile must use the remapped pair.

## Tuning and validation

- Make `GROUP_M` a `tl.constexpr` and tune values such as 1, 4, and 8 together with tile shape,
  `num_warps`, and `num_stages`.
- Keep `GROUP_M=1` as the ordering baseline. A larger value can reduce locality when the grouped
  working set no longer fits in L2 or when the other operand is more valuable to reuse.
- Preserve exact tail handling. Pointer loads and stores still need masks; descriptor accesses still
  need valid block shapes and padding behavior.
- Compare latency across the complete workload set. If available, use profiler L2 hit-rate and DRAM
  traffic metrics to confirm the proposed mechanism rather than inferring cache behavior from source
  order alone.
