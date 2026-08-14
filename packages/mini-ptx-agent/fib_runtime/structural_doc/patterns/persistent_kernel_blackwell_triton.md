# Optimization Pattern: Persistent Device-Descriptor GEMM on Blackwell Triton

On Blackwell, a high-performance standard-Triton GEMM can compose four source-level controls:

1. a one-dimensional persistent grid capped by the useful SM count;
2. device-created tensor descriptors for TMA-backed A and B loads;
3. grouped output-tile ordering for L2 locality; and
4. automatic warp specialization on the persistent matmul loop.

This is a compiler-managed path. Standard Triton does not expose TMEM allocation, TMA mbarriers,
producer and consumer warp IDs, or `tcgen05` issue and wait groups. Do not add manual versions of
those mechanisms around descriptor operations.

## Host launch and configuration

Query the active tensor's device rather than hard-coding an SM count. Launch no more programs than
there are output tiles, and pass the actual launched count as the persistent-loop stride:

```python
def run(a, b, c):
    torch.cuda.set_device(a.device)
    M, K = a.shape
    N = c.shape[1]

    block_m = 128
    block_n = 256
    block_k = 64
    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    device_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_programs = min(device_sms, num_tiles)

    _persistent_tma_gemm[(num_programs,)](
        a,
        b,
        c,
        M,
        N,
        K,
        NUM_PROGRAMS=num_programs,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EPILOGUE_SUBTILE=2,
        FLATTEN=True,
        WARP_SPECIALIZE=True,
        num_warps=8,
        num_stages=4,
    )
```

Device-created descriptors require Triton's descriptor allocator to be installed before launch.
That allocator provides descriptor infrastructure only; the caller still owns and supplies `c`.
Keep a full-grid pointer-tensor GEMM and a non-specialized descriptor GEMM as measured baselines.

Useful Blackwell starting points include `(BLOCK_M, BLOCK_N, BLOCK_K)` values such as
`(128, 256, 64)`, `(256, 128, 64)`, `(128, 256, 128)`, and `(128, 128, 128)`. These are candidates,
not universal winners. Tune them with 4 or 8 warps, at least 2 stages for the specialized path,
`GROUP_M`, and epilogue subtiling. Do not enable automatic warp specialization for a configuration
with fewer than 4 warps or fewer than 2 stages.

## Device-created descriptors

Create descriptors once per program, outside the persistent loop. The descriptor must match the
physical storage. For a common `C = A @ B.T` contract where A is `[M, K]`, B is physically
`[N, K]`, and C is `[M, N]`:

```python
a_desc = tl.make_tensor_descriptor(
    a,
    shape=[M, K],
    strides=[K, 1],
    block_shape=[BLOCK_M, BLOCK_K],
    padding_option="zero",
)
b_desc = tl.make_tensor_descriptor(
    b,
    shape=[N, K],
    strides=[K, 1],
    block_shape=[BLOCK_N, BLOCK_K],
    padding_option="zero",
)
c_desc = tl.make_tensor_descriptor(
    c,
    shape=[M, N],
    strides=[N, 1],
    block_shape=[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE],
)
```

Use runtime strides instead when the public input contract permits layouts other than these dense
row-major forms. Never describe a physical `[N, K]` B tensor as `[K, N]`; load a
`[BLOCK_N, BLOCK_K]` tile and transpose that block for `tl.dot`.

## Persistent grouped tile loop

Apply grouped ordering to every persistent-loop `tile_id`. Reconstruct all descriptor offsets and
tile-local state on every iteration:

```python
start_pid = tl.program_id(0)
grid_m = tl.cdiv(M, BLOCK_M)
grid_n = tl.cdiv(N, BLOCK_N)
num_tiles = grid_m * grid_n
num_pid_in_group = GROUP_M * grid_n

for tile_id in tl.range(
    start_pid,
    num_tiles,
    NUM_PROGRAMS,
    flatten=FLATTEN,
    warp_specialize=WARP_SPECIALIZE,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = (k_tile * BLOCK_K).to(tl.int32)
        a_tile = a_desc.load([offset_m, offset_k])
        b_tile = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a_tile, b_tile.T, acc)

    # Convert and store this tile, optionally through epilogue subtiles.
```

The launch grid and `NUM_PROGRAMS` must agree. Passing the device's total SM count as the loop
stride while intentionally launching fewer programs can leave tiles uncovered. Every flat tile ID
must map to exactly one `(pid_m, pid_n)` pair, including the final short M group.

`WARP_SPECIALIZE` and `FLATTEN` are compile-time switches. The supported path is this simple
descriptor-load, dot, and store loop. Arbitrary branches, atomics, complicated loop-carried state,
or values shared across incompatible prologue and epilogue positions can make the transformation
illegal or unprofitable. Compile and test `WARP_SPECIALIZE=False`; do not assume specialization won
because the specialized kernel compiled.

## Epilogue subtiling and descriptor stores

An FP32 accumulator for a large output tile can keep substantial state live through conversion and
storage. Split the N dimension into a power-of-two number of subtiles and store them sequentially.
For `EPILOGUE_SUBTILE` in `{1, 2, 4}`, each stored block has shape
`[BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]`:

```python
@triton.jit
def _subtile_accumulator(
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBTILE_FACTOR: tl.constexpr,
):
    tl.static_assert(SUBTILE_FACTOR > 0)
    tl.static_assert((SUBTILE_FACTOR & (SUBTILE_FACTOR - 1)) == 0)
    if SUBTILE_FACTOR == 1:
        return (acc,)
    else:
        tl.static_assert(BLOCK_N % 2 == 0)
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        left, right = tl.split(acc)
        return _subtile_accumulator(
            left, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        ) + _subtile_accumulator(
            right, BLOCK_M, BLOCK_N // 2, SUBTILE_FACTOR // 2
        )


subtiles = _subtile_accumulator(
    acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE
)

for i in tl.static_range(EPILOGUE_SUBTILE):
    offset_n_i = offset_n + i * (BLOCK_N // EPILOGUE_SUBTILE)
    c_desc.store([offset_m, offset_n_i], subtiles[i].to(tl.bfloat16))
```

Keep the explicit `else` in the recursive helper. Triton specializes the constexpr branch; placing
the recursive code after an early `return` can still make the frontend compile a zero-factor
recursive call.

The conversion above is for a BF16 output contract. Use the declared public output dtype instead
when the task requires another type; do not infer it from accumulator precision.

The descriptor block shape, subtile shape, and offset increment must match. Subtiling is a tuning
choice rather than a numerical change: compare factors 1, 2, and 4, and check generated code for
register pressure or spills. A direct pointer store remains valid when descriptor-store constraints
or setup costs do not pay off.

## Validation checklist

- Test `WARP_SPECIALIZE=False` and `True` independently; use at least 4 warps and 2 stages for `True`.
- Test `EPILOGUE_SUBTILE=1` before 2 or 4 so errors are not hidden inside reshape and split logic.
- Verify exact tile coverage for `num_tiles < num_programs`, a final short M group, and partial M/N.
- Verify dense and any allowed non-dense physical layouts against their descriptor strides.
- Measure the full-grid pointer baseline, persistent descriptor baseline, and specialized variant.
- Inspect generated TTGIR/PTX/SASS when an instruction-level claim matters. Source-level `tl.dot`
  and descriptor use do not by themselves prove a particular TMA, TMEM, or `tcgen05` lowering.
