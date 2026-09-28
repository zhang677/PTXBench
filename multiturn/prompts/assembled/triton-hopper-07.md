# Triton Contract for NVIDIA Hopper (SM90/SM90a)

This document is a self-contained implementation contract for standard Triton kernels targeting
NVIDIA Hopper. It targets the Triton 3.7.1 language surface. It covers the
Python/JIT boundary, the SPMD execution model, block tensors, types, pointer and descriptor memory
access, control flow, reductions, atomics, matrix multiplication, autotuning, debugging, and the
Hopper-specific TMA and Tensor Core path.

Use the standard frontend:

```python
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
```

This is not a Gluon, CuTe DSL, CUDA C++, or PTX contract. Standard Triton deliberately hides
shared-memory allocation, WGMMA descriptors, mbarriers, and instruction issue/wait groups. Express
the computation with `tl.load`, tensor descriptors, and `tl.dot`; the compiler owns those lowerings.

## 1. Kernel contract and coverage map

Before writing a kernel, state:

1. The logical shape, dtype, device, and element strides of every input and output.
2. The public destination-passing `run(inputs..., outputs...)` signature.
3. The program tile and grid mapping, including the meaning of every `tl.program_id` axis.
4. Which dimensions can be partial and the mask, descriptor padding, or launch rule that protects them.
5. Which values are runtime scalars and which are `tl.constexpr` specialization parameters.
6. Accumulator dtype, reduction order, output conversion, and allowed numerical error.
7. Candidate `num_warps`, `num_stages`, tile shapes, and any persistent scheduling policy.
8. Whether each memory path uses ordinary pointer tensors or a tensor descriptor/TMA.
9. Alignment, contiguity, divisibility, aliasing, and mutation assumptions.
10. The exact Hopper-only features on which the implementation depends.

Do not change one field in isolation. A tile change also changes the grid, pointer shapes, masks,
descriptor block shapes, dot legality, register pressure, shared-memory use, and useful stage count.

| Capability | Standard Triton mechanism |
|---|---|
| GPU program entry | `@triton.jit` |
| Compile-time parameters | annotation `: tl.constexpr` |
| Grid identity | `tl.program_id`, `tl.num_programs` |
| Block construction | `tl.arange`, `tl.full`, `tl.zeros` |
| Global memory | pointer arithmetic plus `tl.load` / `tl.store` |
| Hopper TMA | host `TensorDescriptor` or `tl.make_tensor_descriptor` |
| Tensor Core math | `tl.dot` |
| Reductions and scans | `tl.sum`, `tl.max`, `tl.reduce`, `tl.cumsum`, and peers |
| Synchronizing updates | `tl.atomic_*` |
| Tuning | `triton.Config`, `@triton.autotune`, `@triton.heuristics` |
| Diagnostics | `tl.static_assert`, `tl.device_assert`, `tl.device_print` |

## 2. Host code, JIT code, and launch

### 2.1 The two execution worlds

Ordinary Python in `run` executes on the host. A function decorated with `@triton.jit` is parsed
and compiled as device code. A JIT function may use Python primitives understood by Triton,
`triton.language` builtins, its arguments, compile-time constants, and other JIT functions. It may
not call arbitrary Python, NumPy, or PyTorch computation from device code.

Use PyTorch in `run` only for tensor metadata, device/stream selection, and launching. The caller
already supplies output storage; do not allocate, replace, or return a different output.

```python
@triton.jit
def _scale_kernel(x, y, n_elements, scale: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(y + offsets, values * scale, mask=mask)


def run(x, scale, y):
    torch.cuda.set_device(x.device)
    n_elements = y.numel()
    grid = (triton.cdiv(n_elements, 256),)
    _scale_kernel[grid](x, y, n_elements, scale=scale, BLOCK=256)
```

Scalar `scale` should remain a runtime argument if it varies often. Mark it `tl.constexpr` only when
specializing on its value is intentional. Every distinct constexpr value can create another binary.

### 2.2 Launch syntax

```python
kernel[grid](*runtime_args, **meta_parameters)
```

`grid` is a tuple of one to three positive dimensions or a callable receiving the merged launch
metadata dictionary:

```python
grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),
                     triton.cdiv(N, META["BLOCK_N"]))
kernel[grid](..., M, N, BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3)
```

`num_warps`, `num_stages`, `num_ctas`, and `maxnreg` are compiler launch options, not declared
kernel parameters. Meta-parameters named in `triton.Config.kwargs` must not also be passed at launch.

The first launch of a new specialization compiles it. Later equivalent launches use Triton's cache.
Specialization includes target, argument types, constexpr values, and relevant compiler options.
Changing a runtime shape does not force recompilation unless that shape is constexpr or otherwise
specialized by the JIT policy.

### 2.3 JIT helpers and specialization controls

Device helpers must also use `@triton.jit`:

```python
@triton.jit
def _relu(x):
    return tl.maximum(x, 0.0)
```

Useful decorator arguments include `do_not_specialize`,
`do_not_specialize_on_alignment`, `debug`, `noinline`, and `launch_metadata`. Use specialization
controls only after measuring compile count and generated code; disabling specialization can remove
facts needed for optimization.

## 3. SPMD execution and block tensors

### 3.1 Program instances, not CUDA threads

A Triton launch creates a grid of program instances. Each program evaluates operations on scalar or
N-dimensional block tensors. Triton maps block elements onto warps and emits vector, memory, and
Tensor Core operations. `tl.program_id(axis)` identifies a program along grid axis 0, 1, or 2;
`tl.num_programs(axis)` returns that grid extent.

There is no source-level `threadIdx` for standard Triton. Do not translate a CUDA kernel line by
line. Choose one useful tile per program and express all lanes of that tile as tensors.

### 3.2 Creating blocks

```python
offs = tl.arange(0, BLOCK)                 # [BLOCK]
rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
matrix_offsets = rows[:, None] * stride_m + cols[None, :] * stride_n
zeros = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
filled = tl.full((BLOCK,), 1.0, tl.float32)
```

`tl.arange(start, end)` uses a half-open interval. `start` and `end` are compile-time powers of two,
with `end > start`; normally use `tl.arange(0, BLOCK)` with power-of-two `BLOCK`. Block shapes are
compile-time shapes and are subject to Triton's power-of-two and maximum-element restrictions.

`x[:, None]` and `x[None, :]` insert singleton dimensions. General local-tensor slicing and dynamic
indexing are not supported. Valid tensor indexing is built from full `:` dimensions and `None`.
Use masks, `tl.where`, reshape/split, or a different tile construction instead of `acc[:, :k]`.

### 3.3 Broadcasting

Triton broadcasting follows NumPy: left-pad the shorter shape with ones, then each aligned dimension
must match or one side must be 1. Broadcasting is conceptual and does not itself load or copy data.

```python
row = tl.arange(0, BLOCK_M)[:, None]       # [BM, 1]
col = tl.arange(0, BLOCK_N)[None, :]       # [1, BN]
coords = row * stride_m + col * stride_n   # [BM, BN]
```

Explicit helpers include `tl.broadcast`, `tl.broadcast_to`, and `tl.expand_dims`.

### 3.4 Shape transformations

The common operations are:

- `tl.reshape(x, shape, can_reorder=False)`: same number of elements; preserves linear order unless
  `can_reorder=True` explicitly grants reordering.
- `tl.permute(x, dims)` or `tl.trans(x, dims)`: permute axes. `x.T` swaps the last two axes.
- `tl.ravel(x, can_reorder=False)`: flatten.
- `tl.join(a, b)`: add a final size-2 dimension; `tl.split` is its inverse.
- `tl.cat(a, b, dim=...)`, `tl.interleave`, `tl.flip`, `tl.squeeze`, `tl.unsqueeze`.

`tl.view` is deprecated in Triton 3.7; use `tl.reshape(..., can_reorder=True)` when reordering is
actually legal. Shape transformations change the compiler's logical mapping, not global storage.

## 4. Types, operators, and numerical semantics

### 4.1 Dtypes

Core public scalar dtypes include:

- `tl.int1`, signed `tl.int8/int16/int32/int64`, and unsigned `tl.uint8/uint16/uint32/uint64`;
- `tl.float16`, `tl.bfloat16`, `tl.float32`, `tl.float64`;
- NVIDIA-oriented FP8 names exposed by Triton 3.7, including `tl.float8e4nv` and `tl.float8e5`.

Pointer argument element types are inferred from host tensors. Use `x.to(dtype)` or
`tl.cast(x, dtype, fp_downcast_rounding="rtne"|"rtz")` for numerical conversion. With
`bitcast=True`, bits are reinterpreted rather than numerically converted; source and destination
bit widths must be compatible.

Float-to-integer conversion is defined only if the truncated value fits the target. NaN, infinity,
and out-of-range values are undefined; explicitly handle NaN and clamp before casting.

### 4.2 Promotion

For tensor-tensor binary operations and the value arms of `tl.where`, kind order is
boolean < integer < floating point, then wider types win. Equal-width `float16` and `bfloat16`, or
different FP8 types, promote to `float16`; equal-width mixed signedness prefers unsigned.

A scalar literal or constexpr of equal or lower kind does not widen a tensor. A higher-kind scalar
uses the smallest standard type that can represent it, then participates in promotion. Make
important precision choices explicit rather than relying on promotion in accumulators.

### 4.3 Operators

Arithmetic, comparisons, shifts, and bitwise operators are elementwise and broadcast. Use `&`, `|`,
and `~` for tensor boolean masks; Python `and`, `or`, and `not` are scalar control-flow operators and
must not replace elementwise mask operators.

Integer tensor `//` truncates toward zero and `%` follows the corresponding C rule for mixed signs.
Compile-time scalar-only division follows Python semantics. This difference matters in negative
index calculations.

`tl.where(cond, x, y)` evaluates both value expressions. It cannot make an invalid load safe:

```python
# Wrong: the load may execute for every lane.
value = tl.where(mask, tl.load(ptr), 0.0)

# Correct.
value = tl.load(ptr, mask=mask, other=0.0)
```

## 5. Pointer memory operations

### 5.1 Element pointers and element strides

Pointer arithmetic is in elements of the pointer's pointee type. PyTorch strides are already in
elements:

```python
ptrs = base + rows[:, None] * stride_row + cols[None, :] * stride_col
```

Never multiply a PyTorch stride by element size. Handle negative or zero strides only when the
definition permits them and the address calculation is proven safe.

### 5.2 Loads

```python
value = tl.load(pointer, mask=None, other=None,
                boundary_check=(), padding_option="",
                cache_modifier="", eviction_policy="", volatile=False)
```

For a scalar pointer, `mask` and `other` are scalars. For a tensor of pointers they broadcast to the
pointer shape. A false mask lane performs no memory access and returns `other` converted to the
pointee type. Always provide a neutral `other` when masked values feed arithmetic or reductions.

NVIDIA load cache modifiers are `""`, `".ca"`, `".cg"`, and `".cv"`. Eviction policies include
`"evict_first"` and `"evict_last"`. These are hints, not correctness mechanisms; benchmark them.

### 5.3 Stores

```python
tl.store(pointer, value, mask=None, boundary_check=(),
         cache_modifier="", eviction_policy="")
```

`value` and `mask` broadcast to the pointer shape, and the value converts to the pointee type. A
false mask lane performs no store. NVIDIA store modifiers include `".wb"`, `".cg"`, `".cs"`, and
`".wt"`. Two programs writing the same address without an atomic operation constitute a race even
if they usually write the same value.

### 5.4 Block pointers

Triton 3.7 still provides `tl.make_block_ptr` and `tl.advance`, but `make_block_ptr` is deprecated in
favor of tensor descriptors. For a block pointer, `tl.load`/`tl.store` do not accept `mask` or
`other`; use `boundary_check=(...)` and load `padding_option="zero"` or `"nan"`.

`tl.advance(ptr, offsets)` returns a new block pointer and has no side effect:

```python
block_ptr = tl.advance(block_ptr, (0, BLOCK_K))
```

Do not write a new Hopper guide around a deprecated block-pointer pipeline unless compatibility
with a pre-3.8 deployment requires it.

## 6. Tensor descriptors and Hopper TMA

### 6.1 Host descriptor path

On Hopper, standard Triton tensor-descriptor loads and stores lower to TMA-backed operations. A host
descriptor is created before launch:

```python
a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K])  # B stored [N, K]
c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])
kernel[grid](a_desc, b_desc, c_desc, M, N, K, ...)
```

Inside the kernel:

```python
a = a_desc.load([offset_m, offset_k])
b = b_desc.load([offset_n, offset_k])
c_desc.store([offset_m, offset_n], result)
```

Descriptor loads take offsets only, with no mask. Out-of-bounds loads use descriptor padding;
stores ignore out-of-bounds elements. Offsets and block shapes must satisfy the descriptor/TMA
alignment contract.

### 6.2 Descriptor invariants

For `TensorDescriptor` in Triton 3.7:

- rank is 1 through 5 for a host descriptor; device-side `tl.make_tensor_descriptor` documents 2
  through 5 dimensions;
- base is 16-byte aligned;
- the last stride is 1;
- every leading stride is 16-byte aligned in bytes;
- shapes are positive and descriptor block shapes are compile-time valid block shapes;
- padding is `"zero"` or `"nan"`, with NaN padding only for floating-point tensors.

The descriptor describes physical storage. If B is passed physically as `[N, K]`, load a
`[BLOCK_N, BLOCK_K]` tile and use `b.T` for an `[K, N]` dot operand. Do not silently apply row-major
strides to a transposed view.

Autotuning often changes block shapes. Mutate host descriptor `block_shape` in a config `pre_hook`
or construct one descriptor per configuration; a dummy block shape must not reach execution.

### 6.3 Device-created descriptors

```python
@triton.jit
def kernel(a_ptr, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    tile = desc.load([tl.program_id(0) * BLOCK_M, 0])
```

Device descriptor creation needs Triton's descriptor allocator configured on the host:

```python
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)
```

In a destination-passing benchmark, this allocator is infrastructure storage for descriptors, not
an output allocation. Prefer host descriptors when possible because they avoid repeated device-side
descriptor creation. Use device descriptors when shape construction or a Hopper-specific compiler
path requires them.

### 6.4 What Triton hides

Descriptor loads are logically synchronous at the source level. Do not add manual mbarrier waits,
proxy fences, or shared-memory arrays around them. `num_stages` and loop structure tell the compiler
how much to pipeline. Inspect generated TTGIR/PTX/SASS if instruction-level proof is required; the
presence of a descriptor in source is not by itself proof that every access used TMA.

## 7. Control flow and compile-time programming

### 7.1 Runtime and constexpr branches

An argument annotated `: tl.constexpr` is available during compilation:

```python
@triton.jit
def kernel(x, y, n, USE_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offs, mask=offs < n, other=0.0)
    if USE_BIAS:                 # branch removed at compile time
        value += 1.0
    tl.store(y + offs, value, mask=offs < n)
```

Runtime scalar conditions may form device control flow. Tensor conditions represent lane values and
usually belong in masks or `tl.where`. Define variables before a loop or conditional if they are
used afterward, and assign them on every path; Triton does not use Python's dynamically defined
variable semantics.

### 7.2 Loops

- Python `range` is suitable when the compiler can represent the bounds.
- `tl.static_range` requires constexpr bounds and aggressively unrolls.
- `tl.range` accepts compiler controls: `num_stages`, `loop_unroll_factor`,
  `disallow_acc_multi_buffer`, `flatten`, `warp_specialize`, and `disable_licm`.

`tl.range(..., num_stages=S)` attempts to pipeline most eligible loads in that loop. Kernel launch
`num_stages=S` primarily controls the pipeline of loads feeding dot operations. They are distinct
controls. Excess unrolling or staging can cause register spills or shared-memory overflow.

The general public contract for automatic `warp_specialize=True` is a Blackwell-only simple-matmul
feature. Triton 3.7.1 also contains a narrower Hopper lowering used by its device-descriptor
persistent-matmul path. Treat that Hopper path as a constrained compiler transformation, not as a
general promise that an arbitrary loop can be warp-specialized.

### 7.3 Restricted Hopper warp-specialized descriptor loop

The supported Hopper recipe is more restrictive than CuTe DSL or a hand-written CUDA producer and
consumer pipeline:

- Create the A, B, and C tensor descriptors inside the `@triton.jit` kernel with
  `tl.make_tensor_descriptor`. The corresponding official Hopper path does not use host-created
  `TensorDescriptor` arguments when warp specialization is enabled.
- Apply `warp_specialize` to a simple persistent matmul loop whose body consists primarily of
  descriptor loads, `tl.dot`, and a descriptor store.
- Keep `flatten=False`. The Hopper path is not compatible with the flattened persistent loop used by
  the Blackwell variant.
- Make the switch a `tl.constexpr` and retain `warp_specialize=False` as the correctness and
  performance baseline.
- Let the compiler own warp roles, shared-memory staging, mbarriers, register redistribution, and
  instruction scheduling. Standard Triton does not expose those parts of the transformation.

The structural form is:

```python
@triton.jit
def _descriptor_persistent_matmul(
    a_ptr,
    b_ptr,  # physically [N, K]
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))
```

Launch a persistent grid capped at `min(NUM_SMS, num_tiles)`. As described in the device-created
descriptor section, install Triton's descriptor allocator before the first launch. The allocator is
only infrastructure storage for descriptors; the caller still supplies `c_ptr`.

Compile and test both values of `WARP_SPECIALIZE`. A successful launch is not proof that the intended
partitioning occurred: inspect generated TTGIR/PTX/SASS and profiler results. If the loop grows beyond
the supported form, the transformation fails, or the specialized version regresses, retain the
ordinary software-pipelined configuration.

## 8. Math, reductions, scans, sorting, and randomness

Elementwise math includes `tl.abs`, `tl.minimum`, `tl.maximum`, `tl.clamp`, `tl.exp`, `tl.exp2`,
`tl.log`, `tl.log2`, `tl.sin`, `tl.cos`, `tl.sqrt`, `tl.rsqrt`, `tl.erf`, `tl.sigmoid`, `tl.fma`,
`tl.floor`, and `tl.ceil`. Backend implementations may use approximations; write tolerance and NaN
requirements explicitly.

Reductions include `tl.sum`, `tl.max`, `tl.min`, `tl.argmax`, `tl.argmin`, `tl.xor_sum`, and
`tl.reduce_or`. `axis=None` reduces all dimensions where supported. Use `keep_dims=True` when the
result must broadcast back over the reduced axis. Integer and boolean sums are widened, and
floating sums are accumulated in at least FP32 by default; specify `dtype=` when overflow or exact
promotion matters.

Custom `tl.reduce` and `tl.associative_scan` combiners must be JIT functions and must be associative
for reassociation to be valid. Built-in scans are `tl.cumsum` and `tl.cumprod`. Other block-local
operations include `tl.sort`, `tl.topk`, `tl.gather`, and `tl.histogram`; their legal shapes and cost
are compiler dependent, so avoid using them as substitutes for a better tiled algorithm.

Counter-based random helpers include `tl.randint`, `tl.rand`, and `tl.randn` plus 4-value variants.
Give each logical output element a deterministic, non-overlapping counter when reproducibility is
required.

## 9. Atomics and synchronization

Triton 3.7 exposes `tl.atomic_add`, `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`,
`atomic_xor`, `atomic_xchg`, and `atomic_cas`. They return the value that existed before the update.

Atomic memory semantics are `"relaxed"`, `"acquire"`, `"release"`, or `"acq_rel"`; scopes are
`"cta"`, `"gpu"`, or `"sys"`. Defaults are acquire-release and GPU scope. Select the weakest scope
and ordering that still proves correctness, but do not use a relaxed atomic as a publication barrier.

There is no grid-wide barrier in an ordinary Triton launch. `tl.debug_barrier()` is a program/CTA
debugging barrier, not a grid synchronization primitive. Split global phases into separate kernel
launches unless an algorithm has a proven atomic protocol. Launches on the same CUDA stream are
ordered.

## 10. Matrix multiplication with `tl.dot`

### 10.1 Semantic contract

```python
tl.dot(input, other, acc=None, input_precision=None,
       allow_tf32=None, max_num_imprecise_acc=None,
       out_dtype=tl.float32)
```

Inputs have equal rank at least 2, equal batch prefixes, and matching reduction dimensions. A common
case is `[M, K] @ [K, N] -> [M, N]`; 3-D and higher inputs perform batched dot after compatible
batch-shape handling. If supplied, `acc` has the exact output shape and is added to the product.

Supported common inputs include INT8, FP8, FP16, BF16, and FP32. INT8 uses an INT32 accumulator;
low-precision floating point should normally accumulate in FP32. Convert only after the complete
reduction.

For FP32 inputs on NVIDIA Tensor Cores, `input_precision` may be `"tf32"`, `"tf32x3"`, or
`"ieee"`. The default is TF32 when supported. `allow_tf32` is deprecated. TF32 truncation can bias
results; use explicit precision and, where applicable, descriptor `round_f32_to_tf32=True`.

### 10.2 Canonical tiled GEMM

```python
@triton.jit
def _gemm(a, b, c, M, N, K,
          stride_am, stride_ak, stride_bk, stride_bn,
          stride_cm, stride_cn,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        a_tile = tl.load(a + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                         mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b_tile = tl.load(b + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                         mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a_tile, b_tile, acc)

    out_ptr = c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(out_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

This is a correctness skeleton, not a universal best kernel. Tune tile sizes, program order,
descriptor use, stages, and warps. For repeated large GEMMs, grouped program ordering can improve L2
reuse. For small tile domains, a persistent grid capped at the actual SM count can reduce scheduling
overhead, but every loop tile still needs exact bounds and unique ownership.

### 10.3 Hopper Tensor Core lowering

On Hopper, eligible `tl.dot` operations may lower through Hopper Tensor Core/WGMMA machinery. The
standard frontend does not expose the WGMMA async group protocol. Shapes, dtypes, layouts, alignment,
`num_warps`, `num_stages`, and compiler version determine the lowering. Do not claim WGMMA execution
from source text alone; inspect generated artifacts or profiler metrics.

Hopper WGMMA is warp-group based, so four or eight warps are useful starting points for substantial
dot tiles. More warps can improve parallelism but also increase per-program resource use. Tune rather
than hard-code the assumption that eight is faster.

Hopper FP8 layout support is more restrictive than Blackwell. In particular, a non-transposed FP8
right operand is not a generally portable Hopper fast path. Prefer an explicitly compatible physical
layout, often store B as `[N, K]`, descriptor-load `[BLOCK_N, BLOCK_K]`, and pass `b.T` to `tl.dot`.

## 11. Autotuning and performance controls

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def kernel(..., n_elements, BLOCK: tl.constexpr):
    ...
```

The `key` controls when configurations are rebenchmarked. An overly specific key increases tuning
cost; an overly broad key reuses a poor choice. `reset_to_zero` or `restore_value` is mandatory when
trial configurations mutate state non-idempotently. A config `pre_hook` can update descriptor block
shapes. `prune_configs_by` can remove resource-invalid candidates before compilation.

`@triton.heuristics(values={...})` derives meta-parameters without benchmarking. Do not define the
same symbol in a `Config`, heuristic, and launch call.

Tune at least:

- tile shape and program ordering;
- `num_warps` (usually start with 4 and 8 for Hopper Tensor Core kernels);
- `num_stages` (often 2 through 5, bounded by shared memory and registers);
- pointer versus descriptor/TMA paths;
- accumulator footprint, epilogue shape, and persistent versus full grid.

Performance hints `tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy`, and `tl.assume` assert
facts to the compiler. They do not check or repair data. A false assertion can miscompile the kernel.

## 12. Hopper resource and scheduling facts

For an H100 SXM-class target, use these planning values as a starting point and query the actual
device when possible:

| Resource | Planning value |
|---|---:|
| SMs | 132 |
| Maximum threads per CTA | 1024 |
| Maximum registers per thread | 255 32-bit registers |
| Register allocation ceiling per CTA | 65,536 32-bit registers |
| Shared memory per SM | 256 KiB physical, up to roughly 228 KiB usable by a CTA configuration |
| L2 | 50 MiB |
| HBM3 | 80 GiB, approximately 3.35 TB/s peak |
| Portable cluster size | up to 8 CTAs; platform-specific nonportable sizes may be larger |

Treat these as limits, not targets. Triton-reported registers, spills, shared memory, and achieved
occupancy are the evidence for a specific kernel. `num_ctas > 1` requests clustered compilation
where supported; it does not provide source-level cluster indexing, DSMEM, or barriers. Use it only
for a compiler path known to exploit it and benchmark against `num_ctas=1`.

Hopper-specific optimization sequence:

1. Establish correct masked pointer code with FP32 accumulation.
2. Tune tiles, warps, stages, and L2 program ordering.
3. Test host tensor descriptors/TMA when shape and stride constraints fit.
4. Confirm the intended Tensor Core and TMA instructions with generated-code or profiler evidence.
5. Check spills and occupancy before adding stages or larger accumulator tiles.

## 13. Debugging and validation

Use `tl.static_assert` for compile-time invariants and `tl.static_print` for compile-time values.
`tl.device_assert` is active when Triton debug mode is enabled; `tl.device_print` prints runtime
values but is expensive. `TRITON_INTERPRET=1` can expose indexing and logic errors, but interpreter
floating behavior, unsupported dtypes, and race behavior are not proof of GPU correctness.

`tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)` is an escape hatch for
scalar-per-element assembly. Constraints and `pack` must match the register contract. Inline asm is
not the standard way to manage TMA or WGMMA collectives; use it only with architecture-specific
proof and fallback/version boundaries.

Validation order:

1. Parse/import and compile each intended specialization.
2. Test zero, one, exact-tile, partial-tile, and multi-tile dimensions.
3. Test all supported dtypes, strides, and alias patterns.
4. Compare with the reference using the required numerical tolerance.
5. Run memory and race checking where supported.
6. Benchmark warmed kernels and report compile/autotune time separately.
7. Inspect registers, local-memory spills, shared memory, occupancy, and generated instructions.

## 14. Common failure modes

### 14.1 Compilation failures

- `tl.arange` endpoints or block dimensions are not valid compile-time powers of two.
- A runtime value is used where a shape, axis, loop control, or dtype must be constexpr.
- A local tensor is dynamically sliced or indexed.
- A constexpr is supplied both by `triton.Config` and the launch.
- `tl.dot` ranks, batch prefixes, reduction dimensions, dtypes, or accumulator shape disagree.
- Descriptor rank, alignment, last stride, leading-stride alignment, or block shape is invalid.
- Register or shared-memory use exceeds the target limit.

### 14.2 Incorrect results

- A false mask lane participates because `other` was omitted or was not the reduction identity.
- `tl.where` was incorrectly expected to guard a load.
- Strides were treated as bytes rather than elements.
- B was logically transposed without matching its physical strides or descriptor.
- The K tail was read without zero padding before a dot.
- The accumulator was downcast inside the K loop.
- Two program instances store the same output without a valid reduction protocol.
- A compiler hint asserted alignment or divisibility that the actual pointer did not satisfy.
- Autotune trials mutated an output or counter and it was not reset.

### 14.3 Performance failures

- The program tile is too small to amortize launch and address computation.
- A large accumulator causes spills or collapses occupancy.
- `num_stages` exceeds the useful memory-latency window or shared-memory budget.
- Program order destroys L2 reuse.
- A descriptor path pays setup cost without enough regular data movement to recover it.
- Excessive masking, scalar control flow, sorting, atomics, or device printing serializes work.
- Benchmarking includes first-compile or autotune cost in kernel latency.

When fixing a failure, preserve the full contract: logical coordinates, physical strides, masks or
padding, specialization boundary, and destination ownership must remain mutually consistent.




# Optimization pattern: Producer-Consumer asynchrony through warp-specialization and pingpong scheduling

### Warp-specialization

As with FlashAttention-2, the forward pass of **FlashAttention-3** is embarrassingly parallel in the batch size, number of heads, and query sequence length. Thus, it will suffice to give a CTA-level view of the algorithm, which operates on a tile $Q_i$ of the query matrix to compute the corresponding tile $O_i$ of the output. To simplify the description, we first give the warp-specialization scheme with a circular SMEM buffer that does not have in addition the GEMM–softmax overlapping. Let $d$ be the head dimension, $N$ the sequence length, and fix a query block size $B_r$ to divide $Q$ into $T_r = \lceil N / B_r \rceil$ blocks $Q_1, \ldots, Q_{T_r}$.

```latex
\begin{algorithm}
\caption{Algorithm 1: FlashAttention-3 forward pass without intra-consumer overlapping -- CTA view}
\begin{algorithmic}[1]
\Require Matrices $Q_i \in \mathbb{R}^{B_r \times d}$ and $K, V \in \mathbb{R}^{N \times d}$ in HBM, key block size $B_c$ with $T_c = \lceil N / B_c \rceil$.
\State Initialize pipeline object to manage barrier synchronization with $s$-stage circular SMEM buffer.
\If{in producer warpgroup}
    \State Deallocate predetermined number of registers.
    \State Issue load $Q_i$ from HBM to shared memory.
    \State Upon completion, commit to notify consumer of the load of $Q_i$.
    \For{$0 \le j < T_c$}
        \State Wait for the $(j \bmod s)$th stage of the buffer to be consumed.
        \State Issue loads of $K_j, V_j$ from HBM to shared memory at the $(j \bmod s)$th stage of the buffer.
        \State Upon completion, commit to notify consumers of the loads of $K_j, V_j$.
    \EndFor
\Else
    \State Reallocate predetermined number of registers as function of number of consumer warps.
    \State On-chip, initialize $O_i = (0) \in \mathbb{R}^{B_r \times d}$ and $\ell_i, m_i = (0), (-\infty) \in \mathbb{R}^{B_r}$.
    \State Wait for $Q_i$ to be loaded in shared memory.
    \For{$0 \le j < T_c$}
        \State Wait for $K_j$ to be loaded in shared memory.
        \State Compute $S_i^{(j)} = Q_i K_j^{\top}$ (SS-GEMM). Commit and wait.
        \State Store $m_i^{old} = m_i$ and compute $m_i = \max\!\left(m_i^{old},\, \mathrm{rowmax}(S_i^{(j)} / \sqrt{d})\right)$.
        \State Compute $\widetilde{P}_i^{(j)} = \exp\!\left(S_i^{(j)} - m_i\right)$ and $\ell_i = \exp(m_i^{old} - m_i)\,\ell_i + \mathrm{rowsum}(\widetilde{P}_i^{(j)})$.
        \State Wait for $V_j$ to be loaded in shared memory.
        \State Compute $O_i = \mathrm{diag}(\exp(m_i^{old} - m_i)) O_i + \widetilde{P}_i^{(j)} V_j$ (RS-GEMM). Commit and wait.
        \State Release the $(j \bmod s)$th stage of the buffer for the producer.
    \EndFor
    \State Compute $O_i = \mathrm{diag}(\ell_i)^{-1} O_i$ and $L_i = m_i + \log(\ell_i)$.
    \State Write $O_i$ and $L_i$ to HBM as the $i$th block of $O$ and $L$.
\EndIf
\end{algorithmic}
\end{algorithm}
```

For our implementation of Algorithm 1 on Hopper, we use `setmaxnreg_inc_sync_fn` and `setmaxnreg_dec_sync_fn` for (de)allocations, TMA for loads of $Q_i$ and $\{K_j, V_j\}_{0 \le j < T_c}$, and WGMMA to execute the GEMMs in the consumer mainloop, with the SS or RS prefix indicating whether the first operand is sourced from shared memory or register file. For interpreting the execution flow of Algorithm 1, note that issuing TMA loads does not stall on the completion of other loads due to asynchrony. Moreover, in the producer mainloop, no waits will be issued for the first $s$ iterations as the buffer gets filled.

### Pingpong scheduling

The asynchronous nature of WGMMA and TMA, along with warp-specialization, opens up the opportunity to overlap the softmax computation of one warpgroup with the GEMM of another warpgroup. To motivate this, notice that non-matmul operations have much lower throughput than matmul operations on modern hardware accelerators. As an example, the H100 SXM5 GPU has 989 TFLOPS of FP16 matmul but only 3.9 TFLOPS of special functions such as exponential[^1] (necessary for softmax). For the attention forward pass in FP16 with head dimension 128, there are $512\times$ more matmul FLOPs compared to exponential operations, but the exponential has $256\times$ lower throughput, so exponential can take $50\%$ of the cycle compared to matmul. The situation is even worse with FP8, where the matmul throughput doubles but the exponential throughput stays the same.

Since the exponential is performed by a separate hardware unit (the multi-function unit), ideally we'd want the exponential calculation to be scheduled when the Tensor Cores are performing the matmul. To do so, we use synchronization barriers (`named_barrier_sync_fn` primitives) to force the GEMMs (GEMM1 — $PV$ of one iteration, and GEMM0 — $QK^{\top}$ of the next iteration) of warpgroup 1 to be scheduled before the GEMMs of warpgroup 2. As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs. Then the roles swap, with warpgroup 2 doing softmax while warpgroup 1 doing GEMMs (hence, "pingpong" scheduling). We generally find pingpong scheduling to improve performance (e.g., from 570 TFLOPS to 620–640 TFLOPS for FP16 forward with head dimension 128 and sequence length 8192).

### Attention variants

For multi-query attention and grouped query attention, we follow the approach in **FlashAttention-2** and adjust the tensor indexing to avoid duplicating $K$ and $V$ in HBM.

[^1]: The CUDA programming guide specifies that 16 operations of special functions can be performed per streaming multiprocessor (SM) per clock cycle. We multiply 16 by 132 SMs and 1830 MHz clock speed to get 3.9 TFLOPS of special functions.


