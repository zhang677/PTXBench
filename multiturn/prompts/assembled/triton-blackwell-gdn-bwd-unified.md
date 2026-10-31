# Triton Contract for NVIDIA Blackwell B200 (SM100/SM100a)

This document is a self-contained implementation contract for standard Triton kernels targeting
NVIDIA Blackwell. It targets the Triton 3.7.1 language surface. It covers the
Python/JIT boundary, the SPMD execution model, block tensors, types, pointer and descriptor memory
access, control flow, reductions, atomics, matrix multiplication, autotuning, debugging, and the
Blackwell-specific TMA, fifth-generation Tensor Core, block-scaled dot, and automatic warp-specialization paths.

Use the standard frontend:

```python
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
```

This is not a Gluon, CuTe DSL, CUDA C++, or PTX contract. Standard Triton deliberately hides
shared-memory allocation, TMEM allocation, `tcgen05` descriptors, mbarriers, and instruction
issue/wait groups. Express
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
10. The exact Blackwell-only features on which the implementation depends.

Do not change one field in isolation. A tile change also changes the grid, pointer shapes, masks,
descriptor block shapes, dot legality, register pressure, shared-memory use, and useful stage count.

| Capability | Standard Triton mechanism |
|---|---|
| GPU program entry | `@triton.jit` |
| Compile-time parameters | annotation `: tl.constexpr` |
| Grid identity | `tl.program_id`, `tl.num_programs` |
| Block construction | `tl.arange`, `tl.full`, `tl.zeros` |
| Global memory | pointer arithmetic plus `tl.load` / `tl.store` |
| Blackwell TMA | host `TensorDescriptor` or `tl.make_tensor_descriptor` |
| Tensor Core math | `tl.dot` |
| Native block-scaled Tensor Core math | `tl.dot_scaled` |
| Automatic warp specialization | `tl.range(..., warp_specialize=True)` |
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

Do not write a new Blackwell guide around a deprecated block-pointer pipeline unless compatibility
with a pre-3.8 deployment requires it.

## 6. Tensor descriptors and Blackwell TMA

### 6.1 Host descriptor path

On Blackwell, standard Triton tensor-descriptor loads and stores lower to TMA-backed operations. A host
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
descriptor creation. Use device descriptors when shape construction or a Blackwell-specific compiler
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

For this Triton version, automatic `warp_specialize=True` is a Blackwell-only feature and currently
works on simple matmul loops. It partitions memory, MMA, and vector work into asynchronous warp
roles and increases the total number of warps. Treat it as a compiler transformation with a narrow
legality envelope, not as a promise that arbitrary control flow will be partitioned.

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

### 10.3 Blackwell Tensor Core and Tensor Memory lowering

On Blackwell, eligible `tl.dot` operations may lower through fifth-generation Tensor Core
`tcgen05.mma` machinery. Accumulators can reside in Blackwell Tensor Memory (TMEM), but standard
Triton does not expose TMEM allocation, addresses, copy shapes, or the `tcgen05` commit/wait protocol.
Shapes, dtypes, layouts, alignment, `num_warps`, `num_stages`, loop structure, and compiler version
determine the lowering. Do not claim `tcgen05` or TMEM use from source text alone; inspect generated
artifacts or profiler metrics.

Four and eight warps are useful non-specialized starting points for substantial dot tiles. Automatic
warp specialization adds producer/consumer roles beyond that logical starting point, so its resource
usage must be measured independently. `maxnreg` can sometimes balance role-specific register demand,
but an arbitrary cap can force spills.

Blackwell supports a broader set of FP8 operand layouts than Hopper. Still prefer an explicit,
coalesced physical layout, often store B as `[N, K]`, descriptor-load `[BLOCK_N, BLOCK_K]`, and pass
`b.T` to `tl.dot`. This mapping also aligns naturally with the block-scaled RHS convention below.

### 10.4 Blackwell automatic warp specialization

The source-level switch is an attribute on the loop, not a launch option:

```python
for k0 in tl.range(0, k_tiles, num_stages=NUM_STAGES,
                   warp_specialize=WARP_SPECIALIZE):
    a = a_desc.load([offs_m, k0 * BLOCK_K])
    b = b_desc.load([offs_n, k0 * BLOCK_K])
    acc = tl.dot(a, b.T, acc)
```

`WARP_SPECIALIZE` should be constexpr and normally participates in the autotune key or configuration
set. The supported Triton 3.7 path is a simple descriptor-load/dot loop. Complicated loop-carried
state, arbitrary branches, unsupported operations, or values shared in incompatible prologue and
epilogue positions can prevent or invalidate the transformation. Always keep a
`warp_specialize=False` configuration as a correctness and performance baseline.

Warp specialization can increase shared memory and register pressure even when it overlaps TMA,
MMA, and epilogue work. Tune tile shape, stage count, `maxnreg`, and epilogue subtiling together.

## 11. Blackwell block-scaled matrix multiplication

### 11.1 `tl.dot_scaled` contract

```python
tl.dot_scaled(lhs, lhs_scale, lhs_format,
              rhs, rhs_scale, rhs_format,
              acc=None, fast_math=False,
              lhs_k_pack=True, rhs_k_pack=True,
              out_dtype=tl.float32)
```

This computes a 2-D matrix product whose inputs use microscaling formats. Triton 3.7 accepts format
strings `"e2m1"`, `"e4m3"`, `"e5m2"`, `"bf16"`, and `"fp16"`. FP4 E2M1 data is packed two values
per `uint8`, with the first logical element in the low nibble. FP8 may be held in `uint8` or the
corresponding FP8 dtype. In this version `out_dtype` is FP32.

For OCP MX scales represented as E8M0 `uint8`, each scale normally covers 32 K elements. If logical
`lhs` is `[M, K]`, `lhs_scale` is `[M, K // 32]`. If logical `rhs` is `[K, N]`, `rhs_scale` is
`[N, K // 32]`; do not transpose the RHS scale tensor to `[K // 32, N]`. `lhs_k_pack=False` means
packed FP4 storage is along M instead of K; `rhs_k_pack=False` means it is along N instead of K.

Blackwell provides native hardware acceleration for supported block-scaled combinations. Triton may
emulate this operation on architectures without native support, so source-level `tl.dot_scaled` is
not by itself proof of native instruction use.

### 11.2 Descriptor-loaded block-scaled loop

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for k0 in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
    a = a_desc.load([offs_m, k0 * (BLOCK_K // A_ELEMS_PER_BYTE)])
    b = b_desc.load([offs_n, k0 * (BLOCK_K // B_ELEMS_PER_BYTE)])
    a_scale = a_scale_desc.load([...])
    b_scale = b_scale_desc.load([...])
    # Reshape/permute the preshuffled scale tiles into logical 2-D scale blocks.
    a_scale = a_scale.reshape(...).trans(...).reshape(BLOCK_M, BLOCK_K // 32)
    b_scale = b_scale.reshape(...).trans(...).reshape(BLOCK_N, BLOCK_K // 32)
    acc = tl.dot_scaled(a, a_scale, A_FORMAT,
                        b.T, b_scale, B_FORMAT, acc)
```

`BLOCK_K` above counts logical low-precision elements, while descriptor K offsets count storage
elements. Keep the conversion by `*_ELEMS_PER_BYTE` explicit. Scale preshuffling is part of the
physical data contract: do not apply the logical 2-D scale shape directly to a packed 5-D descriptor
without the matching reshape/transpose inverse.

Blackwell FP4 Tensor Core layouts have additional operand-orientation constraints. For symmetric
FP4, a physically `[N, K/2]` packed RHS loaded as `[BLOCK_N, BLOCK_K/2]` and passed as `b.T` is the
safe standard recipe. Mixed FP8/FP4 has broader hardware choices, but the descriptor layout and
`rhs_k_pack` flag must still describe the same packing.

## 12. Autotuning and performance controls

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
- `num_warps` (usually start with 4 and 8 for non-specialized Tensor Core kernels);
- `num_stages` (often 2 through 5, bounded by shared memory and registers);
- pointer versus descriptor/TMA paths;
- automatic warp specialization, `maxnreg`, and epilogue subtiling;
- standard `tl.dot` versus native-eligible `tl.dot_scaled` where the data format permits it;
- accumulator footprint, epilogue shape, and persistent versus full grid.

Performance hints `tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy`, and `tl.assume` assert
facts to the compiler. They do not check or repair data. A false assertion can miscompile the kernel.

## 13. Blackwell resource and scheduling facts

For an NVIDIA B200 target, use these planning values as a starting point and query the actual
device when possible:

| Resource | Planning value |
|---|---:|
| SMs | 148 |
| Maximum threads per CTA | 1024 |
| Maximum registers per thread | 255 32-bit registers |
| Register allocation ceiling per CTA | 65,536 32-bit registers |
| Shared memory per SM | configurable up to 228 KiB, with 1 KiB reserved from a CTA's addressable maximum |
| Tensor Memory | 512 columns by 128 rows of 32-bit cells per SM, managed by the compiler |
| L2 | 126 MiB |
| HBM3e | 180 GiB, approximately 8 TB/s peak |
| Cluster size | portable up to 8 CTAs; B200 nonportable configurations can reach 16 |

Treat these as limits, not targets. Triton-reported registers, spills, shared memory, and achieved
occupancy are the evidence for a specific kernel. `num_ctas > 1` requests clustered compilation
where supported; it does not provide source-level cluster indexing, DSMEM, or barriers. Use it only
for a compiler path known to exploit it and benchmark against `num_ctas=1`.

Blackwell-specific optimization sequence:

1. Establish correct masked pointer code with FP32 accumulation.
2. Tune tiles, warps, stages, and L2 program ordering.
3. Test host tensor descriptors/TMA when shape and stride constraints fit.
4. Test `warp_specialize=True` only on its supported simple-loop shape and retain the baseline.
5. Use `tl.dot_scaled` only when packing, scale layout, and numerical semantics match the problem.
6. Confirm the intended `tcgen05`, TMEM, block-scaled MMA, and TMA instructions with generated-code
   or profiler evidence.
7. Check spills, shared memory, and occupancy before adding stages or larger accumulator tiles.

## 14. Debugging and validation

Use `tl.static_assert` for compile-time invariants and `tl.static_print` for compile-time values.
`tl.device_assert` is active when Triton debug mode is enabled; `tl.device_print` prints runtime
values but is expensive. `TRITON_INTERPRET=1` can expose indexing and logic errors, but interpreter
floating behavior, unsupported dtypes, and race behavior are not proof of GPU correctness.

`tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)` is an escape hatch for
scalar-per-element assembly. Constraints and `pack` must match the register contract. Inline asm is
not the standard way to manage TMA, TMEM, or `tcgen05` collectives; use it only with architecture-specific
proof and fallback/version boundaries.

Validation order:

1. Parse/import and compile each intended specialization.
2. Test zero, one, exact-tile, partial-tile, and multi-tile dimensions.
3. Test all supported dtypes, strides, and alias patterns.
4. Compare with the reference using the required numerical tolerance.
5. Run memory and race checking where supported.
6. Benchmark warmed kernels and report compile/autotune time separately.
7. Inspect registers, local-memory spills, shared memory, occupancy, and generated instructions.

## 15. Common failure modes

### 15.1 Compilation failures

- `tl.arange` endpoints or block dimensions are not valid compile-time powers of two.
- A runtime value is used where a shape, axis, loop control, or dtype must be constexpr.
- A local tensor is dynamically sliced or indexed.
- A constexpr is supplied both by `triton.Config` and the launch.
- `tl.dot` ranks, batch prefixes, reduction dimensions, dtypes, or accumulator shape disagree.
- Descriptor rank, alignment, last stride, leading-stride alignment, or block shape is invalid.
- Register or shared-memory use exceeds the target limit.

### 15.2 Incorrect results

- A false mask lane participates because `other` was omitted or was not the reduction identity.
- `tl.where` was incorrectly expected to guard a load.
- Strides were treated as bytes rather than elements.
- B was logically transposed without matching its physical strides or descriptor.
- The K tail was read without zero padding before a dot.
- The accumulator was downcast inside the K loop.
- Two program instances store the same output without a valid reduction protocol.
- A compiler hint asserted alignment or divisibility that the actual pointer did not satisfy.
- Autotune trials mutated an output or counter and it was not reset.

### 15.3 Performance failures

- The program tile is too small to amortize launch and address computation.
- A large accumulator causes spills or collapses occupancy.
- `num_stages` exceeds the useful memory-latency window or shared-memory budget.
- Program order destroys L2 reuse.
- A descriptor path pays setup cost without enough regular data movement to recover it.
- Excessive masking, scalar control flow, sorting, atomics, or device printing serializes work.
- Benchmarking includes first-compile or autotune cost in kernel latency.

When fixing a failure, preserve the full contract: logical coordinates, physical strides, masks or
padding, specialization boundary, and destination ownership must remain mutually consistent.




# GDN backward operator and numerical reference

Implement the direct GDN backward operator with the saved-input semantics,
output shapes, and dtypes described below. All six outputs must meet the
workload evaluator's numerical accuracy requirements. Bitwise reproduction of
the reference or its intermediate values is not required.

The arithmetic recipe below explains the reference's rounding-sensitive
behavior; it is a correctness guide, not an exclusive implementation contract.
Alternative formulations, reduction groupings, operation associations, and
fusion are allowed when they preserve the operator semantics and meet the
required output accuracy. Use the detailed recipe to diagnose numerical
mismatches rather than assuming every intermediate must be reproduced exactly.

Kernel decomposition, physical tile sizes, thread layouts, storage,
recomputation, and pipelining are implementation choices. A completed numerical
intermediate expresses a dependency; it does not require a separate launch,
memory write, or whole-block synchronization. Independent work may overlap,
and values may remain in registers or other suitable storage across operations.
The logical groups and loop orders describe the reference arithmetic, not a
required execution schedule.

Use symbolic batch size `B`, sequence length `T`, shared Q/K head count `H`,
and value/state head count `HV`. The key and value widths are `K=V=128`;
the saved inverse and cumulative gates use chunk size `C=64`.

## Direct operator contract

```text
dq, dk, dv, db, dg, dh0 = run(
    q, k, v, g, beta, A, initial_state, do, dht, scale
)
```

| Name | Shape | Dtype | Meaning |
|---|---:|---|---|
| `q`, `k` | `[B,T,H,K]` | BF16 | Saved query and key inputs |
| `v` | `[B,T,HV,V]` | BF16 | Saved value input |
| `g` | `[B,T,HV]` | FP32 | Saved chunk-local cumulative base-2 log decay |
| `beta` | `[B,T,HV]` | BF16 | Saved activated update gate |
| `A` | `[B,T,HV,C]` | BF16 | Saved row-wise WY inverse |
| `initial_state` | `[B,HV,V,K]` | FP32 | Initial state, V-first and K-last |
| `do` | `[B,T,HV,V]` | BF16 | Upstream output gradient |
| `dht` | `[B,HV,V,K]` | FP32 | Upstream final-state gradient |
| `scale` | scalar | FP32 | Fixed output scale |
| `dq`, `dk` | `[B,T,H,K]` | BF16 | Query and key gradients after head reduction |
| `dv` | `[B,T,HV,V]` | BF16 | Value gradient |
| `db` | `[B,T,HV]` | BF16 | Gradient of activated `beta` |
| `dg` | `[B,T,HV]` | FP32 | Gradient before the natural-log cumulative sum |
| `dh0` | `[B,HV,V,K]` | FP32 | Initial-state gradient |

Consume saved `g` and `A` directly. Do not reconstruct the cumulative sum or
inverse, normalize Q/K, or differentiate through a raw-gate activation. Require
positive head counts, `T % C == 0`, and `HV % H == 0`. For value head `hv`, gather Q/K head
`hv // (HV/H)`. Reduce only `dq` and `dk` to `H` heads; keep `dv`, `db`, and
`dg` at `HV` heads.

## Layout, gates, and numerical notation

Suppress batch and head indices for one chunk. Use

```text
Q, Kc : [C,K]       # BF16 Q/K rows gathered for this value head
Vv, dO : [C,V]     # BF16 value and upstream output-gradient rows
beta  : [C]        # BF16
g     : [C]        # FP32
A     : [C,C]      # BF16 logical chunk view of the saved inverse
S     : [K,V]      # internal live state
```

Transpose the last two logical axes when loading or storing external
`initial_state`, `dht`, and `dh0 [V,K]`. Token rows are `i,j in [0,C)`, key
coordinates are `k`, and value coordinates are `v`. Live state scans carry
across chunks within a sequence; gate cumulative sums reset at every chunk.

Let `lambda_i` be the natural-log decay increment before the forward
chunk-local cumulative sum:

```text
ell_i     = sum_{j<=i} lambda_j = ln(2) * g_i
gamma_i   = E(g_i)
gamma_end = E(g_{C-1})
eta_i     = E(g_{C-1} - g_i)
R_ij      = E(g_i - g_j)
```

The reference arithmetic uses the following notation:

```text
F(x_bf)      exact promotion of stored BF16 x_bf to FP32
RBF(x)       one round-to-nearest-even conversion of FP32 x to BF16
E(x)         approximate base-2 exponential on FP32 x,
             flushing subnormal inputs and results to zero
BFMM(X,Y)    BF16 tensor-core operands, FP32 accumulation and result
BFMMA(X,Y,Acc) BF16 tensor-core product accumulated into existing FP32 Acc
TFMM(X,Y)    raw FP32 operands entering tensor cores at TF32 input precision
MUL32(a,b)  FP32 multiply, one round-to-nearest-even result
ADD32(a,b)  FP32 add, one round-to-nearest-even result
FMA32(a,b,c) scalar FP32 fused multiply-add, one round-to-nearest-even result
lower(X)     inclusive lower triangle, j <= i; zero elsewhere
strict(X)    strict lower triangle, j < i; zero elsewhere
```

In this recipe, `TFMM` uses TF32 input precision without explicit TF32
pre-rounding. Accumulation is FP32. `BFMMA`
continues an existing accumulator, whereas standalone `BFMM` starts from zero.
`FMA32` combines completed FP32 operands with one rounding; `MUL32` followed by
`ADD32` rounds twice. Folding scalar arithmetic into a matrix accumulator or
adding separately rounded products can therefore change the result.

The recipe promotes scalar BF16 operands to FP32 and rounds at named `RBF` boundaries. Suffixes `_bf` and `_fp` describe
these reference values, not mandatory storage. Retain these distinctions when
using the recipe directly. An optimized alternative may change internal
precision, grouping, or fusion, provided all outputs still meet the required
accuracy; greater intermediate precision alone does not guarantee agreement.

## Semantic chunk algebra

The saved inverse corresponds to

```text
G = Kc @ Kc.T
L = strict(Diag(beta) @ (G * R))
A = inverse(I + L)
```

Using the supplied `A`, the semantic forward quantities are

```text
Bv    = Diag(beta) @ Vv
Bk    = Diag(beta * gamma) @ Kc
U     = A @ Bv
W     = A @ Bk
X     = U - W @ S
N     = Q @ Kc.T
P_sem = lower(N * R)
O     = scale * (Diag(gamma) @ Q @ S + P_sem @ X)
Z     = Diag(eta) @ X
Snext = gamma_end * S + Kc.T @ Z
```

These equations explain the dependencies; the reference recipe below shows
rounding and folds `scale` into selected BF16 intermediates. Use the supplied
`A`.
The returned gate gradient is with respect to `lambda`, not the saved numeric
base-2 tensor `g`:

```text
dg_j = sum_{i=j..C-1} bar_ell_i
```

Apply this inclusive reverse cumulative sum independently within each chunk,
with no extra `ln(2)` or `1/ln(2)` factor.

## Reference arithmetic recipe

### WY operands and forward state scan

For every chunk row:

```text
Bv[i,v]       = RBF(F(beta[i]) * F(Vv[i,v]))
key_beta[i,k] = RBF(F(beta[i]) * F(Kc[i,k]))
Bk_fwd[i,k]   = RBF(F(key_beta[i,k]) * gamma_i)
beta_gamma_i = F(beta[i]) * gamma_i
Kbg_fp[i,k]   = F(Kc[i,k]) * beta_gamma_i
Kbg_bwd[i,k]  = RBF(Kbg_fp[i,k])
U             = RBF(BFMM(A, Bv))
W             = RBF(BFMM(A, Bk_fwd))
```

The reference forward key path has two BF16 rounds, while `Kbg_bwd` has
one final round for the saved-WY `barA` contraction. These values need not
agree. The recipe uses unrounded `Kbg_fp` for the WY gate gradient and
`key_beta` for the second inverse key path; merging these paths can affect
output accuracy.

The reference starts with FP32 `S_live = transpose(initial_state)` and visits
chunks in increasing order:

```text
S_snap_bf = RBF(S_live)
WS_fp     = BFMM(W, S_snap_bf)
x_fp      = F(U) - WS_fp
X_bf      = RBF(x_fp)
Z_bf      = RBF(eta[:,None] * x_fp)
KS_fp     = BFMM(Kc.T, Z_bf)
S_live    = FMA32(gamma_end, S_live, KS_fp)
```

Backward uses the values represented by `S_snap_bf` and `X_bf`, which may
be retained or recomputed. In the recipe, `Z_bf` uses unrounded `x_fp`, and
the live state stays FP32 between chunks. `KS_fp` completes from a zero
accumulator before the state FMA. Starting that matrix product from
`gamma_end * S_live` has different rounding; assess such changes by their
output accuracy. These dependencies do not prescribe storage or synchronization.

### Output interaction and reverse state scan

For every chunk:

```text
N_fp         = BFMM(Q, Kc.T)
P_bf         = RBF(lower(scale * N_fp * R))
barX_local_bf = RBF(BFMM(P_bf.T, dO))
```

The recipe applies the inclusive mask before rounding `P_bf`, which already
contains `scale`; applying another scale to `barX_local_bf` would change the
operator.

The reference starts with FP32 `dS_live = transpose(dht)` and visits chunks in
decreasing order:

```text
dS_snap_bf  = RBF(dS_live)
barZ_fp     = BFMM(Kc, dS_snap_bf)
barX_fp     = FMA32(eta[:,None], barZ_fp, F(barX_local_bf))
barX_bf     = RBF(barX_fp)

state_path_fp  = BFMM(W.T, barX_bf)
output_path_fp = TFMM((F(Q) * gamma[:,None]).T, F(dO))
add_fp         = FMA32(scale, output_path_fp, -state_path_fp)
dS_live        = FMA32(gamma_end, dS_live, add_fp)      # H100 reference
# B200 reference: dS_live = ADD32(MUL32(gamma_end, dS_live), add_fp)
```

The reference forms `add_fp` from completed contractions with one FP32 FMA,
and keeps `Q*gamma` in FP32 until the TF32 operand boundary; `barX_fp` uses
one FP32 FMA. The reverse live-state update depends on the reference's code
generation: the H100 reference contracts it into one FP32 FMA, while the B200
reference rounds the decayed state with `MUL32` before `ADD32`. Either form is
acceptable on either GPU when all outputs meet the required accuracy; the
distinction is useful when diagnosing state-gradient mismatches. After chunk index `0`, the resulting `transpose(dS_live)` is the
FP32 output `dh0`.

### Direct state/output gradients

```text
barP_fp = BFMM(dO, X_bf.T)
barN_bf = RBF(lower(barP_fp * R) * scale)

dq_main_fp = BFMM(dO, S_snap_bf.T)
dq_main_fp[i,:] = (dq_main_fp[i,:] * gamma_i) * scale
dk_main_fp = BFMM(X_bf, dS_snap_bf.T)
dk_main_fp[i,:] *= eta_i
dk_state_fp = value of dk_main_fp at this point
Deta_i = sum_k(dk_state_fp[i,k] * F(Kc[i,k]))

dq_main_fp = BFMMA(barN_bf, Kc, dq_main_fp)
dk_main_fp = BFMMA(barN_bf.T, Q, dk_main_fp)

dg0_i = sum_k(dq_main_fp[i,k] * F(Q[i,k]))
        - sum_k(dk_main_fp[i,k] * F(Kc[i,k]))
eta_end_fp = sum_{i,k}(dk_state_fp[i,k] * F(Kc[i,k]))
state_dot_fp = sum_{k,v}(F(S_snap_bf[k,v]) * F(dS_snap_bf[k,v]))
end_fp = FMA32(gamma_end, state_dot_fp, eta_end_fp)
dg0_{C-1} += end_fp
```

In the recipe, `barN_bf` is the causal score-gradient boundary and the causal
products continue the FP32 Q/K accumulators. `dg0` uses the main gradients
before saved-WY contributions or output casts. Row dots and the endpoint
reduction use FP32 accumulation. The reference endpoint reduces the complete
`[C,K]` product; summing separately rounded `Deta` rows can give a different
result. Alternative reductions remain subject to output accuracy.

The `-k*dk_state` term already contributes `-Deta_i`. Restore `sum_i(Deta_i)`
at the final row because every `eta_i` depends on the chunk-end gate. The
recipe keeps both endpoint terms and `dg0` in FP32 before the suffix scan.

### Chunk-end state dot

`state_dot_fp` contracts the BF16 chunk-entry state snapshot with the BF16
chunk-exit adjoint snapshot over all `[K,V]` elements and returns one FP32
scalar per chunk and value head. Promote both operands to FP32 before
multiplying; every BF16-by-BF16 product is then exact in FP32, and products and
partial sums are accumulated in FP32. The reduction order, tiling over `K` or
`V`, and FMA contraction are not numerically significant here. Do not round
the products, partial sums, or the result to BF16, and do not substitute the
FP32 live states for the BF16 snapshots.

Add the `gamma_end`-scaled contribution once per chunk/head, even if the Q/K
gradient computation is split into key tiles.

### WY right-hand-side VJP

```text
barW_bf  = -RBF(BFMM(barX_bf, S_snap_bf.T))
barBv_fp = BFMM(A.T, barX_bf)
barBk_fp = BFMM(A.T, barW_bf)

dv_bf[i,v] = RBF(F(beta[i]) * barBv_fp[i,v])
dk_rhs_fp[i,k] = barBk_fp[i,k] * beta_gamma_i
dk_rhs_bf[i,k] = RBF(dk_rhs_fp[i,k])

dbeta_rhs_i = 0
dg_rhs_i = 0
for each logical 64-coordinate key group kt in increasing order:
    dbeta_rhs_i += sum_{k in kt}((barBk_fp[i,k] * F(Kc[i,k])) * gamma_i)
    dg_rhs_i += sum_{k in kt}(barBk_fp[i,k] * Kbg_fp[i,k])
for each logical 64-coordinate value group vt in increasing order:
    dbeta_rhs_i += sum_{v in vt}(barBv_fp[i,v] * F(Vv[i,v]))
```

The recipe negates `barW` after its BF16 cast and keeps `barBv` and `barBk`
FP32. Its 64-coordinate groups describe the reference partials, with key
partials added before value partials; they do not fix physical tiles or rule
out other accurate reductions. The direct WY gate term uses unrounded
`Kbg_fp` and already represents an adjoint of the cumulative natural-log
decay, so an extra gamma factor would change the operator. `dk_rhs_bf`
feeds the later inverse VJP.

### Saved-inverse VJP

```text
barA_fp = 0
for each logical 64-coordinate key group kt in increasing order:
    barA_fp = BFMMA(barW_bf[:,kt], Kbg_bwd[:,kt].T, barA_fp)
for each logical 64-coordinate value group vt in increasing order:
    barA_fp = BFMMA(barX_bf[:,vt], Bv[:,vt].T, barA_fp)
barA_lower_bf = RBF(strict(barA_fp))
T_bf          = RBF(BFMM(barA_lower_bf, A.T))
Ltmp_fp       = BFMM(A.T, T_bf)
C_L_fp        = strict(-Ltmp_fp)
D_bf          = RBF(strict(C_L_fp * R))

G_fp      = BFMM(Kc, Kc.T)
first_fp  = BFMM(D_bf, Kc)
second_fp = BFMM(D_bf.T, key_beta)

dbeta_fp = dbeta_rhs
for each logical 64-coordinate key group kt in increasing order:
    dbeta_fp[i] += sum_{k in kt}(first_fp[i,k] * F(Kc[i,k]))
dk_inv_fp[i,k] = FMA32(F(beta[i]), first_fp[i,k], second_fp[i,k])
dk_wy_fp[i,k]  = dk_inv_fp[i,k] + F(dk_rhs_bf[i,k])
rbar[i,j] = F(D_bf[i,j]) * (G_fp[i,j] * F(beta[i]))
```

The reference accumulates key products followed by value products into one
FP32 `barA` accumulator. `G_fp` also continues one accumulator across key
groups. Independently rounded full products followed by addition can differ.
The recipe rounds strict-lower `barA` and the first inverse product to BF16;
the second inverse product stays FP32. Its decay ratio precedes the `D_bf`
rounding, and `beta` enters at the displayed later operations.

The reference adds inverse key-group partials directly to `dbeta`, and forms
`dk_wy_fp` as `FMA32(F(beta), first_fp, second_fp) + F(dk_rhs_bf)` from completed
products. Accumulating `beta * first_fp` inside `second_fp`, or combining
main/WY branches before their output reductions, changes rounding. These are
numerical considerations when selecting an alternative, not a requirement to
use separate buffers, launches, or synchronization for each expression.

### Final accumulation and outputs

The direct `gamma` and `eta` paths are already in `dg0`. For each row, form
the remaining saved-WY decay contribution in FP32:

```text
row_i = sum_j(rbar[i,j])
col_i = sum_j(rbar[j,i])
re_i = row_i - col_i
dg2_i = dg_rhs_i + re_i
dg_cum_fp[i] = dg0_i + dg2_i
```

`rbar` already contains `R` through `D_bf`; do not multiply by the ratio again.
The reference reduces rows and columns separately in FP32 before subtracting;
an interleaved add/subtract reduction has different rounding and needs to meet
the same output accuracy requirements.
Any separate `etabar` accumulator stays zero, because adding that path again
would double-count the `eta` terms in `dg0`.

Perform one chunk-local FP32 inclusive reverse cumulative sum:

```text
dg_fp = chunk_local_reverse_cumsum_fp32(dg_cum_fp, C)
```

The sum resets at each chunk boundary and introduces no logarithm-base factor.
Store `dg_fp` directly as FP32 `dg`. Let `r=HV/H`; the value heads belonging
to Q/K head `h` are `hv=h*r .. (h+1)*r-1`. Define the elementwise reduction

```text
head_reduce_bf16(x) = RBF(sum_fp32_{hv=h*r..(h+1)*r-1}(F(RBF(x[...,hv,:]))))

dq[...,h,:] = head_reduce_bf16(dq_main_fp)
dk_main_head_bf = head_reduce_bf16(dk_main_fp)
dk_wy_head_bf   = head_reduce_bf16(dk_wy_fp)
dk[...,h,:] = RBF(F(dk_main_head_bf) + F(dk_wy_head_bf))

dv[...,hv,:]    = dv_bf
db[...,hv]      = RBF(dbeta_fp)
dg[...,hv]      = dg_fp
dh0[...,hv,:,:] = transpose(dS_live)

return dq, dk, dv, db, dg, dh0
```

The reference rounds each value-head Q/K contribution to BF16, reduces
main/WY branches independently to BF16, and adds the two reduced `dk` values
with a final BF16 rounding. This explains accuracy differences that can arise
from fusing the reductions; it does not require intermediate storage or
exclude another formulation that meets the output accuracy requirements.
Output dtypes remain BF16 for `dq`, `dk`, `dv`, and `db`, and FP32 for `dg`
and `dh0`.


