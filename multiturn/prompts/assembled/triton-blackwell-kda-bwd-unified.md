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




# KDA backward operator and numerical reference

Implement the direct KDA backward operator with the saved-input semantics,
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

The equations are parameterized by batch size `B`, sequence length `T`, heads
`H`, widths `K,V`, and chunk size `C`. The reference workload uses
`B=4, T=4096, H=96, K=V=128, C=64`, with V-first state, `safe_gate=True`,
recomputation enabled, no packed sequences or context parallelism, and
`scale=1/sqrt(K)`.

## Direct operator contract

```text
dq, dk, dv, db, dg, dh0 = run(
    q, k, v, g, beta, Aqk, Akk, initial_state, do, dht, scale
)
```

| Name | Shape | Dtype | Meaning |
|---|---:|---|---|
| `q`, `k` | `[B,T,H,K]` | BF16 | Saved, already-normalized query and key |
| `v` | `[B,T,H,V]` | BF16 | Saved value |
| `g` | `[B,T,H,K]` | FP32 | Saved chunk-local cumulative base-2 log decay |
| `beta` | `[B,T,H]` | BF16 | Saved activated update coefficient |
| `Aqk` | `[B,T,H,C]` | BF16 | Saved scaled causal Q/K factor |
| `Akk` | `[B,T,H,C]` | BF16 | Saved row-wise WY inverse factor |
| `initial_state` | `[B,H,V,K]` | FP32 | Initial state, V-first and K-last |
| `do` | `[B,T,H,V]` | BF16 | Upstream output gradient |
| `dht` | `[B,H,V,K]` | FP32 | Upstream final-state gradient |
| `scale` | scalar | FP32 | Output scale |
| `dq`, `dk` | `[B,T,H,K]` | FP32 | Raw direct query and key gradients |
| `dv` | `[B,T,H,V]` | BF16 | Value gradient |
| `db` | `[B,T,H]` | FP32 | Gradient of activated `beta` |
| `dg` | `[B,T,H,K]` | FP32 | Gradient before the natural-log cumulative sum |
| `dh0` | `[B,H,V,K]` | FP32 | Initial-state gradient |

Consume saved `g`, `Aqk`, and `Akk` directly. Do not reconstruct them or apply
Q/K normalization, beta sigmoid, raw-gate activation, or parameter VJPs. This
direct contract keeps `dq`, `dk`, `db`, and `dg` in FP32; a public autograd
wrapper may cast them and therefore has a different output boundary.

The active workload has one value head per Q/K head, so there is no grouped-head
reduction. Require `T % C == 0` for this dense direct contract.

## Layout, gates, and numerical notation

Suppress batch and head indices for one chunk. Use

```text
Q, Kd : [C,K]
Vv    : [C,V]
S     : [V,K]       # external and live state, V-first / K-last
Aqk, Akk : [C,C]    # logical chunk views of the saved tensors
```

Token rows are `i,j in [0,C)`, key coordinates are `d`, and value coordinates
are `r`. All scans and cumulative sums reset at logical-sequence boundaries;
chunks must not cross such boundaries.

Let `ell_i[d]` be the natural-log decay increment before the forward chunk-local
cumulative sum. The saved tensor and its derived factors are

```text
g_i[d]     = (sum_{s<=i} ell_s[d]) / ln(2)
gamma_i[d] = E(g_i[d])
eta_i[d]   = E(g_{C-1}[d] - g_i[d])
R_ij[d]    = E(g_i[d] - g_j[d])
```

where the decay is independently defined for every key coordinate. The
reference arithmetic uses the following notation:

```text
F(x_bf)       exact promotion of BF16 x_bf to FP32
RBF(x)        one round-to-nearest-even conversion of FP32 x to BF16
E(x)          base-2 exponential on FP32 x
BFMM(X,Y)     BF16 tensor-core operands, FP32 accumulation and result
BFMMA(X,Y,Acc) BF16 tensor-core product accumulated into existing FP32 Acc
TFMM(X,Y)     FP32 operands entering tensor cores at TF32 input precision
lower(X)      inclusive lower triangle, j <= i
strict(X)     strict lower triangle, j < i
```

In this recipe, `TFMM` uses TF32 input precision. Reductions accumulate in
FP32. `BFMMA` continues an existing tensor-core accumulator, whereas standalone
`BFMM` starts from zero. A separately completed product followed by scalar
arithmetic, including scalar FP32 FMA, rounds differently from using the scalar
term as the tensor-core accumulator. Folding scalar arithmetic into a matrix
accumulator, or changing grouping across the displayed rounding boundaries, can
therefore change the result.

## Semantic chunk algebra

The token recurrence, with state `S [V,K]`, is

```text
alpha_t[d] = exp(ell_t[d])
Sbar_t     = S_{t-1} * alpha_t[None,:]
residual_t = beta_t * (v_t - Sbar_t @ k_t)
S_t        = Sbar_t + residual_t[:,None] * k_t[None,:]
o_t        = S_t @ (scale * q_t)
```

For the parallel chunk form define

```text
L_ij  = beta_i * sum_d(k_i[d] * k_j[d] * R_ij[d])  if j < i else 0
A     = inverse(I + L)
P0_ij = sum_d(q_i[d] * k_j[d] * R_ij[d])            if j <= i else 0

Kgamma = Kd * gamma
Keta   = Kd * eta
Qgamma = Q * gamma
Bv     = beta[:,None] * Vv
Bk     = beta[:,None] * Kgamma
U      = A @ Bv
W      = A @ Bk
X      = U - W @ S.T
O      = scale * (Qgamma @ S.T + P0 @ X)
Snext  = gamma_{C-1}[None,:] * S + X.T @ Keta
```

`Aqk` is a BF16 materialization of `lower(scale*P0)`. `Akk` is the supplied
BF16 materialization of the row-wise inverse `A`; its forward construction may
have used mixed-precision block solves, so it is not interchangeable with a
fresh exact inverse. These statements describe the saved inputs, not work to
repeat inside backward.

The returned `dg` is with respect to `ell`, not the stored numeric `g`. After
forming the cumulative adjoint, return

```text
dg_j[d] = sum_{i=j..C-1} bar_ell_i[d]
```

with no extra `ln(2)` or `1/ln(2)` factor.

## Reference arithmetic recipe

Suffixes `_bf` and `_fp` describe the precision of these reference values at
a dependency boundary, not mandatory storage. Retain these distinctions when
using the recipe directly. An optimized alternative may change internal
precision, grouping, or fusion, provided all outputs still meet the required
accuracy; greater intermediate precision alone does not guarantee agreement.

### WY operands and forward state scan

For every chunk row:

```text
Bv_bf[i,r]       = RBF(F(beta[i]) * F(Vv[i,r]))
key_beta_bf[i,d] = RBF(F(Kd[i,d]) * F(beta[i]))
Bk_bf[i,d]       = RBF(F(key_beta_bf[i,d]) * E(g[i,d]))
Qgamma_bf[i,d]   = RBF(F(Q[i,d]) * E(g[i,d]))
Keta_bf[i,d]     = RBF(F(Kd[i,d]) * E(g[C-1,d] - g[i,d]))

U_bf = RBF(BFMM(Akk, Bv_bf))
W_bf = RBF(BFMM(Akk, Bk_bf))
```

`Bk_bf` has two BF16 boundaries: the BF16-by-BF16 `K*beta` product rounds
before it is promoted for multiplication by the FP32 exponential, and the
result rounds again before the matrix product. `Qgamma_bf` and `Keta_bf`
are independently rounded BF16 operands that may be produced where consumed.
The safe-gate column path independently forms the same `key_beta_bf` value
before applying its anchored exponential.

Initialize `S_live_fp` from FP32 `initial_state`. For chunks in increasing
order:

```text
H_bf      = RBF(S_live_fp)                     # save chunk-entry state
pred_fp   = BFMM(W_bf, H_bf.T)
X_fp      = F(U_bf) - pred_fp
X_bf      = RBF(X_fp)                          # save corrected value
update_fp = BFMM(X_bf.T, Keta_bf)              # complete product from zero
S_live_fp = E(g[C-1])[None,:] * S_live_fp + update_fp
```

The update consumes BF16 `X_bf`; the live state remains FP32 between chunks.

The reference completes `update_fp` from a zero tensor-core accumulator and
then combines it with the live state; the scalar update may use FP32 FMA.
Using `gamma_end * S_live_fp` as the product's initial accumulator, as in
`BFMMA(X_bf.T, Keta_bf, gamma_end * S_live_fp)`, has different rounding.
A final `dg` failure can originate in the forward `H_bf` and `X_bf` snapshots;
passing `dv` and `dh0` does not establish that those snapshots match.

### Output interaction and reverse state scan

The VJP through the local output interaction is

```text
dP0_fp        = lower(scale * BFMM(do, X_bf.T))
barX_local_bf = RBF(BFMM(Aqk.T, do))
```

`dP0_fp` is the FP32 adjoint of unscaled `P0`. Saved `Aqk` already contains
`scale`, so do not apply another scale in the `barX_local` path.

Initialize `dS_live_fp` from FP32 `dht`. For chunks in decreasing order:

```text
dH_bf  = RBF(dS_live_fp)
barX_fp = F(barX_local_bf) + BFMM(Keta_bf, dH_bf.T)
barX_bf = RBF(barX_fp)

add_fp = scale * BFMM(do.T, Qgamma_bf) - BFMM(barX_bf.T, W_bf)
dS_live_fp = E(g[C-1])[None,:] * dS_live_fp + add_fp
```

`dH_bf` feeds both the corrected-value and later key-gradient paths. Keep the
live adjoint FP32; after processing the first sequence chunk, store it directly
as FP32 `dh0 [V,K]`.

### Direct state/output gradients

```text
dq_state_fp = BFMM(do, H_bf) * E(g) * scale
dk_state_fp = BFMM(X_bf, dH_bf) * E(g[C-1] - g)

dg_state_fp[i,d] = F(Q[i,d])  * dq_state_fp[i,d]
                   - F(Kd[i,d]) * dk_state_fp[i,d]

state_dot_fp[d] = sum_r(F(H_bf[r,d]) * F(dH_bf[r,d]))
end_k[d] = state_dot_fp[d] * E(g[C-1,d])
           + sum_i(F(Kd[i,d]) * dk_state_fp[i,d])
dg_state_fp[C-1,d] += end_k[d]
```

`state_dot_fp[d]` contracts the BF16 chunk-entry state snapshot with the BF16
chunk-exit adjoint snapshot over the value axis, giving one FP32 value per key
coordinate. Promote both operands to FP32 before multiplying; every
BF16-by-BF16 product is then exact in FP32, and products and partial sums are
accumulated in FP32. The reduction order, value tiling, and FMA contraction are
not numerically significant here. Do not round the products, partial sums, or
the result to BF16, and do not substitute the FP32 live states for the BF16
snapshots. The Q/K and `barW` matrix products over V continue their FP32
tensor-core accumulators across value groups.

### WY right-hand side and saved-inverse VJP

```text
barW_bf  = -RBF(BFMM(barX_bf, H_bf))
barBv_fp = BFMM(Akk.T, barX_bf)
barBk_fp = BFMM(Akk.T, barW_bf)

dv_bf[i,r] = RBF(F(beta[i]) * barBv_fp[i,r])
gamma_beta[i,d] = E(g[i,d]) * F(beta[i])
Kgamma_fp[i,d] = F(Kd[i,d]) * E(g[i,d])
dk_rhs_fp[i,d] = barBk_fp[i,d] * gamma_beta[i,d]

db_rhs_fp[i] = 0
for each value tile vt in increasing order:
    db_rhs_fp[i] += sum_{r in vt}(barBv_fp[i,r] * F(Vv[i,r]))
for each key tile kt in increasing order:
    db_rhs_fp[i] += sum_{d in kt}(barBk_fp[i,d] * Kgamma_fp[i,d])
dg_rhs_fp[i,d] = (Kgamma_fp[i,d] * barBk_fp[i,d]) * F(beta[i])
```

Keep `barBv`, `barBk`, `dk_rhs`, `db_rhs`, and `dg_rhs` in FP32. `dv_bf` is the
final BF16 value gradient. The value and key tiles above are FP32 partial
groups whose width depends on the reference's selected configuration; they do
not fix physical storage or transfer tiles, and regrouping these FP32 partials
is subject only to output accuracy. The intra-chunk pair reductions below use
logical groups of 32 key coordinates.
Preserve the unrounded `Kgamma_fp` in the scalar VJPs and the displayed factor
association.

For the inverse VJP, first form a separately rounded key operand without beta:

```text
Kgamma_bf = RBF(Kgamma_fp)

barA0_fp = 0
for each value tile vt in increasing order:
    barA0_fp = BFMMA(barX_bf[:,vt], Vv[:,vt].T, barA0_fp)
for each key tile kt in increasing order:
    barA0_fp = BFMMA(barW_bf[:,kt], Kgamma_bf[:,kt].T, barA0_fp)

D_bf[i,j] = RBF(barA0_fp[i,j] * F(beta[j]))  if j < i
              BF16(0)                         otherwise
tmp_bf    = RBF(BFMM(D_bf, Akk.T))
dL_fp     = strict(-BFMM(Akk.T, tmp_bf))
```

The value and key products share one FP32 tensor-core accumulator. Add all
value tiles first, then each key tile in order; do not form independent full
value/key products and add once.
Both `D_bf` and `tmp_bf` are checker-visible BF16 boundaries; do not retain both
inverse products in FP32.

### Safe-gate intra-chunk VJP

The pair dependencies below use `R_ij` as algebraic shorthand. Evaluate the
ratios using the anchored schedule below, with FP32 `dP0_fp` and `dL_fp`:

```text
dq_pair[i,d] = sum_{j<=i} dP0[i,j] * Kd[j,d] * R_ij[d]

dk_first_unscaled[i,d] = sum_{j<i} dL[i,j] * Kd[j,d] * R_ij[d]
dk_first[i,d] = dk_first_unscaled[i,d] * F(beta[i])

dk_second[j,d] = sum_{i>=j} dP0[i,j] * Q[i,d] * R_ij[d]
                 + sum_{i>j} dL[i,j] * F(key_beta_bf[i,d]) * R_ij[d]

db_inv_tile[kt,i] = sum_{d in kt}(dk_first_unscaled[i,d] * F(Kd[i,d]))

dg_pair[i,d] = F(Q[i,d]) * dq_pair[i,d]
               + (dk_first[i,d] - dk_second[i,d]) * F(Kd[i,d])
```

For the active safe-gate arithmetic, group the 64 rows into four logical `BC=16`
subchunks and the 128 key coordinates into four logical `BK=32` tiles. These
groups determine gate anchors and numerical partials, not thread-block or
memory-transfer shapes. Diagonal blocks use row 8 as their midpoint gate anchor;
cross-subchunk row paths use the first row of the destination block, and column
paths use the last row of the source block. Factor each exponential ratio around
its anchor. For example,

```text
# One anchored row-path block; a is its gate-anchor row.
Kanch_fp[j,d] = F(Kd[j,d]) * E(g[a,d] - g[j,d])

dq_block_fp = TFMM(dP0_block_fp, Kanch_fp)
dk_first_unscaled_block_fp = TFMM(dL_block_fp, Kanch_fp)

dq_block_fp[i,d] *= E(g[i,d] - g[a,d])
dk_first_unscaled_block_fp[i,d] *= E(g[i,d] - g[a,d])
```

The analogous column-path contractions also use `TFMM`; cross-subchunk loops
continue their existing FP32 tensor-core accumulators.

The anchored factors are algebraically equal to `R_ij[d]` but have different
rounding from a direct exponent difference. In both diagonal and cross-block
column paths, form `key_beta_bf = RBF(Kd * beta)` first, promote it, and then
multiply by the anchored exponential. Keep that scaled operand FP32 at the
TF32 matrix-product boundary; do not replace the intermediate BF16 product
with `F(Kd) * F(beta)` or cast the scaled operand back to BF16. Row paths
instead apply beta to `dk_first_unscaled` after the contractions.

```text
# Key-beta contribution to one column-path block.
key_beta_bf[i,d] = RBF(F(Kd[i,d]) * F(beta[i]))
key_col_fp[i,d] = F(key_beta_bf[i,d]) * E(g[i,d] - g[a,d])

dk_second_L_block_fp = TFMM(dL_block_fp.T, key_col_fp)
dk_second_L_block_fp[j,d] *= E(g[a,d] - g[j,d])
```

The FP32 matrix operands enter tensor cores at TF32 input precision. Pair
contributions remain FP32 when combined with the state/WY gradients; they
are not separately stored BF16 tensors.

Reduce `dk_first_unscaled * Kd` before multiplying the first key path by beta.
Compute `db_inv_tile` independently for each of the four key tiles, yielding four
FP32 partials. Preserve the FP32 reduction grouping of those partials and
then add `db_rhs_fp` once. Do not cast either term to BF16.

`dg_pair` names the algebraic contribution. Preserve the final expression
below when combining it with the existing state/WY gradient, since regrouping
the terms changes FP32 rounding and scalar FMA contraction.

### Final accumulation and outputs

```text
dq_fp     = dq_pair + dq_state_fp
dk_base_fp = dk_state_fp + dk_rhs_fp
dk_fp     = (dk_first + dk_base_fp) + dk_second
db_fp     = reduce_key_tile_partials_fp32(db_inv_tile) + db_rhs_fp
dg_base_fp = dg_state_fp + dg_rhs_fp
dg_cum_fp = F(Q) * dq_pair + ((dk_first - dk_second) * F(Kd) + dg_base_fp)
dg_fp     = chunk_local_reverse_cumsum_fp32(dg_cum_fp, C)

return dq_fp, dk_fp, dv_bf, db_fp, dg_fp, dh0_fp
```

The reverse cumulative sum is inclusive, independent for every key coordinate,
and resets at each chunk boundary. There is no head reduction or output
postprocessing. The output dtypes are FP32 for `dq`, `dk`, `db`, `dg`, and
`dh0`, and BF16 only for `dv`.


