#!/usr/bin/env python3
"""Multiturn kernel generation using mini-swe-agent infrastructure.

The model outputs CUDA kernel code directly each turn (no bash commands).
Docker container handles compilation, the profiling service handles evaluation.
Uses DefaultAgent + DockerEnvironment for trajectory logging/inspection compatibility.

System prompt: loads tvm_ffi_usage.txt + embedded PTX instruction tables.

Usage:
    python run_v1.py --definition gemm_n6144_k4096 --model gemini-3.1-pro-preview \
        --test-path ../mini_swe_agent_docker/envs/test_profile_cuda_gemm_n6144_k4096.py \
        --log-path trajectory.json --max-turns 5 --target-speedup 1.5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SCRIPT_DIR, SYSTEM_INSTRUCTIONS, run_main


def build_system_prompt(gpu_arch: str = "hopper") -> str:
    """Build system prompt from instructions + tvm_ffi_usage.txt + PTX tables.

    Wraps in {% raw %}...{% endraw %} because base_prompt contains PTX inline
    assembly with {%...} syntax that conflicts with Jinja2 template rendering.
    """
    base_prompt_path = SCRIPT_DIR / f"tvm_ffi_usage.txt"
    base_prompt = base_prompt_path.read_text() if base_prompt_path.exists() else ""
    if gpu_arch == "hopper":
        base_prompt += "\n\nYou are targeting Hopper GPU architecture. Remember to use SM90+ features for best performance.\n"
    elif gpu_arch == "blackwell":
        base_prompt += "\n\nYou are targeting Blackwell GPU architecture. Remember to use SM100+ features for best performance.\n"
    else:
        raise ValueError(f"Unsupported GPU architecture: {gpu_arch}")
    base_prompt += r"""
## 1. Tensor Core Compute

### 1.1 Ampere (SM80) — MMA

| PTX Instruction | Description |
|---|---|
| `mma.sync.aligned.{shape}.{layout}.{satfinite}.{dtype}.{atype}.{btype}.{ctype}` | Warp-level matrix multiply-accumulate |
| `mma.sp.sync.aligned.*` | Sparse MMA (structured sparsity) |
| `ldmatrix.sync.aligned.m8n8.x{N}.{trans}.shared.{b16\|b8}` | Load matrix fragment from shared memory |
| `stmatrix.sync.aligned.m8n8.x{N}.{trans}.shared.b16` | Store matrix fragment to shared memory |

### 1.2 Hopper (SM90) — WGMMA

| PTX Instruction | Description |
|---|---|
| `wgmma.fence.sync.aligned` | Fence before WGMMA input reads |
| `wgmma.commit_group.sync.aligned` | Commit a group of WGMMA operations |
| `wgmma.wait_group.sync.aligned N` | Wait until at most N WGMMA groups pending |
| `wgmma.mma_async.sync.aligned.m{M}n{N}k{K}.{otype}.{itype}.{itype}` (SS variant) | Async warpgroup MMA, both operands in shared memory |
| `wgmma.mma_async.sync.aligned.m{M}n{N}k{K}.{otype}.{itype}.{itype}` (RS variant) | Async warpgroup MMA, A in registers, B in shared memory |

### 1.3 Blackwell (SM100) — tcgen05

| PTX Instruction | Description |
|---|---|
| `tcgen05.alloc.cta_group::{N}.sync.aligned.shared::cta.b32` | Allocate tensor memory for a CTA group |
| `tcgen05.dealloc.cta_group::{N}.sync.aligned.b32` | Deallocate tensor memory |
| `tcgen05.relinquish_alloc_permit.cta_group::{N}.sync.aligned` | Relinquish allocation permit (for non-MMA CTAs) |
| `tcgen05.fence::before_thread_sync` | Fence before thread-level sync touching tmem |
| `tcgen05.fence::after_thread_sync` | Fence after thread-level sync touching tmem |
| `tcgen05.ld.sync.aligned.{shape}.x{N}.b32` | Load from tensor memory (with optional `.pack::16b`) |
| `tcgen05.st.sync.aligned.{shape}.x{N}.b32` | Store to tensor memory (with optional `.unpack::16b`) |
| `tcgen05.wait::ld.sync.aligned` | Wait for tensor memory loads to complete |
| `tcgen05.wait::st.sync.aligned` | Wait for tensor memory stores to complete |
| `tcgen05.mma.cta_group::{N}.kind::{kind}` | Tensor core MMA |
| `tcgen05.mma.sp.cta_group::{N}.kind::{kind}` | Sparse tensor core MMA |
| `tcgen05.mma.cta_group::{N}.kind::{kind}.block_scale.scale_vec::{V}X` | Block-scaled MMA (e.g. nvfp4) |
| `tcgen05.mma.sp.cta_group::{N}.kind::{kind}.block_scale.scale_vec::{V}X` | Sparse block-scaled MMA |
| `tcgen05.commit.cta_group::{N}.mbarrier::arrive::one.shared::cluster.b64` | Commit MMA group with mbarrier arrive (with optional `.multicast::cluster`) |
| `tcgen05.cp.cta_group::{N}.{shape}` | Copy within tensor memory (with optional `.multicast`) |
| `tcgen05.shift.cta_group::{N}.down` | Shift tensor memory contents down |

---

## 2. Asynchronous Copy

### 2.1 cp.async (Ampere, SM80)

| PTX Instruction | Description |
|---|---|
| `cp.async.{ca\|cg}.shared.global` | Async copy global→shared (4/8/16B), with optional L2 hints |
| `cp.async.commit_group` | Commit a group of cp.async operations |
| `cp.async.wait_group N` | Wait until at most N cp.async groups pending |

### 2.2 TMA — cp.async.bulk.tensor (Hopper/Blackwell, SM90+)

| PTX Instruction | Description |
|---|---|
| `cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes` | TMA global→shared (unicast) |
| `cp.async.bulk.tensor.{dim}d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster` | TMA global→shared (multicast to cluster) |
| `cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group` | TMA shared→global (store) |
| `cp.async.bulk.prefetch.tensor.{dim}d.L2.global.tile` | TMA prefetch into L2 |
| `cp.reduce.async.bulk.tensor.{dim}d.global.shared::cta.{red_op}.tile.bulk_group` | TMA shared→global with reduction (add/min/max) |
| `cp.async.bulk.commit_group` | Commit a group of TMA operations |
| `cp.async.bulk.wait_group N` | Wait until at most N TMA groups pending (with optional `.read`) |

Both TMA unicast and multicast support `.cta_group::{N}` qualifier (Blackwell) and `.L2::cache_hint`.

### 2.3 C++ Legacy Path

| PTX Instruction | Description |
|---|---|
| `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` | Bulk async copy (C++ emitted) |
| `cp.async.mbarrier.arrive.shared.b64` | Mbarrier arrive after cp.async (C++ emitted) |

---

## 3. Synchronization and Barriers

### 3.1 Basic Barriers

| PTX Instruction | Description |
|---|---|
| `bar.arrive %id, %count` | Named barrier arrive |
| `bar.sync %id, %count` | Named barrier sync (used for warpgroup sync with count=128) |

### 3.2 Memory Fences

| PTX Instruction | Description |
|---|---|
| `fence.{sc\|acq_rel\|acquire\|release}.{cta\|cluster\|gpu\|sys}` | Memory fence with specified semantics and scope |
| `fence.proxy.async` | Async proxy fence (no space qualifier) |
| `fence.proxy.async.{global\|shared::cta\|shared::cluster}` | Async proxy fence for specific address space |
| `fence.mbarrier_init.release.cluster` | Fence for mbarrier initialization (cluster scope) |

### 3.3 Cluster Barriers (SM90+)

| PTX Instruction | Description |
|---|---|
| `barrier.cluster.arrive.{sem}.aligned` | Cluster barrier arrive |
| `barrier.cluster.wait.{acquire}.aligned` | Cluster barrier wait |

### 3.4 MBarrier (SM90+)

| PTX Instruction | Description |
|---|---|
| `mbarrier.init.shared.b64 [addr], count` | Initialize mbarrier with expected arrival count |
| `mbarrier.arrive.shared.b64` | Local mbarrier arrive |
| `mbarrier.arrive.shared::cluster.b64` | Remote (cross-CTA) mbarrier arrive |
| `mbarrier.arrive.expect_tx.shared.b64` | Local mbarrier arrive with expected TX byte count |
| `mbarrier.arrive.expect_tx.shared::cluster.b64` | Remote mbarrier arrive with expected TX byte count |
| `mbarrier.try_wait.parity.shared::cta.b64` | Try-wait on mbarrier parity (polling loop with `bra.uni`) |

Remote mbarrier operations use `mapa.shared::cluster.u32` to compute the remote CTA address.

---

## 4. Math Operations

### 4.1 Scalar Approximations

| PTX Instruction | Description |
|---|---|
| `ex2.approx.ftz.f32` | Fast approximate 2^x |
| `rcp.approx.ftz.f32` | Fast approximate reciprocal |
| `rsqrt.approx.ftz.f32` | Fast approximate reciprocal square root |
| `lg2.approx.ftz.f32` | Fast approximate log2 |

### 4.2 Three-Input Min/Max (SM100a+)

| PTX Instruction | Description |
|---|---|
| `max.f32 %0, %1, %2, %3` | Three-input maximum |
| `min.f32 %0, %1, %2, %3` | Three-input minimum |

### 4.3 Packed f32x2 Operations (SM100a+)

| PTX Instruction | Description |
|---|---|
| `add.{rz}.ftz.f32x2` | Packed 2×f32 addition |
| `sub.{rz}.ftz.f32x2` | Packed 2×f32 subtraction |
| `mul.{rz}.ftz.f32x2` | Packed 2×f32 multiplication |
| `fma.{rz}.ftz.f32x2` | Packed 2×f32 fused multiply-add |

### 4.4 Integer Dot Product

| PTX Instruction | Description |
|---|---|
| `dp4a.u32.s32` | 4-element dot product (unsigned × signed) |
| `dp4a.s32.u32` | 4-element dot product (signed × unsigned) |

---

## 5. Memory Access and Atomics

### 5.1 Global Loads

| PTX Instruction | Description |
|---|---|
| `ld.global.nc.f32` | Predicated non-coherent global load (L2 cached) |
| `ld.global.acquire.gpu.{b32\|b64}` | Global load with acquire semantics (SM70+) |
| `ld.global.cg.{b32\|b64}` | Global load with cache-global hint (pre-SM70 fallback) |

### 5.2 Global Stores

| PTX Instruction | Description |
|---|---|
| `st.global.release.sys.b32` | Global store with release.sys semantics |
| `st.global.release.gpu.b32` | Global store with release.gpu semantics |

### 5.3 Shared Memory

| PTX Instruction | Description |
|---|---|
| `st.shared.b32` | Shared memory store |

### 5.4 Atomic Operations

| PTX Instruction | Description |
|---|---|
| `atom.release.gpu.global.add.u32` | Atomic add (unsigned) with release semantics |
| `atom.release.gpu.global.add.s32` | Atomic add (signed) with release semantics |

### 5.5 Global Reductions (SM100)

| PTX Instruction | Description |
|---|---|
| `red.global.v4.f16.add.noftz` | Vectorized 4×f16 global reduction (add) |
| `red.global.v4.f32.add` | Vectorized 4×f32 global reduction (add) |

### 5.6 Multi-Memory / NVSwitch (Multi-GPU)

| PTX Instruction | Description |
|---|---|
| `multimem.ld_reduce.acquire.sys.global.add.acc::f32.v8.f16` | NVSwitch multi-memory load-reduce (8×f16 → f32 accumulation) |
| `multimem.st.release.sys.global.v4.f32` | NVSwitch multi-memory store release (4×f32) |

---

## 6. Register, Address, and Control Flow

### 6.1 Register Management

| PTX Instruction | Description |
|---|---|
| `setmaxnreg.{inc\|dec}.sync.aligned.u32 N` | Dynamically increase/decrease register allocation |
| `mov.u32 %0, %{special_reg}` | Fetch special register (e.g. `%tid.x`, `%ctaid.x`) |
| `mov.u32 %0, %globaltimer_lo` | Read low 32 bits of global timer |

### 6.2 Address Space Conversion

| PTX Instruction | Description |
|---|---|
| `cvta.to.shared.u64` | Convert generic pointer to shared memory address |
| `cvt.u32.u64` | Truncate 64-bit to 32-bit (for shared memory pointer) |
| `mapa.u64` | Map address to another CTA's address space (64-bit) |
| `mapa.shared::cluster.u32` | Map shared address to remote CTA in cluster (32-bit) |

### 6.3 Warp-Level Control

| PTX Instruction | Description |
|---|---|
| `elect.sync %%rx\|%%px, mask` | Elect one thread from active mask |
| `setp.ne.b32` / `setp.eq.u32` | Set predicate (used for conditional MMA/barrier ops) |
| `bra.uni` | Unconditional uniform branch (mbarrier wait loop) |
| `trap` | Trigger trap (assertion failure) |

---

## 7. Third-Party

| PTX Instruction | Description |
|---|---|
| `st.global.release.sys.b32` | Store with release.sys |
| `ld.global.acquire.sys.b32` | Load with acquire.sys |
| `st.global.volatile.b32` | Volatile store (pre-SM70 fallback) |
| `ld.global.volatile.b32` | Volatile load (pre-SM70 fallback) |

---

## Architecture Coverage Summary

| Architecture | Key PTX Instruction Families |
|---|---|
| **Ampere (SM80)** | `mma.sync`, `ldmatrix`, `stmatrix`, `cp.async`, `dp4a` |
| **Hopper (SM90)** | All of SM80 + `wgmma.*`, `cp.async.bulk.tensor.*` (TMA), `mbarrier.*`, `barrier.cluster.*`, `fence.*`, `setmaxnreg`, `elect.sync`, `mapa.*` |
| **Blackwell (SM100)** | All of SM90 + `tcgen05.*` (alloc/dealloc/ld/st/mma/commit/cp/shift), 3-input `max`/`min`, packed `f32x2` ops, `red.global.v4.*`, `multimem.*` |
"""
    return "{% raw %}" + SYSTEM_INSTRUCTIONS + base_prompt + "{% endraw %}"


if __name__ == "__main__":
    run_main(build_system_prompt)
