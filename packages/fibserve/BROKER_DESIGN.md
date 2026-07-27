# GPU Residency Broker Design

Status: proposed design for GPU Server v2

## 1. Purpose

GPU Server v2 runs every submitted program in a fresh request-scoped process, while
consecutive requests may refer to exactly the same immutable blobs. Copying the same
tensor bytes from CPU memory to GPU memory for every request can dominate interactive
latency.

This document defines a per-device residency broker that keeps immutable blob contents
resident across requests without keeping request Python processes, modules, objects, or
mutable application state alive.

The central rule is:

> The broker owns persistent allocations. A request owns only a temporary capability and
> a process-local mapping to allocations named by its validated manifest.

The design extends the process, cache, isolation, and plugin sections of
[`NEW_SERVICE.md`](./NEW_SERVICE.md). It assumes that the scheduler can see the queued
request manifests and determine that request N and request N+1 refer to the same complete
SHA-256 blob identifiers.

## 2. Goals

- Avoid repeated CPU-to-GPU copies when consecutive jobs consume the same immutable
  tensor blobs.
- Preserve one-request-at-a-time benchmark execution on each GPU.
- Preserve fresh-process isolation for submitted Python code and native libraries.
- Keep the HTTP frontend free of CUDA, Metal, and other GPU runtime initialization.
- Make allocations accessible only to requests whose manifests name the corresponding
  blobs.
- Support NVIDIA discrete GPUs, Jetson, and Apple silicon through backend-specific
  sharing mechanisms behind one plugin interface.
- Recover conservatively after timeouts, process crashes, illegal GPU accesses, and
  device resets.
- Keep `/blobs` an immutable, SHA-256-verified CPU-byte store.

## 3. Non-goals

- This design does not provide isolation against a malicious caller in the first version.
  It preserves the trusted-caller isolation contract in `NEW_SERVICE.md`.
- It does not preserve mutable Python objects, imported modules, compiled function
  objects, arbitrary device pointers, or application sessions across requests.
- It does not infer tensor shape or dtype from arbitrary raw bytes.
- It does not overlap GPU copies for N+1 with the measured GPU work of N.
- It does not make DLPack an interprocess transport. DLPack is used only after a resource
  has been imported into the receiving process.
- It does not promise that all platforms can retain private device memory. A backend may
  use shared system memory or fall back to a per-request copy.

## 4. Isolation Model

Retaining physical bytes and physically erasing every request's bytes are mutually
exclusive requirements. The v2 contract therefore distinguishes physical allocation
lifetime from request accessibility.

The broker may retain an immutable physical allocation after a request completes. A
request may access only mappings granted by its own request lease. The following state
remains request-private:

- process and process group;
- Python interpreter and imported modules;
- current working directory and uploaded file layout;
- standard output and standard error capture;
- compiled application objects and JIT cache directories;
- mutable outputs, scratch buffers, and temporary device allocations;
- process-local device virtual addresses and tensor wrapper objects;
- diagnostic tool state.

The following state may survive a request:

- immutable CPU blob-cache entries;
- immutable broker-owned GPU or shared-memory residency entries;
- cache metadata such as size, last use, generation, and lease count.

No request receives a broker pointer or broker-side Python object. It receives an
operating-system or platform capability that it imports into its own address space.

## 5. Process Architecture

```text
                         CPU-only frontend
                    validation, blobs, FIFO queue
                                |
                  dispatch + validated resource plan
                                |
                    per-device slot supervisor
                         /                 \
                        /                   \
       long-lived residency broker       fresh request process group
       trusted service code only           |
       owns immutable allocations           +-- main.py runner
       never imports submitted code         |
       never runs submitted kernels         +-- optional tool target
                                                  ncu / sanitizer / debugger
```

There is one slot supervisor and one residency broker for each configured device. A
broker is not a general worker. It has a narrow trusted control protocol and performs
only allocation, population, mapping-handle export, synchronization, eviction, and
health operations.

Each request gets a new process group. Submitted code and plugin tool targets execute
only in this group. Killing the group removes the request's process-local mappings and
runtime state without destroying broker-owned cache entries.

The broker should be a separate process rather than a thread in the frontend because:

- the frontend must not initialize a GPU runtime;
- GPU runtime failure is contained per device;
- broker restart can invalidate one device generation without discarding frontend state;
- ownership and lifetime remain explicit.

## 6. Why the Broker Must Be the Original Owner

The system must not implement this chain:

```text
job N allocation -> broker imports it -> job N exits -> broker re-exports it -> job N+1
```

Legacy CUDA IPC is mapping, not ownership transfer. Its exporter must keep the exported
allocation valid while importers use it. Relaying framework tensors through multiple
processes also makes allocator ownership and failure cleanup ambiguous.

Instead, use this chain:

```text
broker allocates and populates immutable bytes
    -> job N imports a request-local mapping
    -> job N mapping closes
    -> job N+1 imports a new request-local mapping
```

If a future feature needs a job to produce a persistent tensor, the broker must allocate
the destination before the job starts. The job receives exclusive write access, writes
the destination, synchronizes, and commits it. After verification, subsequent leases are
read-only. The broker never adopts an arbitrary allocation created inside submitted code.

## 7. Manifest Tensor Declarations

The existing file manifest identifies immutable bytes but does not always identify a
tensor interpretation. A raw blob can be viewed with different shapes, dtypes, strides,
and offsets. Add an optional `tensors` object to the job body:

```json
{
  "language": "python",
  "entry": {
    "file": "main.py",
    "function": "main"
  },
  "files": {
    "main.py": {
      "blob": "<main-sha256>"
    },
    "data/input.bin": {
      "blob": "<input-sha256>"
    }
  },
  "tensors": {
    "input": {
      "file": "data/input.bin",
      "dtype": "float16",
      "shape": [4096, 4096],
      "strides": null,
      "offset_bytes": 0,
      "access": "read"
    }
  }
}
```

Rules:

- `tensors` is optional and does not change normal file materialization.
- Each tensor entry refers to a path already present in `files`.
- The server resolves the path to the file's validated blob hash.
- The declared byte range must fit entirely inside the blob.
- The first version accepts only C-contiguous views; `strides` must be null or the
  canonical C-contiguous strides.
- `access` accepts only `read` for persistent entries in the first version.
- Multiple tensor names may create different validated views of the same resident bytes.
- Safetensors-aware plugins may derive view metadata from the validated safetensors
  header, but the resulting view must pass the same checks.
- A transformed representation uses a separate cache identity containing the transform
  name and version. It must not masquerade as the original SHA-256 blob.

The basic residency key for an untransformed blob is:

```text
(backend_kind, physical_device_id, device_generation, blob_sha256, blob_size)
```

Tensor view metadata is validated separately. This permits multiple views to reuse the
same underlying bytes.

## 8. Request-Facing Plugin API

Submitted code refers to a logical tensor name, not a device pointer or IPC handle:

```python
from plugins import tensor


def main():
    x = tensor("input")
    return run_kernel(x)
```

`plugins.tensor("input")` performs only request-local work:

1. Look up `input` in the immutable resource table prepared for this request.
2. Import or map the already granted platform capability.
3. Construct a process-local tensor wrapper with the declared dtype and shape.
4. Keep the imported mapping and request lease alive for at least as long as the wrapper.

Conceptual implementation:

```python
def tensor(name):
    resource = current_request.resources[name]
    mapping = backend.import_allocation(
        resource.capability,
        size=resource.allocation_size,
        access=resource.access,
    )
    return backend.make_tensor(
        mapping=mapping,
        offset_bytes=resource.offset_bytes,
        shape=resource.shape,
        dtype=resource.dtype,
        strides=resource.strides,
    )
```

The returned `torch.Tensor`, `tvm_ffi.Tensor`, or other framework value is local to the
request process. Its virtual address may differ from the broker's mapping and from every
other request's mapping. All mappings refer to the same immutable physical storage.

The tensor wrapper's deleter closes only the local import or mapping. It does not free
the broker's physical allocation. The broker lease is also tied to the request control
connection, so cleanup does not depend on Python destructors running after a crash.

## 9. Lease and Capability Protocol

The frontend sends the selected slot supervisor a validated resource plan containing
only hashes already acquired from the CPU blob cache. The broker protocol should expose
operations equivalent to:

```text
PREPARE(request_id, resource_plan)
GRANT(request_id, runner_identity)
RELEASE(request_id)
EVICT(cache_key)
HEALTH_CHECK()
RESET_GENERATION(reason)
```

### 9.1 Prepare

`PREPARE` atomically acquires all required residency entries for a request. If any entry
cannot be prepared, the broker releases the partial acquisition and returns a structured
error. A prepared entry cannot be evicted before the request releases it.

On a cache hit, preparation increments a lease count and performs no GPU copy.

On a cache miss, the broker allocates storage and populates it from the immutable CPU
blob. GPU population occurs only during the slot's exclusive `GPU_PREP` phase.

### 9.2 Grant

`GRANT` authenticates the fresh request process and sends only the capabilities named by
that request's resource plan. On POSIX systems, file-descriptor capabilities should be
sent over a Unix-domain socket with `SCM_RIGHTS`, not serialized as integer descriptor
numbers.

The control channel should verify the expected process identity where the operating
system provides peer credentials. Capabilities should be close-on-exec unless a specific
tool target is intentionally granted access.

### 9.3 Release

`RELEASE` removes all leases for the request. Release happens after normal runner cleanup
and also automatically when the authenticated request control connection closes or the
supervisor reaps the request process group.

Lease release does not evict an entry. It merely makes an unused entry eligible for LRU
eviction.

## 10. Scheduling and Benchmark Isolation

Each device slot has the following state machine:

```text
IDLE
  -> CPU_PREP       validate files and acquire CPU blob references
  -> GPU_PREP       populate missing resident entries; exclusive GPU use
  -> RUNNING        execute one request process group; exclusive GPU use
  -> CLEANUP        synchronize, reap processes, release request leases
  -> IDLE

Any state
  -> RECOVERING     kill runner, probe/reset device, possibly restart broker
  -> IDLE or UNHEALTHY
```

CPU preparation for N+1 may overlap GPU execution of N. GPU preparation for N+1 must not
overlap measured execution of N. Even a copy on a separate CUDA stream can consume copy
engines, memory bandwidth, cache capacity, and power or clock headroom.

When N and N+1 use the same blobs:

1. N imports and runs against broker entries.
2. While N runs, the scheduler validates N+1 and detects identical residency keys.
3. N+1 acquires leases, but the broker performs no GPU operation.
4. N synchronizes and its process group exits.
5. N's local mappings disappear and its leases are released.
6. N+1 receives new capabilities and imports new process-local mappings.

A cache hit therefore avoids the copy without introducing background GPU interference.

## 11. Normal Completion and Failure Cleanup

### 11.1 Normal completion

Before reporting success, trusted runner code must synchronize the device and finish
return-value serialization. The supervisor then:

1. closes request-local tensor wrappers and tool targets;
2. terminates any remaining request children;
3. reaps the request process group;
4. releases broker leases;
5. deletes the request work directory;
6. makes the slot available to the next request.

### 11.2 Timeout or crash

On timeout, pipe failure, abnormal exit, or failed synchronization:

1. stop granting new work to the slot;
2. terminate the full request process group;
3. wait for the configured grace period;
4. force-kill and reap remaining processes;
5. release all leases by request ID;
6. perform a broker/device health probe;
7. either resume with the current generation or reset it.

Python destructors and plugin cleanup callbacks are not trusted to run in this path.

### 11.3 Device generation reset

Each broker instance maintains a monotonically increasing generation. A fatal device
error, broker restart, device reset, or failed post-crash health check invalidates every
entry and capability from the old generation.

The broker must never reuse an old capability after generation reset. Queued requests
must prepare their resources again against the new generation.

## 12. Platform Backends

The control-plane contract is portable. Allocation and import mechanisms are not.

### 12.1 NVIDIA discrete GPU on Linux

Preferred implementation: CUDA Driver API virtual memory management.

Broker responsibilities:

- query virtual-memory-management and POSIX-FD handle support;
- allocate dedicated, correctly aligned physical memory with `cuMemCreate`;
- reserve/map a broker address and populate it from the CPU blob;
- export a POSIX file-descriptor handle;
- retain its allocation reference while the entry is cached.

Runner responsibilities:

- receive the FD with `SCM_RIGHTS`;
- import it with `cuMemImportFromShareableHandle`;
- reserve a process-local virtual address;
- map the allocation and set device access to read-only;
- wrap the resulting process-local pointer as a tensor;
- unmap, release the imported handle, and close the FD during cleanup.

Legacy CUDA IPC may be a compatibility fallback, but it is not the preferred isolation
mechanism. It requires careful exporter lifetime management, does not provide the same
allocation-level access control, and allocator suballocation can expose a larger backing
allocation than intended.

The current implementation's `PersistentSubprocessWorker.run_ref()` retains baseline
tensors in the caller and `run_solution()` passes those tensors to a persistent child.
The v2 broker generalizes that reuse while moving GPU ownership out of the frontend and
allowing the submitted-code process to be replaced for every request.

### 12.2 Jetson

The implementation must not assume legacy CUDA memory IPC, because CUDA memory-sharing
IPC is not generally supported on Tegra. Preferred choices are capability-dependent:

1. NvSciBuf-backed raw or tensor storage imported as CUDA external memory, with reduced
   read-only permissions for consumer requests.
2. Shared system-memory storage directly accessible by the iGPU when the device exposes
   the required pageable-memory or host-registration capabilities.
3. Pinned or unified-memory strategies selected according to the specific Tegra
   coherency model.
4. A per-request device copy as the correctness fallback.

Jetson CPU and iGPU use the same physical SoC DRAM, so a shared-system-memory backend can
avoid a duplicate physical CPU-to-device copy. Cacheability and coherency differ by SoC;
the backend must probe capabilities rather than infer them from a product name.

### 12.3 Apple silicon

Apple silicon uses unified system memory. The preferred general-buffer design is:

- allocate page-aligned shared VM storage whose lifetime is held by the broker;
- send the shared-memory capability to the request process;
- map it in the request process;
- create a process-local `MTLBuffer` using a no-copy shared-storage path;
- wrap the buffer with the framework-specific tensor adapter.

For texture-compatible storage, IOSurface and `MTLSharedTextureHandle` provide explicit
cross-process sharing mechanisms. `MTLSharedEvent` can provide cross-process ordering
where the backend needs an explicit GPU completion primitive.

General `MTLBuffer` cross-process behavior and framework wrapping must be verified on the
minimum supported macOS release. If the required no-copy mapping cannot be provided, the
backend falls back to a private request buffer and copy without changing the plugin API.

Older discrete-GPU Macs do not share Apple silicon's memory model and may require managed
storage synchronization or a copy. The API promises semantic portability, not zero-copy
on every Mac.

### 12.4 Capability discovery

Each broker reports a backend mode in health and request metadata:

```text
cuda_vmm
cuda_legacy_ipc
jetson_nvscibuf
jetson_shared_system
metal_shared_system
private_copy
```

This makes performance differences observable without changing program behavior.

## 13. DLPack and Framework Wrapping

An IPC or OS handle is not a tensor. After import, the request process has a local pointer
or platform resource plus tensor metadata. A native adapter creates a local managed
tensor object whose lifetime owns the imported mapping.

For CUDA, the adapter can expose a `DLManagedTensor` containing the process-local CUDA
pointer, device, dtype, shape, and strides. The managed-tensor deleter unmaps the local
view and releases the imported handle. Frameworks then consume the value through their
DLPack interface.

For Metal and Jetson-specific external resources, the backend supplies an equivalent
framework adapter. DLPack describes the local tensor after import; it does not carry the
cross-process capability.

The service must test lifetime behavior for every supported framework. A framework
tensor must not outlive its imported mapping, and an early framework conversion must not
release the broker lease while device work is still pending.

## 14. Diagnostic and Evaluation Plugins

Plugin names are user-facing APIs, not execution locations. For example:

```python
from plugins import profile, run_solution, sanitize
```

The long-lived broker never executes any solution passed to these functions.

- `run_solution()` executes in the request runner or a request-scoped helper.
- `profile()` asks the request supervisor to launch an NCU-controlled target.
- `sanitize()` asks it to launch a compute-sanitizer-controlled target.
- `debug()` asks it to launch a cuda-gdb-controlled target when supported.

Tool targets receive their resource capabilities directly. For example:

```text
broker --request lease--> ncu --launches--> request-scoped target
```

If NCU must launch the target itself, the supervisor either passes only the required FDs
with an explicit `pass_fds` list or gives the target an authenticated one-use broker
connection. The system must not serialize a Python CUDA tensor through the runner, NCU,
and target as a chain of owners.

All tool processes belong to the request process group and are removed before the slot
accepts another request. JIT and compiler caches remain redirected into the request work
directory unless a separate immutable compilation-cache design is introduced.

## 15. Mutability and Persistent Outputs

Persistent cache entries are immutable. Normal outputs and scratch memory are always
request-private.

In-place kernels cannot receive a persistent entry as a writable input. They must use a
private working copy. A device-to-device copy may still avoid CPU transfer, but it does
not provide zero-copy execution.

A future persistent-output protocol may be added as a separate operation:

```text
ALLOCATE_OUTPUT(request_id, tensor_spec)
  -> exclusive writable mapping
COMMIT_OUTPUT(request_id, output_id)
  -> synchronize, validate, hash, seal immutable
ABORT_OUTPUT(request_id, output_id)
  -> discard allocation
```

Committing a tensor to the public `/blobs` namespace requires canonical CPU bytes and a
verified SHA-256 digest. A live mutable GPU allocation must not be inserted into `/blobs`
under an unverified identifier. The initial implementation should omit persistent output
commit and support only immutable uploaded inputs.

## 16. Cache Admission, Accounting, and Eviction

The broker tracks at least:

```text
cache_key
logical_blob_bytes
allocated_bytes
backend_kind
device_generation
state: LOADING | READY | EVICTING | FAILED
lease_count
last_used_monotonic
```

`allocated_bytes` may exceed blob size because backend granularity and alignment are part
of physical allocation. Admission uses allocated bytes, not logical bytes.

The per-device memory budget reserves space for:

- broker residency entries;
- the active request's private inputs, outputs, compilation, and scratch memory;
- driver/runtime overhead;
- a configurable safety margin.

On pressure, evict the least-recently-used READY entry whose lease count is zero. Never
evict LOADING or leased entries. If a new entry and the request working-set reserve still
do not fit after eviction, fail preparation before starting submitted code.

The CPU blob cache and GPU residency cache have separate reference counts and capacities.
An active residency entry pins its CPU source only while population or recovery requires
it; policy may retain the CPU blob independently according to the file-cache LRU.

## 17. Observability

Add per-device broker information to `/health`:

```json
{
  "gpu_id": "0",
  "status": "idle",
  "broker": {
    "backend": "cuda_vmm",
    "generation": 7,
    "entries": 12,
    "logical_bytes": 4294967296,
    "allocated_bytes": 4303355904,
    "leased_entries": 0
  }
}
```

Structured events should include:

```text
broker_started
broker_stopped
broker_generation_reset
residency_hit
residency_miss
residency_load_started
residency_load_finished
residency_evicted
residency_prepare_failed
lease_granted
lease_released
lease_reclaimed_after_crash
```

Each event includes request ID where applicable, device ID, backend kind, generation,
cache key or blob hash, logical bytes, allocated bytes, and duration. Do not log raw
capability contents or device pointers.

Request timing should distinguish:

```text
queue_ms
cpu_prepare_ms
gpu_prepare_ms
execute_ms
serialize_ms
cleanup_ms
residency_hits
residency_misses
```

## 18. Security and Robustness Requirements

Even for trusted callers, the implementation should enforce the following boundaries:

- The broker protocol accepts cache identities and validated metadata only from the slot
  supervisor, not arbitrary paths from submitted code.
- A request can resolve only tensor names present in its own immutable resource table.
- The broker sends only capabilities for the authenticated request ID and process.
- Persistent inputs are read-only wherever the platform can enforce access permissions.
- Writable cache aliases are forbidden.
- Broker control descriptors are not inherited by arbitrary request children.
- Request process groups and descendants are reaped before the next request starts.
- Old-generation capabilities are rejected.
- Cache entries use dedicated allocations where allocator suballocation could expose
  unrelated data.
- Eviction optionally scrubs memory when deployment confidentiality policy requires it.

Because the first version accepts trusted submitted code, these measures reduce accidental
cross-request contamination but are not a complete malicious-code sandbox. Strong hostile
code isolation additionally requires OS sandboxing, restricted device interfaces, syscall
policy, and stronger descendant control.

## 19. Implementation Plan

### Phase 1: protocol and private-copy backend

- Add tensor declarations and validation to the job schema.
- Add `plugins.tensor(name)` with a request-local resource table.
- Add the broker control protocol and generation handling.
- Implement a private-copy backend as a portable correctness baseline.
- Add lifecycle, crash, lease, and observability tests.

### Phase 2: CUDA VMM backend

- Implement dedicated VMM allocation, population, FD export/import, and local mapping.
- Implement read-only consumer mappings.
- Add cache admission and LRU eviction.
- Test normal exit, timeout, SIGKILL, illegal access, broker restart, and device reset.
- Integrate request-scoped evaluation and diagnostic targets.

### Phase 3: Jetson backend

- Probe representative supported Jetson generations.
- Implement NvSciBuf or shared-system-memory import according to capabilities.
- Measure cached, uncached, pinned, and coherent access behavior.
- Preserve private-copy fallback for unsupported devices.

### Phase 4: Apple silicon backend

- Implement shared-VM allocation and process-local Metal buffer wrapping.
- Validate framework/DLPack lifetime integration.
- Add IOSurface/shared-texture support where useful.
- Test process crash, mapping cleanup, shared-event ordering, and fallback behavior.

### Phase 5: optional persistent outputs

- Add broker-allocated exclusive writable outputs.
- Define canonical byte representation and verified commit.
- Seal committed entries immutable before allowing a second request to acquire them.

## 20. Acceptance Criteria

The first residency-enabled release is complete when:

- two consecutive requests using the same declared blob produce one residency load and
  two independent request-local mappings;
- the second request performs no CPU-to-GPU population copy on a cache hit;
- no N+1 GPU operation overlaps the measured execution phase of N;
- killing N does not kill the broker or invalidate a healthy cache entry;
- a fatal device error resets the broker generation and invalidates all old handles;
- a request cannot resolve a tensor absent from its own manifest;
- persistent inputs cannot be modified on backends that advertise read-only enforcement;
- tool targets import only the capabilities leased to their request;
- frontend health and HTTP processing do not initialize a GPU runtime;
- Jetson and macOS use a capability-selected zero-copy/shared-memory backend or report an
  explicit private-copy fallback;
- logs and metrics distinguish residency hits, misses, preparation, execution, and
  recovery.

## 21. References

- [CUDA Programming Guide: Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)
- [CUDA Programming Guide: Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
- [CUDA Driver API: Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
- [CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)
- [CUDA interoperability with NvSciBuf](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html)
- [Apple Metal shared storage](https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared)
- [Apple Metal no-copy buffer creation](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29)
- [Apple IOSurface](https://developer.apple.com/documentation/iosurface)
- [Apple MTLSharedTextureHandle](https://developer.apple.com/documentation/metal/mtlsharedtexturehandle)
