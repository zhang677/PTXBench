#### [9.7.9.25. Data Movement and Conversion Instructions: Asynchronous copy](#data-movement-and-conversion-instructions-asynchronous-copy)

An asynchronous copy operation performs the underlying operation asynchronously in the background,
thus allowing the issuing threads to perform subsequent tasks.

An asynchronous copy operation can be a *bulk* operation that operates on a large amount of data, or
a *non-bulk* operation that operates on smaller sized data. The amount of data handled by a bulk
asynchronous operation must be a multiple of 16 bytes.

An asynchronous copy operation typically includes the following sequence:

- Optionally, reading from the tensormap.
- Reading data from the source location(s).
- Writing data to the destination location(s).
- Writes being made visible to the executing thread or other threads.

##### [9.7.9.25.1. Completion Mechanisms for Asynchronous Copy Operations](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms)

A thread must explicitly wait for the completion of an asynchronous copy operation in order to
access the result of the operation. Once an asynchronous copy operation is initiated, modifying the
source memory location or tensor descriptor or reading from the destination memory location before
the asynchronous operation completes, exhibits undefined behavior.

This section describes two asynchronous copy operation completion mechanisms supported in PTX:

Async-group mechanism and mbarrier-based mechanism.

Asynchronous operations may be tracked by either of the completion mechanisms or both mechanisms.

The tracking mechanism is instruction/instruction-variant specific.

###### [9.7.9.25.1.1. Async-group mechanism](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group)

When using the async-group completion mechanism, the issuing thread specifies a group of
asynchronous operations, called *async-group* , using a *commit* operation and tracks the completion
of this group using a *wait* operation. The thread issuing the asynchronous operation must create
separate *async-groups* for bulk and non-bulk asynchronous operations.

A *commit* operation creates a per-thread *async-group* containing all prior asynchronous operations
tracked by *async-group* completion and initiated by the executing thread but none of the asynchronous
operations following the commit operation. A committed asynchronous operation belongs to a single *async-group* .

When an *async-group* completes, all the asynchronous operations belonging to that group are
complete and the executing thread that initiated the asynchronous operations can read the result of
the asynchronous operations. All *async-groups* committed by an executing thread always complete in
the order in which they were committed. There is no ordering between asynchronous operations within
an *async-group* .

A typical pattern of using *async-group* as the completion mechanism is as follows:

- Initiate the asynchronous operations.
- Group the asynchronous operations into an *async-group* using a *commit* operation.
- Wait for the completion of the async-group using the wait operation.
- Once the *async-group* completes, access the results of all asynchronous operations in that *async-group* .

###### [9.7.9.25.1.2. Mbarrier-based mechanism](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier)

A thread can track the completion of one or more asynchronous operations using the current phase of
an *mbarrier object* . When the current phase of the *mbarrier object* is complete, it implies that
all asynchronous operations tracked by this phase are complete, and all threads participating in
that *mbarrier object* can access the result of the asynchronous operations.

The *mbarrier object* to be used for tracking the completion of an asynchronous operation can be
either specified along with the asynchronous operation as part of its syntax, or as a separate
operation. For a bulk asynchronous operation, the *mbarrier object* must be specified in the
asynchronous operation, whereas for non-bulk operations, it can be specified after the asynchronous
operation.

A typical pattern of using mbarrier-based completion mechanism is as follows:

- Initiate the asynchronous operations.
- Set up an *mbarrier object* to track the asynchronous operations in its current phase, either as part of the asynchronous operation or as a separate operation.
- Wait for the *mbarrier object* to complete its current phase using `mbarrier.test_wait` or `mbarrier.try_wait` .
- Once the `mbarrier.test_wait` or `mbarrier.try_wait` operation returns `True` , access the results of the asynchronous operations tracked by the *mbarrier object* .

##### [9.7.9.25.2. Async Proxy](#async-proxy)

The `cp{.reduce}.async.bulk` operations are performed in the *asynchronous proxy* (or *async proxy*).

Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the *async proxy* , `fence.proxy.async` should be used to synchronize memory between *generic proxy* and the *async proxy* .

The completion of a `cp{.reduce}.async.bulk` operation is followed by an implicit *generic-async* proxy fence. So the result of the asynchronous operation is made visible to the generic proxy as soon as its completion is observed.

*Async-group* OR *mbarrier-based* completion mechanism must be used to wait for the completion of the `cp{.reduce}.async.bulk` instructions.

#### [9.7.13.15. Parallel Synchronization and Communication Instructions: mbarrier](#parallel-synchronization-and-communication-instructions-mbarrier)

`mbarrier` is a barrier created in shared memory that supports :

- Synchronizing any subset of threads within a CTA
- One-way synchronization of threads across CTAs of a cluster. As noted in [mbarrier support with shared memory](#parallel-synchronization-and-communication-instructions-mbarrier-smem) , threads can perform only *arrive* operations but not *wait* on an mbarrier located in `shared::cluster` space.
- Waiting for completion of asynchronous memory operations initiated by a thread and making them visible to other threads.

An *mbarrier object* is an opaque object in memory which can be initialized and invalidated using :

- mbarrier.init
- mbarrier.inval
Operations supported on *mbarrier object* s are :

- mbarrier.expect\_tx
- mbarrier.complete\_tx
- mbarrier.arrive
- mbarrier.arrive\_drop
- mbarrier.test\_wait
- mbarrier.try\_wait
- mbarrier.pending\_count
- cp.async.mbarrier.arrive
Performing any *mbarrier* operation except `mbarrier.init` on an uninitialized *mbarrier object* results in undefined behavior.

Performing any *non-mbarrier* or `mbarrier.init` operations on an initialized *mbarrier object* results in undefined behavior.

Unlike `bar{.cta}` / `barrier{.cta}` instructions which can access a limited number of barriers per CTA, *mbarrier objects* are user defined and are only limited by the total shared memory size available.

*mbarrier* operations enable threads to perform useful work after the arrival at the *mbarrier* and before waiting for the *mbarrier* to complete.

##### [9.7.13.15.1. Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment)

An mbarrier object is an opaque object with the following type and alignment requirements :

| Type  |   Alignment (bytes) | Memory space    |
|--------------|---------------------|-----------------|
| ``` .b64 ``` |     8 | ``` .shared ``` |

##### [9.7.13.15.2. Contents of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-contents)

An opaque *mbarrier object* keeps track of the following information :

- Current phase of the *mbarrier object*
- Count of pending arrivals for the current phase of the *mbarrier object*
- Count of expected arrivals for the next phase of the *mbarrier object*
- Count of pending asynchronous memory operations (or transactions) tracked by the current phase of the *mbarrier object* . This is also referred to as *tx-count* .

An *mbarrier object* progresses through a sequence of phases where each phase is defined by threads
performing an expected number of [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operations.

The valid range of each of the counts is as shown below:

| Count name      | Minimum value   | Maximum value   |
|------------------------|-----------------|-----------------|
| Expected arrival count | 1 | 2 ^ 20  - 1      |
| Pending arrival count  | 0 | 2 ^ 20  - 1      |
| tx-count | -(2 ^ 20  - 1)   | 2 ^ 20  - 1      |

##### [9.7.13.15.3. Lifecycle of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-lifecycle)

The *mbarrier object* must be initialized prior to use.

An *mbarrier object* is used to synchronize threads and asynchronous memory operations.

An *mbarrier object* may be used to perform a sequence of such synchronizations.

An *mbarrier object* must be invalidated to repurpose its memory for any purpose,
including repurposing it for another mbarrier object.

##### [9.7.13.15.4. Phase of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase)

The phase of an *mbarrier object* is the number of times the *mbarrier object* has been used to
synchronize threads and
[asynchronous](#program-order-async-operations) operations. In each phase {0, 1, 2, ...}, threads perform in program order :

- [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operations to complete the current phase and
- *test\_wait* / *try\_wait* operations to check for the completion of the current phase.

An *mbarrier object* is automatically reinitialized upon completion of the current phase for
immediate use in the next phase. The current phase is incomplete and all prior phases are complete.

For each phase of the mbarrier object, at least one *test\_wait* or *try\_wait* operation must be
performed which returns
`True` for `waitComplete` before an [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation
in the subsequent phase.

##### [9.7.13.15.5. Tracking asynchronous operations by the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations)

Starting with the Hopper architecture ( `sm_9x` ), *mbarrier object* supports a new count, called *tx-count* , which is used for tracking the completion of asynchronous memory operations or transactions.

*tx-count* tracks the number of asynchronous transactions, in units specified by the asynchronous memory operation, that are outstanding and yet to be complete.

The *tx-count* of an *mbarrier object* must be set to the total amount of asynchronous memory
operations, in units as specified by the asynchronous operations, to be tracked by the current
phase. Upon completion of each of the asynchronous operations, the
[complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation) operation will be performed on the *mbarrier object* and thus progress the mbarrier towards the completion of the current phase.

###### [9.7.13.15.5.1. expect-tx operation](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)

The *expect-tx* operation, with an `expectCount` argument, increases the *tx-count* of an *mbarrier object* by the value specified by `expectCount` . This sets the current phase of the *mbarrier object* to expect and track the completion of additional asynchronous transactions.

###### [9.7.13.15.5.2. complete-tx operation](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)

The *complete-tx* operation, with an `completeCount` argument, on an *mbarrier object* consists of the following:
- mbarrier signaling:
Signals the completion of asynchronous transactions that were tracked by the current phase. As a result of this, *tx-count* is decremented by `completeCount` .
- mbarrier potentially completing the current phase:
If the current phase has been completed then the mbarrier transitions to the next phase. Refer to [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion) for details on phase completion requirements and phase transition process.

##### [9.7.13.15.6. Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)

The requirements for completion of the current phase are described below. Upon completion of the
current phase, the phase transitions to the subsequent phase as described below.

Current phase completion requirements
An *mbarrier object* completes the current phase when all of the following conditions are met:

- The count of the pending arrivals has reached zero.
- The *tx-count* has reached zero.

Phase transition
When an *mbarrier* object completes the current phase, the following actions are performed
atomically:

- The *mbarrier object* transitions to the next phase.
- The pending arrival count is reinitialized to the expected arrival count.

##### [9.7.13.15.7. Arrive-on operation on mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)

An *arrive-on* operation, with an optional *count* argument, on an *mbarrier object* consists of the following 2 steps :

- mbarrier signalling: Signals the arrival of the executing thread OR completion of the asynchronous instruction which signals the arrive-on operation initiated by the executing thread on the *mbarrier object* . As a result of this, the pending arrival count is decremented by *count* . If the *count* argument is not specified, then it defaults to 1.
- mbarrier potentially completing the current phase: If the current phase has been completed then the mbarrier transitions to the next phase. Refer to [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion) for details on phase completion requirements and phase transition process.

##### [9.7.13.15.8. mbarrier support with shared memory](#parallel-synchronization-and-communication-instructions-mbarrier-smem)

The following table summarizes the support of various mbarrier operations on *mbarrier objects* located at different shared memory locations:

| mbarrier operations   | ``` .shared::cta ```   | ``` .shared::cluster ``` |
|----|------------------------|---------------------------------|
| ``` mbarrier.arrive ```      | Supported| Supported, cannot return result |
| ``` mbarrier.expect_tx ```   | Supported| Supported  |
| ``` mbarrier.complete_tx ``` | Supported| Supported  |
| Other mbarrier operations    | Supported| Not supported     |

Below is the hardware parameter of NVIDIA H100
| Component | Value |
|-----------|---------------|
| SM Count | 132 |
| Peak FP8 | ~1978 TFLOPS |
| Peak BF16 | ~989 TFLOPS |
| Cluster Size | Up to 16 SMs |
| Max Registers / CTA | 65536 |
| Max Registers / Thread | 255 |
| Max CTA size (# of threads) | 1024 |

| Level | Capacity | Latency | Bandwidth |
|-------|----|------|---------|
| Global (HBM) | 80 GB | ~500 cycles | 3.35 TB/s |
| L2 Cache | 50 MB | ~100 cycles | ~12TB/s |
| Shared Memory | 256 KB / SM (228 KB usable with 28KB L1 Cache) | ~30 cycles | 128 B/cycle/SM |
| Registers | 256 KB / SM  | 1 cycle | Unlimited (matched with compute) |

/*
    // Get the SM ID of the current thread
*/
__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

/*
    elect.sync elects one predicated active leader thread from among a set of threads specified by membermask. 
    The predicate destination p is set to True for the leader thread, and False for all other threads.
    Election of a leader thread happens deterministically, i.e. the same leader thread is elected for the same membermask every time.
*/
__device__ __forceinline__ bool elect_one_sync_fn() {
    // Elect a single thread in the warp. Returns 1 for elected, 0 for others.
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

/*

    A warpgroup is a set of four contiguous warps (128 threads) such that the warp-rank of the first warp is a multiple of 4.
    warp-rank of a warp is defined as: (%tid.x + %tid.y * %ntid.x  + %tid.z * %ntid.x * %ntid.y) / 32
    setmaxnreg provides a hint to the system to update the maximum number of per-thread registers owned by the executing warp to the value specified by the imm-reg-count operand.
    Qualifier .dec is used to release extra registers such that the absolute per-thread maximum register count is reduced from its current value to imm-reg-count. 
    Qualifier .inc is used to request additional registers such that the absolute per-thread maximum register count is increased from its current value to imm-reg-count.

    A pool of available registers is maintained per-CTA. Register adjustments requested by the setmaxnreg instructions are handled by supplying extra registers from this pool to the requesting warp or by releasing extra registers from the requesting warp to this pool, depending upon the value of the .action qualifier.

    The setmaxnreg.inc instruction blocks the execution until enough registers are available in the CTA’s register pool. After the instruction setmaxnreg.inc obtains new registers from the CTA pool, the initial contents of the new registers are undefined. The new registers must be initialized before they are used.

    The same setmaxnreg instruction must be executed by all warps in a warpgroup. After executing a setmaxnreg instruction, all warps in the warpgroup must synchronize explicitly before executing subsequent setmaxnreg instructions. If a setmaxnreg instruction is not executed by all warps in the warpgroup, then the behavior is undefined.

    Operand imm-reg-count is an integer constant. The value of imm-reg-count must be in the range 24 to 256 (both inclusive) and must be a multiple of 8.

    Changes to the register file of the warp always happen at the tail-end of the register file.

    The setmaxnreg instruction requires that the kernel has been launched with a valid value of maximum number of per-thread registers specified via the appropriate compilation via the appropriate compile-time option or the appropriate performance tuning directive. Otherwise, the setmaxnreg instruction may have no effect.

    When qualifier .dec is specified, the maximum number of per-thread registers owned by the warp prior to the execution of setmaxnreg instruction should be greater than or equal to the imm-reg-count. Otherwise, the behaviour is undefined.

    When qualifier .inc is specified, the maximum number of per-thread registers owned by the warp prior to the execution of setmaxnreg instruction should be less than or equal to the imm-reg-count. Otherwise, the behaviour is undefined.

    The mandatory .sync qualifier indicates that setmaxnreg instruction causes the executing thread to wait until all threads in the warp execute the same setmaxnreg instruction before resuming execution.

    The mandatory .aligned qualifier indicates that all threads in the warpgroup must execute the same setmaxnreg instruction. In conditionally executed code, setmaxnreg instruction should only be used if it is known that all threads in warpgroup evaluate the condition identically, otherwise the behavior is undefined.
*/

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_dec_sync_fn() {
    // Decrease the maximum number of registers for the warpgroup to NUM_REGS
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}


__device__ __forceinline__ float fast_exp2f_fn(float x) {
    // Find the base-2 exponential of a value. ln2 = 0.6931471805599453
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    // 128-bit vectorized store to shared memory (st.shared.v4.b32).
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    // Pack two FP32 values (as uint32 bit patterns) into one uint32 of two BF16.
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

/*
Cluster is a group of CTAs that run concurrently or in parallel and can synchronize and communicate with each other via shared memory. The executing CTA has to make sure that the shared memory of the peer CTA exists before communicating with it via shared memory and the peer CTA hasn’t exited before completing the shared memory operation.

Threads within the different CTAs in a cluster can synchronize and communicate with each other via shared memory. Cluster-wide barriers can be used to synchronize all the threads within the cluster. Each CTA in a cluster has a unique CTA identifier within its cluster (cluster_ctaid). Each cluster of CTAs has 1D, 2D or 3D shape specified by the parameter cluster_nctaid. Each CTA in the cluster also has a unique CTA identifier (cluster_ctarank) across all dimensions. The total number of CTAs across all the dimensions in the cluster is specified by cluster_nctarank. Threads may read and use these values through predefined, read-only special registers %cluster_ctaid, %cluster_nctaid, %cluster_ctarank, %cluster_nctarank.
*/

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// Cluster synchronization
/*
    Host-Side Cluster Launch

    ```cpp
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;       // grid.x MUST be divisible by cluster_size
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel_fn, arg0, arg1, ...);
    // The 2nd arg (kernel_fn) must be the kernel function pointer directly (for template deduction). Do NOT cast it to other types
    ```
    Note: Grid not divisible by cluster size — causes `cudaErrorInvalidConfiguration`.
*/

/*
    Performs barrier synchronization and communication within a cluster.
*/

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_arrive_fn() {
    asm volatile("barrier.cluster.arrive;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_wait_fn() {
    asm volatile("barrier.cluster.wait;\n" ::: "memory");
}


// MBarrier operations

/*  
    mbarrier.init initializes the mbarrier object at the location specified by the address operand [addr] with the unsigned 32-bit integer count. 
    Initialization of the mbarrier object involves:
    Initializing the current phase to 0.
    Initializing the expected arrival count to count.
    Initializing the pending arrival count to count.
    Initializing the tx-count to 0.
*/
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {

    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

/*
    fence.op_restrict.release.cluster;
    The fence instruction establishes an ordering between memory accesses requested by this thread. When .op_restrict is .mbarrier_init, the synchronizing effect of the fence only applies to the prior mbarrier.init operations executed by the same thread on mbarrier objects in .shared::cta state space. 
*/

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

/*
    mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64           state, [addr]{, count};
    mbarrier.arrive{.sem.scope}{.shared::cluster}.b64         _, [addr] {,count}
    mbarrier.arrive.expect_tx{.sem.scope}{.shared{::cta}}.b64 state, [addr], txCount;
    mbarrier.arrive.expect_tx{.sem.scope}{.shared::cluster}.b64   _, [addr], txCount;
    mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64  state, [addr], count;

    .sem   = { .release, .relaxed }
    .scope = { .cta, .cluster }

    A thread executing mbarrier.arrive performs an [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation on the mbarrier object at the location specified by the address operand [addr].
    The optional qualifier .expect_tx specifies that an [expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation) operation is performed prior to the arrive-on operation. The 32-bit unsigned integer operand txCount specifies the expectCount argument to the expect-tx operation. When both qualifiers .arrive and .expect_tx are specified, then the count argument of the arrive-on operation is assumed to be 1.
    mbarrier.arrive operation on an mbarrier object located in .shared::cta returns an opaque 64-bit register capturing the phase of the mbarrier object prior to the arrive-on operation in the destination operand state. Contents of the state operand are implementation specific. Optionally, sink symbol '_' can be used for the state argument.
    mbarrier.arrive operation on an mbarrier object located in .shared::cluster but not in .shared::cta cannot return a value. Sink symbol ‘_’ is mandatory for the destination operand for such cases.
    If the .sem qualifier is absent, .release is assumed by default. The .relaxed qualifier does not provide any memory ordering semantics and visibility guarantees.

*/

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    // Cluster-scoped arrive_expect_tx: signals a barrier in target_cta's shared memory.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
}

__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta) {
    // Arrive at a barrier on a remote CTA (target_cta) in the cluster.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta)); // mapa maps the address of the shared variable in the target CTA.
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a));
}

/*
    The mbarrier.try_wait operation tests for the completion of the current or the immediately preceding phase of an mbarrier object at the location specified by the operand [addr].
    mbarrier.try_wait is a potentially blocking instruction which tests for the completion of the phase. If the phase is not complete, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes OR before the phase completes following a system-dependent time limit.
    mbarrier.try_wait test for completion of the phase indicated by the 32-bit unsigned integer operand phaseParity, which is the integer parity of either the current phase or the immediately preceding phase of the mbarrier object.
    The .parity variant of the instructions test for the completion of the phase indicated by the operand phaseParity, which is the integer parity of either the current phase or the immediately preceding phase of the mbarrier object. An even phase has integer parity 0 and an odd phase has integer parity of 1. So the valid values of phaseParity operand are 0 and 1.
    try_wait operation is valid only for :
    the current incomplete phase, for which waitComplete returns False.
    the immediately preceding phase, for which waitComplete returns True.
    The following ordering of memory operations hold for the executing thread when mbarrier.try_wait having acquire semantics returns True :
    All memory accesses (except async operations) requested prior, in program order, to mbarrier.arrive having release semantics during the completed phase by the participating threads of the CTA are performed and are visible to the executing thread.
    All cp.async operations requested prior, in program order, to cp.async.mbarrier.arrive during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread.
    All cp.async.bulk asynchronous operations using the same mbarrier object requested prior, in program order, to mbarrier.arrive having release semantics during the completed phase by the participating threads of the CTA are performed and made visible to the executing thread.
    All memory accesses requested after the mbarrier.try_wait, in program order, are not performed and not visible to memory accesses performed prior to mbarrier.arrive having release semantics, in program order, by other threads participating in the mbarrier.
*/
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

/*
    Accessing the same memory location across multiple proxies needs a cross-proxy fence.
*/
__device__ __forceinline__ void fence_proxy_async_fn() {
    // Generic async proxy fence. Required after manual shared memory writes (generic proxy) before WGMMA consumption (asynchronous "async" proxy)
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    // Async proxy fence on shared memory (fence.proxy.async.shared::cta).
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/*
    barrier{.cta}.sync{.aligned}      a{, b};
    barrier{.cta}.arrive{.aligned}    a, b;
    Performs barrier synchronization and communication within a CTA. Each CTA instance has sixteen barriers numbered 0..15.
    barrier{.cta} instructions can be used by the threads within the CTA for synchronization and communication.

    Operands a and b have type .u32. Source operand a specifies a logical barrier resource as an immediate constant or register with value 0 through 15. Operand b specifies the number of threads participating in the barrier. If no thread count is specified, all threads in the CTA participate in the barrier. When specifying a thread count, the value must be a multiple of the warp size. Note that a non-zero thread count is required for barrier{.cta}.arrive.

    Depending on operand b, either specified number of threads (in multiple of warp size) or all threads in the CTA participate in barrier{.cta} instruction. The barrier{.cta} instructions signal the arrival of the executing threads at the named barrier.

    barrier{.cta} instruction causes executing thread to wait for all non-exited threads from its warp and marks warps’ arrival at barrier. In addition to signaling its arrival at the barrier, the barrier{.cta}.sync instruction causes executing thread to wait for non-exited threads of all other warps participating in the barrier to arrive. barrier{.cta}.arrive does not cause executing thread to wait for threads of other participating warps.

    When a barrier completes, the waiting threads are restarted without delay, and the barrier is reinitialized so that it can be immediately reused.

    The barrier{.cta}.sync or or barrier{.cta}.arrive instruction guarantees that when the barrier completes, prior memory accesses requested by this thread are performed relative to all threads participating in the barrier. The barrier{.cta}.sync instruction further guarantees that no new memory access is requested by this thread before the barrier completes.

    A memory read (e.g., by ld or atom) has been performed when the value read has been transmitted from memory and cannot be modified by another thread participating in the barrier. A memory write (e.g., by st, red or atom) has been performed when the value written has become visible to other threads participating in the barrier, that is, when the previous value can no longer be read.

    Instruction barrier{.cta} has optional .aligned modifier. When specified, it indicates that all threads in CTA will execute the same barrier{.cta} instruction. In conditionally executed code, an aligned barrier{.cta} instruction should only be used if it is known that all threads in CTA evaluate the condition identically, otherwise behavior is undefined.

    Different warps may execute different forms of the barrier{.cta} instruction using the same barrier name and thread count. One example mixes barrier{.cta}.sync and barrier{.cta}.arrive to implement producer/consumer models. The producer threads execute barrier{.cta}.arrive to announce their arrival at the barrier and continue execution without delay to produce the next value, while the consumer threads execute the barrier{.cta}.sync to wait for a resource to be produced. The roles are then reversed, using a different barrier, where the producer threads execute a barrier{.cta}.sync to wait for a resource to consumed, while the consumer threads announce that the resource has been consumed with barrier{.cta}.arrive. Care must be taken to keep a warp from executing more barrier{.cta} instructions than intended (barrier{.cta}.arrive followed by any other barrier{.cta} instruction to the same barrier) prior to the reset of the barrier.
*/
__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
     // Named barrier for math warpgroup sync (before TMA store). Can be used for sync threads within a warpgroup that has `count` number of threads.
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void named_barrier_arrive_fn(int bar_id, int count) {
    asm volatile("barrier.arrive.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

// Grid-level barrier using atomic operations
__device__ unsigned int __grid_sync_count = 0;
__device__ volatile int __grid_sync_sense = 0;

__device__ __forceinline__ void grid_sync_fn() {
    __syncthreads();
    __threadfence();
    // Only (0,0,0) thread of each block participates in the grid barrier
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned int num_blocks = gridDim.x * gridDim.y * gridDim.z;
        // Atomically increment the arrival counter
        unsigned int arrived = atomicAdd(&__grid_sync_count, 1);
        if (arrived == num_blocks - 1) {
            // Last block: reset counter and flip sense
            __grid_sync_count = 0;
            __threadfence();
            __grid_sync_sense ^= 1;
        } else {
            // Wait for the sense to flip
            int expected = __grid_sync_sense ^ 1;
            while (__grid_sync_sense != expected) {
                // Spin-wait
            }
        }
    }
    __syncthreads();
}

/*

    // ---- Shared-Memory Bank Swizzling ----
    Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle. When loading and storing shared memory, bank conflicts arise if the same bank is used multiple times within a transaction, resulting in reduced bandwidth.
    The swizzle patterns define the mapping of the 16-byte chunks along the swizzle width to subgroups of four banks.
    The tables define the mapping of the 16-byte chunks along the 128 bytes to eight subgroups of four banks.

    In the below examples, positions with the same x lie in the same bank
    CU_TENSOR_MAP_SWIZZLE_NONE
    ```
    __shared__ int4 smem[8][8];
    smem[y][x] <-> smem[y][x]
    ```
    CU_TENSOR_MAP_SWIZZLE_128B
    ```
    __shared__ __align__(1024) int4 smem[8][8];
    smem[y][x] <-> smem[y][(y % 8) ^ x]
    ```
    CU_TENSOR_MAP_SWIZZLE_64B
    ``` 
    __shared__ __align__(512) int4 smem[4][8];
    smem[y][x] <-> smem[y][(y % 4) ^ x]
    ```
    CU_TENSOR_MAP_SWIZZLE_32B
    ```
    __shared__ __align__(256) int4 smem[2][8];
    smem[y][x] <-> smem[y][(y % 2) ^ x]
    ```

    // ---- Create TMA descriptors ----
    CUresult cuTensorMapEncodeTiled ( CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill)
    enum CUtensorMapDataType: Tensor map data type
    Values:
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    
    enum CUtensorMapFloatOOBfill: Tensor map out-of-bounds fill type
    Values:
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA

    enum CUtensorMapL2promotion: Tensor map L2 promotion type
    Values:
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B

    enum CUtensorMapSwizzle: Tensor map swizzling mode of shared memory banks
    Values:
    CU_TENSOR_MAP_SWIZZLE_NONE = 0
    CU_TENSOR_MAP_SWIZZLE_32B,        // Swizzle 16B chunks within 32B  span
    CU_TENSOR_MAP_SWIZZLE_64B,        // Swizzle 16B chunks within 64B  span
    CU_TENSOR_MAP_SWIZZLE_128B,       // Swizzle 16B chunks within 128B span
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,         // Swizzle 32B chunks within 128B span
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B, // Swizzle 32B chunks within 128B span, additionally swap lower 8B with upper 8B within each 16B for every alternate row
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,         // Swizzle 64B chunks within 128B span

    The box loaded by TMA will be used by mma in terms of atoms. Therefore, TMA setup is also related with the LBO and SBO of the mma instructions that will consume the data
    One core matrix is a matrix with a MN-mode and a K-mode. The chunks lay out along the major mode, and the spans lay out along the minor mode. 
    8 spans along the minor mode form a "core matrix". Each core matrix has a strided direction and a contiguous (leading) direction, 
    such that its length is 8 in the strided direction and 16 bytes in the contiguous direction. An Atom is composed of core matrices along the major mode.
    
    | Swizzling Mode	| Major-ness	| Atom Layout: MN-mode x K-mode |
    |----------------|-------------------|---------------------------|
    | 128B| MN| 128B × 8|
    | 128B| K  | 8 × 128B|
    | 64B | MN| 64B × 8 |
    | 64B | K  | 8 × 64B |
    | 32B | MN| 32B × 8 |
    | 32B | K  | 8 × 32B |
    | None| MN| 16B × 8 |
    | None| K  | 8 × 16B |

    Below uses M as example, N is the same.
    K-Major descriptor under 128B swizzled layouts. Just replace 128 with 64 or 32 for other swizzling mode
    ```
    constexpr uint32_t BOX_MMODE_DIM = BLOCK_M;
    constexpr uint32_t BOX_KMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
    constexpr uint32_t SBO = 8 * 128U; // 8 spans along M-mode (the strided dimension)
    // LBO is not used in the all swizzled case because wgmma's K-mode has 32B, which equals the smallest swizzling bytes

    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        K, M,
        BOX_KMODE_DIM, BOX_MMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```

    K-Major descriptor under non-swizzled layout:
    ```
    constexpr uint32_t BOX_MMODE_DIM = BLOCK_M;
    constexpr uint32_t BOX_KMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
    constexpr uint32_t SBO = 8 * 16U; // 8 spans along M-mode
    constexpr uint32_t LBO = BOX_MMODE_DIM * 16U; // Each core matrix has 16 bytes in the leading dimension
    
    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        K, M,
        BOX_KMODE_DIM, BOX_MMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```
    For M-Major the situation can be little bit more confusing because LBO and SBO mean visually different things for Swizzle vs no Swizzle case.
    MN-Major descriptor under 128B swizzled layouts. Just replace 128 with 64 or 32 for other swizzling mode
    ```
    constexpr uint32_t BOX_KMODE_DIM = BLOCK_K;
    constexpr uint32_t BOX_MMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
    constexpr uint32_t SBO = 8 * 128U; // 8 spans along K-mode
    constexpr uint32_t LBO = BOX_KMODE_DIM * 128U // The spans lay out along the minor mode. LBO jumps to the next groups of spans
    create_tma_2d_descriptor_2B(    
        &desc,
        ptr,
        M, K,
        BOX_MMODE_DIM, BOX_KMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```

    MN-Major descriptor under non-swizzled layouts
    ```
    constexpr uint32_t BOX_KMODE_DIM = BLOCK_K;
    constexpr uint32_t BOX_MMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
    constexpr uint32_t LBO = 8 * 16U; // 8 spans along K-mode
    constexpr uint32_t SBO = BOX_KMODE_DIM * 16U; // Switch LBO and SBO compared with the swizzle layouts

    create_tma_2d_descriptor_2B(
        &desc,
        ptr,
        M, K,
        BOX_MMODE_DIM, BOX_KMODE_DIM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    ```
    

    // The descriptor lives on the host stack and is passed by value. The __grid_constant__ attribute tells the compiler to place it in constant memory accessible to TMA hardware. Passing a plain struct by value WITHOUT __grid_constant__ causes illegal memory access. 
    // ---- Kernel signature ----
    __global__ void cta_gemm_kernel( 
        const __grid_constant__ CUtensorMap tma_A, ...)
    {
        // Use &tma_A directly — it's in grid-constant memory, accessible to TMA        
        tma_load_2d_fn(&tma_A, &mbar[s], smem_dst, coord0, coord1);    
        ...
    }

    // ---- Host setup ----      
    void run(tvm::ffi::TensorView A, ...) {     
        ...
        // 1. Create TMA descriptors for A
        CUtensorMap dA;
        __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
        create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, BOX_KMODE_DIM, BOX_MMODE_DIM, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        ...
        // 2. CTA Launch — pass by value, no cudaMalloc needed
        cta_gemm_kernel<<<grid, block, 0, stream>>>(dA, ...);
    }
    Note: `create_tma_2d_descriptor_2B` returns `CUresult`, not `cudaError_t`. Use a separate check macro.
*/
CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        dataType,
        2, // tensorRank
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, // BE CAREFUL: inner_box_bytes ≤ swizzle_size (for bf16 and CU_TENSOR_MAP_SWIZZLE_128B boxDim[0] ≤ 64). boxDim array specifies number of elements to be traversed along each of the tensorRank dimensions. Elements in boxDim must be non-zero, less than or equal to 256. These dimensions must 16 byte-aligned
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    // Prefetch TMA descriptor.
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

/*
    cp.async.bulk.tensor is a non-blocking instruction which initiates an asynchronous copy operation of tensor data from the location in .src state space to the location in the .dst state space.

    // global -> shared::cta
    cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
         [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}

    .dst =       { .shared::cta }
    .src =       { .global }
    .dim =       { .1d, .2d, .3d, .4d, .5d }
    .completion_mechanism = { .mbarrier::complete_tx::bytes }
    .cta_group = { .cta_group::1, .cta_group::2 } // Default is .cta_group::1
    .load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 } // Default is .tile
    .level::cache_hint =    { .L2::cache_hint }


    // global -> shared::cluster
    cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
         [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}
         {, ctaMask} {, cache-policy}

    .dst =       { .shared::cluster }
    .src =       { .global }
    .dim =       { .1d, .2d, .3d, .4d, .5d }
    .completion_mechanism = { .mbarrier::complete_tx::bytes }
    .cta_group = { .cta_group::1, .cta_group::2 } // Default is .cta_group::1
    .load_mode = { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 } // Default is .tile
    .level::cache_hint =    { .L2::cache_hint }
    .multicast = { .multicast::cluster  }

    The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation. The tensorMap is accessed in tensormap proxy. 
    The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory. 
    The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk.tensor variant uses mbarrier based completion mechanism. Upon the completion of the asynchronous copy operation, the complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.
    The optional qualifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit ctaMask operand corresponds to the %cluster_ctarank of the destination CTA. The source data is multicast to the same offset as dstMem in the shared memory of each destination CTA. 
    When .cta_group is specified as .cta_group::1, the mbarrier signal is also multicasted to the same offset as mbar in the shared memory of the destination CTA.
    When .cta_group::1 is specified, the mbarrier object mbar that is specified must be in the shared memory of the same CTA as the shared memory destination dstMem.
*/

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    // c0 and c1 are tensorCoords; globalDim={dim0, dim1} → coords={coord_in_dim0, coord_in_dim1}.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}


/*
    // shared::cta -> global
    cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
         [tensorMap, tensorCoords], [srcMem] {, cache-policy}

    .dst =       { .global }
    .src =       { .shared::cta }
    .dim =       { .1d, .2d, .3d, .4d, .5d }
    .completion_mechanism = { .bulk_group }
    .load_mode = { .tile, .tile::scatter4, .im2col_no_offs }
    .level::cache_hint =    { .L2::cache_hint }
*/

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    // TMA store accesses shared memory across async and generic proxies
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}


/*
    cp.async.bulk.commit_group instruction creates a new per-thread bulk async-group and batches all prior cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions satisfying the following conditions into the new bulk async-group:
    - The prior cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions use bulk_group based completion mechanism, and
    - They are initiated by the executing thread but not committed to any bulk async-group.

    If there are no uncommitted cp{.reduce}.async.bulk{.prefetch}{.tensor} instructions then cp.async.bulk.commit_group results in an empty bulk async-group.

    An executing thread can wait for the completion of all cp{.reduce}.async.bulk{.prefetch}{.tensor} operations in a bulk async-group using cp.async.bulk.wait_group.

    There is no memory ordering guarantee provided between any two cp{.reduce}.async.bulk{.prefetch}{.tensor} operations within the same bulk async-group.
*/

__device__ __forceinline__ void tma_store_commit_fn() {
    // TMA store arrive/commit.
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

/*
    cp.async.bulk.wait_group instruction will cause the executing thread to wait until only N or fewer of the most recent bulk async-groups are pending and all the prior bulk async-groups committed by the executing threads are complete. For example, when N is 0, the executing thread waits on all the prior bulk async-groups to complete. Operand N is an integer constant.

    By default, cp.async.bulk.wait_group instruction will cause the executing thread to wait until completion of all the bulk async operations in the specified bulk async-group. A bulk async operation includes the following:

    - Optionally, reading from the tensormap.
    - Reading from the source locations.
    - Writing to their respective destination locations.
    - Writes being made visible to the executing thread.
*/

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    // cp.async.bulk.wait_group requires immediate operand, use switch for common values
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

/*
    // global -> shared::cluster
    cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}
  [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}

    .dst =       { .shared::cluster }
    .src =       { .global }
    .completion_mechanism = { .mbarrier::complete_tx::bytes }
    .level::cache_hint =    { .L2::cache_hint }
    .multicast = { .multicast::cluster }

    // shared::cta -> global
    cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}
  [dstMem], [srcMem], size {, cache-policy} {, byteMask}

    .dst =       { .global }
    .src =       { .shared::cta }
    .completion_mechanism = { .bulk_group }
    .level::cache_hint =    { .L2::cache_hint }

    cp.async.bulk is a non-blocking instruction which initiates an asynchronous bulk-copy operation from the location specified by source address operand srcMem to the location specified by destination address operand dstMem.
    The 32-bit operand size specifies the amount of memory to be copied, in terms of number of bytes. size must be a multiple of 16. If the value is not a multiple of 16, then the behavior is undefined. The memory range [dstMem, dstMem + size - 1] must not overflow the destination memory space and the memory range [srcMem, srcMem + size - 1] must not overflow the source memory space. Otherwise, the behavior is undefined. The addresses dstMem and srcMem must be aligned to 16 bytes.
    When the destination of the copy is .shared::cta the destination address has to be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined.
    The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk variant uses mbarrier based completion mechanism. The complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.
    The modifier .bulk_group specifies that the cp.async.bulk variant uses bulk [async-group](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group) based completion mechanism.
    The copy operation in cp.async.bulk is treated as a weak memory operation and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope. The copy operation is performed in the async proxy.
*/

__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
}

__device__ __forceinline__ void tma_copy_1d_s2g_fn(void const* smem, void* gmem, int32_t bytes) {
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :: "l"(gmem), "r"(smem_int), "r"(bytes) : "memory");
}

/*
    The warpgroup level matrix multiply and accumulate operation has either of the following forms,
    where matrix `D` is called accumulator:

    - D = A * B + D
    - D = A * B, where the input from accumulator D is disabled.

    The `wgmma` instructions perform warpgroup level matrix multiply-and-accumulate operation by
    having all threads in a warpgroup collectively perform the following actions:

    1. Load matrices A, B and D into registers or into shared memory.
    2. Perform the following `fence` operations:
        - `wgmma.fence` operations to indicate that the register/shared-memory across the warpgroup have been written into.
        - `fence.proxy.async` operation to make the generic proxy operations visible to the async proxy.
    3. Issue the asynchronous matrix multiply and accumulate operations using the `wgmma.mma_async` operation on the input matrices. The `wgmma.mma_async` operation is performed in the async proxy.
    4. Create a wgmma-group and commit all the prior outstanding `wgmma.mma_async` operations into the group, by using `wgmma.commit_group` operation.
    5. Wait for the completion of the required wgmma-group.
    6. Once the wgmma-group completes, all the `wgmma.mma_async` operations have been performed and completed.

    All the wgmma instructions have to be executed in a warpgroup (128 threads), not just a warp (32 threads).
*/

__device__ __forceinline__ void wgmma_fence_fn() {
    // WGMMA fence synchronization. Ensure memory operations complete before WGMMA reads
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_fn() {
    // WGMMA commit group. Group WGMMA operations for collective waiting
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void wgmma_wait_fn() {
    // WGMMA wait group. Block until N groups remain pending
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void wgmma_fence_operand_fn(float& r) {
    // Prevent compiler from optimizing away accumulator dependency.
    asm volatile("" : "+f"(r) :: "memory");
}

__device__ __forceinline__ void wgmma_fence_operand_array_fn(float* a, int n) {
    // Fence all registers in an accumulator array.
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a[i]) :: "memory");
    }
}

/*

    wgmma.mma_async.sync.aligned.shape.dtype.bf16.bf16  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;
    wgmma.mma_async.sync.aligned.shape.dtype.bf16.bf16  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;
    .shape   = {.m64n{N}k16 where N={8, 16, 32, ..., 256}};
    .dtype   = {.f16, .f32};

    The 5 immediate operands:
      scale-d     = 0  — Overwrite mode (D = A×B); 1 — Accumulate mode (D = A×B + D)
      imm-scale-a = 1  — No scaling; The valid values of imm-scale-a and imm-scale-b are -1 and 1
      imm-scale-b = 1  — No scaling
      imm-trans-a = 0  — core matrix A is K-major (would be 1 for MN-major)
      imm-trans-b = 0  — core matrix B is K-major (would be 1 for MN-major)
    
    Note that imm-trans-a applies only when A comes from shared memory. For register-sourced A in wgmma.m64n{N}k16,
    the per-thread A fragment follows the same 64x16 warpgroup-distributed logical coordinate pattern as the
    accumulator tile of wgmma.m64n16k16, with each pair of bf16 A elements packed into one 32-bit register.
    Refer to `get_d_coord_` below for the register-fragment coordinate mapping.

    Note that tensor core and tma NEVER do transpose! trans-a and trans-b tell the tensor core how to interpret the core matrices in SMEM.
    Such information is used to decide the correct LBO and SBO, and doesn't imply whether the original tensor is transposed or not.
*/

/*
wgmma descriptor bit layout:
[13:0]   — matrix-descriptor-encode(Matrix start address)
[29:16]  — matrix-descriptor-encode(LBO (Leading Byte Offset))
[45:32]  — matrix-descriptor-encode(SBO (Stride Byte Offset))
[51:49]  - Matrix base offset. This is valid for all swizzling modes except the no-swizzle mode.
[63:62]  — Swizzle mode: 0=NONE, 1=128B, 2=64B, 3=32B // WGMMA requires CU_TENSOR_MAP_SWIZZLE_128B for TMA and swizzle_mode=1 for the descriptor to match.
where matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4

The value of base offset is 0 when the repeating pattern of the specified swizzling mode starts as per the below table:

| Swizzling mode | Starting address of the repeating pattern |
|---|---|
| 128-Byte swizzle | 1024-Byte boundary |
| 64-Byte swizzle | 512-Byte boundary |
| 32-Byte swizzle | 256-Byte boundary |

Otherwise, the base offset must be a non-zero value, computed using the following formula: base offset = (pattern start addr >> 0x7) & 0x7
This is used to resolve SMEM alignment problems in case SMEM addresses are not aligned to the byte boundary of the repeating pattern for the swizzle mode.

Below uses N as example, M is the same.
K-Major descriptor under 128B swizzled layouts
```
BOX_MMODE_DIM = BLOCK_M;
BOX_KMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
SBO = 8 * 128 = 1024; // 8 spans along M-mode (core matrix)
LBO = 1 # Not used, assumed to be 1.

```
MN-Major descriptor under 128B swizzled layouts
```
BOX_KMODE_DIM = BLOCK_K;
BOX_MMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
SBO = 128 * 8 = 1024; // 8 spans along K-mode (core matrix)
LBO = (BOX_KMODE_DIM / 8) * SBO; // Remember each element in a row is 128bits (16B) when calculating LBO ("offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows."), so LBO is essentially 128B x (# of spans lined up along K-mode). In other words, SBO x (# of core matrices lined up along K-mode)
```

K-Major descriptor under non-swizzled layout
```
BOX_MMODE_DIM = BLOCK_M;
BOX_KMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
SBO = 8 * 16 = 128; // 8 spans along M-mode
LBO = (BOX_MMODE_DIM / 8) * SBO; // BLOCK_M spans along M-mode. Each 8 spans compose a core matrix; each core matrix has SBO bytes. There are BOX_MMODE_DIM / 8U core matrices
```

MN-Major descriptor under non-swizzled layouts
```
BOX_KMODE_DIM = BLOCK_K;
BOX_MMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
LBO = 16 * 8 = 128; // 8 spans along K-mode
SBO = (BOX_KMODE_DIM / 8) * LBO; // Switch LBO and SBO compared with the swizzle layouts
```

wgmma just takes matrices described by m64n{N}k16, and does not loop over these dimensions in boxes.
Therefore, MN-Major descriptor under 128B swizzled layouts works for m/n64 at the largest. m/n>64 requires iterating with ptr offsets.
Given swizzle mode, matrix start address, and the matrix base offset, wgmma can figure out the ID of the starting chunk. 
Therefore, a K-major 128B swizzle tile contains eight 16B chunks, and each k16 wgmma consumes only two chunks; 
the user only needs to provide the descriptor whose matrix start address points to the desired logical K slice, 
while keeping the swizzle mode and matrix base offset consistent with how the data was laid out in shared memory.
*/

__device__ static inline uint64_t make_wgmma_desc(void* ptr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle_mode) {
    uint64_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t desc = 0;
    desc |= (addr & 0x3FFFF) >> 4; // [13:0] address
    desc |= ((uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16);  // [29:16] LBO
    desc |= ((uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32);  // [45:32] SBO
    desc |= ((uint64_t)swizzle_mode << 62); // [63:62] swizzle
    return desc;
}

/*
Shared-A/shared-B 64n64k16 WGMMA. For other shapes, change the number of output registers
Computes D = A * B + D for logical shapes:
A: 64x16 bf16, B: 16x64 bf16, D: 64x64 f32.
Each of the 128 warpgroup threads owns 32 f32 accumulator registers.
float acc[32];
for(int _i=0; _i<32; _i++) acc[_i]=0.0f; 
*/
template<int trans_a, int trans_b>
__device__ __forceinline__ void wgmma_m64n64k16_ss_fn(float* c, uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32,%33,p,1,1,%34,%35;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),
          "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),
          "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),
          "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
        : "l"(da), "l"(db), "n"(trans_a), "n"(trans_b));
}

/*
Register-A/shared-B 64n64k16 WGMMA. For other shapes, change the number of output registers
A is provided as a warpgroup-distributed register fragment.
Each thread supplies 4 packed b32 registers, containing 8 bf16 A elements.
B is provided through a WGMMA shared-memory descriptor.
Each thread owns 32 f32 accumulator registers for D.
*/
template<int trans_b>
__device__ __forceinline__ void wgmma_m64n64k16_rs_fn_(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35}, %36, p, 1, 1, %37;\n}\n"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "n"(trans_b));
}

/*
wgmma accumulator D's layout can be viewed as a hierachical tiling. Tile shape's unit is an element. Tiles are layed out in row-major and there are 64 rows because m=64.
Layer 0: 16 rows x 8 cols, indexed by [t2, r2]
Layer 1: 8 rows x 2 cols, indexed by [r1, t0]
Layer 2: 1 row x 1 col, indexed by [t1, r0]
Layer i-1 is composed by Layer i tiles layed out in row-major
*/
__device__ __forceinline__ void get_d_coord_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}

__device__ __forceinline__ void store_acc_global_n256_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    // Store WGMMA accumulator to global memory for n256.
    #pragma unroll
    for (int r = 0; r < 128; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}

__device__ __forceinline__ void store_acc_global_n64_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    // Store WGMMA accumulator to global memory for n64.
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}

/*
# ==============================================================================
# Multi-accumulator support for 2x2 tiling (128x128 with 4x m64n64k16)
# ==============================================================================
*/

/*
// Each thread holds 4 x 32 accumulators in a 2x2 layout:
float acc_00[32];
float acc_01[32];
float acc_10[32];
float acc_11[32];
for(int _i=0;_i<32;_i++){acc_00[_i]=0.0f;acc_01[_i]=0.0f;acc_10[_i]=0.0f;acc_11[_i]=0.0f;};
*/

__device__ __forceinline__ void wgmma_fence_4acc_fn(float* a0, float* a1, float* a2, float* a3, int n) {
    // This forces the compiler to treat each accumulator register as live at this point — it cannot reorder, eliminate, or coalesce the registers across this barrier
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a0[i]), "+f"(a1[i]), "+f"(a2[i]), "+f"(a3[i]) :: "memory");
    }
}

// Store 4 m64n64 accumulators (2x2 layout) to global float32.
__device__ __forceinline__ void store_4acc_f32_fn(
    float* C, float* a00, float* a01, float* a10, float* a11,
    int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_d_coord_(tid, r, lm, ln);
        if (bm + lm < M && bn + ln < N)
            C[(int64_t)(bm + lm) * N + bn + ln] = a00[r];
        if (bm + lm < M && bn + 64 + ln < N)
            C[(int64_t)(bm + lm) * N + bn + 64 + ln] = a01[r];
        if (bm + 64 + lm < M && bn + ln < N)
            C[(int64_t)(bm + 64 + lm) * N + bn + ln] = a10[r];
        if (bm + 64 + lm < M && bn + 64 + ln < N)
            C[(int64_t)(bm + 64 + lm) * N + bn + 64 + ln] = a11[r];
    }
}

// Store WGMMA accumulator to SMEM for n256.
__device__ __forceinline__ void store_acc_smem_bf16_n256_fn(
    __nv_bfloat16* sC, float* ac, int ltid, int row_offset) {
    int warp = ltid >> 5;
    int lane_id = ltid & 31;
    int row0 = row_offset + warp * 16 + (lane_id >> 2);
    int row1 = row0 + 8;
    int col_base = (lane_id & 3) * 2;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int col = col_base + i * 8;
        sC[row0 * 256 + col + 0] = __float2bfloat16(ac[i * 4 + 0]);
        sC[row0 * 256 + col + 1] = __float2bfloat16(ac[i * 4 + 1]);
        sC[row1 * 256 + col + 0] = __float2bfloat16(ac[i * 4 + 2]);
        sC[row1 * 256 + col + 1] = __float2bfloat16(ac[i * 4 + 3]);
    }
}

/*
```
stmatrix
```

Collectively store one or more matrices to shared memory.

Syntax

```
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;
.shape  = {.m8n8};
.num    = {.x1, .x2, .x4};
.ss     = {.shared{::cta}};
.type   = {.b16, .b8};
```

Description
Collectively store one or more matrices across all threads in a warp to the location indicated by
the address operand `p` , in `.shared` state space. If no state space is provided, generic
addressing is used, such that the address in `p` points into `.shared` space. If the generic
address doesn't fall in `.shared` state space, then the behavior is undefined.

The `.shape` qualifier indicates the dimensions of the matrices being loaded. Each matrix element
holds 16-bit or 8-bit data as indicated by the `.type` qualifier.

The values `.x1` , `.x2` and `.x4` for `.num` indicate one, two or four matrices
respectively.

The mandatory `.sync` qualifier indicates that `stmatrix` causes the executing thread to wait
until all threads in the warp execute the same `stmatrix` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same `stmatrix` instruction. In conditionally executed code, an `stmatrix` instruction should only be
used if it is known that all threads in the warp evaluate the condition identically, otherwise the behavior is undefined.

The behavior of `stmatrix` is undefined if all threads do not use the same qualifiers, or if any thread in the warp has exited.

The source operand `r` is a brace-enclosed vector expression consisting of 1, 2, or 4 32-bit
registers as per the value of `.num` . Each component of the vector expression holds a fragment
from the corresponding matrix.

Consecutive instances of row need not be stored contiguously in memory. The eight addresses required
for each matrix are provided by eight threads, depending upon the value of `.num` as shown in the following table. Each address corresponds to the start of a matrix row. Addresses addr0-addr7
correspond to the rows of the first matrix, addresses addr8-addr15 correspond to the rows of the second matrix, and so on.

| ``` .num ```   | Threads 0-7   | Threads 8-15   | Threads 16-23   | Threads 24-31   |
|----------------|---------------|----------------|-----------------|-----------------|
| ``` .x1 ```    | addr0-addr7   | -              | -               | -               |
| ``` .x2 ```    | addr0-addr7   | addr8-addr15   | -               | -               |
| ``` .x4 ```    | addr0-addr7   | addr8-addr15   | addr16-addr23   | addr24-addr31   |

When storing 8x8 matrices, a group of four consecutive threads stores 16 bytes. The matrix addresses
must be naturally aligned accordingly.

Each thread in a warp stores fragments of a row, with thread 0 storing the first fragment from its
register
`r` , and so on. A group of four threads stores an entire row of the matrix as shown in [Figure 107](#mma-stmatrix-fragments) .

> ```
> stmatrix fragment layout for one 8x8 matrix with 16-bit elements:
> 
>   Each thread Tn stores register r to shared memory:
>     Row = n / 4
>     Cols = ((n%4)*2) to ((n%4)*2+1)
> 
>   A group of 4 consecutive threads stores an entire row.
>   When .num = .x2, second matrix stored from next source register.
> ```
When `.num` = `.x2` , the elements of the second matrix are storedd from the next source register
in each thread as per the layout in above table. Similarly, when
`.num` = `.x4` , elements of the third and fourth matrices are stored from the subsequent source registers in each thread.
*/
__device__ __forceinline__ void stsm_x2_fn_(
    __nv_bfloat162 v0, __nv_bfloat162 v1, void* p) {
    // stmatrix is efficient on both Ampere and Hopper
    uint32_t s0 = *reinterpret_cast<uint32_t*>(&v0);
    uint32_t s1 = *reinterpret_cast<uint32_t*>(&v1);
    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(p)), "r"(s0), "r"(s1));
}

// Store WGMMA accumulator to SMEM with 128B swizzle for n256.
/*
int32_t tid = threadIdx.x;
int32_t wg = (tid / 128);
int32_t ltid = (tid % 128);
int32_t lane = (tid % 32);
int32_t warp_in_wg = (ltid / 32);
*/
__device__ __forceinline__ void store_accum_n256_swizzle_fn(
    __nv_bfloat16* D_smem, float* acc, int warp_in_wg, int lane, int m_offset) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int ao = i / 8, iao = i % 8;
        int row = iao / 8 + lane, col = iao;
        col ^= row % 8;
        uint8_t* sp = reinterpret_cast<uint8_t*>(D_smem) +
            warp_in_wg * (16 * 128) + m_offset * 128 + ao * 128 * 128 + row * 128 + col * 16;
        __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[i * 4], acc[i * 4 + 1]);
        __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[i * 4 + 2], acc[i * 4 + 3]);
        stsm_x2_fn_(v0, v1, sp);
    }
}


Below are some common errors and solutions.
# Common Causes of Incorrect Results
1. Wrong wgmma descriptor stride: Revise the tma doc

2. Swizzle mismatch between TMA and wgmma descriptor

3. Missing fence.proxy.async (manual loads before WGMMA)
```cpp
// WRONG
smem[i] = global[i];
__syncthreads();
wgmma_m64n64k16_fn_(acc, da, db);  // WGMMA may see stale data!
// CORRECT
smem[i] = global[i];
__syncthreads();
fence_proxy_async_fn()
__syncwarp();
wgmma_fence_fn();
wgmma_m64n64k16_fn_(acc, da, db);
```

4. Barrier race conditions
```cpp
// WRONG: empty arrives before math finishes
mbarrier_arrive_fn(empty_barriers[s]);
wgmma_wait_fn();  // Race!
// CORRECT
wgmma_wait_fn();
mbarrier_arrive_fn(empty_barriers[s]);
```

# Execution timeout
Hangs and Deadlocks
Cause: Barrier expected arrivals don't match actual arrivals.
Debug:
```cpp
  // Add debug prints (disable in production!)
  if (tid == 0) {
      printf("Block %d: init barrier %d with %d arrivals\n",
             blockIdx.x, s, expected_arrivals);
}
```
Common mistakes: - init_smem_barrier_fn(empty_barriers[s], 4) but only 3 consumer warps call arrive() - Cluster barriers initialized with wrong count - Not all threads reach arrive() due to early exit.

# Register Spill
Symptom: Performance drops dramatically
Causes: 1. Too many accumulators live simultaneously 2. Loop unrolling creates too many variables 3. Missing #pragma unroll causing inefficient code
Solutions: 1. Process tiles sequentially with accumulator reuse 2. Use #pragma nounroll where appropriate 3. Reduce NUM_MATH_REGS request

Debug hints end.



# Optimization Pattern: Postpone the division
\begin{algorithm}[t]
\caption{\textsc{FlashAttention-2} forward pass}
\label{alg:flashattention2-forward}
\begin{algorithmic}[1]
\Require Matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ in HBM, block sizes $B_c, B_r$.
\State Divide $\mathbf{Q}$ into $T_r = \left\lceil \frac{N}{B_r} \right\rceil$ blocks $\mathbf{Q}_1, \ldots, \mathbf{Q}_{T_r}$ of size $B_r \times d$ each, and divide $\mathbf{K}, \mathbf{V}$ into $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks $\mathbf{K}_1, \ldots, \mathbf{K}_{T_c}$ and $\mathbf{V}_1, \ldots, \mathbf{V}_{T_c}$, of size $B_c \times d$ each.
\State Divide the output $\mathbf{O} \in \mathbb{R}^{N \times d}$ into $T_r$ blocks $\mathbf{O}_i, \ldots, \mathbf{O}_{T_r}$ of size $B_r \times d$ each, and divide the logsumexp $L$ into $T_r$ blocks $L_i, \ldots, L_{T_r}$ of size $B_r$ each.
\For{$1 \leq i \leq T_r$}
    \State Load $\mathbf{Q}_i$ from HBM to on-chip SRAM.
    \State On chip, initialize $\mathbf{O}_i^{(0)} = (0)_{B_r \times d} \in \mathbb{R}^{B_r \times d}$, $\ell_i^{(0)} = (0)_{B_r} \in \mathbb{R}^{B_r}$, $m_i^{(0)} = (-\infty)_{B_r} \in \mathbb{R}^{B_r}$.
    \For{$1 \leq j \leq T_c$}
        \State Load $\mathbf{K}_j, \mathbf{V}_j$ from HBM to on-chip SRAM.
        \State On chip, compute $\mathbf{S}_i^{(j)} = \mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c}$.
        \State On chip, compute $m_i^{(j)} = \max(m_i^{(j-1)}, \mathrm{rowmax}(\mathbf{S}_i^{(j)})) \in \mathbb{R}^{B_r}$, $\tilde{\mathbf{P}}_i^{(j)} = \exp(\mathbf{S}_i^{(j)} - m_i^{(j)}) \in \mathbb{R}^{B_r \times B_c}$ pointwise, $\ell_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} \ell_i^{(j-1)} + \mathrm{rowsum}(\tilde{\mathbf{P}}_i^{(j)}) \in \mathbb{R}^{B_r}$.
        \State On chip, compute $\mathbf{O}_i^{(j)} = \mathrm{diag}(e^{m_i^{(j-1)} - m_i^{(j)}})^{-1}\mathbf{O}_i^{(j-1)} + \tilde{\mathbf{P}}_i^{(j)}\mathbf{V}_j$.
    \EndFor
    \State On chip, compute $\mathbf{O}_i = \mathrm{diag}(\ell_i^{(T_c)})^{-1}\mathbf{O}_i^{(T_c)}$.
    \State On chip, compute $L_i = m_i^{(T_c)} + \log(\ell_i^{(T_c)})$.
    \State Write $\mathbf{O}_i$ to HBM as the $i$-th block of $\mathbf{O}$.
    \State Write $L_i$ to HBM as the $i$-th block of $L$.
\EndFor
\State \Return the output $\mathbf{O}$ and the logsumexp $L$.
\end{algorithmic}
\end{algorithm}

# Optimization Pattern: Work partition between warps
Split Q across 4 warps while keeping K and V accessible by all warps. After each warp performs matrix multiply to get a slice of QK⊤, they just need to multiply with their shared slice of V to get their corresponding slice of the output. There is no need for communication between warps. The reduction in shared memory reads/writes could yield speedup


