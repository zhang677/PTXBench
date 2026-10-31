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

#### [9.7.16.1. Tensor Memory](#tensor-memory)

The 5 th generation TensorCore has dedicated on-chip memory that is specialized for use by TensorCore operations. This Tensor Memory is organized as a two-dimensional matrix where the horizontal rows are called lanes and the vertical columns are called columns.

On architecture `sm_100a` / `sm_100f` , the 5 th generation TensorCore's Tensor Memory has a two-dimensional structure of 512 columns and 128 rows per CTA, with each cell being 32-bits in size.

Restrictions on threads accessing the Tensor Memory via the load and store operations are specified in [Access restrictions](#tcgen05-tensor-memory-ld-st-access-restrictions) .

##### [9.7.16.1.1. Tensor Memory Addressing](#tensor-memory-addressing)

Tensor Memory addresses are 32-bit wide and specify two components.

1. Lane index
2. Column index
The layout is as follows:

| 31~16   | 15~0   |
|------------------------------|------------------------------|
| Lane index      | Column index    |

[Figure 182](#tensor-memory-layout) shows the view of the Tensor Memory Layout within CTA.

Figure 182 Tensor Memory Layout and Addressing
> ```
> Tensor Memory Layout and Addressing within CTA:
> 
>   Tensor Memory is organized as lanes × columns:
>   - 128 lanes (rows), indexed by lane index
>   - Each (lane, column) cell holds one 32-bit value
>   - At most 512 columns (2KB) per row
> 
>   Address format (32-bit):
>   ┌────────────────────┬────────────────────┐
>   │ Bits [31:16]       │ Bits [15:0]        │
>   │ Lane index         │ Column index       │
>   └────────────────────┴────────────────────┘
> ```

##### [9.7.16.1.2. Tensor Memory Allocation](#tensor-memory-allocation)

The Tensor Memory is dynamically allocated. The Tensor Memory must be allocated by a single warp in a CTA using the [Tensor Memory Allocation and Management Instructions](#tcgen05-memory-alloc-manage-instructions) .

The allocation and deallocation of [Tensor Memory](#tensor-memory) is performed in terms of columns. The unit of allocation is 32 columns and the number of columns being allocated must be a power of 2. When a column is allocated, all 128 lanes of the column are allocated.

All of the Tensor Memory that was allocated in a kernel, must be explicitly deallocated before the kernel exits.

#### [9.7.16.4. Matrix Descriptors](#tcgen05-matrix-descriptors)

##### [9.7.16.4.1. Shared memory descriptor](#tcgen05-shared-memory-descriptor)
The shared memory descriptor describes the properties of multiplicand matrix in shared memory including its location in the shared memory of the current *CTA* . It is a 64-bit value contained in a register with the following layout:

| Bit-field   |   Size in bits | Description |
|-------------|----------------|-------------------|
| 0-13 |   14 | matrix-descriptor-encode (Matrix start address)  |
| 16-29|   14 | matrix-descriptor-encode (Leading dimension byte offset relative(LBO)) |
| 32-45|   14 | matrix-descriptor-encode (Stride dimension byte offset (SBO))      |
| 46-48|    3 | Fixed constant value of 0b001   |
| 49-51|    3 | Matrix base offset|
| 52|    1 | Leading dimension stride mode: - 0: byte offset relative |
| 53-60|    8 | Fixed constant value of 0 |
| 61-63|    3 | Specifies the swizzling mode to be used:  0. No swizzling  1. 128-Byte with 32B atomic swizzling  2. 128-Byte swizzling  4. 64-Byte swizzling  6. 32-Byte swizzling  Note: Values 3, 5 and 7 are invalid |
where matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4

The value of base offset is 0 when the repeating pattern of the specified swizzling mode starts as per shown in the table below

| Swizzling mode   | Starting address of the repeating pattern   |
|------------------|---------------------------------------------|
| 128-Byte swizzle | 1024-Byte boundary     |
| 64-Byte swizzle  | 512-Byte boundary      |
| 32-Byte swizzle  | 256-Byte boundary      |

Otherwise, the base offset must be a non-zero value, computed using the following formula: `base offset = (pattern start addr >> 0x7) & 0x7`
The following must be 16-byte aligned:
1. Matrix start address
2. Leading dimension byte offset
3. Stride dimension byte offset

One Atom is a matrix with a MN-mode and a K-mode. A "row" is an array with the same index at MN-mode. A "column" is an array with with same index at K-mode.
The chunks lay out along the major mode, and the spans lay out along the minor mode. 8 spans along the minor mode form a "core matrix".

LBO and SBO are defined as follows for matrices whose element types are normalized to 128-bits. In the tcgen05 instruction descriptor, "No Transpose" means K-Major and "Transpose" means MN-Major.
[Leading dimension byte offset relative](LBO)

| Major-ness   | Case | Definition|
|--------------|------|-------------|
| K-Major      | No-Swizzling | offset from the first column to the second columns of the 8x2 tile in the 128-bit element type normalized matrix. |
| K-Major      | Swizzling | Not used, assumed to be 1. |
| MN-Major     | No-Swizzling | offset from the first 8 columns to the next 8 columns. |
| MN-Major     | Swizzling | offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows.  |

[Stride dimension byte offset](SBO)

| Major-ness   | Case | Definition|
|--------------|------|--------|
| K-Major      | All | The offset from the first 8 rows to the next 8 rows.  |
| MN-Major     | No-Swizzling | offset from the first row to the next row. |
| MN-Major     | Swizzling | offset from the first 8 columns to the next 8 columns |


| Swizzling Mode	| Major-ness	| Atom Layout: MN-mode x K-mode |
|----------------|-------------------|---------------------------|
| 128B Swizzling with 32B atomicity | MN only | 128B x 4|
| 128B| MN| 128B × 8|
| 128B| K  | 8 × 128B|
| 64B | MN| 64B × 8 |
| 64B | K  | 8 × 64B |
| 32B | MN| 32B × 8 |
| 32B | K  | 8 × 32B |
| None| MN| 16B × 8 |
| None| K  | 8 × 16B |

Below uses N as example, M is the same.
K-Major descriptor under 128B swizzled layouts
```
ATOM_MMODE_DIM = BLOCK_M;
ATOM_KMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
SBO = 8 * 128 = 1024; // 8 spans along M-mode (core matrix)
LBO = 1 # Not used, assumed to be 1.

```
MN-Major descriptor under 128B swizzled layouts
```
ATOM_KMODE_DIM = BLOCK_K;
ATOM_MMODE_DIM = 128 / sizeof(nv_bfloat16) = 64; // Span size = 128B
SBO = 128 * 8 = 1024; // 8 spans along K-mode (core matrix)
LBO = (ATOM_KMODE_DIM / 8) * SBO; // Remember each element in a row is 128bits (16B) when calculating LBO ("offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows."), so LBO is essentially 128B x (# of spans lined up along K-mode). In other words, SBO x (# of core matrices lined up along K-mode)
```

K-Major descriptor under non-swizzled layout
```
ATOM_MMODE_DIM = BLOCK_M;
ATOM_KMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
SBO = 8 * 16 = 128; // 8 spans along M-mode
LBO = (ATOM_MMODE_DIM / 8) * SBO; // BLOCK_M spans along M-mode. Each 8 spans compose a core matrix; each core matrix has SBO bytes. There are ATOM_MMODE_DIM / 8U core matrices
```

MN-Major descriptor under non-swizzled layouts
ATOM_KMODE_DIM = BLOCK_K;
ATOM_MMODE_DIM = 16 / sizeof(nv_bfloat16) = 8; // Span size = 16B
LBO = 16 * 8 = 128; // 8 spans along K-mode
SBO = (ATOM_KMODE_DIM / 8) * LBO; // Switch LBO and SBO compared with the swizzle layouts

For cta_group::1, umma just takes in the atom described by {ATOM_KMODE_DIM, ATOM_MMODE_DIM} (K-major) or {ATOM_MMODE_DIM, ATOM_KMODE_DIM} (MN-major), and does not loop over these dimensions.
For cta_group::2, umma takes in doubled atom dimensions, but the descriptor is the same for both CTAs as they have the same layout and access pattern, just different tiles.


##### [9.7.16.4.2. Instruction descriptor](#tcgen05-instruction-descriptor)

The instruction descriptor describes the shapes, types and other details of all the matrices and the matrix-multiplication-and-accumulation operation. It is a 32-bit value in registers and the exact layout is dependent on the MMA-Kind:

<table>
  <caption>
    Table 42. Instruction descriptor format for
    <code>.kind::tf32</code>, <code>.kind::f16</code>,
    <code>.kind::f8f6f4</code>, and <code>.kind::i8</code>
  </caption>

  <thead>
    <tr>
      <th rowspan="2">Bits</th>
      <th rowspan="2">Size (bits)</th>
      <th rowspan="2">Description</th>
      <th colspan="4">Values</th>
    </tr>
    <tr>
      <th><code>.kind::tf32</code></th>
      <th><code>.kind::f16</code></th>
      <th><code>.kind::f8f6f4</code></th>
      <th><code>.kind::i8</code></th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>0-1</td>
      <td>2</td>
      <td>
        <a href="#tcgen05-sparse-matrices-sparsity-selector">Sparsity selector</a>,
        if Sparsity is enabled
      </td>
      <td colspan="4">0-3</td>
    </tr>

    <tr>
      <td>2</td>
      <td>1</td>
      <td>Sparsity</td>
      <td colspan="4">
        Dense = 0<br>
        Sparse = 1
      </td>
    </tr>

    <tr>
      <td>3</td>
      <td>1</td>
      <td>Saturate for integer types</td>
      <td colspan="3">0 (NA)</td>
      <td>
        No Saturate = 0<br>
        Saturate = 1
      </td>
    </tr>

    <tr>
      <td>4-5</td>
      <td>2</td>
      <td>dtype (Matrix D type)</td>
      <td>F32 = 1</td>
      <td colspan="2">
        F16 = 0<br>
        F32 = 1
      </td>
      <td>S32 = 2</td>
    </tr>

    <tr>
      <td>6</td>
      <td>1</td>
      <td>Reserved</td>
      <td colspan="4">0</td>
    </tr>

    <tr>
      <td>7-9</td>
      <td>3</td>
      <td>atype (Matrix A type)</td>
      <td rowspan="2">TF32 = 2</td>
      <td rowspan="2">
        F16 = 0<br>
        BF16 = 1
      </td>
      <td rowspan="2">
        E4M3 = 0<br>
        E5M2 = 1<br>
        E2M3 = 3<br>
        E3M2 = 4<br>
        E2M1 = 5
      </td>
      <td rowspan="2">
        Unsigned 8b = 0<br>
        Signed 8b = 1
      </td>
    </tr>

    <tr>
      <td>10-12</td>
      <td>3</td>
      <td>btype (Matrix B type)</td>
    </tr>

    <tr>
      <td>13</td>
      <td>1</td>
      <td>Negate A Matrix</td>
      <td rowspan="2" colspan="3">
        No Negate = 0<br>
        Negate = 1
      </td>
      <td rowspan="2">No Negate = 0</td>
    </tr>

    <tr>
      <td>14</td>
      <td>1</td>
      <td>Negate B Matrix</td>
    </tr>

    <tr>
      <td>15</td>
      <td>1</td>
      <td>Transpose A Matrix</td>
      <td rowspan="2" colspan="4">
        No Transpose = 0<br>
        Transpose = 1
      </td>
    </tr>

    <tr>
      <td>16</td>
      <td>1</td>
      <td>Transpose B Matrix</td>
    </tr>

    <tr>
      <td>17-22</td>
      <td>6</td>
      <td>N, Dimension of Matrix B (3 LSBs not included)</td>
      <td colspan="4">N &gt;&gt; 3</td>
    </tr>

    <tr>
      <td>23</td>
      <td>1</td>
      <td>Reserved</td>
      <td colspan="4">0</td>
    </tr>

    <tr>
      <td>24-28</td>
      <td>5</td>
      <td>M, Dimension of Matrix A (4 LSBs not included)</td>
      <td colspan="4">M &gt;&gt; 4</td>
    </tr>

    <tr>
      <td>29</td>
      <td>1</td>
      <td>Reserved</td>
      <td colspan="4">0</td>
    </tr>

    <tr>
      <td>30-31</td>
      <td>2</td>
      <td>
        Maximum shift while attempting B matrix-reuse in <code>.ws</code>
      </td>
      <td colspan="4">
        no shift = 0<br>
        maximum shift of 8 = 1<br>
        maximum shift of 16 = 2<br>
        maximum shift of 32 = 3
      </td>
    </tr>
  </tbody>
</table>

For .kind::tf32, the transpose operations on matrices A and B are supported only with 128B swizzling mode with 32B swizzle-atomicity.

For all other MMA-Kinds, the transpose operations on matrices A and B are not supported on 128B swizzling mode with 32B swizzle-atomicity.

#### [9.7.16.5. Issue Granularity](#tcgen05-issue-granularity)

Each of the `tcgen05` operation has different requirements for the number of
threads/warps that needs to issue them.

The following table lists the execution granularity requirements of each of the `tcgen05` operation:

| tcgen05 operation | .cta_group   | Issue Granularity        |
|-------------------|--------------|-------------------------|
| ``` . mma , . cp , . shift , . commit ```  | ::1| An issue from a single thread in the current CTA would initiate the base operation.   |
|   | ::2| Issue from a single thread from the  [CTA-Pair](#tcgen05-cta-pair)  would initiate  the base operation.  When the current CTA issues the operation, the peer  CTA should be active and should not have exited. |
| ``` . alloc , . dealloc , . relinquish_alloc_permit ``` | ::1| Issue from a single warp in the current CTA would initiate the allocation management instruction.      |
|   | ::2| Issue from two warps, one in each of the current CTA  and its  [Peer CTA](#tcgen05-peer-cta)  , in order to  collectively perform the operation, i.e., the first  warp to perform the operation could block until the  the second warp in the  [Peer CTA](#tcgen05-peer-cta)  also performs the operation (see examples below). |
| ``` . ld , . st , . wait :: { ld , st } ```| N/A| Issue from a warp in the current CTA can access only 1/4 of the Tensor Memory of the current CTA. So, a warpgroup is needed to access the entire Tensor Memory of the current CTA.   |
| ``` . fence ::* ```  | N/A| A thread needs to fence all its accesses to the tensor memory that it wants to order with other accesses to the tensor memory from other threads.      |

The following example shows that:

- Before attempting to deallocate Tensor Memory, it suffices to ensure that there are no concurrent Tensor Memory accesses from the [Peer CTA](#tcgen05-peer-cta) .
- Warps can immediately exit after deallocating Tensor Memory; no extra synchronization required.

| CTA0 Warp| CTA1 Warp|
|--------------|--------------|
| barrier.cluster.arrive; barrier.cluster.wait; tcgen05.dealloc.2cta.sync.aligned; exit; | barrier.cluster.arrive; barrier.cluster.wait; tcgen05.dealloc.2cta.sync.aligned; exit; |

This example uses a cluster barrier for illustration purposes but in practice other
synchronization mechanisms are often used.

The following example illustrates a scenario in which the program exhibits non-deterministic
behavior due to incorrect synchronizaton because
`.dealloc` may or may not block:

| CTA0 Warp| CTA1 Warp|
|--------------|--------------|
| barrier.cluster.arrive; barrier.cluster.wait; tcgen05.dealloc.2cta.sync.aligned; exit; | tcgen05.dealloc.2cta.sync.aligned; barrier.cluster.arrive; barrier.cluster.wait; exit; |

##### [9.7.16.5.1. CTA Pair](#tcgen05-cta-pair)

Any 2 CTAs within the cluster whose `%cluster_ctarank` differs by the last bit only
is said to form a CTA pair.

Within a CTA pair, the CTA whose last bit in the `%cluster_ctarank` is:

- 0 is termed the even numbered CTA within the CTA pair.
- 1 is termed as the odd numbered CTA within the CTA pair.

Most of the `tcgen05` operations can either execute at a single CTA level granularity OR
at a CTA pair level granularity. When a
`tcgen05` operation is performed at CTA pair
granularity, the Tensor Memory of both the CTAs within the CTA pair are accessed. The set
of threads that need to issue the
`tcgen05` operation is listed in the [Issue Granularity](#tcgen05-issue-granularity) .

##### [9.7.16.5.2. Peer CTA](#tcgen05-peer-cta)

The peer CTA of the odd CTA within the CTA pair is the even CTA in the same pair.

Similarly, the peer CTA of the even CTA within the CTA pair is the odd CTA in the same pair.

##### [9.7.16.2.3. Data Movement Shape](#tcgen05-data-movement-shape)

The data movement shape indicates the dimension of the data to be moved to or from the Tensor Memory. These shapes are described as a tuple `lane x size` where:

- `lane` indicates the number of rows in the Tensor Memory; and
- `size` indicates the amount of data, in units of bits (b), across the columns in the Tensor Memory.

The following shapes are supported by various tcgen05 operations:

| Shape      | tcgen05.<op>   |
|-----------------------------|-----------------------------|
| ``` .16x64b ```  ,  ``` .16x128b ```  ,  ``` .16x256b ```  ,  ``` .16x32bx2 ```  ,  ``` .32x32b ```   | ``` .ld ```  /  ``` .st ``` |
| ``` .4x256b ```  ,  ``` .32x128b ```  ,  ``` .64x128b ```  ,  ``` .128x256b ```  ,  ``` .128x128b ``` | ``` .cp ```    |
| ``` .31x256b ```  (implicit)    | ``` .shift ``` |

###### [9.7.16.6.2.1. Implicitly pipelined tcgen05 Instructions](#tcgen05-memory-consistency-model-pipelined-instructions-implicit)

Instructions `tcgen05.commit` and `tcgen05.wait` are implicitly pipelined with respect
to previously issued `tcgen05.{mma,cp,shift}` and `tcgen05.{ld,st}` instructions respectively that they track from the same thread.

9.7.16.6.2.1.1. [mbarrier based completion mechanism](#tcgen05-memory-consistency-model-mbarrier-completion)

Completion of the following instruction's asynchronous operations is observed
through the mbarrier based waiting mechanism:

1. tcgen05.mma
2. tcgen05.cp
3. tcgen05.shift
`tcgen05.commit` is used to track the completion of the above asynchronous instructions.

Following are the implicitly pipelined `tcgen05` instruction pairing that uses mbarrier based completion mechanism:

- `tcgen05.mma.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
- `tcgen05.cp.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
- `tcgen05.shift.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)

##### [9.7.16.8.1. Access restrictions ](#tcgen05-tensor-memory-ld-st-access-restrictions)

Not all threads of the CTA can access the entire Tensor Memory via the `tcgen05.ld` and `tcgen05.st` operations.

The Tensor Memory of a CTA is divided into 4 equal chunks such that each warp of a warpgroup
in the CTA can access a chunk of the Tensor Memory. All the columns of the Tensor Memory can
be accessed by all the four warps of a warpgroup. A lane of the Tensor Memory can be accessed
by a single warp in the warpgroup. The following table describes the access restriction.

|   ID of the warp within the warpgroup | Accessible Lanes   |
|---------------------------------------|--------------------|
|           0 | 0-31  |
|           1 | 32-63 |
|           2 | 64-95 |
|           3 | 96-127|

##### [9.7.16.8.2. Packing and Unpacking ](#tcgen05-tensor-memory-ld-st-packing-unpacking)

Optionally, the following pack and unpack operations can be performed during the load and store:

1. Packing: two 16-bit chunks can be packed into a single 32-bit chunk in the register in `tcgen05.ld`
2. Unpacking: a single 32-bit chunk in the register can be unpacked into two 16-bit chunks in `tcgen05.st`

Below is the hardware parameter of NVIDIA B200 (SM100)
| Component | Value |
|-----------|---------------|
| SM Count | 148 |
| Peak MMA FP8 | ~4500 TFLOPS |
| Peak MMA BF16 | ~2250 TFLOPS (8192 ops/cycle/SM) |
| Peak MUFU (multifunction unit) | 16 ops/cycle/SM |
| Cluster Size | Portable: 8 CTAs; B200 nonportable: up to 16 CTAs |
| Max Registers / CTA | 65536 32-bit|
| Max Registers / Thread | 255 |
| Max CTA size (# of threads) | 1024 |

| Level | Capacity | Latency | Bandwidth |
|-------|----|------|---------|
| Global (HBM) | 180 GB | ~400 cycles | 8 TB/s |
| L2 Cache | 126 MB | ~100 cycles | ~16TB/s |
| Shared Memory | 0, 8, 16, 32, 64, 100, 132, 164, 196, and 228 KB per SM. A single CTA can address up to 227 KB of shared memory as 1 KB is reserved by CUDA | ~20 cycles | 128 B/cycle/SM |
| Tensor Memory (TMEM) | 256 KB (512 columns × 128 rows x 32-bit cells) / SM |  ~10 cycles | Tensor Core internal |
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

// The TMA setup is also related with the LBO (Leading dimension byte offset relative) and SBO (Stride dimension byte offset) of the shared memory descriptor.

Below uses M as example, N is the same. The swizzled layouts are always recommended for better performance
K-Major descriptor under 128B swizzled layouts
```
constexpr uint32_t ATOM_MMODE_DIM = BLOCK_M;
constexpr uint32_t ATOM_KMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
constexpr uint32_t SBO = 8U * 128U; // 8 spans along M-mode

create_tma_2d_descriptor_2B(
    &desc,
    ptr,
    K, M,
    ATOM_KMODE_DIM, ATOM_MMODE_DIM,
    CU_TENSOR_MAP_SWIZZLE_128B, 
    CU_TENSOR_MAP_L2_PROMOTION_NONE, 
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

K-Major descriptor under non-swizzled layout:
```
constexpr uint32_t ATOM_MMODE_DIM = BLOCK_M;
constexpr uint32_t ATOM_KMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
constexpr uint32_t SBO = 8U * 16U; // 8 spans along M-mode
constexpr uint32_t LBO = (ATOM_MMODE_DIM / 8U) * SBO; // BLOCK_M spans along M-mode. Each 8 spans compose a core matrix; each core matrix has SBO bytes. There are ATOM_MMODE_DIM / 8U core matrices

create_tma_2d_descriptor_2B(
    &desc,
    ptr,
    K, M,
    ATOM_KMODE_DIM, ATOM_MMODE_DIM,
    CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```
For M-Major the situation can be little bit more confusing because LBO and SBO mean visually different things for Swizzle vs no Swizzle case.
MN-Major descriptor under 128B swizzled layouts
```
// Swizzle 128B -> stride-1 row slabs in MN-major order
constexpr uint32_t ATOM_KMODE_DIM = BLOCK_K;
constexpr uint32_t ATOM_MMODE_DIM = 128U / sizeof(nv_bfloat16); // Span size = 128B
constexpr uint32_t SBO = 128 * 8U; // 8 spans along K-mode 
constexpr uint32_t LBO = (ATOM_KMODE_DIM / 8U) * SBO; // Remember each element in a row is 128bits (16B) now, so LBO is essentially 128B x (# of spans lined up along K-mode). In other words, SBO x (# of core matrices lined up along K-mode)

create_tma_2d_descriptor_2B(
    &desc,
    ptr,
    M, K,
    ATOM_MMODE_DIM, ATOM_KMODE_DIM,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

MN-Major descriptor under non-swizzled layouts
```
constexpr uint32_t ATOM_KMODE_DIM = BLOCK_K;
constexpr uint32_t ATOM_MMODE_DIM = 16U / sizeof(nv_bfloat16); // Span size = 16B
constexpr uint32_t LBO = 16 * 8U; // 8 spans along K-mode
constexpr uint32_t SBO = (ATOM_KMODE_DIM / 8U) * LBO; // Switch LBO and SBO compared with the swizzle layouts

create_tma_2d_descriptor_2B(
    &desc,
    ptr,
    M, K,
    ATOM_MMODE_DIM, ATOM_KMODE_DIM,
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
    create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
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

// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
        [tensorMap, tensorCoords], [srcMem] {, cache-policy}

.dst =       { .global }
.src =       { .shared::cta }
.dim =       { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .bulk_group }
.load_mode = { .tile, .tile::scatter4, .im2col_no_offs }
.level::cache_hint =    { .L2::cache_hint }

The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation. The tensorMap is accessed in tensormap proxy. 
The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory. 

The modifier `.completion_mechanism` specifies the completion mechanism that is supported on the
instruction variant. The completion mechanisms that are supported for different variants are
summarized in the following table:

| .completion-mechanism   | ``` .dst ```  | ``` .src ``` | Completion mechanism (Needed for completion of entire Async operation) |
|---------|---------|-------------|----------------|
| ``` .mbarrier::... ```  | ``` .shared::cta ```     | ``` .global ```      | mbarrier based     |
| ``` .mbarrier::... ``` | ``` .shared::cluster ``` | ``` .global ```      | mbarrier based       | 
| ``` .bulk_group ```     | ``` .global ```          | ``` .shared::cta ``` | *Bulk async-group*  based    |

The modifier `.cta_group` can only be specified with the mbarrier based completion mechanism. The modifier `.cta_group` is used to signal either the odd numbered CTA or the even numbered CTA among
the [CTA-Pair](#tcgen05-cta-pair). When `.cta_group::1` is specified, the mbarrier object `mbar` that is specified must be in the shared memory of the same CTA as the shared memory destination `dstMem` .
When `.cta_group::2` is specified, the mbarrier object `mbar` can be in shared memory of either the same CTA as the shared memory destination `dstMem` or in its [peer-CTA](#tcgen05-peer-cta) . If `.cta_group` is not specified, then it defaults to `.cta_group::1` .

The modifier `.mbarrier::complete_tx::bytes` specifies that the cp.async.bulk.tensor variant uses mbarrier based completion mechanism. Upon the completion of the asynchronous copy operation, the complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.
    
The optional modifier `.multicast::cluster` allows copying of data from global memory to shared
memory of multiple CTAs in the cluster. Operand `ctaMask` specifies the destination CTAs in the
cluster such that each bit position in the 16-bit `ctaMask` operand corresponds to the `%ctaid` of the destination CTA. The source data is multicast to the same offset as `dstMem` in the shared
memory of each destination CTA. When `.cta_group` is specified as:
- `.cta_group::1` : The mbarrier signal is also multicasted to the same offset as `mbar` in the shared memory of the destination CTA.
- `.cta_group::2` : The mbarrier signal is multicasted either to all the odd numbered CTAs or the even numbered CTAs within the corresponding [CTA-Pair](#tcgen05-cta-pair) . For each destination CTA specified in the `ctaMask` , the mbarrier signal is sent either to the destination CTA or its [peer-CTA](#tcgen05-peer-cta) based on CTAs `%cluster_ctarank` parity of shared memory where the mbarrier object `mbar` resides.

The copy operation in `cp.async.bulk.tensor` is treated as a weak memory operation and the complete-tx operation on the mbarrier has `.release` semantics at the `.cluster` scope.
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    // cta_group::2 TMA 2D load. When the following umma is cta_group::2, it is faster than tma_load_2d in each CTA because it avoids the expensive cluster sync
    // This function doesn't multicast; it just allows the barrier to be in one of the peer CTAs' shared memory.
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; // The cta_group::2 TMA instruction requires the barrier to be specified in the leader's (peer 0) SMEM. By AND-ing with 0xFEFFFFFF, the helper canonicalizes whichever CTA's bar pointer was passed in to the peer-0 address
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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

Dynamic [Tensor Memory](#tensor-memory) allocation management instructions

```
tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32  [dst], nCols;
tcgen05.dealloc.cta_group.sync.aligned.b32  taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
.cta_group = { .cta_group::1, .cta_group::2 }
```

`tcgen05.alloc` is a blocking instruction which dynamically allocates the specified number of columns in the [Tensor Memory](#tensor-memory) and writes
the address of the allocated Tensor Memory into shared memory at the location specified by address operand dst. The `tcgen05.alloc` blocks if the
requested amount of Tensor Memory is not available and unblocks as soon as the requested amount of Tensor Memory becomes available for allocation.

`tcgen05.dealloc` is a potentially blocking instruction which deallocates the Tensor Memory specified by the Tensor Memory address `taddr` . The operand `taddr` must point to a previous Tensor Memory allocation.

If `.cta_group::2` is specified,
- issuing warp and [peer CTA](#tcgen05-peer-cta) warp must synchronize Tensor Memory accesses before attempting to collectively deallocate the Tensor Memory , and
- `tcgen05.dealloc` may block to collectively performs the deallocation with the other peer CTA's warp.

All of the Tensor Memory that was allocated using `tcgen05.alloc` instruction in a kernel must be explicitly deallocated using `tcgen05.dealloc` before the kernel exits.

The unsigned 32-bit operand `nCols` specify the number of columns to be allocated or de-allocated. The unit of allocation and de-allocation is 32 columns and all of lanes
per column. The number of columns must be a power of 2. The operand `nCols` must be within the range [32, 512]. The number of columns allocated should not increase between
any two allocations in the execution order within the CTA.

Instruction `tcgen05.relinquish_alloc_permit` specifies that the CTA of the executing
thread is relinquishing the right to allocate [Tensor Memory](#tensor-memory) . So,
it is illegal for a CTA to perform `tcgen05.alloc` after any of its constituent threads execute `tcgen05.relinquish_alloc_permit` .

Qualifier `.cta_group` specifies the number of CTAs involved in the allocation and de-allocation operation. When `.cta_group::1` is specified, one warp from the CTA must perform the allocation and de-allocation. When
`.cta_group::2` is specified, one warp from each of the peer CTAs must collectively perform the allocation and de-allocation. Refer to the
[Issue Granularity](#tcgen05-issue-granularity) section.

When `.cta_group::2` is specified, the issuing warp must make sure that peer CTA is launched and its warps eventually participate in collective operations.

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group` qualifier.

The mandatory `.sync` qualifier indicates that the instruction causes the executing thread
to wait until all threads in the warp execute the same instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the
same instruction. In conditionally executed code, the instruction should only be used if it
is known that all threads in the warp evaluate the condition identically, otherwise behavior
is undefined.

The behavior of the instruction is undefined if all the threads in the warp do not use the
same values of `nCols` , or if any thread in the warp has exited.

*/

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

/*
Asynchronous collective load from tensor memory into registers.

tcgen05.ld.sync.aligned.shape1.num{.pack}.b32    r, [taddr];
tcgen05.ld.sync.aligned.shape2.num{.pack}.b32    r, [taddr], immHalfSplitoff;
.shape1 = { .16x64b, .16x128b, .16x256b, .32x32b }
.shape2 = { .16x32bx2 }
.num    = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }
.pack   = { .pack::16b }

Instruction `tcgen05.ld` asynchronously loads data from the Tensor Memory at the location specified by the 32-bit address operand `taddr` into the destination register `r`, collectively across all threads of the warps.

All the threads in the warp must specify the same value of `taddr` , which must be the
base address of the collective load operation. Otherwise, the behavior is undefined.

The `.shape` qualifier and the `.num` qualifier together determines the total dimension of the data which is loaded from the Tensor Memory. 
The `.shape` qualifier indicates the base dimension of data to be accessed as described in the [Data Movement Shape](#tcgen05-data-movement-shape). 
The `.num` qualifier indicates the repeat factor on the base dimension resulting in the total dimension of the data that is accessed.

The shape `.16x32bx2` performs two accesses into Tensor Memory of the shape `.16x32b` .
The base address of the first access is specified by taddr and the base address of the
second access is specified by `taddr+immHalfSplitoff` , where `immHalfSplitoff` is an immediate argument.

The destination operand `r` is a brace-enclosed vector expression consisting of one or more 32-bit
registers as per the value of `.shape` and `.num`. The size of the
vector for various combinations of `.num` and `.shape` is shown in [Table 49](#tcgen05-num-shapes-ld) .

| .num   | .shape   | .shape   | .shape   |
|---------------|-------------------------------|----------|----------|
| .num   | .16x32bx2 / .16x64b / .32x32b | .16x128b | .16x256b |
| ``` .x1 ```   | 1  | 2 | 4 |
| ``` .x2 ```   | 2  | 4 | 8 |
| ``` .x4 ```   | 4  | 8 | 16|
| ``` .x8 ```   | 8  | 16| 32|
| ``` .x16 ```  | 16 | 32| 64|
| ``` .x32 ```  | 32 | 64| 128      |
| ``` .x64 ```  | 64 | 128      | NA|
| ``` .x128 ``` | 128| NA| NA|

The optional qualifier `.pack::16b` can be used to pack two 16-bit elements from adjacent
columns into a single 32-bit element during the load as shown in the section
[Packing and Unpacking](#tcgen05-tensor-memory-ld-st-packing-unpacking) .

The mandatory `.sync` qualifier indicates that `tcgen05.ld` causes the executing thread
to wait until all threads in the warp execute the same `tcgen05.ld` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the
same `tcgen05.ld` instruction. In conditionally executed code, a `tcgen05.ld` instruction
should only be used if it is known that all threads in the warp evaluate the condition
identically, otherwise behavior is undefined.

The behavior of `tcgen05.ld` is undefined if all threads do not use the same values of `taddr` ,
or if any thread in the warp has exited.

*/

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    // Load 4×32b from TMEM column
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    // Load 8×32b from TMEM column
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_bf16_row_fn(
    __nv_bfloat16* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    // Read TMEM columns, convert FP32->BF16, store to global D. Each thread reads its own row (m_base + tid) for BN columns starting at n_base.
    uint32_t m_idx = m_base + tid;
    if (m_idx >= M) return;
    for (uint32_t col = 0; col < BN; col += 4) {
 uint32_t r0, r1, r2, r3;
 asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
: "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
 asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
 float f0 = __uint_as_float(r0);
 float f1 = __uint_as_float(r1);
 float f2 = __uint_as_float(r2);
 float f3 = __uint_as_float(r3);
 uint32_t nc = n_base + col;
 __nv_bfloat16* out = D + (uint64_t)m_idx * N + nc;
 if (nc     < N) out[0] = __float2bfloat16(f0);
 if (nc + 1 < N) out[1] = __float2bfloat16(f1);
 if (nc + 2 < N) out[2] = __float2bfloat16(f2);
 if (nc + 3 < N) out[3] = __float2bfloat16(f3);
    }
}

/*
Coalesced epilogue: TMEM -> SMEM staging -> vectorized coalesced global writes.

Phase 1: Each of 128 threads loads its row from TMEM (FP32), converts to BF16,
writes to shared memory. Phase 2: Coalesced 8-byte stores to global D.

REQUIRES blockDim.x == 128 (exactly 4 warps). The row mapping in Phase 2
uses "row = step*4 + warp_id"; other thread counts will produce wrong output.

Args:
    D: Global output pointer (BF16).
    smem_out: BM*BN BF16 buffer in shared memory (caller must allocate).
    M, N: Full matrix dimensions.
    m_block, n_block: Block start indices.
    bm, bn: Block tile sizes (BM, BN).
*/
__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    // Phase 1: TMEM -> SMEM
    // Each of the 128 threads owns one row (tid 0..127 -> row 0..127). Load 4
    // FP32 cols at a time from TMEM, convert to BF16, write to smem_out[tid*BN+col].
    for (uint32_t col = 0; col < BN; col += 4) {
 uint32_t r0, r1, r2, r3;
 asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
: "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
 asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
 uint32_t base = threadIdx.x * BN + col;
 smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
 smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
 smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
 smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    // Phase 2: SMEM -> Global (coalesced vectorized 8-byte writes)
    // REQUIRES 4 warps: each step processes 4 rows (one per warp). row = step*4 + warp_id.
    // Each warp: 32 lanes write 4 BF16 (8 bytes) each, covering 32*4=128 cols per row.
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
 uint32_t row = step * 4 + warp_id;
 if (row >= BM) continue;
 uint32_t global_row = m_block * BM + row;
 uint32_t col_start = lane_id * 4;
 uint32_t global_col = n_block * BN + col_start;
 if (global_row < M && global_col + 3 < N) {
     uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
     *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
 }
    }
}

/*

Waits for the completion of all prior asynchronous `tcgen05.ld` / `tcgen05.st` instructions.

```
tcgen05.wait_operation.sync.aligned;
.wait_operation = { .wait::ld, .wait::st }
```
Instruction `tcgen05.wait::ld`/`tcgen05.wait::st` causes the executing thread to block until all prior `tcgen05.ld`/`tcgen05.st` operations issued by the executing thread have completed.
The mandatory `.sync` and `.aligned` qualifiers have the same semantics as described in `tcgen05.ld` / `tcgen05.st`.

Examples:
```ptx
Example 1:

tcgen05.ld.sync.aligned.32x32b.x2.b32     {r0, r1}, [taddr0];

// Prevents subsequent tcgen05.mma from racing ahead of the tcgen05.ld

tcgen05.wait::ld.sync.aligned;

tcgen05.mma.cta_group::1.kind::f16   [taddr0],  a-desc,  b-desc, idesc, p;
```
*/

__device__ __forceinline__ void tmem_load_fence_fn() {
    // Fence for TMEM loads (tcgen05.wait::ld).
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

/*

Specialized fence for the asynchronous tcgen05 operations.

```
tcgen05.fence::before_thread_sync ;
tcgen05.fence::after_thread_sync  ;
```

The instruction `tcgen05.fence::before_thread_sync` (`tcgen05.fence::after_thread_sync`) orders all the prior (subsequent) asynchronous `tcgen05` operations with respect to the subsequent (prior) `tcgen05` and the execution
ordering operations.

The `tcgen05.fence::*` instructions compose with execution ordering instructions across a thread scope and provide ordering between `tcgen05` instructions across the same scope.

The `tcgen05.fence::before_thread_sync` (`tcgen05.fence::after_thread_sync`) instructions behave as code motion fence for prior (subsequent) `tcgen05` instructions as they cannot be hoisted across. 

Examples:
```ptx
// Producer thread:

tcgen05.cp.cta_group::1.128x256b  [taddr0], sdesc0; // Initiates an asynchronous copy operation from shared memory to the Tensor Memory.

tcgen05.fence::before_thread_sync;
st.relaxed.b32 [flag], 1;

// Consumer thread:

loop:
ld.relaxed.b32 r, [flag];
setp.eq.u32 p, r, 1;
@!p bra loop;

tcgen05.fence::after_thread_sync;
tcgen05.mma.cta_group.kind   [taddr0], adesc, bdesc, idesc, p;
```
*/


__device__ __forceinline__ void tcgen05_fence_after_fn() {
    // tcgen05 fence after thread sync.
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    // tcgen05 fence before thread sync.
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

/*

Perform the 5 th generation of matrix multiply and accumulate operation.

```
tcgen05.mma.cta_group.kind   [d-tmem],  a-desc,  b-desc, idesc,
  { disable-output-lane }, enable-input-d {, scale-input-d};
tcgen05.mma.cta_group.kind   [d-tmem], [a-tmem], b-desc, idesc,
  { disable-output-lane }, enable-input-d {, scale-input-d};
.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }
.cta_group = { .cta_group::1, .cta_group::2 }
```

Instruction `tcgen05.mma` is an asynchronous instruction which initiates an *MxNxK* matrix
multiply and accumulate operation, `D = A*B+D` where the `A` matrix is *MxK* , the `B` matrix is *KxN* , and the `D` matrix is *MxN* .

The operation of the form `D = A*B` is issued when the input predicate argument `enable-input-d` is false.

The 32-bit register operand `idesc` is the instruction descriptor as described
in [Instruction descriptor](#tcgen05-instruction-descriptor) , specifies
the shapes, exact types, sparsity and other details of the input matrices,
output matrix and the matrix multiply and accumulate operation.

The qualifier `.cta_group::1` specifies that the matrix multiply and
accumulate operation is performed on the Tensor Memory of the executing thread's CTA only. The qualifier
`.cta_group::2` specifies that the matrix multiply and accumulate operation is performed on the
Tensor Memory of the executing thread's CTA and its peer CTA.

The instruction `tcgen05.mma` has single thread semantics, unlike the collective
instructions `mma.sync` or `wgmma.mma_async` . So, a single thread issuing the `tcgen05.mma` will result in the initiation of the whole matrix multiply and
accumulate operation. Refer to the section [Issue Granularity](#tcgen05-issue-granularity).

The qualifier `.kind` specifies the general kind of the element types of the multiplicand
matrices. The exact types of the elements of the input and output matrices for each MMA-kind
are specified in the [Instruction descriptor](#tcgen05-instruction-descriptor) .

The address operand `d-tmem` specifies the address of the destination and the accumulation
matrix `D` in the Tensor Memory . The address operand `a-tmem` specifies the address of the matrix `A` in the Tensor Memory.

The 64-bit register operand `a-desc` and `b-desc` are the matrix descriptors which
represent the matrices `A` and `B` in shared memory respectively. The format of the
matrix descriptor is described in [Matrix Descriptors](#tcgen05-matrix-descriptors) .

Supported shapes:
| .kind::* | CTA Group | dtype | atype/btype | Shapes supported (M×N×K) |
|----------|-----------|-------|-------------|---------------------------|
| .kind::f16 | 1 | .f32 | .f16, .bf16 | 64xNxK, 128xNxK, N = {8, 16, 24, …, 256} with steps of 8, K = 16|
| .kind::f16 | 2 | .f32 | .f16, .bf16 | 128xNxK, 256xNxK, N = {16, 32, … 256} steps of 16, K = 16|

How `cta_group::2` UMMA really works:
```
CTA0 loads: A[m0:m0+m_block, :], B[n:n+n_block // 2, :]    CTA1 loads: A[m1:m1+m_block, :], B[n+n_block // 2:n+n_block, :]
      ↓ UMMA cta_group::2 ↓
SM0 TMEM = A[m0:m0+m_block, :] × B[n:n+n_block, :] (tile 0 result)      SM1 TMEM = A[m1:m1+m_block, :] × B[n:n+n_block, :] (tile 1 result)
```
Under `.cta_group::2` CTA0 and CTA1 has asymmetric roles. Both issue TMA loads into their own SMEM but CTA0 holds the barrier. The UMMA instruction
is issued by CTA0 only. After the UMMA, each CTA reads its own SM's TMEM and stores to its own tile's output
*/

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    // SM100 UMMA instruction: cta_group::2, kind::f16 (BF16×BF16→FP32 or FP16xFP16→FP32). accum=0 clears accumulator, accum=1 accumulates.
    // Each MMA only contracts K=16. The TMA descriptors can be set up for larger K, so the caller needs to issue multiple MMA instructions and manage the accumulation across them.
    // Also, MN-Major descriptor under 128B swizzled layouts works for m (or n) = 64 at the largest. m (or n) > 64 requires iterating with ptr offsets
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

/*

Makes the mbarrier object track the completion of all prior async-tcgen05 operations initiated
by the executing thread.

```
tcgen05.commit.cta_group.completion_mechanism{.shared::cluster}{.multicast}.b64 [mbar] {, ctaMask};
.completion_mechanism = { .mbarrier::arrive::one }
.cta_group     = { .cta_group::1, .cta_group::2 }
.multicast     = { .multicast::cluster }
```

Single-CTA completion:

```ptx
tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];
```

For `tcgen05.commit`, the explicit shared-memory state-space qualifier is
`.shared::cluster`, including with `cta_group::1`. `.shared::cta` is invalid.
The CTA-group qualifier and address-space qualifier serve different purposes.

The instruction `tcgen05.commit` is an asynchronous instruction which makes the mbarrier object,
specified by the address operand `mbar` , track the completion of all the prior asynchronous `tcgen05` operations, as listed in [mbarrier based completion mechanism](#tcgen05-memory-consistency-model-mbarrier-completion) ,
initiated by the executing thread. Upon the completion of the tracked asynchronous `tcgen05` operations, the signal specified by the `.completion_mechanism` is triggered by the system on the mbarrier object.

This instruction accesses its `mbarrier` operand using generic-proxy.

The instruction `tcgen05.commit.cta_group::2` tracks for the completion of all prior asynchronous `tcgen05` operations with `.cta_group::2` issued by the current thread.

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group` qualifier.

The qualifier `.mbarrier::arrive::one` indicates that upon the completion of the prior
asynchronous `tcgen05` operation issued by the current thread, an arrive-on operation, with
the count argument of 1, is signaled on the mbarrier object. The scope of the arrive-on operation
is the cluster scope.

The optional qualifier `.multicast::cluster` allows signaling on the mbarrier objects of multiple
CTAs in the cluster. Operand `ctaMask` specifies the CTAs in the cluster such that each bit
position in the 16-bit `ctaMask` operand corresponds to the `%cluster_ctarank` of the destination
CTA. The mbarrier signal is multicast to the same offset as `mbar` in the shared memory of each destination CTA.

*/

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    // UMMA commit with cta_group::2 multicast barrier arrive.
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3)); // It hardcodes ctaMask=0x3 (CTAs 0 and 1 only). This works only for cluster_size=2.
}

// This SWIZZLE_128B helper leaves the matrix base offset (bits [51:49]) zero.
// It requires the swizzle pattern to start on a 1024-byte boundary;
// 128-byte alignment alone is insufficient. For other pattern starts, set
// bits [51:49] to (pattern_start_addr >> 7) & 7 before using the descriptor.
// pattern_start_addr is the swizzle pattern's shared-memory byte address,
// not an arbitrary interior submatrix pointer.
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    // Build SM100 UMMA SMEM descriptor (version=1, SWIZZLE_128B).
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; // LBO is stored in bits [29:16]
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; // Another pattern is to set SBO=0 with K-walked manually via the base address
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    // Build SM100 UMMA instruction descriptor (BF16×BF16→FP32, K-major for A and B).
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major), 1 (A is M-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major), 1 (B is N-Major)
    d |= ((N / 8) << 17);     // n_dim. For cta_group::2, N should be the combined dimension (BN_per_cta × 2).
    d |= ((M / 16) << 24);    // m_dim. For cta_group::2, M should be the combined dimension (BM_per_cta × 2).
    return d;
}



#pragma once

#include <cstdint>

// One dense tcgen05.kind::tf32 instruction contracts K=8.  For K>8, stage one
// canonical K-major tile per K=8 slice, issue the first call with accumulate=0,
// and issue later calls with accumulate=1.  Store the original FP32 bit
// patterns in shared memory: tcgen05 performs the TF32 input conversion.  On
// the validated B200 path this discards the low 13 FP32 mantissa bits rather
// than using round-to-nearest-even.  Do not call __float_to_tf32 before this
// instruction.

// Physical index, in float elements, for one canonical no-swizzle K-major
// [MnDim, 8] tile.  Each K half is a contiguous [MnDim, 4] collection of
// 16-byte spans; rows are grouped eight at a time as required by the SM100
// matrix descriptor.
template <int MnDim>
__host__ __device__ constexpr uint32_t tfmm_b200_kmajor_smem_index(
    uint32_t mn, uint32_t k) {
  static_assert(MnDim % 8 == 0, "TFMM MN dimension must be a multiple of 8");
  return (k >> 2) * MnDim * 4 + (mn >> 3) * 32 + (mn & 7) * 4 +
         (k & 3);
}

// Descriptor for the canonical layout produced by
// tfmm_b200_kmajor_smem_index.  LBO selects the second four-element K span;
// SBO advances by one group of eight MN coordinates.
template <int MnDim>
__device__ __forceinline__ uint64_t tfmm_b200_make_kmajor_smem_desc(
    const void* smem_ptr) {
  static_assert(MnDim % 8 == 0, "TFMM MN dimension must be a multiple of 8");
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  constexpr uint32_t lbo = MnDim * 16;
  constexpr uint32_t sbo = 8 * 16;
  uint64_t desc = static_cast<uint64_t>((addr & 0x3ffffu) >> 4);
  desc |= static_cast<uint64_t>((lbo & 0x3ffffu) >> 4) << 16;
  desc |= static_cast<uint64_t>((sbo & 0x3ffffu) >> 4) << 32;
  desc |= uint64_t{1} << 46;  // SM100 matrix-descriptor version = 1.
  return desc;                // layout_type = 0: no swizzling.
}

template <int M, int N>
__host__ __device__ constexpr uint32_t tfmm_b200_make_instr_desc() {
  static_assert(M == 64 || M == 128,
                "1-CTA TF32 UMMA requires M=64 or M=128");
  static_assert((M == 64 && N >= 8 && N <= 256 && N % 8 == 0) ||
                    (M == 128 && N >= 16 && N <= 256 && N % 16 == 0),
                "invalid N for 1-CTA TF32 UMMA");
  return (1u << 4) |                 // D format: FP32
         (2u << 7) |                 // A format: TF32 from FP32 storage
         (2u << 10) |                // B format: TF32 from FP32 storage
         (static_cast<uint32_t>(N / 8) << 17) |
         (static_cast<uint32_t>(M / 16) << 24);
}

// Single-thread issue: call this from exactly one elected thread after every
// producer of the A and B shared-memory tiles has synchronized.  The four zero
// mask registers keep all output lanes enabled, matching NVIDIA's SM100 TF32
// UMMA wrapper.  The instruction is asynchronous; commit it to an mbarrier and
// wait before loading the result from TMEM or reusing operand storage.
__device__ __forceinline__ void tfmm_b200_ss_fn(
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t instr_desc,
    uint32_t accumulate) {
  const uint32_t mask0 = 0;
  const uint32_t mask1 = 0;
  const uint32_t mask2 = 0;
  const uint32_t mask3 = 0;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::tf32 "
      "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n"
      "}\n"
      :
      : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(instr_desc),
        "r"(accumulate), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3)
      : "memory");
}



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


