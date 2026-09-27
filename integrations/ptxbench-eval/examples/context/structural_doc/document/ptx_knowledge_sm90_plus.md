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