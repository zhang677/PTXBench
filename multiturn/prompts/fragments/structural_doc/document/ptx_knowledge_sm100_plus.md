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