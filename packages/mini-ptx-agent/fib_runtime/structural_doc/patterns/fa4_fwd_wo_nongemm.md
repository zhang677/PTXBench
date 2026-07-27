# 3.1 Attention forward pass (FA-4 for Blackwell)

We first do a roofline analysis to show the bottlenecks of attention forward pass, which motivates our new pipeline design, as well as changes in the FlashAttention algorithm to increase the throughput of the exponential unit and avoid most of the softmax rescaling steps.

## 3.1.1 Feeds and Speeds

We provide intuition for our kernel design and optimizations by first analyzing the roofline, based on the throughput of the matmul units (tensor cores), shared memory (smem), and exponential unit. We note that this is a simplified analysis that does not consider all resources in the GPU (e.g., floating point math, register bandwidth, L2 bandwidth). Nevertheless, it can identify bottlenecks.

Let the shape of the tile along the length dimension of the sequence of Q and K be M×N, and let the head dimension be d. We analyze the compute and memory traffic requirements to identify the performance bottleneck.

**MMA compute.** The forward pass performs two matrix multiply-accumulate (MMA) operations per iteration: QK⊤ (computing M×N output from M×d and d×N inputs) and PV (computing M×d output from M×N and N×d inputs). Each MMA requires 2MNd floating-point operations. With a tensor core throughput of 8192 FLOPs per cycle, the total compute time is

$$T_{MMA} = \frac{4MNd}{8192} \text{ cycles.} \quad (1)$$

**Shared memory traffic.** Of the two MMAs, one is shared-shared (SS) where both operands are read from shared memory (QK⊤), while the other is tensor-shared (TS) where operand A is read from tensor memory and operand B from shared memory (PV). Since each MMA instruction operates on tiles of size 128 × 128, computing an M×N output requires ⌈M/128⌉ × ⌈N/128⌉ MMA instructions. Crucially, when multiple MMA instructions are needed, the shared memory operands are read multiple times.

For QK⊤ (SS), computing the M×N output requires ⌈M/128⌉ × ⌈N/128⌉ MMA instructions, each reading a 128×d chunk of Q and a d×128 chunk of K⊤ from shared memory. The total shared memory reads are ⌈M/128⌉ × ⌈N/128⌉ × (128d + 128d) = ⌈M/128⌉⌈N/128⌉ × 256d elements.

For PV (TS), computing the M×d output requires ⌈M/128⌉ × ⌈d/128⌉ MMA instructions, each reading a N×128 chunk of V from shared memory, totaling ⌈M/128⌉ × ⌈d/128⌉ × 128N elements.

At 2 bytes per element (bf16) and 128 bytes per cycle bandwidth, the shared memory (T_smem) read time is

$$T_{smem} = (2 \left\lceil \frac{M}{128} \right\rceil \left\lceil \frac{N}{128} \right\rceil 256d + 2 \left\lceil \frac{M}{128} \right\rceil \left\lceil \frac{d}{128} \right\rceil 128N) \times \frac{1}{128} = \frac{3MNd}{8192} \text{ cycles} \quad (2)$$

(assuming M, N, d are multiples of 128).

**Exponential unit.** The exponential unit computes elementwise operations required for the softmax computation. The forward pass requires exponential operations on M×N values (corresponding to the attention matrix S). With a throughput of 16 operations per cycle, the exponential unit requires

$$T_{exp} = \frac{MN}{16} \text{ cycles.} \quad (3)$$

Table 1 summarizes the analysis for two typical tile configurations. For M = N = d = 128, the resources are well-balanced, with shared memory (768 cycles) being slightly lower than both MMA compute and exponential unit (both 1024 cycles). For the larger tile size M = 256, N = d = 128, the shared memory traffic increases to 1536 cycles due to reading MMA operands multiple times, while MMA compute and exponential unit double to 2048 cycles. This analysis motivates our kernel design to (1) use large tile sizes and maximize overlap between MMA operations and softmax computations (2) increase the throughput of exponential by using other hardware units (3) reduce the time of unnecessary non-matmul operations.

**Table 1:** Roofline analysis (cycles) for the attention forward pass. For both tile sizes, MMA compute and exponential unit are the primary bottlenecks.

| Resource          | 128³ | 256×128² |
|-------------------|------|----------|
| MMA compute       | 1024 | 2048     |
| Shared memory     | 768  | 1536     |
| Exponential unit  | 1024 | 2048     |

## 3.1.2 New pipeline to overlap matmul and softmax

Since the Blackwell architecture doubled the tensor core flops again, taking care to overlap softmax and tensor core operations is even more crucial than on Hopper. We follow a ping-pong schedule similar to FA-3, where two tiles of the output are computed per thread block. While one tile's tensor core operations are executed, the other tile computes softmax. While Hopper tensor cores hold the accumulator in registers, with four threads per row in an interleaved pattern, Blackwell tensor cores hold their accumulators in tensor memory. Additionally, a single accumulator tile on Blackwell is 128 by 128 elements large, where Hopper's tile size was 64 by 128.

The natural way to distribute work across these tiles is then to have two warpgroups of 128 threads each, with each thread processing an entire row. This eliminates the need for inter-warp shuffles to reduce the row max, and for multiple statistics registers per thread. Just like with FA-3, we explicitly synchronize the two softmax warpgroups to not overlap in their critical section, which is the part of exponential computation. Each softmax warpgroup proceeds by first loading the entire row into registers, then computing the maximum, then computing the softmax (i.e., subtract the max, rescale, exponentiate, convert to input precision), and finally computing the row sum.

Another difference from FA-3 is that since we transfer P via tensor memory rather than register file, we can decouple the rescaling of the output to a separate "correction" warpgroup and thus take it out of the critical path.

Several tensor memory partitionings are possible to achieve this pipeline overlap. All must allocate two tiles worth of output, leaving (at head dimension 128) half the tensor memory to store S and P. That memory can store two copies of S or four copies of P (assuming the input of the FP16 or BF16 tensor core). This leaves us with roughly two partitioning options for the remaining tensor memory: one tile of S and two tiles of P, or two tiles of S that overlap with P. We choose the latter because it allows us to start our software pipeline by immediately computing two S tiles. It also leaves some tensor memory to communicate rescale statistics to the correction warpgroup.

One issue of the larger Blackwell tile sizes and the chosen thread assignment is that, unless we re-load from tensor memory, we must hold an entire row of 128 elements in register. Given that we use two softmax warpgroups, one correction warpgroup, and one warpgroup to drive tensor cores and TMA units, assigning sufficient registers to softmax and preventing register spills is critical. For BF16 input data types, we need to hold 128 registers for the input, and potentially 64 registers for the output (plus miscellaneous and temporary registers). To reduce register pressure, we stage out storing P: The first three quarters are stored once (and trigger the corresponding MMA operations), and the last quarter is stored separately.
