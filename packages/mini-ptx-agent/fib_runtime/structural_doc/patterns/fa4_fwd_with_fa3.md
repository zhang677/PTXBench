# Optimization pattern: Producer-Consumer asynchrony through warp-specialization and pingpong scheduling (FA-3)

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
        \State Store $m_i^{\text{old}} = m_i$ and compute $m_i = \max\!\left(m_i^{\text{old}},\, \mathrm{rowmax}(S_i^{(j)})\right)$.
        \State Compute $\widetilde{P}_i^{(j)} = \exp\!\left(S_i^{(j)} - m_i\right)$ and $\ell_i = \exp(m_i^{\text{old}} - m_i)\,\ell_i + \mathrm{rowsum}(\widetilde{P}_i^{(j)})$.
        \State Wait for $V_j$ to be loaded in shared memory.
        \State Compute $O_i = \mathrm{diag}(\exp(m_i^{\text{old}} - m_i))^{-1} O_i + \widetilde{P}_i^{(j)} V_j$ (RS-GEMM). Commit and wait.
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


# 3.1 Attention forward pass (FA-4 for Blackwell)

## 3.1.2 New pipeline to overlap matmul and softmax

Since the Blackwell architecture doubled the tensor core flops again, taking care to overlap softmax and tensor core operations is even more crucial than on Hopper. We follow a ping-pong schedule similar to FA-3, where two tiles of the output are computed per thread block. While one tile's tensor core operations are executed, the other tile computes softmax. While Hopper tensor cores hold the accumulator in registers, with four threads per row in an interleaved pattern, Blackwell tensor cores hold their accumulators in tensor memory. Additionally, a single accumulator tile on Blackwell is 128 by 128 elements large, where Hopper's tile size was 64 by 128.

The natural way to distribute work across these tiles is then to have two warpgroups of 128 threads each, with each thread processing an entire row. This eliminates the need for inter-warp shuffles to reduce the row max, and for multiple statistics registers per thread. Just like with FA-3, we explicitly synchronize the two softmax warpgroups to not overlap in their critical section, which is the part of exponential computation. Each softmax warpgroup proceeds by first loading the entire row into registers, then computing the maximum, then computing the softmax (i.e., subtract the max, rescale, exponentiate, convert to input precision), and finally computing the row sum.

Another difference from FA-3 is that since we transfer P via tensor memory rather than register file, we can decouple the rescaling of the output to a separate "correction" warpgroup and thus take it out of the critical path.

Several tensor memory partitionings are possible to achieve this pipeline overlap. All must allocate two tiles worth of output, leaving (at head dimension 128) half the tensor memory to store S and P. That memory can store two copies of S or four copies of P (assuming the input of the FP16 or BF16 tensor core). This leaves us with roughly two partitioning options for the remaining tensor memory: one tile of S and two tiles of P, or two tiles of S that overlap with P. We choose the latter because it allows us to start our software pipeline by immediately computing two S tiles. It also leaves some tensor memory to communicate rescale statistics to the correction warpgroup.

One issue of the larger Blackwell tile sizes and the chosen thread assignment is that, unless we re-load from tensor memory, we must hold an entire row of 128 elements in register. Given that we use two softmax warpgroups, one correction warpgroup, and one warpgroup to drive tensor cores and TMA units, assigning sufficient registers to softmax and preventing register spills is critical. For BF16 input data types, we need to hold 128 registers for the input, and potentially 64 registers for the output (plus miscellaneous and temporary registers). To reduce register pressure, we stage out storing P: The first three quarters are stored once (and trigger the corresponding MMA operations), and the last quarter is stored separately.

## 3.1.3 Emulation of the exponential function

**Exponential throughput bottleneck.** On modern GPUs, the exponential function is computed by the multi-function unit (MUFU), which has significantly lower throughput than the tensor cores used for matrix multiplication. On B200 and GB200 GPUs, MUFU provides 16 operations per clock per SM, compared to 8192 operations per clock per SM for matrix multiplication. Since softmax computation requires many exponential evaluations, this disparity makes the exponential function a critical bottleneck in attention kernels.

**Software emulation via polynomial approximation.** To increase exponential throughput, we implement a software emulation of 2^x using floating-point FMA units, which can operate in parallel with MUFU. We use the classical range reduction technique (Cody-Waite) and then the polynomial approximation [16]. The key insight is to decompose the exponential computation:

$$2^x = 2^{\lfloor x \rfloor} \cdot 2^{x - \lfloor x \rfloor} \quad (4)$$

where ⌊x⌋ is the integer part and x − ⌊x⌋ ∈ [0, 1) is the fractional part.

The integer part 2^⌊x⌋ can be computed efficiently using bit manipulation of the IEEE 754 floating-point representation. Since the exponent field directly represents powers of two, computing 2^⌊x⌋ amounts to a shift and add operation on the exponent bits, which can be done using integer ALU instructions.

For the fractional part, we approximate 2^x_frac where x_frac ∈ [0, 1) using a polynomial:

$$2^{x_{frac}} \approx \sum_{i=0}^{n} p_i x_{frac}^i \quad (5)$$

with p_0 = 1.0 and the remaining coefficients chosen to minimize the relative approximation error over [0, 1), calculated using the Sollya software package [4]. The polynomial evaluation uses Horner's method with FMA instructions, achieving high throughput.

The complete algorithm proceeds as follows:

1. Clamp x to be at least −127 to avoid underflow
2. Compute ⌊x⌋ using round-down mode: add 2^23 + 2^22 to x (forcing the fractional bits into the mantissa), then subtract it back with round-down mode
3. Compute fractional part: x_frac = x − ⌊x⌋
4. Evaluate polynomial to get 2^x_frac
5. Combine integer and fractional parts: shift ⌊x⌋ into the exponent field and add the mantissa bits of 2^x_frac

By distributing exponential computations across both MUFU and FMA units, this approach effectively increases the exponential throughput, alleviating a key bottleneck in attention computation.

**Partial emulation.** Although polynomial emulation increases exponential throughput, it comes at a cost: additional registers (to hold intermediate values and coefficients), higher register bandwidth consumption, and longer latency compared to the MUFU instruction. Using emulation for all exponential evaluations would increase register pressure and could cause spills that negate the throughput benefit. Instead, we apply emulation to only a subset of the entries in each softmax row (10–25%), with the remaining entries computed via hardware `MUFU.EX2`. The exact fraction is tuned empirically based on the ratio of MMA and exponential throughput for a given tile configuration.

**Numerical accuracy.** Table 2 compares the accuracy of polynomial approximations of different degrees against the hardware `MUFU.EX2` instruction, measured on 4M random inputs in [0, 1). We report two metrics: the FP32-level error (before any quantization) and the BF16-level error (after rounding the FP32 output to BF16), both measured against a FP64 reference.

At the FP32 level, the degree-3 polynomial has a maximum relative error of 8.8 × 10^−5, roughly 600× higher than hardware. However, after rounding to BF16, the errors become nearly indistinguishable: the quantization error of BF16 (∼3.9 × 10^−3) dominates the polynomial approximation error for all degrees ≥ 3. The degree-3 polynomial matches hardware to within 1 BF16 ULP on 99% of inputs, which is sufficient for attention computation where the softmax output is consumed with BF16 precision. Higher-degree polynomials close the FP32 gap: degree 5 matches hardware to within 2× in maximum relative error, at the cost of two additional FMA instructions per evaluation.

**Table 2:** Accuracy of 2^x polynomial emulation on [0, 1), measured against FP64 reference on 4M random inputs. FP32 columns measure the raw polynomial output; BF16 columns measure after rounding to BF16. The BF16 quantization error dominates for all degrees ≥ 3.

|                          | FP32 vs FP64           |                        | BF16 vs FP64           |                        |
|--------------------------|------------------------|------------------------|------------------------|------------------------|
| Method                   | Max rel err            | Mean rel err           | Max rel err            | Mean rel err           |
| Ideal (FP64 → BF16)      | —                      | —                      | 3.89 × 10^−3           | 1.41 × 10^−3           |
| Hardware `MUFU.EX2`      | 1.41 × 10^−7           | 3.04 × 10^−8           | 3.89 × 10^−3           | 1.41 × 10^−3           |
| Degree 3                 | 8.77 × 10^−5           | 5.43 × 10^−5           | 3.90 × 10^−3           | 1.41 × 10^−3           |
| Degree 4                 | 3.05 × 10^−6           | 1.84 × 10^−6           | 3.89 × 10^−3           | 1.41 × 10^−3           |
| Degree 5                 | 1.44 × 10^−7           | 5.48 × 10^−8           | 3.89 × 10^−3           | 1.41 × 10^−3           |

## 3.1.4 Skipping online softmax rescaling

**FlashAttention online softmax.** FlashAttention computes attention softmax(QK⊤)V in blocks to minimize memory traffic. For numerical stability, the algorithm maintains running statistics as it processes blocks. When computing block j, let S_j = QK_j⊤ be the attention scores for that block. The online softmax algorithm tracks:

$$m_j = \max(m_{j-1}, \mathrm{rowmax}(S_j))$$

$$\ell_j = e^{m_{j-1} - m_j} \ell_{j-1} + \mathrm{rowsum}(e^{S_j - m_j})$$

where m_j is the running max and ℓ_j is the running sum of exponentials (normalizer). The intermediate output O_j is updated as: O_j = e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j. The rescaling factor e^{m_{j-1} - m_j} ensures numerical stability by renormalizing previous results when larger values are encountered.

**Conditional rescaling.** The step e^{m_{j-1} - m_j} O_{j-1} requires a vector multiplication. We make two simple observations:

1. Rescaling is only necessary when m_j > m_{j-1}, i.e., when new larger values are found.
2. We can tolerate some "slack" in the rescaling: only rescale when m_j − m_{j-1} > τ, where τ is a threshold (typically set to log_2(256) = 8.0, corresponding to a rescaling factor of 256.0). As long as we keep track of the statistics (the total scaling we have done), we can still get the true denominator at the end to get the right final output.

In FlashAttention-4, we modify the algorithm as:

$$O_j = \begin{cases} e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j & \text{if } m_j - m_{j-1} > \tau \\ O_{j-1} + e^{S_j - m_{j-1}} V_j & \text{otherwise} \end{cases} \quad (6)$$

When m_j − m_{j-1} ≤ τ, we skip updating m and continue using m_{j-1}. This maintains the correctness because at the end of the computation, all accumulated values are renormalized by the true maximum m_final and the final normalizer ℓ_final:

$$\text{Output} = \frac{1}{\ell_{final}} O_{final}$$

This modification significantly reduces the number of rescaling operations while maintaining numerical accuracy, as the final normalization step corrects any small deviations introduced by skipping intermediate rescaling.

In practice, to avoid warp divergence, we rescale when any of the threads in the warp needs rescaling.
