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