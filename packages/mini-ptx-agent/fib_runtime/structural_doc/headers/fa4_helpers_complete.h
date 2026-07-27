/*
    fa4_helpers.h — Software emulation of 2^x via FMA, as used in FlashAttention-4
    (SM100/Blackwell).

    Rationale: MUFU.EX2 on B200/GB200 runs at 16 ops/SM/cycle, far slower
    than the FMA pipeline. We emulate 2^x on the FMA pipe using the
    decomposition

        2^x  =  2^floor(x) * 2^(x - floor(x))

    with the fractional part approximated by a Sollya-fitted minimax
    polynomial on [0, 1). The integer part is folded back into the IEEE-754
    exponent field via a shift-and-add on the bit pattern. Both pipes run
    in parallel, so a fraction of softmax entries is steered onto the
    polynomial path while the rest stay on MUFU.

    Sollya invocation (degree n):
        fpminimax(exp(x * log(2.0)), 1, [|1, 24...|], [0; 1], relative);

    Important: every step's rounding mode matters.
      - Step 2 (floor via magic constant) MUST use add.rm (round toward -inf).
      - Step 3 (recover floor) uses add.rn (round-to-nearest); the
        subtraction is exact because the integer part fits in 8 bits.
      - Step 5 (splice) uses add.s32 (LEA pipe) rather than add.u32 (IMAD/FMA
        pipe), so the splice does not contend with the polynomial FMAs.
*/

#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


// ----------------------------------------------------------------------------
// Polynomial coefficients (Sollya minimax of 2^x on [0, 1))
//
// p[0] = 1.0 is fixed by construction. Higher-degree variants close the
// FP32 gap at the cost of additional FMAs.
//
// At the BF16 quantization level, degree-3 matches hardware to within 1 BF16 ULP on 99 % of inputs, 
// which is what the SM100 FlashAttention kernel uses by default.
// ----------------------------------------------------------------------------

template <int DEGREE> struct PolyEx2;

template <> struct PolyEx2<3> {
    static constexpr float coeff[4] = {
        1.0f,
        0.695146143436431884765625f,    // p1
        0.227564394474029541015625f,    // p2
        0.077119089663028717041015625f  // p3
    };
};

template <> struct PolyEx2<4> {
    static constexpr float coeff[5] = {
        1.0f,
        0.693042695522308349609375f,
        0.2412912547588348388671875f,
        5.2225358784198760986328125e-2f,
        1.3434938155114650726318359375e-2f
    };
};

template <> struct PolyEx2<5> {
    static constexpr float coeff[6] = {
        1.0f,
        0.693151414394378662109375f,
        0.24016360938549041748046875f,
        5.5802188813686370849609375e-2f,
        9.01452265679836273193359375e-3f,
        1.86810153536498546600341796875e-3f
    };
};


// ----------------------------------------------------------------------------
// Step 2 primitive: add with round-toward-minus-infinity, FTZ.
//
// PTX: add.rm.ftz.f32
//
// Used with the magic constant (2^23 + 2^22) so that the integer floor of
// x lands in the low 8 bits of the result's mantissa. Round-toward-minus-
// infinity is essential; round-to-nearest would round half-integers up and
// break the floor semantics.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float add_round_down_fn(float x, float y) {
    float z;
    asm("add.rm.ftz.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(y));
    return z;
}


// ----------------------------------------------------------------------------
// Step 5 primitive: splice the integer and fractional parts of 2^x.
//
// Inputs:
//   x_rounded : floor(x) + magic_constant, as float (still biased)
//   frac_ex2  : 2^(x - floor(x)) in [1, 2) (an IEEE-754 normal)
//
// Algorithm:
//   1. Reinterpret x_rounded as int32. The low 8 bits hold floor(x).
//   2. shl.b32 by 23 to move floor(x) into the IEEE-754 exponent field.
//   3. add.s32 with the bit pattern of frac_ex2. Because frac_ex2 has
//      biased exponent 127 (since it is in [1, 2)), adding floor(x) into
//      that field yields 2^(floor(x) + x_frac) = 2^x.
//   4. Reinterpret the integer result back as float.
//
// Why add.s32 (signed) rather than add.u32 (unsigned)? They produce the
// same bit pattern, but the compiler lowers add.s32 to a SASS LEA on the
// ALU pipeline, while add.u32 lowers to IMAD on the FMA pipeline. The
// polynomial Horner already saturates the FMA pipe, so the splice is
// steered onto the ALU pipe to avoid contention.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float combine_int_frac_ex2_fn(float x_rounded, float frac_ex2) {
    float out;
    asm("{\n\t"
        ".reg .s32 xri, fei, xre, oi;\n\t"
        "mov.b32 xri, %1;\n\t"
        "mov.b32 fei, %2;\n\t"
        "shl.b32 xre, xri, 23;\n\t"
        "add.s32 oi, xre, fei;\n\t"
        "mov.b32 %0, oi;\n\t"
        "}\n"
        : "=f"(out)
        : "f"(x_rounded), "f"(frac_ex2));
    return out;
}


// ----------------------------------------------------------------------------
// Step 4 primitive: Horner-form polynomial evaluation with FMA.
//
// Evaluates p[0] + x*(p[1] + x*(p[2] + ... + x*p[n])) using n FMA
// instructions. Templated on degree so the loop is fully unrolled and
// the coefficient table indexes are constexpr.
//
// Lowers to a chain of fma.rn.ftz.f32 instructions.
// ----------------------------------------------------------------------------
template <int DEGREE>
__device__ __forceinline__ float evaluate_polynomial_fn(float x) {
    const float* p = PolyEx2<DEGREE>::coeff;
    float out = p[DEGREE];
    #pragma unroll
    for (int i = DEGREE - 1; i >= 0; --i) {
        out = fmaf(out, x, p[i]);
    }
    return out;
}


// ----------------------------------------------------------------------------
// Step 4 primitive (packed): Horner-form polynomial evaluation on a
// f32x2 pair using fma.rn.ftz.f32x2.
//
// On Blackwell, packed-f32x2 FMA issues at the same per-thread cycle
// throughput as scalar FMA, so this halves the polynomial cost when
// evaluating two values at once.
//
// We marshal through .b64 because PTX f32x2 ops take 64-bit operand
// registers containing two packed f32 values. Returns the packed result
// as a float2.
// ----------------------------------------------------------------------------
template <int DEGREE>
__device__ __forceinline__ float2 evaluate_polynomial_packed_fn(float2 x) {
    const float* p = PolyEx2<DEGREE>::coeff;
    // Initialize (out_lo, out_hi) = (p[DEGREE], p[DEGREE]).
    float2 out = make_float2(p[DEGREE], p[DEGREE]);
    #pragma unroll
    for (int i = DEGREE - 1; i >= 0; --i) {
        float coef = p[i];
        uint64_t out_b64, x_b64, c_b64, res_b64;
        asm("mov.b64 %0, {%1, %2};" : "=l"(out_b64) : "f"(out.x), "f"(out.y));
        asm("mov.b64 %0, {%1, %2};" : "=l"(x_b64)   : "f"(x.x),   "f"(x.y));
        asm("mov.b64 %0, {%1, %2};" : "=l"(c_b64)   : "f"(coef),  "f"(coef));
        asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
            : "=l"(res_b64)
            : "l"(out_b64), "l"(x_b64), "l"(c_b64));
        asm("mov.b64 {%0, %1}, %2;" : "=f"(out.x), "=f"(out.y) : "l"(res_b64));
    }
    return out;
}


// ----------------------------------------------------------------------------
// Packed primitives: add.rm.ftz.f32x2 and sub.rn.ftz.f32x2.
//
// Same pack/unpack-around-asm pattern as evaluate_polynomial_packed_fn.
// `add_packed_rd` rounds down (used for the magic-constant floor),
// `sub_packed_rn` rounds nearest (used to recover floor and compute frac).
// ----------------------------------------------------------------------------
__device__ __forceinline__ float2 add_packed_rd_f32x2_fn(float2 a, float2 b) {
    uint64_t a_b64, b_b64, r_b64;
    asm("mov.b64 %0, {%1, %2};" : "=l"(a_b64) : "f"(a.x), "f"(a.y));
    asm("mov.b64 %0, {%1, %2};" : "=l"(b_b64) : "f"(b.x), "f"(b.y));
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(r_b64) : "l"(a_b64), "l"(b_b64));
    float2 r;
    asm("mov.b64 {%0, %1}, %2;" : "=f"(r.x), "=f"(r.y) : "l"(r_b64));
    return r;
}

__device__ __forceinline__ float2 sub_packed_rn_f32x2_fn(float2 a, float2 b) {
    uint64_t a_b64, b_b64, r_b64;
    asm("mov.b64 %0, {%1, %2};" : "=l"(a_b64) : "f"(a.x), "f"(a.y));
    asm("mov.b64 %0, {%1, %2};" : "=l"(b_b64) : "f"(b.x), "f"(b.y));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(r_b64) : "l"(a_b64), "l"(b_b64));
    float2 r;
    asm("mov.b64 {%0, %1}, %2;" : "=f"(r.x), "=f"(r.y) : "l"(r_b64));
    return r;
}


// ----------------------------------------------------------------------------
// Hardware MUFU.EX2: ex2.approx.ftz.f32
//
// Single-instruction baseline; runs on the multi-function unit (MUFU) at
// 16 ops/SM/cycle on B200/GB200. The partial-emulation selector keeps
// this path for ~75-90 % of softmax entries.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float ex2_mufu_fn(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


// ----------------------------------------------------------------------------
// Scalar emulation driver: 2^x via Cody-Waite + polynomial + bit splice.
//
// Mirrors flash_attn/cute/utils.py:ex2_emulation. Five steps:
//
//   1. Clamp x to [-127, +inf) so floor(x) fits in 8 bits and the
//      subsequent IEEE exponent shift does not produce subnormals.
//   2. add.rm.ftz with magic constant 2^23 + 2^22 = 0x4B400000. The
//      integer floor of x now sits in the low 8 bits of x_rounded's
//      mantissa.
//   3. Subtract the magic constant back (round-to-nearest, exact since
//      floor(x) fits in 8 bits) and compute x_frac = x_clamped - floor(x)
//      in [0, 1).
//   4. Evaluate the degree-N minimax polynomial on x_frac to get
//      2^x_frac in [1, 2).
//   5. Splice: bit-shift floor(x) into the IEEE-754 exponent field of
//      the polynomial result. The integer-add on the exponent field
//      multiplies the polynomial result by 2^floor(x).
//
// Default degree 3 (3 FMAs); degree 5 (5 FMAs) recovers hardware-level
// FP32 accuracy.
// ----------------------------------------------------------------------------
template <int DEGREE = 3>
__device__ __forceinline__ float ex2_emulation_fn(float x) {
    constexpr float MAGIC = 12582912.0f;          // 2^23 + 2^22, IEEE 0x4B400000
    constexpr float CLAMP_LO = -127.0f;
    float x_clamped      = fmaxf(x, CLAMP_LO);                       // step 1
    float x_rounded      = add_round_down_fn(x_clamped, MAGIC);      // step 2
    float x_rounded_back = x_rounded - MAGIC;                        // step 3a
    float x_frac         = x_clamped - x_rounded_back;               // step 3b
    float frac_ex2       = evaluate_polynomial_fn<DEGREE>(x_frac);   // step 4
    return combine_int_frac_ex2_fn(x_rounded, frac_ex2);             // step 5
}


// ----------------------------------------------------------------------------
// Packed emulation driver: 2^x and 2^y in parallel on the FMA pipe.
//
// Same five steps as ex2_emulation_fn but each arithmetic op is f32x2. 
// The splice (step 5) is unrolled into two scalar calls because the ALU pipe has no
// packed-int variants (shl.b32x2 / add.s32x2 do not exist).
// ----------------------------------------------------------------------------
template <int DEGREE = 3>
__device__ __forceinline__ float2 ex2_emulation_packed_fn(float2 xy) {
    constexpr float MAGIC = 12582912.0f;
    constexpr float CLAMP_LO = -127.0f;
    float2 xy_clamped = make_float2(fmaxf(xy.x, CLAMP_LO),
                                    fmaxf(xy.y, CLAMP_LO));            // step 1
    float2 magic_f32x2 = make_float2(MAGIC, MAGIC);
    float2 xy_rounded      = add_packed_rd_f32x2_fn(xy_clamped, magic_f32x2);   // step 2
    float2 xy_rounded_back = sub_packed_rn_f32x2_fn(xy_rounded, magic_f32x2);   // step 3a
    float2 xy_frac         = sub_packed_rn_f32x2_fn(xy_clamped, xy_rounded_back); // step 3b
    float2 frac_ex2        = evaluate_polynomial_packed_fn<DEGREE>(xy_frac);     // step 4
    float2 out;
    out.x = combine_int_frac_ex2_fn(xy_rounded.x, frac_ex2.x);                   // step 5
    out.y = combine_int_frac_ex2_fn(xy_rounded.y, frac_ex2.y);
    return out;
}


// ----------------------------------------------------------------------------
// All-inline-PTX reference of the packed emulation, mirroring
// flash_attn/cute/utils.py:e2e_asm2. Degree-3 only — the coefficients are
// baked in as PTX hex literals (matching POLY_EX2[3]):
//   p_3 = 0f3D9DF09D = 0.077119089663...
//   p_2 = 0f3E6906A4 = 0.227564394474...
//   p_1 = 0f3F31F519 = 0.695146143436...
//   p_0 = 0f3F800000 = 1.0
//   magic   = 0f4B400000 = 2^23 + 2^22
//   clampLo = 0fC2FE0000 = -127.0
//
// Useful as a sanity check: the SASS this emits should match the
// composition ex2_emulation_packed_fn<3>.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float2 ex2_emulation_packed_asm_fn(float x, float y) {
    float ox, oy;
    asm("{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        // step 1: clamp x, y to >= -127
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        // load magic constant 0x4B400000
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        // step 2: x_rounded = x_clamped + magic   (round-toward-minus-inf)
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        // step 3a: x_rounded_back = x_rounded - magic   (round-to-nearest)
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        // step 3b: x_frac = x_clamped - x_rounded_back  (in [0, 1))
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        // load polynomial coefficients p3, p2, p1, p0
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        // step 4: Horner — ((p3*xf + p2)*xf + p1)*xf + p0
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        // step 5: splice — unpack to scalars, shl<<23, add.s32, repack
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
        : "=f"(ox), "=f"(oy)
        : "f"(x), "f"(y));
    return make_float2(ox, oy);
}


// ----------------------------------------------------------------------------
// Pre-emulation bias step.
//
// In the baseline kernel the
// scale and the subtract-row-max would be fused into the same loop that
// calls ::exp2f, but emulation cannot share FMA slots with the polynomial,
// so the bias is computed in a separate pass. After this loop, every
// element holds  (scale_log2 * acc[i]) + (max_offset - row_max * scale_log2),
// which is then fed to apply_exp2_emulation_row_fn.
//
// `max_offset` is the rescale-slack constant — by adding a positive
// offset to every entry, the emulation always operates on x_frac in
// [0, 1) with x well above the clamp floor.
//
// N must be even (the inner loop pairs adjacent elements for packed FMA).
// ----------------------------------------------------------------------------
template <int N>
__device__ __forceinline__ void scale_subtract_rowmax_fn(
    float (&acc)[N], float row_max, float scale_log2, float max_offset
) {
    static_assert(N % 2 == 0, "row length must be even");
    float row_max_scaled = row_max * scale_log2;
    float bias = max_offset - row_max_scaled;
    #pragma unroll
    for (int i = 0; i < N; i += 2) {
        // (acc[i], acc[i+1]) = (acc[i], acc[i+1]) * scale_log2 + (bias, bias)
        // via fma.rn.ftz.f32x2
        float2 in       = make_float2(acc[i], acc[i + 1]);
        float2 sc       = make_float2(scale_log2, scale_log2);
        float2 bi       = make_float2(bias, bias);
        uint64_t in_b64, sc_b64, bi_b64, out_b64;
        asm("mov.b64 %0, {%1, %2};" : "=l"(in_b64) : "f"(in.x), "f"(in.y));
        asm("mov.b64 %0, {%1, %2};" : "=l"(sc_b64) : "f"(sc.x), "f"(sc.y));
        asm("mov.b64 %0, {%1, %2};" : "=l"(bi_b64) : "f"(bi.x), "f"(bi.y));
        asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
            : "=l"(out_b64) : "l"(in_b64), "l"(sc_b64), "l"(bi_b64));
        float2 out;
        asm("mov.b64 {%0, %1}, %2;" : "=f"(out.x), "=f"(out.y) : "l"(out_b64));
        acc[i]     = out.x;
        acc[i + 1] = out.y;
    }
}


// ----------------------------------------------------------------------------
// Partial-emulation selector.
//
// Returns true
// if the pair (k, j) should go through ex2_emulation_packed_fn; false if
// it should go through ex2_mufu_fn. All arguments are template
// (compile-time) so the selector resolves to a single straight-line
// branch in the compiled code — no runtime divergence.
//
// Guards (all must hold to emulate):
//   (k % FREQ) >= (FREQ - RES)   : k is inside the per-period emu window
//   j >= START_FRG               : fragment is not in the early-MUFU region
//   j <  FRG_CNT - 1             : fragment is not the final one
//
// FREQ == 0 disables emulation entirely (the caller should special-case
// that, e.g. by skipping the branch via if constexpr).
//
// Default RES = 4 matches the in-code default in apply_exp2_convert;
// hd=256 configs in _TUNING_CONFIG override to RES = 6.
// ----------------------------------------------------------------------------
template <int FREQ, int RES, int START_FRG, int FRG_CNT>
__device__ __forceinline__ constexpr bool should_emulate_ex2_fn(int k, int j) {
    if (FREQ == 0) return false;
    bool in_emu_window = (k % FREQ) >= (FREQ - RES);
    bool eligible_frag = (j >= START_FRG) && (j < FRG_CNT - 1);
    return in_emu_window && eligible_frag;
}


// ----------------------------------------------------------------------------
// Fused fragment-aware softmax exp2: applies 2^x to a row, routing each
// pair through emulation or MUFU according to should_emulate_ex2_fn.
//
// Inputs are pre-biased
// (scale_subtract_rowmax_fn has already run). Outputs the BF16-cast
// result into `out_bf16` and overwrites `acc` with the post-exp values.
//
// Layout assumption matches the Python: the row of length N is divided
// into FRG_CNT contiguous fragments of FRG_TILE = N / FRG_CNT elements.
// The pair index k iterates inside one fragment in steps of 2; the
// fragment index j iterates across fragments. For SM100 with
// n_block_size = 128 and frg_tile = 32, this gives FRG_CNT = 4 and the
// inner loop has FRG_TILE/2 = 16 pair-iterations.
//
// FREQ = 0 forces every entry onto MUFU (the SM103 / masked / FP8 cases).
// ----------------------------------------------------------------------------
template <int N, int FRG_CNT, int FREQ, int RES = 4, int START_FRG = 0,
          int EMU_DEGREE = 3>
__device__ __forceinline__ void apply_exp2_convert_row_fn(
    float (&acc)[N], __nv_bfloat16 (&out_bf16)[N]
) {
    static_assert(N % 2 == 0, "row length must be even");
    static_assert(N % FRG_CNT == 0, "row must split evenly into fragments");
    constexpr int FRG_TILE = N / FRG_CNT;
    static_assert(FRG_TILE % 2 == 0, "fragment width must be even");

    #pragma unroll
    for (int j = 0; j < FRG_CNT; ++j) {
        #pragma unroll
        for (int k = 0; k < FRG_TILE; k += 2) {
            int idx = j * FRG_TILE + k;
            if constexpr (FREQ == 0) {
                acc[idx]     = ex2_mufu_fn(acc[idx]);
                acc[idx + 1] = ex2_mufu_fn(acc[idx + 1]);
            } else {
                if constexpr (should_emulate_ex2_fn<FREQ, RES, START_FRG, FRG_CNT>(k, j)) {
                    float2 pr = ex2_emulation_packed_fn<EMU_DEGREE>(
                        make_float2(acc[idx], acc[idx + 1])
                    );
                    acc[idx]     = pr.x;
                    acc[idx + 1] = pr.y;
                } else {
                    acc[idx]     = ex2_mufu_fn(acc[idx]);
                    acc[idx + 1] = ex2_mufu_fn(acc[idx + 1]);
                }
            }
        }
        // Convert the fragment to BF16 in one shot at the end of the j-loop.
        // The conversion of fragment j overlaps with the exp2 computation of
        // fragment j+1 in the kernel's wider pipeline.
        #pragma unroll
        for (int k = 0; k < FRG_TILE; ++k) {
            int idx = j * FRG_TILE + k;
            out_bf16[idx] = __float2bfloat16(acc[idx]);
        }
    }
}


// ----------------------------------------------------------------------------
// Per-config tuning constants from indexed by
// (use_2cta, is_causal, head_dim, is_sm103); SM103 entries all set
// FREQ = 0 because its native MUFU is fast enough that the FMA-pipe
// diversion no longer pays for the extra register pressure.
//
// Emulated entry fraction = (RES / FREQ) * ((FRG_CNT - 1 - START_FRG) / FRG_CNT)
// with FRG_CNT = 4 for n_block_size = 128.
//
//   hd=128 2cta non-causal : FREQ=10, RES=4, START=1 -> 20.0  %
//   hd=128 1cta causal     : FREQ=16, RES=4, START=1 -> 12.5  %
//   hd=192 2cta non-causal : FREQ=16, RES=4, START=0 -> 18.75 %
//   hd=192 1cta causal     : FREQ=32, RES=4, START=1 ->  6.25 %
//   hd=256 2cta any        : FREQ=14, RES=6, START=0 -> 32.14 %
//   any SM103              : FREQ= 0                 ->  0.0  %
// ----------------------------------------------------------------------------

struct Fa4Sm100TuneHd128_2ctaNoncausal { static constexpr int FREQ = 10; static constexpr int RES = 4; static constexpr int START_FRG = 1; };
struct Fa4Sm100TuneHd128_1ctaCausal    { static constexpr int FREQ = 16; static constexpr int RES = 4; static constexpr int START_FRG = 1; };
struct Fa4Sm100TuneHd192_2ctaNoncausal { static constexpr int FREQ = 16; static constexpr int RES = 4; static constexpr int START_FRG = 0; };
struct Fa4Sm100TuneHd192_1ctaCausal    { static constexpr int FREQ = 32; static constexpr int RES = 4; static constexpr int START_FRG = 1; };
struct Fa4Sm100TuneHd256_2cta          { static constexpr int FREQ = 14; static constexpr int RES = 6; static constexpr int START_FRG = 0; };
struct Fa4Sm100TuneSm103               { static constexpr int FREQ = 0;  static constexpr int RES = 4; static constexpr int START_FRG = 0; };
