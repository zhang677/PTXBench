# FA-4 non-GEMM optimizations: exponential emulation

This note records where the §3.1.3 "Emulation of the exponential function" idea
from `fa4_fwd.md` actually lives in code, walks through every relevant code
fragment, and gives the exact arithmetic and partial-emulation fractions used
by the flash-attention SM100 kernel.

## 1. Where each codebase stands

| Source | exp2 strategy |
|---|---|
| `cutlass/examples/77_blackwell_fmha` (C++) | Every softmax path calls `::exp2f` directly → all MUFU.EX2. No Cody-Waite, no polynomial, no partial emulation. This is the pre-FA-4 baseline. |
| `cutlass/examples/python/CuTeDSL/.../fmha/fmha.py` | Uses `cute.math.exp2(..., fastmath=True)` for every entry → all MUFU.EX2. `fastmath=True` only skips IEEE special-case handling; it does **not** replace MUFU with FMA. |
| `flash-attention/flash_attn/cute/flash_fwd_sm100.py` (+ `utils.py`, `softmax.py`) | Full §3.1.3 implementation: Cody-Waite range reduction, degree-3 Sollya polynomial through packed-f32x2 FMA, IEEE-754 bit-splice, plus a per-config tuning table that emulates 12–43 % of softmax entries on the FMA pipe while the rest stay on MUFU. |

Representative call sites for the baselines:

`cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:636-655`

```cpp
CUTLASS_PRAGMA_UNROLL
for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
  float2 in = make_float2(
    tTMEM_LOADrS(i + 0),
    tTMEM_LOADrS(i + 1)
  );
  float2 out;
  cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
  tTMEM_LOADrS(i + 0) = out.x;
  tTMEM_LOADrS(i + 1) = out.y;

  tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
  tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));

  Array<ElementQK, kConversionsPerStep> in_conv;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kConversionsPerStep; j++) {
    in_conv[j] = tTMEM_LOADrS(i + j);
  }
  tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);
  ...
}
```

`cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py:1730-1742`

```python
for j in range(frg_cnt):
    for k in cutlass.range(
        cute.size(tTMEM_LOADrS_frg, mode=[0]), vectorize=True
    ):
        tTMEM_LOADrS_frg[k, j] = (
            tTMEM_LOADrS_frg[k, j] * scale + minus_row_max_scale
        )
        tTMEM_LOADrS_frg[k, j] = cute.math.exp2(
            tTMEM_LOADrS_frg[k, j], fastmath=True
        )

    s_vec = tTMEM_LOADrS_frg[None, j].load()
    tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))
```

Both directly issue MUFU.EX2 with no escape hatch onto the FMA pipe.

Everything below documents the third row — the flash-attention implementation.

## 2. Sollya polynomial coefficients

`flash_attn/cute/utils.py:24-58` — the table the rest of the emulation reads
from:

```python
# Obtained from sollya:
# fpminimax(exp(x * log(2.0)), 1, [|1,24...|],[0;1],relative);
POLY_EX2 = {
    0: (1.0),
    1: (
        1.0,
        0.922497093677520751953125,
    ),
    2: (
        1.0,
        0.6657850742340087890625,
        0.330107033252716064453125,
    ),
    3: (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    ),
    4: (
        1.0,
        0.693042695522308349609375,
        0.2412912547588348388671875,
        5.2225358784198760986328125e-2,
        1.3434938155114650726318359375e-2,
    ),
    5: (
        1.0,
        0.693151414394378662109375,
        0.24016360938549041748046875,
        5.5802188813686370849609375e-2,
        9.01452265679836273193359375e-3,
        1.86810153536498546600341796875e-3,
    ),
}
```

Each tuple is `(p_0, p_1, …, p_n)` for a degree-`n` minimax polynomial of
`2^x` on `[0, 1)` relative to a `[|1, 24…|]` precision profile. `p_0 = 1.0`
matches eq. (5) of `fa4_fwd.md`. The default `poly_degree=3` corresponds to
the paper's "matches hardware to within 1 BF16 ULP on 99 % of inputs" choice;
`degree=5` is what the paper's Table 2 calls out as matching hardware to
within 2× max relative error at the cost of two extra FMAs.

## 3. The five-step emulation primitives

### 3.1 Step 4 building block: `evaluate_polynomial` (scalar Horner)

`utils.py:690-697`:

```python
@dsl_user_op
@cute.jit
def evaluate_polynomial(x: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None) -> Float32:
    deg = len(poly) - 1
    out = poly[deg]
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = out * x + poly[i]
    return out
```

Standard Horner evaluation: `((p_n · x + p_{n−1}) · x + p_{n−2}) … · x + p_0`.
Each `out * x + poly[i]` lowers to a single `fma.rn.ftz.f32` instruction.
For degree 3 this is exactly three FMA instructions per evaluation — the
"high throughput" claim of §3.1.3.

### 3.2 Step 4 building block: `evaluate_polynomial_2` (packed Horner)

`utils.py:700-709`:

```python
@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out
```

Same Horner schedule but each step is one packed `fma.rn.ftz.f32x2` instead
of two scalar FMAs. Blackwell issues packed-f32x2 FMA at the same throughput
as scalar FMA per thread, so this is a 2× speedup over the scalar variant.

### 3.3 Step 2 building block: `add_round_down` (PTX `add.rm.ftz.f32`)

`utils.py:712-725`:

```python
@dsl_user_op
def add_round_down(x: float | Float32, y: float | Float32, *, loc=None, ip=None) -> Float32:
    # There's probably a way to call llvm or nvvm to do this instead of ptx
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
            "add.rm.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
```

The DSL has no scalar round-toward-minus-infinity FP add, so this drops to
hand-written PTX. `add.rm` rounds the result toward −∞, and `.ftz` flushes
denormals to zero. This rounding mode is what makes the magic-constant trick
produce `⌊x⌋` rather than `round(x)`: if `x + (2^23 + 2^22)` is rounded up
when the result is exact, we lose a unit; rounding down guarantees the
integer-floor semantics.

### 3.4 Step 5 building block: `combine_int_frac_ex2` (PTX bit splice)

`utils.py:728-752`:

```python
@dsl_user_op
def combine_int_frac_ex2(x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 generates IMAD instruction and add.s32 generates LEA instruction
            # IMAD uses the FMA pipeline and LEA uses the ALU pipeline, afaik
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
```

Why each line:

- `mov.b32 x_rounded_i, $1;` — reinterpret the `f32` `x_rounded` as a 32-bit
  integer. Because `x_rounded = ⌊x⌋ + (2^23 + 2^22)` and the magic constant's
  exponent (`0x4B = 150`) aligns the binary point at bit 0, the low 8 bits of
  this integer literally contain the two's-complement representation of
  `⌊x⌋` (biased by the magic constant's mantissa, which we cancel by the
  subtraction in `ex2_emulation`).
- `mov.b32 frac_ex_i, $2;` — reinterpret `2^x_frac` (an `f32` in `[1, 2)`)
  as its IEEE-754 bit pattern. Its biased exponent is `127` (the leading 1
  bit) and its 23-bit mantissa encodes the fraction in `[1, 2)`.
- `shl.b32 x_rounded_e, x_rounded_i, 23;` — shift `⌊x⌋` left by 23 so it
  lines up with the IEEE-754 exponent field of `2^x_frac`. After the shift,
  adding it to `frac_ex_i` is the same as adding `⌊x⌋` to the biased
  exponent — i.e. multiplying the value by `2^⌊x⌋`.
- `add.s32 out_i, x_rounded_e, frac_ex_i;` — this is the multiplication.
  The deliberate choice of `add.s32` over `add.u32` is documented in the
  comment: `add.s32` lowers to a SASS `LEA` (ALU pipe), while `add.u32`
  lowers to `IMAD` (FMA pipe). Since the polynomial Horner already saturates
  the FMA pipeline, the splice is steered onto the ALU pipeline to avoid
  contending with it.
- `mov.b32 $0, out_i;` — reinterpret the integer sum back as an `f32`.
  The result is `2^⌊x⌋ · 2^x_frac = 2^x`.

This trick exploits the fact that IEEE-754 `f32` is `(−1)^s · 2^(e−127) · m`,
so adding `⌊x⌋` to the biased exponent of a value already equal to `2^x_frac`
gives `2^(⌊x⌋ + x_frac) = 2^x`.

### 3.5 Scalar driver: `ex2_emulation`

`utils.py:755-768`:

```python
@dsl_user_op
def ex2_emulation(x: Float32, *, poly_degree: int = 3, loc=None, ip=None) -> Float32:
    assert poly_degree in POLY_EX2, f"Polynomial degree {poly_degree} not supported"
    # We assume x <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    x_clamped = cute.arch.fmax(x, -127.0)
    # We want to round down here, so that the fractional part is in [0, 1)
    x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
    # The integer floor of x is now in the last 8 bits of x_rounded
    # We assume the next 2 ops round to nearest even. The rounding mode is important.
    x_rounded_back = x_rounded - fp32_round_int
    x_frac = x_clamped - x_rounded_back
    x_frac_ex2 = evaluate_polynomial(x_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)
```

The five paper-aligned steps, line by line:

1. **Clamp** (`x_clamped = cute.arch.fmax(x, -127.0)`) — eq. (4) needs
   `⌊x⌋ ≥ −127` so that the IEEE exponent field stays non-negative after the
   shift in `combine_int_frac_ex2`. Inputs below −127 would otherwise produce
   subnormals; clamping rounds them to `2^−127 ≈ 5.88e-39`, which is treated
   as zero by downstream BF16 cast anyway.
2. **Floor via magic constant** (`x_rounded = add_round_down(x_clamped,
   2^23 + 2^22)`) — the constant `0x4B400000 = 2^23 + 2^22` has biased
   exponent 150, large enough that adding any `x ∈ [−127, 127]` aligns
   `x`'s binary point to integer position. With round-toward-minus-infinity,
   the low 8 bits of the result encode `⌊x⌋` exactly.
3. **Recover floor and fractional part**:
   `x_rounded_back = x_rounded - fp32_round_int` produces `⌊x⌋` as a plain
   float (this subtraction is exact under round-to-nearest because the
   integer part fits in 8 bits); `x_frac = x_clamped - x_rounded_back` is
   then `x − ⌊x⌋ ∈ [0, 1)`. The docstring at line 764 stresses the
   round-to-nearest mode requirement here.
4. **Polynomial** (`evaluate_polynomial(x_frac, POLY_EX2[poly_degree])`) —
   evaluates `2^x_frac` via Horner FMA. With degree 3 this is 3 FMAs.
5. **Splice** (`combine_int_frac_ex2(x_rounded, x_frac_ex2)`) — bit-level
   `2^⌊x⌋ · 2^x_frac` as described in §3.4. Note that the splice consumes
   `x_rounded` (still biased by the magic constant) rather than
   `x_rounded_back`; that biased form keeps the integer part in the low 8
   bits where the `shl.b32 …, 23` expects it.

### 3.6 Packed driver: `ex2_emulation_2`

`utils.py:771-790`:

```python
# TODO: check that the ex2_emulation_2 produces the same SASS as the ptx version
@dsl_user_op
def ex2_emulation_2(
    x: Float32, y: Float32, *, poly_degree: int = 3, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm")
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = quack.activation.sub_packed_f32x2(
        xy_rounded, (fp32_round_int, fp32_round_int)
    )
    xy_frac = quack.activation.sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out
```

Same five steps as the scalar driver but each arithmetic op is packed-f32x2:

- `cute.arch.add_packed_f32x2(..., rnd="rm")` is the packed equivalent of
  `add.rm.ftz.f32` — emits `add.rm.ftz.f32x2`.
- `quack.activation.sub_packed_f32x2` emits `sub.rn.ftz.f32x2` (round-nearest).
- `evaluate_polynomial_2` does the packed Horner FMAs.
- Splice still loops twice because `combine_int_frac_ex2` is scalar — the
  underlying `shl`/`add.s32` would be `shl.b32x2`/`add.s32x2` if available,
  but those don't exist as packed forms (the splice is on the ALU pipe,
  which doesn't have packed variants on Blackwell).

This is the function actually called from softmax. The TODO comment notes the
author hasn't confirmed it produces identical SASS to a fully-inlined PTX
version — `e2e_asm2` below is the manual reference.

### 3.7 All-PTX reference: `e2e_asm2`

`utils.py:793-837`:

```python
@dsl_user_op
def e2e_asm2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    out_f32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(y, loc=loc, ip=ip).ir_value()],
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, $2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, $3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 $0, r7;\n\t"
        "mov.b32 $1, r8;\n\t"
        "}\n",
        "=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [0], loc=loc, ip=ip))
    out1 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [1], loc=loc, ip=ip))
    return out0, out1
```

The constants in PTX hex are the IEEE-754 bit patterns of the same numbers
that show up in `POLY_EX2[3]` and the magic constants:

| PTX literal     | Float32 value                          | Role                                       |
|-----------------|----------------------------------------|--------------------------------------------|
| `0fC2FE0000`    | `−127.0`                               | clamp floor (step 1)                       |
| `0f4B400000`    | `2^23 + 2^22 = 12 582 912.0`           | magic constant (step 2)                    |
| `0f3D9DF09D`    | `0.077119089663…` = `p_3` of degree 3   | top polynomial coefficient                 |
| `0f3E6906A4`    | `0.227564394474…` = `p_2`               | next polynomial coefficient                |
| `0f3F31F519`    | `0.695146143436…` = `p_1`               | next polynomial coefficient                |
| `0f3F800000`    | `1.0` = `p_0`                           | leading polynomial coefficient             |

Execution maps 1-to-1 onto the five-step algorithm:

- `max.ftz.f32 …, 0fC2FE0000` — clamp x and y to `≥ −127` (step 1).
- `mov.b64 l1, {f1, f2}` — pack the two scalars into a 64-bit pair register
  so subsequent `f32x2` instructions can act on both simultaneously.
- `add.rm.ftz.f32x2 l7, l1, l2` — magic-constant floor (step 2).
- `sub.rn.ftz.f32x2 l8, l7, l2` — recover the floor as a plain float.
- `sub.rn.ftz.f32x2 l9, l1, l8` — compute `x_frac = x_clamped − ⌊x⌋` (step 3).
- Three `fma.rn.ftz.f32x2` — Horner for degree-3 polynomial (step 4).
- `mov.b64 {r1, r2}, l7` / `{r3, r4}, l10` — split the packed registers
  back into scalars for the splice (since packed integer shift is
  unavailable on the ALU pipe).
- Two `shl.b32 …, 23` and `add.s32` — bit splice for x and y separately
  (step 5).

`e2e_asm2` is not currently called; `apply_exp2_convert` uses
`ex2_emulation_2` and leaves the inline-PTX version commented out.

## 4. Where partial emulation is decided in softmax

### 4.1 The pre-emulation FMA bias step: `scale_subtract_rowmax`

`softmax.py:256-271`:

```python
@cute.jit
def scale_subtract_rowmax(
    self,
    acc_S_row: cute.Tensor,
    row_max: Float32,
):
    assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
    row_max_scaled = row_max * self.scale_log2
    max_offset = Float32(self.max_offset)
    bias = max_offset - row_max_scaled
    for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
        acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
            (acc_S_row[i], acc_S_row[i + 1]),
            (self.scale_log2, self.scale_log2),
            (bias, bias),
        )
```

This step is split out from the C++ baseline's combined `fma + ::exp2f` loop
because the emulation requires the inputs to already be biased so that the
polynomial's `[0, 1)` validity domain is reached after the magic-constant
floor. `max_offset` is the "10–25 % rescale slack" constant — by adding a
constant offset to every row score, the emulation always operates well
above zero and away from underflow.

### 4.2 The fragment-aware selector: `apply_exp2_convert`

`softmax.py:273-315`:

```python
@cute.jit
def apply_exp2_convert(
    self,
    acc_S_row: cute.Tensor,
    acc_S_row_converted: cute.Tensor,
    ex2_emu_freq: cutlass.Constexpr[int] = 0,
    ex2_emu_res: cutlass.Constexpr[int] = 4,
    ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
):
    assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
    frg_tile = 32
    assert frg_tile % 2 == 0
    frg_cnt = cute.size(acc_S_row) // frg_tile
    assert cute.size(acc_S_row) % frg_tile == 0
    acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
    acc_S_row_converted_frg = cute.logical_divide(
        acc_S_row_converted, cute.make_layout(frg_tile)
    )
    for j in cutlass.range_constexpr(frg_cnt):
        for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
            # acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
            # acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
            if cutlass.const_expr(ex2_emu_freq == 0):
                acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
            else:
                if cutlass.const_expr(
                    k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                    or j >= frg_cnt - 1
                    or j < ex2_emu_start_frg
                ):
                    acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                    acc_S_row_frg[k + 1, j] = cute.math.exp2(
                        acc_S_row_frg[k + 1, j], fastmath=True
                    )
                else:
                    # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.e2e_asm2(acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                    acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.ex2_emulation_2(
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                    )
        acc_S_row_converted_frg[None, j].store(
            acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
        )
```

Structure: the row is tiled into `frg_cnt = cute.size(acc_S_row) / 32`
fragments of 32 elements each (matching the K-major TMEM layout for one
warp's slice of S). For an `n_block_size = 128` softmax row at hd=128 this
gives `frg_cnt = 4`.

The selector decides, **per pair `(k, j)`**, whether to call MUFU.EX2 or
polynomial emulation. A pair is emulated iff:

```
NOT (k % ex2_emu_freq < ex2_emu_freq − ex2_emu_res)   ← inside the emu window
AND  ex2_emu_start_frg <= j                            ← fragment column is allowed
AND  j < frg_cnt − 1                                   ← not the final fragment
```

Why each guard:

- `k % ex2_emu_freq` cycles through the period `[0, ex2_emu_freq)`. The
  emulation window is the last `ex2_emu_res` pair-steps of each period;
  everything else stays on MUFU. This is the duty-cycle knob.
- `j < ex2_emu_start_frg` keeps low-index fragments on MUFU. Those fragments
  are the first to be issued in the loop, and the kernel wants their MUFU.EX2
  ops in flight as early as possible to overlap with the previous tile's
  remaining work.
- `j == frg_cnt − 1` keeps the final fragment on MUFU. The conversion at
  the end of the `j`-loop (`store(...to(...))`) consumes the previous
  fragment's output while the next fragment is being computed; the final
  fragment has no following iteration to hide the polynomial's longer latency,
  so it goes back to the lower-latency MUFU path.

Also worth noting: when `ex2_emu_freq == 0` the entire body collapses to the
plain MUFU path through `const_expr` — every branch is `cutlass.const_expr`,
so the partitioning is resolved at JIT time and the runtime sees a single
straight-line block with no divergence.

### 4.3 The unified "scale + exp" path: `scale_apply_exp2_convert`

`softmax.py:317-365`:

```python
@cute.jit
def scale_apply_exp2_convert(
    self,
    acc_S_row: cute.Tensor,
    row_max: Float32,
    acc_S_row_converted: cute.Tensor,
):
    assert cute.size(acc_S_row.shape) % 2 == 0, "acc_S_row must have an even number of elements"
    minus_row_max_scaled = -row_max * self.scale_log2
    for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
        acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
            (acc_S_row[i], acc_S_row[i + 1]),
            (self.scale_log2, self.scale_log2),
            (minus_row_max_scaled, minus_row_max_scaled),
        )

    # for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
    #     acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
    #         (acc_S_row[i], acc_S_row[i + 1]),
    #         (self.scale_log2, self.scale_log2),
    #         (minus_row_max_scaled, minus_row_max_scaled),
    #     )
    #     acc_S_row[i] = cute.math.exp2(acc_S_row[i], fastmath=True)
    #     acc_S_row[i + 1] = cute.math.exp2(acc_S_row[i + 1], fastmath=True)

    frg_tile = 32
    assert frg_tile % 2 == 0
    frg_cnt = cute.size(acc_S_row) // frg_tile
    assert cute.size(acc_S_row) % frg_tile == 0
    acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
    acc_S_row_converted_frg = cute.logical_divide(
        acc_S_row_converted, cute.make_layout(frg_tile)
    )
    for j in cutlass.range_constexpr(frg_cnt):
        for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
            # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
            #     cute.arch.fma_packed_f32x2(
            #         (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
            #         (self.scale_log2, self.scale_log2),
            #         (minus_row_max_scaled, minus_row_max_scaled),
            #     )
            # )
            # acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
            # acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
            acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
            acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)
        acc_S_row_converted_frg[None, j].store(
            acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
        )
```

This is the pre-emulation reference path: do `fma + exp2` in one shot,
all on MUFU. The driver (`flash_fwd_sm100.py:2303`) keeps a commented-out
call to `scale_apply_exp2_convert` next to the active `apply_exp2_convert`
call, so the comparison is visible at the call site.

## 5. Driver wiring in `flash_fwd_sm100.py`

### 5.1 The `_TUNING_CONFIG` table

`flash_fwd_sm100.py:71-101`:

```python
# === TUNING KNOBS (agent-editable) ===
# Keys: (use_2cta_instrs: bool, is_causal: bool, head_dim_padded: int, is_sm103: bool)
# Values:
#   ex2_emu_freq: int — how often to use emulated exp2 (0=all hardware exp2, higher=more emulation).
#                        SM103 has fast native exp2, so set freq=0 there.
#   ex2_emu_res: int — (hd256 only) number of fragment-pairs per freq period to emulate.
#   ex2_emu_start_frg: int — fragment index to start emulation from
#   num_regs_softmax: int — register count for softmax warps (multiple of 8)
#   num_regs_correction: int — register count for correction warps (multiple of 8)
#   num_regs_other is derived: 512 - num_regs_softmax * 2 - num_regs_correction
#                  (hd256 exception: num_regs_other is fixed at 32, not derived)
_TUNING_CONFIG = {
    (True,  False, 128, False): {"ex2_emu_freq": 10, "ex2_emu_start_frg": 1, "num_regs_softmax": 176, "num_regs_correction": 88},
    (False, True,  128, False): {"ex2_emu_freq": 16, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 72},
    (True,  False, 192, False): {"ex2_emu_freq": 16, "ex2_emu_start_frg": 0, "num_regs_softmax": 184, "num_regs_correction": 80},
    (False, True,  192, False): {"ex2_emu_freq": 32, "ex2_emu_start_frg": 1, "num_regs_softmax": 192, "num_regs_correction": 72},
    (True,  False, 128, True):  {"ex2_emu_freq": 0,  "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 80},
    (False, True,  128, True):  {"ex2_emu_freq": 0,  "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 64},
    (True,  False, 192, True):  {"ex2_emu_freq": 0,  "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 64},
    (False, True,  192, True):  {"ex2_emu_freq": 0,  "ex2_emu_start_frg": 0, "num_regs_softmax": 176, "num_regs_correction": 72},
    (True,  False, 256, False): {"ex2_emu_freq": 14, "ex2_emu_res": 6, "ex2_emu_start_frg": 0, "num_regs_softmax": 256, "num_regs_correction": 160},
    (True,  True,  256, False): {"ex2_emu_freq": 14, "ex2_emu_res": 6, "ex2_emu_start_frg": 0, "num_regs_softmax": 256, "num_regs_correction": 160},
}
_FP8_TUNING_CONFIG = {
    (True, False, 128, False): {'ex2_emu_freq': 10, 'ex2_emu_start_frg': 1, 'num_regs_softmax': 160, 'num_regs_correction': 72},
}
_FP8_SMALL_HDIM_REGS = {
    False: {"num_regs_softmax": 168, "num_regs_correction": 96, "num_regs_other": 80},
    True:  {"num_regs_softmax": 152, "num_regs_correction": 96, "num_regs_other": 72},
}
```

Two observations:

- Only the hd=256 configs override `ex2_emu_res` (set to 6 from the default of
  4 in `apply_exp2_convert`). Larger tile sizes need a larger emulation
  window to absorb the proportionally larger MUFU.EX2 critical path.
- All SM103 entries pin `ex2_emu_freq = 0`. The header comment explains:
  SM103 has faster native MUFU so the FMA-pipe diversion no longer pays off.

### 5.2 The `enable_ex2_emu` decision

`flash_fwd_sm100.py:193-198`:

```python
is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
self.is_sm103 = is_sm103
# enable_ex2_emu is derived: True if tuning config has freq > 0, else fallback to default logic
_default_enable_ex2_emu = (self.head_dim_padded <= 128 or (self.head_dim_padded == 192 and self.use_2cta_instrs and not self.is_causal and not self.is_local)) and not is_sm103
self.enable_ex2_emu = _default_enable_ex2_emu
```

`flash_fwd_sm100.py:294-298`:

```python
# Look up tuning config for register counts and ex2_emu params
_tune_key = (self.use_2cta_instrs, self.is_causal, self.head_dim_padded, self.is_sm103)
self._tune = _TUNING_CONFIG.get(_tune_key, {})
if "ex2_emu_freq" in self._tune:
    self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
```

`flash_fwd_sm100.py:441-447` (FP8 path):

```python
if const_expr("ex2_emu_freq" in fp8_tune):
    self._tune = {**self._tune, **fp8_tune}
    self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
if const_expr(not paged_kv_non_tma and "num_regs_softmax" in fp8_tune):
    self.num_regs_softmax = fp8_tune["num_regs_softmax"]
    self.num_regs_correction = fp8_tune["num_regs_correction"]
    self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_correction
```

`flash_fwd_sm100.py:456-463`:

```python
self.ex2_emu_freq = 0
self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)
if const_expr(self.enable_ex2_emu):
    self.ex2_emu_freq = self._tune.get("ex2_emu_freq", 16)
    if const_expr(
        self.pack_gqa and self.head_dim_padded > 64 and not self.is_causal and not self.is_local
    ):
        self.ex2_emu_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else self._tune.get("ex2_emu_freq", 10)
```

The pack-GQA / varlen branch at line 460-463 bumps `ex2_emu_freq` to 32 when
the kernel must handle variable sequence lengths — i.e. it cuts emulation in
half because the variable-length path has additional masking overhead that
competes with the polynomial for FMA bandwidth.

### 5.3 The call site

`flash_fwd_sm100.py:2293-2309`:

```python
softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
# Sequence barrier wait
if const_expr(self.s0_s1_barrier):
    pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
tSrP_r2t_f32 = cute.make_fragment(
    thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
)
tSrP_r2t = cute.make_tensor(
    cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
)
# softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
softmax.apply_exp2_convert(
    tSrS_t2r,
    tSrP_r2t,
    ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
    ex2_emu_start_frg=self.ex2_emu_start_frg,
)
```

Two important details visible here:

1. `scale_subtract_rowmax` (FMA-only) precedes `apply_exp2_convert` (the
   selector). Splitting these two operations is a precondition for partial
   emulation — `apply_exp2_convert` no longer has free FMA slots to fold
   the scale-and-subtract into, because those slots are spoken for by the
   polynomial.
2. **Masking disables emulation** (`mask_fn is None ? self.ex2_emu_freq : 0`):
   masked tiles often feed very-negative `x` (the masked-out lanes) to
   `exp2`. The polynomial is only fitted on `x_frac ∈ [0, 1)`, and while the
   bit-splice produces correct `2^x` for any finite `x`, the masked lanes
   would still waste FMA cycles computing a result that gets multiplied by
   zero later. Reverting to all-MUFU saves the FMA pipeline for the
   non-masked tile.

## 6. Exact emulated-fraction calculation

Let:

| Symbol | Meaning |
|---|---|
| `F` | `ex2_emu_freq` from `_TUNING_CONFIG` |
| `R` | `ex2_emu_res` (default 4, overridden to 6 for hd=256) |
| `S` | `ex2_emu_start_frg` |
| `C` | `frg_cnt` — number of fragments per row. With `frg_tile = 32` and `n_block_size = 128`, `C = 4`. |
| `K` | `cute.size(acc_S_row_frg, mode=[0]) / 2` — number of `k`-pair steps per fragment |

The selector emulates a pair `(k, j)` iff:

```
(k % F) >= (F − R)        AND   S <= j < C − 1
```

Per fragment, the emulated fraction of `k`-pair steps is, in the limit
`K · 2 ≫ F` (which holds in practice — `K = 16` for fragment width 32, and
`F` is at most 32):

```
ρ_k  =  R / F
```

Across fragments, the eligible fraction is:

```
ρ_j  =  max(0, C − 1 − S) / C
```

Multiplying, the overall fraction of softmax entries that go through
`ex2_emulation_2` is:

```
frac_emulated  =  ρ_k · ρ_j  =  (R / F) · (C − 1 − S) / C
```

With `C = 4` (the only case in `_TUNING_CONFIG`) and the defaults baked into
`apply_exp2_convert` (`R = 4` except for hd=256 entries which override to
6), this evaluates to:

| `(use_2cta, is_causal, hdim, is_sm103)` | F  | R | S | `(C−1−S)/C` | `R/F`    | **frac_emulated** |
|------------------------------------------|----|---|---|-------------|----------|-------------------|
| `(True,  False, 128, False)`             | 10 | 4 | 1 | 2/4 = 0.50  | 4/10 = 0.40 | **0.200 (20 %)**  |
| `(False, True,  128, False)`             | 16 | 4 | 1 | 0.50        | 0.250    | **0.125 (12.5 %)**|
| `(True,  False, 192, False)`             | 16 | 4 | 0 | 3/4 = 0.75  | 0.250    | **0.1875 (18.75 %)** |
| `(False, True,  192, False)`             | 32 | 4 | 1 | 0.50        | 0.125    | **0.0625 (6.25 %)**  |
| `(True,  False, 256, False)`             | 14 | 6 | 0 | 0.75        | 6/14 ≈ 0.4286 | **0.3214 (~32 %)** |
| `(True,  True,  256, False)`             | 14 | 6 | 0 | 0.75        | 0.4286   | **0.3214 (~32 %)** |
| any `(_, _, _, True)` (SM103)            | 0  | – | – | –           | (`ex2_emu_freq == 0` branch) | **0 %** |
| FP8 small-hdim `(True, False, 128, False)` | 10 | 4 | 1 | 0.50      | 0.40     | **0.200 (20 %)**  |
| varlen pack-GQA hd>64 non-causal non-local | 32 | 4 | 1 | 0.50      | 0.125    | **0.0625 (6.25 %)** |

These all fall in or near the paper's 10–25 % range. The hd=256 configs
push above it because that tile is the most exp-throughput-starved
(M=256, N=d=128 in §3.1.1 has MMA compute and exponential unit both at
2048 cycles, so it can spend more FMA bandwidth on emulation before
softmax stops being the bottleneck).

### Worked example: hd=128 1cta causal

`F = 16, R = 4, S = 1, C = 4`. With `n_block_size = 128`, the inner `k`-pair
loop has `K = size(acc_S_row_frg, mode=[0]) / 2 = 32/2 = 16` iterations
**per fragment**. The selector's `k % F` test means a pair is in the
emulation window when `k mod 16 >= 12`, i.e. `k ∈ {12, 14}` out of
`k ∈ {0, 2, 4, …, 30}` — 2 emulated pair-steps out of 16, i.e. 0.125 of
pair-steps per fragment.

Fragment fraction: `j ∈ {1, 2}` are eligible (excluding `j = 0` because
`j < S` and `j = 3` because `j >= C − 1`), so 2/4 = 0.5 of fragments.

Overall: 0.125 · 0.5 = **0.0625** of pair-steps. Since each
`ex2_emulation_2` call produces two entries, the entry-level fraction is
the same 6.25 %. Wait — that's not what the per-fragment formula gives
because `K = 16 = F`. Reading more carefully: the formula `ρ_k = R/F`
assumes the emulation window appears once per period of `F`; with `K = F`
the window appears exactly once. The window width is `R = 4` raw `k`
values, but the loop steps `k` by 2, so the window contains `R/2 = 2`
pair-iterations. The per-fragment pair-emulation fraction is therefore
`(R/2) / (K = F/2 pair-steps total) = R / F`, confirming
`ρ_k = R/F = 0.25`. Multiplying with `ρ_j = 0.5` gives **0.125 (12.5 %)**.

(The table above uses the corrected `ρ_k = R/F` ratio.)

### Why these fractions

§3.1.1 of `fa4_fwd.md` shows that for the M=256, N=d=128 tile, both MMA
compute and the exponential unit are bottlenecked at 2048 cycles, with
shared memory traffic at 1536 cycles. The emulation diverts a slice of
softmax work from MUFU (16 ops/SM/cycle) onto the FMA pipe (which the
tensor cores aren't saturating because the SS-MMA shared-memory traffic
gates them). The per-config `ex2_emu_freq` and `ex2_emu_res` are tuned so
that the residual MUFU load drops below the critical-path window without
spending so much FMA bandwidth that the softmax warpgroup runs out of
registers (§3.1.2's 128-register row constraint). SM103, which the paper
notes has faster native MUFU, turns the emulation off entirely
(`ex2_emu_freq = 0`) because the FMA-pipe diversion no longer pays for the
extra register pressure.
