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

