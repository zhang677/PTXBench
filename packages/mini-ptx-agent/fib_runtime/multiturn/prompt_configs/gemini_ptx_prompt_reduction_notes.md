# Gemini PTX Prompt Reduction Notes

Source runs: Gemini rows in `/home/ubuntu/AccRL/benchmark/experiments.csv`.
Kernel corpus: 15 eval runs, 1174 extracted `kernel_t*.cu` files.
Generated kernel comments were stripped before counting usage.

## Header Size

| header | bytes | lines | code lines | comment lines |
| --- | ---: | ---: | ---: | ---: |
| `simt.h` | 7797 | 110 | 43 | 33 |
| `barrier.h` | 17608 | 234 | 87 | 90 |
| `tma_bf16.h` | 20668 | 379 | 68 | 246 |
| `wgmma_bf16.h` | 18837 | 387 | 137 | 182 |

The largest prompt savings are in comments, especially `tma_bf16.h` and
`wgmma_bf16.h`. The code wrappers are much smaller than the prose.

## High-Value PTX Targets

Keep compact wrappers and minimal operand notes for:

| family | evidence |
| --- | --- |
| `mbarrier.*` | `mbarrier.init`, `try_wait`, and `arrive.expect_tx` each appear in over 1000 generated kernels. |
| `fence.*` | `fence.mbarrier_init.release.cluster` and async proxy fences appear across all 15 runs. |
| `wgmma.*` | WGMMA fence/commit/wait and `m64n64k16` MMA are frequent in 7-9 runs. |
| `cp.async.bulk.tensor` | 2D TMA load/store variants appear in 14 runs. |
| `setmaxnreg.*` | Inc/dec appears in all 15 runs, with 713 total occurrences. |
| `ex2.approx.ftz.f32` | Used in 12 runs, mainly softmax-style kernels. |

## Low-Value Or Specialized Targets

These are real code targets but much lower frequency:

| target | occurrences | runs | prompt action |
| --- | ---: | ---: | --- |
| `elect.sync` / `selp.b32` | 18 / 17 | 3 | Keep only as tiny helper or move to optional SIMT appendix. |
| `cp.async.bulk.global.shared::cta.bulk_group` | 6 | 1 | Move 1D TMA copy helpers to optional appendix unless needed by current workload. |
| `barrier.arrive.aligned` | 4 | 2 | Keep wrapper if named barrier producer/consumer patterns matter; otherwise shorten heavily. |
| `barrier.cluster.wait` | 1 | 1 | Covered by `cluster_sync_fn`; prose can be minimal. |

## Comment-Only Intrinsics

The SIMT comment mentions `__ballot_sync`, `__match_any_sync`, `__popc`, and
`__ffs`. In generated Gemini kernels:

| intrinsic | occurrences | kernel files | runs |
| --- | ---: | ---: | ---: |
| `__ballot_sync` | 1 | 1 | 1 |
| `__match_any_sync` | 0 | 0 | 0 |
| `__popc` | 0 | 0 | 0 |
| `__ffs` | 0 | 0 | 0 |

For prompt reduction, remove the long prose for these from the main header.
If kept at all, use a one-line optional note for `__ballot_sync`.

The broad comment scan also found common CUDA/C++ tokens such as
`__nv_bfloat16`, `__grid_constant__`, `__shared__`, `__align__`, and
`__global__`. Those are not PTX instruction targets and should not drive PTX
prompt retention decisions.

`barrier.h` also mentions `__threadfence()` and `__threadfence_system()` only
in comments. A raw scan of the Gemini kernel corpus found zero calls to either,
so the default prompt does not need their CUDA API prose.

## Suggested Prompt Slimming

1. Keep code wrappers for all high-value PTX targets.
2. Replace long PTX manual excerpts with one-line summaries plus the exact asm
   syntax used by the wrapper.
3. Move descriptor construction tutorials, swizzle tables, and `stmatrix`
   documentation to an optional appendix or workload-specific prompt.
4. Remove SIMT prose for unused intrinsics (`__match_any_sync`, `__popc`,
   `__ffs`) from the default prompt.
5. Keep TMA/WGMMA examples only where they encode easy-to-misremember constants:
   TMA descriptor box dimensions, WGMMA LBO/SBO, swizzle mode, and accumulator
   layout.
