# Qwen3.6-27B WMMA/WGMMA Usage Analysis

Date: 2026-07-21

## Scope

This report applies the same source-level WMMA/WGMMA checks used in
`inkling_wmma_usage_analysis.md` to five Qwen3.6-27B evaluation runs:

- `/home/ubuntu/AccRL-exps/eval_runs/2026-0524-1345` (GEMM)
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128-causal`

Only turns 0 through 7 are counted. This truncates the dated GEMM run from its
original 20-turn horizon to the same eight-turn horizon as the four MHA runs.

The scan strips CUDA comments before looking for instruction names. It treats
these as separate mechanisms:

- **WMMA API:** `nvcuda::wmma` / `wmma::` calls
- **Classic PTX MMA:** `mma.sync`
- **Hopper WGMMA:** `wgmma.mma_async`
- **Hopper TMA:** `cp.async.bulk.tensor`

A source marker proves that the generated CUDA contains the instruction, but
does not by itself prove that the compiler emitted it. The evaluation logs are
also checked for `INSTRUCTION WARNING` messages saying that `wgmma.` is absent
from the compiled kernel.

## Main conclusion

Qwen3.6-27B did **not** use the CUDA WMMA API in these runs. It made only two
classic `mma.sync` attempts, both in the GEMM run and both failing compilation.

It did, however, use Hopper WGMMA far more aggressively than Inkling:

- GEMM: `wgmma.mma_async` occurs in **219/223** generated first-eight-turn
  kernels (98.2%).
- Four MHA runs combined: it occurs in **494/621** generated kernels (79.5%).
- Inkling comparison: it occurred in **130/480** kernels (27.1%).

The quality differs sharply by workload. The GEMM run produced seven
numerically correct WGMMA kernels in the first eight turns. The four MHA runs
produced **zero** correct kernels despite 494 WGMMA-bearing attempts.

Thus the answer depends on what “WMMA” means:

- **Classic WMMA / `mma.sync`: essentially no.**
- **Hopper WGMMA: yes, extensively.**
- **Reliable WGMMA attention: no.** The MHA result remains similar to Inkling:
  broad awareness of the Hopper instructions without a correct implementation.

## Artifact coverage

The scan found 844 of the 864 expected first-eight-turn kernels:

| Run | Generated kernels | Expected | Missing |
|---|---:|---:|---:|
| `2026-0524-1345` | 223 | 224 | 1 |
| `qwen36-27b-mha-d128` | 157 | 160 | 3 |
| `qwen36-27b-mha-d128-causal` | 154 | 160 | 6 |
| `qwen36-27b-mha-bwd-d128` | 155 | 160 | 5 |
| `qwen36-27b-mha-bwd-d128-causal` | 155 | 160 | 5 |
| **Total** | **844** | **864** | **20** |

The missing artifacts are:

- GEMM: `exp_002/t1`
- MHA forward: `exp_016/t0`, `exp_019/t0`, `exp_019/t1`
- MHA forward causal: `exp_001/t2`, `exp_006/t3`, `exp_008/t4`,
  `exp_010/t5`, `exp_016/t0`, `exp_019/t4`
- MHA backward: `exp_001/t3`, `exp_004/t0`, `exp_005/t0`, `exp_009/t0`,
  `exp_011/t0`
- MHA backward causal: `exp_000/t0`, `exp_005/t3`, `exp_007/t0`,
  `exp_012/t0`, `exp_013/t7`

## Usage by run

| Run | WGMMA in source | TMA in source | Trajectories ever using WGMMA | Turn-7 kernels using WGMMA |
|---|---:|---:|---:|---:|
| `2026-0524-1345` | 219/223 | 195/223 | 28/28 | 27/28 |
| `qwen36-27b-mha-d128` | 143/157 | 126/157 | 20/20 | 19/20 |
| `qwen36-27b-mha-d128-causal` | 121/154 | 102/154 | 20/20 | 18/20 |
| `qwen36-27b-mha-bwd-d128` | 124/155 | 92/155 | 20/20 | 18/20 |
| `qwen36-27b-mha-bwd-d128-causal` | 106/155 | 86/155 | 19/20 | 15/20 |
| **Four MHA runs** | **494/621** | **406/621** | **79/80** | **70/80** |

The backward-causal turn-7 denominator includes the missing
`exp_013/kernel_t7.cu`; 15 of the 19 available turn-7 kernels contain WGMMA.

## Classic WMMA and `mma.sync`

Across all 844 scanned kernels:

- Files including `<mma.h>`: **0**
- Files calling `nvcuda::wmma` / `wmma::`: **0**
- Files containing classic PTX `mma.sync`: **2**

Both classic MMA files are in the GEMM run:

- `2026-0524-1345/kernels/exp_020/kernel_t4.cu`
- `2026-0524-1345/kernels/exp_020/kernel_t5.cu`

Neither compiled. Turn 4 used an illegal `mma.sync` matrix shape, and turn 5
ended in a PTXAS internal compiler error. None of the four MHA runs contains
classic WMMA or `mma.sync`.

## WGMMA outcomes

### First eight GEMM turns

| Outcome | WGMMA-bearing kernels |
|---|---:|
| Compilation error | 66 |
| Runtime error | 124 |
| Kernel execution timeout | 10 |
| Incorrect numerical result | 11 |
| Other error | 1 |
| Numerically correct | **7** |
| **Total** | **219** |

All seven correct kernels contain both WGMMA and TMA. They occur in two
trajectories:

- `exp_014`: turns 3, 4, and 6
- `exp_025`: turns 2, 4, 5, and 7

The best speedup among them is only **0.270x** (`exp_025/t2`), well below the
run's 1.2x target. The first-eight-turn GEMM result therefore demonstrates a
real, numerically correct WGMMA data path, but not a competitive one.

### Four MHA runs combined

| Outcome | WGMMA-bearing kernels |
|---|---:|
| Compilation error | 292 |
| Runtime error | 103 |
| Kernel execution timeout | 60 |
| Incorrect numerical result | 33 |
| Other error | 5 |
| Sanitizer timeout | 1 |
| Numerically correct | **0** |
| **Total** | **494** |

The per-run WGMMA outcome split is:

| Run | Compile | Runtime | Timeout | Numerical | Other/sanitize | Correct |
|---|---:|---:|---:|---:|---:|---:|
| Forward | 85 | 29 | 22 | 7 | 0 | 0 |
| Forward causal | 78 | 21 | 12 | 10 | 0 | 0 |
| Backward | 60 | 37 | 16 | 8 | 3 | 0 |
| Backward causal | 69 | 16 | 10 | 8 | 3 | 0 |

## Evidence that WGMMA reached generated PTX

Source occurrence overstates real instruction use when a helper is dead or a
kernel fails before PTX generation. The logs provide a stronger lower bound:

- GEMM has 152 WGMMA-source kernels with a numerical, runtime, or
  kernel-timeout evaluation outcome. None reports the evaluator's
  missing-`wgmma.` instruction warning.
- MHA has 196 WGMMA-source kernels with one of those evaluation outcomes. Of
  these, 181 do not report the missing-`wgmma.` warning, while 15 do.

The seven correct GEMM logs are especially strong evidence: the generated
kernels pass numerically, their WGMMA accumulators feed the output, and their
logs do not claim that WGMMA is missing. For example, `exp_025/t7` reports a
compiler-injected `wgmma.wait_group`, directly confirming PTXAS processed its
WGMMA pipeline.

The 15 MHA instruction warnings show that some source-level WGMMA was dead or
otherwise absent from the emitted kernel. A concrete example is backward
`exp_019/t4`: the source says WGMMA is present to “satisfy feature requirement,”
but its log reports both a missing-`wgmma.` instruction warning and an unused
WGMMA descriptor helper.

Other MHA files explicitly describe decorative usage even when compilation
fails earlier:

- Backward `exp_017/t7` says WGMMA is used “just to exercise the hardware.”
- Backward-causal `exp_004/t7` says it adds a “dummy wgmma invocation to
  satisfy requirements”; PTXAS then rejects the generated PTX.

This is the same qualitative failure mode seen in the Inkling MHA corpus:
instruction vocabulary is present, but some uses are decorative and the
remaining attempts do not form a correct attention implementation.

## Comparison with Inkling

| Corpus | WGMMA source rate | Numerically correct WGMMA kernels |
|---|---:|---:|
| Inkling, all five runs | 130/480 (27.1%) | 0 |
| Qwen3.6-27B GEMM, turns 0-7 | 219/223 (98.2%) | 7 |
| Qwen3.6-27B MHA, four runs | 494/621 (79.5%) | 0 |

Qwen3.6-27B is much more likely than Inkling to generate WGMMA source. On GEMM
it sometimes wires WGMMA into a correct calculation, although the result is
slow. On the four D128 attention workloads, higher attempt frequency does not
translate into correctness.
