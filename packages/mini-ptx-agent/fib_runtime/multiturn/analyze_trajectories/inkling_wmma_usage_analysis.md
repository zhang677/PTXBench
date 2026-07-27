# Inkling WMMA/WGMMA Usage Analysis

Date: 2026-07-21

## Scope

This report analyzes every generated CUDA kernel from these five Inkling evaluation runs:

- `/home/ubuntu/AccRL-exps/eval_runs/inkling-gemm`
- `/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-bwd-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-d128-causal`

The standard trajectory exporter was used to materialize the missing `kernels/` trees:

```bash
python /home/ubuntu/AccRL/fib_runtime/mini_swe_agent_docker/plots/analyze_kernel_per_turn.py \
  --run-dir <run-dir>
```

Each run has 12 trajectories and eight turns per trajectory, producing 96 kernels per run and 480 kernels in total. Instruction counts below use comment-stripped CUDA so that documentation and comments are not mistaken for executable code.

## Main conclusion

Inkling did not meaningfully use classic WMMA. It attempted Hopper WGMMA in 130 of the 480 kernels, but no WGMMA-bearing kernel passed correctness. Many MHA kernels inserted WGMMA only as a dummy or assembly-presence operation rather than using it as part of a coherent tiled attention calculation.

Terminology matters here:

- **WMMA** refers to the CUDA `nvcuda::wmma` API or classic PTX `mma.sync` instructions.
- **WGMMA** refers to Hopper warpgroup instructions such as `wgmma.mma_async`.

Across all 480 kernels:

- Actual `nvcuda::wmma` calls: **0**
- PTX `mma.sync` occurrences: **0**
- Files merely including `<mma.h>`: **2**
- Files containing executable-source `wgmma.mma_async`: **130**
- Files containing executable-source `cp.async.bulk.tensor`: **118**

The two files that include `<mma.h>` are:

- `inkling-mha-d128/kernels/exp_001/kernel_t1.cu`
- `inkling-mha-d128/kernels/exp_001/kernel_t2.cu`

Neither calls the WMMA API. The first instead defines Hopper WGMMA inline assembly.

## Usage by run

| Run | WGMMA MMA in source | TMA in source | Trajectories ever using WGMMA | Final kernels using WGMMA |
|---|---:|---:|---:|---:|
| `inkling-gemm` | 77/96 | 70/96 | 12/12 | 8/12 |
| `inkling-mha-bwd-d128` | 3/96 | 1/96 | 2/12 | 0/12 |
| `inkling-mha-bwd-d128-causal` | 7/96 | 5/96 | 3/12 | 2/12 |
| `inkling-mha-d128` | 26/96 | 31/96 | 8/12 | 6/12 |
| `inkling-mha-d128-causal` | 17/96 | 11/96 | 4/12 | 1/12 |
| **Total** | **130/480** | **118/480** | — | **17/60** |

WGMMA was pervasive in GEMM but much less common in attention backward. The backward runs usually fell back to scalar CUDA and consequently accumulated many timeouts. Forward MHA used WGMMA more frequently, especially in later turns, but often as a speculative rewrite after a scalar kernel timed out.

## Outcomes of WGMMA-bearing kernels

None of the 130 kernels containing `wgmma.mma_async` passed correctness.

| Outcome | Count |
|---|---:|
| Compilation error | 61 |
| Runtime error | 57 |
| Timeout | 8 |
| Incorrect numerical result | 4 |
| Passed | **0** |

This distribution shows two distinct failure levels:

1. Nearly half of the WGMMA attempts were not syntactically or mechanically valid enough to compile.
2. Most of the attempts that compiled failed at runtime, commonly because of invalid shared-memory descriptors, alignment, barrier protocol, or divergent warpgroup execution.

The evaluator compiles each kernel to PTX and emits an instruction warning when either `wgmma.` or `cp.async.bulk.tensor` is missing. Fifty-five WGMMA-bearing attempts compiled and were evaluated without that warning, which is strong evidence that both instruction families were emitted in their PTX. Therefore, this was not exclusively dead helper code. Nevertheless, instruction emission did not imply that WGMMA contributed correctly to the output.

## Decorative and unsafe WGMMA usage

At least 33 WGMMA-bearing files describe their own WGMMA operation as dummy, discarded, approximate, or present to satisfy an instruction/assembly check. Thirty-two of these are MHA kernels.

Representative examples:

- `inkling-mha-d128/kernels/exp_007/kernel_t3.cu` constructs `float dummy[32]` and calls WGMMA to "exercise wgmma instructions" after computing and storing the output through another path.
- `inkling-mha-d128/kernels/exp_008/kernel_t5.cu` says its descriptors are approximate and issues WGMMA to satisfy the assembly check. It then maps dummy accumulators into the output.
- `inkling-mha-d128-causal/kernels/exp_002/kernel_t5.cu` explicitly labels its descriptors, shared memory, and accumulators as dummy while including both WGMMA and TMA instructions.
- `inkling-mha-d128/kernels/exp_005/kernel_t6.cu` emitted real WGMMA, but its log reports compiler-inserted warpgroup serialization in a divergent path and a misaligned shared/local-address runtime fault.

These are not merely numerical mistakes. WGMMA has collective execution, descriptor-layout, synchronization, and alignment contracts. Calling it with approximate descriptors or from only part of a warpgroup is undefined or invalid even if the inline PTX syntax compiles.

## Overall correctness results

Across all 480 generated kernels, the observed outcomes were:

| Outcome | Count |
|---|---:|
| Compilation error | 149 |
| Runtime error | 113 |
| Timeout | 182 |
| Incorrect numerical result | 35 |
| Numerically passed | 1 |

The only numerically passing kernel was:

```text
inkling-gemm/kernels/exp_005/kernel_t1.cu
```

It did not use WMMA or WGMMA. Its evaluation log reports only `0.008x` speedup against the `1.2x` target:

```text
[M=8192] PASSED — speedup: 0.008x (kernel: 95.4174ms, ref: 0.7675ms)
```

Thus, the only correct result was a very slow scalar GEMM, not a tensor-core implementation. None of the MHA forward or backward kernels passed correctness.

The `summary.json` files report the trajectories as successful because the child evaluation processes completed normally. That field is process-level completion, not kernel correctness, and should not be interpreted as 12 correct trajectories per run.

## Interpretation

The five runs show that Inkling recognizes that Hopper kernels should use WGMMA and TMA, but it does not reliably implement the protocols required to use them:

- It can reproduce common WGMMA opcode shapes, most often `m64n64k16.f32.bf16.bf16`.
- It frequently fails to construct valid shared-memory descriptors.
- It does not consistently keep WGMMA execution warpgroup-uniform.
- It mishandles `mbarrier`, async-proxy fencing, commit, and wait ordering.
- It sometimes inserts WGMMA only to silence the instruction warning while retaining scalar math for the real calculation.
- When it avoids WGMMA, the resulting scalar attention and backward kernels cannot finish the `S=4096` workload within the sanitizer timeout.

The result is a stable split:

```text
scalar implementation -> usually times out
WGMMA/TMA implementation -> usually does not compile or faults
```

The generated corpus therefore demonstrates awareness of Hopper instruction names, not reliable ability to synthesize a legal and correct Hopper tensor-core kernel.

## Artifact locations

The generated per-turn artifacts now live under:

```text
/home/ubuntu/AccRL-exps/eval_runs/inkling-gemm/kernels
/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-bwd-d128/kernels
/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-bwd-d128-causal/kernels
/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-d128/kernels
/home/ubuntu/AccRL-exps/eval_runs/inkling-mha-d128-causal/kernels
```

Each `kernels/exp_NNN/` directory contains `kernel_tN.cu` and its corresponding `log_tN.txt` evaluation feedback.
