Optimize the CUDA implementation in `/workspace/kernel.cu` for the fixed
PTXBench H100 workload described below.

- Operation: `C = A @ B.T`
- A: `[8192, 5120]`, BF16, row-major
- B: `[7168, 5120]`, BF16, row-major
- C: `[8192, 7168]`, BF16, row-major
- Target GPU: NVIDIA H100 (`sm_90a`)
- Binding: TVM-FFI, destination-passing style

Your first shell command must be exactly the following command; do not inspect
the source or environment first:

```bash
ptxbench eval /workspace/kernel.cu --json
```

After each meaningful change, run the same command again.

Each invocation evaluates the current file and is recorded in the Harbor ATIF
trajectory. Make no more than four evaluation calls. Leave the best candidate
in `/workspace/kernel.cu`. Do not change the task manifest or attempt to select
a different definition or workload.

The evaluator reports compilation, architecture-specific instruction usage
extracted from an emitted `compute_90a` PTX artifact, two memory-sanitizer
checks, numerical correctness, and performance. A candidate is correct only
when `all_passed` is `true`; `min_speedup` is meaningful only for a correct
candidate.
