Create and optimize a CUDA implementation in `/workspace/kernel.cu` for the
fixed PTXBench H100 workload described by the formal task definition below.

Task:
{task_content}

The workspace does not contain a starter kernel. Create `/workspace/kernel.cu`
using the ABI documented in `/opt/ptxbench/reference/README.md`, then evaluate
it with:

```bash
ptxbench eval /workspace/kernel.cu --json
```

After each meaningful change, run the same command again.

Each invocation evaluates the current file and is recorded in the Harbor ATIF
trajectory. Iterate on compiler, correctness, and performance feedback, and
leave the best candidate in `/workspace/kernel.cu`. Do not change the task
manifest or attempt to select a different definition or workload.

The evaluator reports compilation, architecture-specific instruction usage
extracted from an emitted `compute_90a` PTX artifact, two memory-sanitizer
checks, numerical correctness, and performance. A candidate is correct only
when `all_passed` is `true`; `min_speedup` is meaningful only for a correct
candidate.
