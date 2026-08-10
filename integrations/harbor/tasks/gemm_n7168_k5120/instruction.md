Create and optimize a CUDA implementation in `/workspace/kernel.cu` for the
fixed PTXBench H100 workload described by the formal task definition below.

Task:
{
  "name": "gemm_n7168_k5120",
  "op_type": "gemm",
  "axes": {
    "M": {
      "type": "var",
      "description": null
    },
    "N": {
      "type": "const",
      "value": 7168,
      "description": null
    },
    "K": {
      "type": "const",
      "value": 5120,
      "description": null
    }
  },
  "inputs": {
    "A": {
      "shape": [
        "M",
        "K"
      ],
      "dtype": "bfloat16",
      "description": null
    },
    "B": {
      "shape": [
        "N",
        "K"
      ],
      "dtype": "bfloat16",
      "description": null
    }
  },
  "outputs": {
    "C": {
      "shape": [
        "M",
        "N"
      ],
      "dtype": "bfloat16",
      "description": null
    }
  },
  "reference": "import torch\n\ndef run(A, B):\n    C = torch.matmul(A, B.T)\n    return C",
  "description": "General matrix multiply (GEMM) C = A @ B.T. Captured from Qwen3 14B qkv_proj (combined Q+K+V, (40+8+8)*128=7168, hidden=5120).",
  "constraints": []
}

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
