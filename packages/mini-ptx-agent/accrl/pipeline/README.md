# Problem Collection Pipeline

Collect kernel optimization problems (Definitions + Workloads) from open-source repos, verify them, and publish to `AccRL/accrl-training`.

## Quick Start

```bash
# Collect kernels from a repo
/collect-kernels https://github.com/state-spaces/mamba

# Collect TRT-LLM Tier 1 kernels
/collect-trtllm 1

# Verify everything passes
/verify-definitions

# Push to HuggingFace
/sync-hub
```

## Skills Reference

### Collectors

| Skill | What it does | Example |
|-------|-------------|---------|
| `/collect-kernels <repo> [name]` | Generic — reads any repo, writes Definitions | `/collect-kernels https://github.com/NVIDIA/TensorRT-LLM rmsnorm` |
| `/collect-trtllm [tier\|name]` | TRT-LLM specialist — uses tier catalog | `/collect-trtllm 1` or `/collect-trtllm rmsnorm` |
| `/collect-triton [name]` | Triton kernels from `triton-lang/triton` | `/collect-triton` |
| `/collect-mamba [name]` | Mamba/SSM kernels | `/collect-mamba selective_scan` |
| `/collect-fla [name]` | Flash linear attention (GLA, RetNet, RWKV) | `/collect-fla gla` |
| `/collect-quantization <repo> [name]` | Real quantization kernels | `/collect-quantization https://github.com/NVIDIA/TensorRT-LLM per_token_quant` |

### Transforms

| Skill | What it does | Example |
|-------|-------------|---------|
| `/specialize-shapes <name>` | Fix axes to real model dims (Llama, Qwen, etc.) | `/specialize-shapes rmsnorm` |
| `/collect-backward <name>` | Write backward pass for an existing forward def | `/collect-backward rmsnorm` |

### Ops

| Skill | What it does | Example |
|-------|-------------|---------|
| `/verify-definitions [name]` | Build + run references on GPU | `/verify-definitions rmsnorm` |
| `/sync-hub` | Upload to `AccRL/accrl-training` | `/sync-hub` |

## Typical Workflow

### 1. Collect base kernels

```
/collect-trtllm 1
/collect-trtllm 2
/collect-mamba
/collect-fla
```

### 2. Verify

```
/verify-definitions
```

Fix any failures, then re-verify.

### 3. Expand the problem set

```
# Shape-specialize for real models
/specialize-shapes rmsnorm
/specialize-shapes layernorm

# Add backward passes
/collect-backward rmsnorm
/collect-backward causal_conv1d

# Collect quantization kernels
/collect-quantization https://github.com/NVIDIA/TensorRT-LLM
```

### 4. Verify again and push

```
/verify-definitions
/sync-hub
```

## Python CLI

The skills call these utilities under the hood. You can also run them directly:

```bash
# Verify all definitions (requires GPU)
python -m accrl.pipeline.verify --input-dir ~/accrl-training-data

# Verify specific definition
python -m accrl.pipeline.verify --input-dir ~/accrl-training-data --definitions rmsnorm

# Validate traceset structure (no GPU needed)
python -m accrl.pipeline.write_traceset --input-dir ~/accrl-training-data

# Generate workloads for a definition
python -m accrl.pipeline.generate_variants \
  --definition ~/accrl-training-data/definitions/normalization/rmsnorm.json \
  --ranges '{"M": [32, 64, 128, 256], "N": [768, 1024, 2048, 4096]}' \
  --n 25

# Push to HuggingFace Hub
python -m accrl.pipeline.hub_sync --local-dir ~/accrl-training-data
```

## Output Format

All data lands in `~/accrl-training-data/` with the flashinfer-trace folder layout:

```
~/accrl-training-data/
├── definitions/{op_type}/{name}.json      # Definition with reference code
├── workloads/{op_type}/{name}.jsonl       # Trace lines (workload only)
├── solutions/{op_type}/{name}/            # Populated during training
└── traces/{op_type}/{name}.jsonl          # Populated during training
```

This mirrors `flashinfer-ai/flashinfer-trace` but contains a **disjoint** problem set for training.

## Writing a Definition by Hand

If you need to create a definition without a skill:

```python
from pathlib import Path
from flashinfer_bench.data import Definition
from flashinfer_bench.data.definition import AxisVar, AxisConst, TensorSpec
from accrl.pipeline.write_traceset import write_traceset
from accrl.pipeline.generate_variants import generate_workloads

defn = Definition(
    name="rmsnorm",
    op_type="normalization",
    description="RMS Normalization",
    axes={
        "M": AxisVar(description="batch * seq_len"),
        "N": AxisVar(description="hidden dimension"),
    },
    inputs={
        "x": TensorSpec(shape=["M", "N"], dtype="float16", description="input"),
        "gamma": TensorSpec(shape=["N"], dtype="float16", description="scale"),
    },
    outputs={
        "y": TensorSpec(shape=["M", "N"], dtype="float16", description="output"),
    },
    reference=(
        "import torch\n\n"
        "def run(x, gamma):\n"
        "    eps = 1e-5\n"
        "    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)\n"
        "    return (gamma * x * torch.rsqrt(variance + eps),)"
    ),
    tags=["llm", "normalization"],
    constraints=["M > 0", "N > 0"],
)

traces = generate_workloads(
    defn,
    var_axis_ranges={"M": [32, 64, 128, 256, 512], "N": [768, 1024, 2048, 4096]},
    n=25,
    seed=42,
)

write_traceset(defn, traces, Path("~/accrl-training-data").expanduser())
```

Then verify: `python -m accrl.pipeline.verify --input-dir ~/accrl-training-data --definitions rmsnorm`

## TRT-LLM Tier Reference

See `accrl/synthesis/trtllm.md` for the full catalog. Summary:

| Tier | Count | Examples |
|------|-------|---------|
| 1 | 6 | RMSNorm, LayerNorm, Per-Token Quant, MoE Routing |
| 2 | 6 | Causal Conv1D, Top-K, Cumulative Sum, Embedding |
| 3 | 8 | FP4 Quant, Top-P Sampling, Attention Mask |
| 4 | 5 | Selective Scan, LRU, Beam Search |

## Verify TRT-LLM extraction results
```
  # Verify references (definitions):
  python -m accrl.pipeline.trtllm_tier1 --output-dir ~/accrl-training-data --verify

  # Verify CUDA solutions against references:
  python -m accrl.pipeline.trtllm_tier1 --output-dir ~/accrl-training-data --verify-solutions

  # Both together:
  python -m accrl.pipeline.trtllm_tier1 --output-dir ~/accrl-training-data --verify --verify-solutions

  # Standalone reference verification (on an existing traceset):
  python -m accrl.pipeline.verify --input-dir ~/accrl-training-data

  # You can also verify specific definitions:
  python -m accrl.pipeline.verify --input-dir ~/accrl-training-data --definitions trtllm_layernorm trtllm_per_channel_scale 
  ```