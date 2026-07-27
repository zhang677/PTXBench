# KernelBench Backward Pass Collection Pipeline

End-to-end workflow for deriving backward-pass kernel definitions from KernelBench forward problems.

## Overview

**Goal**: Derive backward-pass kernel definitions from KernelBench forward computation problems, producing definitions + workloads suitable for RL training.

**Source**: `~/KernelBench/KernelBench/level{1-4}/`
- Level 1 (100 problems): Single ops — matmul, conv, activations, norms, pooling, loss
- Level 2 (100 problems): Fused ops — Conv2d+ReLU+Bias, Gemm+Multiply+LeakyReLU, etc.
- Level 3 (50 problems): Architectures — multi-layer networks, attention blocks, MLPs
- Level 4 (20 problems): HF models — **skip**, too coarse-grained for kernel-level optimization

**Output**: `~/accrl-training-data/` (definitions + workloads)

## KernelBench Problem Format

Each problem file contains:

```python
# Module-level constants
batch_size = 128
in_features = 1024
out_features = 512

class Model(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return []  # constructor arguments
```

Key elements to parse:
- `Model.__init__()` — layers, parameters, constants
- `Model.forward()` — the computation graph to differentiate
- `get_inputs()` / `get_init_inputs()` — tensor shapes and dtypes
- Module-level constants (batch_size, in_features, etc.)

## Collection Pipeline

Use the `/collect-backward` skill. Six steps:

### Step 1: Read Problem

Parse the KernelBench problem file. Understand the computation graph, parameter shapes, and constants.

### Step 2: Derive Backward Computation

Analytically apply the chain rule to the forward computation. For fused ops (Level 2+), derive the full chain through the composition.

For each differentiable input, explicitly identify which forward tensors that gradient depends on:
- `grad_A` depends on: `grad_output`, `B`
- `grad_B` depends on: `grad_output`, `A`

Verify correctness with `torch.autograd.grad()`:
```python
model = Model(*get_init_inputs())
inputs = get_inputs()
x = inputs[0].requires_grad_(True)
out = model(x)
grad_output = torch.randn_like(out)
grad_x, = torch.autograd.grad(out, x, grad_outputs=grad_output)
```

For models with parameters (weights, biases), also compute parameter gradients:
```python
params = [p for p in model.parameters() if p.requires_grad]
grads = torch.autograd.grad(out, [x] + params, grad_outputs=grad_output)
```

### Step 3: Write PyTorch References (one per differentiable input)

Write a **separate** `def run(...)` function for each differentiable input — **no** `torch.autograd.Function` wrapper. Each `run()` computes the gradient w.r.t. **one** input and returns a **single** tensor.

- **Inputs**: `grad_output` + only the forward tensors needed for *this specific* gradient
- **Output**: single gradient tensor (not a tuple)
- **Skip trivial gradients**: If a gradient is trivially element-wise (pure ReLU mask, simple scaling), skip it

Example (matmul backward, `C = A @ B`) — two separate references:
```python
# Definition: kernelbench_l1_2_backward_A
def run(grad_output, B):
    return grad_output @ B.T   # (M, K)

# Definition: kernelbench_l1_2_backward_B
def run(grad_output, A):
    return A.T @ grad_output   # (K, N)
```

Example (fused Conv2d+ReLU backward) — two definitions (bias gradient is trivial, skip):
```python
# Definition: kernelbench_l2_5_backward_x
def run(grad_output, weight, conv_output):
    grad_relu = grad_output * (conv_output > 0).float()
    return torch.nn.functional.conv_transpose2d(grad_relu, weight, padding=1)

# Definition: kernelbench_l2_5_backward_weight
def run(grad_output, x, conv_output):
    grad_relu = grad_output * (conv_output > 0).float()
    return torch.nn.functional.conv2d(x.transpose(0, 1), grad_relu.transpose(0, 1), padding=1).transpose(0, 1)
```

### Step 4: Create One Definition Per Differentiable Input

Each non-trivial gradient gets its own definition:

- **name**: `kernelbench_l{level}_{number}_backward_{input_name}` (e.g., `kernelbench_l1_2_backward_A`, `kernelbench_l2_5_backward_weight`)
- **op_type**: from the mapping table below
- **tags**: `["backward", "source:kernelbench", "level:{N}"]`
- **axes**: Variable axes from problem dimensions, const axes from fixed parameters (eps, etc.)
- **inputs**: `grad_output` + only the forward tensors needed for *this specific* gradient
- **outputs**: single gradient tensor (not a tuple)
- **reference**: the corresponding `def run(...)` from step 3

**Semantic uniqueness check**: Compare each definition against `~/flashinfer-trace/definitions/` and `~/accrl-training-data/definitions/`. Skip if an equivalent mathematical operation already exists.

### Step 5: Generate Workloads

Use `generate_workloads()` from `accrl.pipeline.generate_variants`. Each split definition gets its own workloads.

Sizing rules (workloads must produce meaningful GPU execution times):
- Total element count >= 1M for memory-bound kernels
- Minimum batch size: 16
- Minimum hidden/embedding dimension: 2048
- Minimum sequence length: 256
- Scale up from KernelBench defaults if they are too small

### Step 6: Write and Verify

Call `write_traceset()` once per split definition:

```python
from pathlib import Path
from accrl.pipeline.write_traceset import write_traceset

output_dir = Path("~/accrl-training-data").expanduser()
for defn, traces in split_definitions:
    write_traceset(defn, traces, output_dir)
```

Then verify each:
```bash
python -m accrl.pipeline.verify --input-dir ~/accrl-training-data --definitions kernelbench_l{level}_{number}_backward_{input_name}
```

## Validation Pipeline

Use the `/check-backward` skill. Four checks (defined in `accrl/pipeline/FEEDBACK.md`):

### Check 1: Uniqueness

The mathematical operation must not already exist in:
- `~/flashinfer-trace/definitions/`
- `~/accrl-training-data/definitions/`

Compare **semantically**, not just by name (e.g., a transposed matmul backward is the same as a regular matmul backward with swapped inputs).

### Check 2: Challenge Level

The backward pass must be non-trivial:
- **Accept**: contains matrix multiplication + vector operations
- **Accept**: contains complex vector operations alone (e.g., normalization backward with recomputation, multi-step chain rule)
- **Reject**: trivial element-wise backward (pure ReLU mask, simple scaling, element-wise multiply by constant)

### Check 3: Performance Bounds

Reference latency must be in **[1ms, 15ms]** for all workloads:
```bash
python -m accrl.pipeline.verify --input-dir ~/accrl-training-data --definitions <name> --profile --min-latency-ms 1.0 --max-latency-ms 15.0
```
- Too fast (<1ms): dominated by kernel launch overhead
- Too slow (>15ms): training episodes too expensive

### Check 4: Baseline Solution Correctness

If a baseline solution was provided alongside the definition:
- Must produce correct outputs (match reference within tolerance)
- Must be faster than the PyTorch reference

## Workflow Commands

```bash
# Collect backward definitions from a specific level
/collect-backward level1

# Validate all backward definitions
/check-backward --all

# Validate a specific definition
/check-backward kernelbench_l1_42_backward_x

# Push accepted definitions to HuggingFace
/sync-hub
```

The collect → check → revise cycle:
1. `/collect-backward` produces definitions
2. `/check-backward --all` validates them
3. Fix any REVISE verdicts (resize workloads, adjust definitions)
4. Re-run `/check-backward` until all ACCEPT
5. `/sync-hub` to publish

## Op Type Mapping

| KernelBench Category | op_type |
|---|---|
| Matmul, Linear, GEMM backward | `gemm` |
| Conv2d, ConvTranspose backward | `conv` |
| BatchNorm, LayerNorm, GroupNorm backward | `normalization` |
| Activation fusions backward | `fused` |
| Pooling backward | `pooling` |
| Attention backward | `attention` |
| Loss function backward | `loss` |
| MLP / multi-layer backward | `mlp` |

## Existing Definitions (Do Not Duplicate)

Op type directories already in `~/accrl-training-data/definitions/`:

- `attention`
- `conv`
- `embedding`
- `normalization`
- `quantization`
- `rnn`
- `routing`
- `sampling`
- `scan`
- `selection`
- `ssm`

Always check semantically against these before creating a new backward definition.

## Key Files

| Purpose | Path |
|---|---|
| Collect skill | `.claude/skills/collect-backward/SKILL.md` |
| Check skill | `.claude/skills/check-backward/SKILL.md` |
| Write traceset | `accrl/pipeline/write_traceset.py` |
| Verify definitions | `accrl/pipeline/verify.py` |
| Generate workloads | `accrl/pipeline/generate_variants.py` |
| Quality rubrics | `accrl/pipeline/FEEDBACK.md` |
| Hub sync | `accrl/pipeline/hub_sync.py` |
| Output data | `~/accrl-training-data/` |
| Existing definitions | `~/flashinfer-trace/definitions/` |
| KernelBench source | `~/KernelBench/KernelBench/level{1-4}/` |
