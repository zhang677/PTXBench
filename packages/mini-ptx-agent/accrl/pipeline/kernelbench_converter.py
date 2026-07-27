"""Convert KernelBench problems to AccRL Definitions using LLM assistance.

For each KernelBench problem:
1. Send the code to an LLM
2. LLM identifies the math operation, flattens nn.Module weights, and proposes variants
3. Parse LLM output into Definition objects
4. Generate workloads, verify, and write to traceset

Usage:
    python -m accrl.pipeline.kernelbench_converter \
        --kernelbench-root /path/to/KernelBench/KernelBench \
        --output-dir ~/accrl-training-data \
        --api-base http://localhost:30001/v1 \
        --model openai/Qwen/Qwen3-Coder-Next \
        --level 1

    # Dry run (no LLM call, just show parsed problems):
    python -m accrl.pipeline.kernelbench_converter \
        --kernelbench-root /path/to/KernelBench/KernelBench \
        --dry-run --level 1
"""

import argparse
import json
import logging
import re
from pathlib import Path

from accrl.pipeline.kernelbench_parser import KernelBenchProblem, parse_all

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer. Your task is to convert PyTorch model \
code from KernelBench into flashinfer-bench Definition format for RL training.

## Definition Format

A Definition describes a kernel optimization problem:
- **name**: `kb_<descriptive_snake_case>` (always start with `kb_`)
- **op_type**: one of: gemm, normalization, attention, quantization, conv, \
activation, elementwise, reduction, sampling, embedding, loss, pooling, fused
- **axes**: dict of dimension parameters. Use `{"type": "var"}` for variable dims, \
`{"type": "const", "value": N}` for fixed dims.
- **inputs**: dict of input tensors with shape (referencing axis names) and dtype
- **outputs**: dict of output tensors with shape and dtype
- **constraints**: list of constraint strings (e.g. "N % 16 == 0")
- **reference**: a standalone `def run(...)` function using only PyTorch ops. \
It must take ALL inputs (including weights) as arguments. No nn.Module, no state.
- **description**: human-readable description of what this kernel does AND what \
makes it an interesting optimization challenge
- **tags**: list of tags

## QUALITY CRITERIA (from FEEDBACK.md — MUST follow)

1. The mathematical operation must NOT already exist in the dataset. Each definition \
must be a genuinely different computational pattern.
2. The problem must be CHALLENGING ENOUGH:
   - It SHOULD contain both matrix multiplication AND vector operations (e.g., \
matmul + bias + activation, matmul + residual + norm)
   - If it doesn't contain matrix multiplication, then the vector operations must \
be complex enough (e.g., fused normalization + activation, not just a single torch.relu)
   - Pure element-wise ops (scalar multiply, add, single activation) are TOO TRIVIAL
   - Pure matmul without any epilogue (bias, activation, norm) is NOT challenging \
enough — the interesting problems have FUSED operations
3. The PyTorch reference must run in 1ms-15ms on every workload. Workloads that are \
too small (launch-overhead dominated, <0.1ms) or too large (>15ms or OOM) are rejected.

## NAMING CONVENTIONS

- Names: `kb_<operation>` (e.g., `kb_fused_linear_relu`, `kb_batched_matmul_transposed`)
- Axis names: UPPERCASE single letters or short names (M, N, K, B, H, D, seq_len, hidden_dim)
- op_type: use `gemm` for all matrix multiply variants (not `matmul`)
- Tags: always include `source:kernelbench` and `level:<N>`

## REFERENCE IMPLEMENTATION RULES

1. STATELESS: `def run(...)` with `@torch.no_grad()`. No nn.Module, no self.
2. Flatten all nn.Module weights into explicit function arguments \
(Linear.weight → weight, Linear.bias → bias, LayerNorm.weight → ln_weight, etc.)
3. Compute in float32 internally, cast output back to input dtype.
4. Inputs/outputs use axis names for shapes, not literal numbers.
5. dtype: bfloat16 for tensors, float32 for scalars/logits, int32 for indices.

## VARIANT GENERATION

A variant is a DIFFERENT OPERATION, not a different workload shape. \
Different shapes of the same operation should be handled via workload_ranges.

GOOD variants (different operations):
- With/without bias (changes number of inputs and compute pattern)
- Different activation epilogues (ReLU vs SiLU vs GELU — different fused ops)
- Fused vs unfused operations (matmul+norm vs separate matmul then norm)
- With/without residual connection (adds an extra input and memory access)
- Transposed inputs (different memory access patterns)

BAD variants (just workload shape differences — DO NOT generate these):
- "tall-skinny matmul" vs "rectangular matmul" (same torch.matmul, different shapes)
- "large K matmul" vs "small K matmul" (same torch.matmul, different shapes)
- "matvec tall-skinny" vs "matvec short-wide" (same torch.mv, different shapes)

## SKIP these problems (return empty list):
- Full model architectures (ResNet, VGG, DenseNet, etc.) — too complex
- Single trivial ops (torch.add, scalar multiply, single activation with no fusion)
- Problems using HuggingFace AutoModel/AutoConfig
- Pure matmul with no epilogue — unless it has a genuinely unique pattern \
(e.g., batched with broadcast, symmetric A@A^T)

## WORKLOAD SIZING GUIDE

Target: each workload should make the PyTorch reference run in 1-15ms.

Rough estimates (B200 GPU, bf16, ~50% effective throughput):
- GEMM [M,K]@[K,N]: ~1ms when 2*M*N*K ≈ 5e11 FLOPS
  - M=N=K=4096 is ~0.3ms, M=N=K=8192 is ~2ms, M=N=K=16384 is ~18ms
  - For batched: B*M*N*K matters. B=32,M=N=K=2048 is ~1ms
- Elementwise [M,N]: ~1ms when total bytes ≈ 4GB (M*N ≈ 2B elements bf16)
- Reduction [M,N]→[M]: ~1ms when M*N ≈ 1B elements
- Normalization [B,D]: ~1ms when B*D ≈ 500M elements
- MatVec [M,K]@[K]: ~1ms when M*K ≈ 2B elements (pure bandwidth-bound)

Minimum workload sizes (MUST be at least this large):
- GEMM: M,N,K >= 2048 each (or equivalent total FLOPS)
- Batched GEMM: B >= 8, M,N,K >= 512
- Elementwise: total elements >= 100M
- MatVec: M*K >= 100M

NEVER include workloads smaller than these minimums. When in doubt, go larger.
"""

USER_PROMPT_TEMPLATE = """\
Convert this KernelBench problem into one or more Definition(s).

## Problem: {name} (Level {level}, ID {problem_id})

```python
{code}
```

## Instructions

1. Analyze the mathematical operation in Model.forward()
2. Identify which nn.Module parameters need to become explicit inputs
3. Propose variants (with/without optional features, different configurations)
4. For each variant, output a complete Definition JSON

Respond with a JSON array of definitions. Each definition should have:
```json
[
  {{
    "name": "string",
    "op_type": "string",
    "description": "string",
    "tags": ["string"],
    "axes": {{"axis_name": {{"type": "var", "description": "..."}} }},
    "inputs": {{"name": {{"shape": ["axis1", "axis2"], "dtype": "bfloat16"}} }},
    "outputs": {{"name": {{"shape": ["axis1", "axis2"], "dtype": "bfloat16"}} }},
    "constraints": ["string"],
    "reference": "import torch\\n\\n@torch.no_grad()\\ndef run(...):\\n    ...",
    "workload_ranges": {{"axis_name": [val1, val2, ...]}}
  }}
]
```

If this problem should be SKIPPED, return `[]`.
"""


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------


def call_llm(
    problem: KernelBenchProblem,
    model: str = "openai/Qwen/Qwen3-Coder-Next",
    api_base: str = "http://localhost:30001/v1",
    temperature: float = 0.3,
    max_tokens: int = 16384,
) -> str:
    """Send a KernelBench problem to the LLM and get back Definition JSON."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        name=problem.name,
        level=problem.level,
        problem_id=problem.problem_id,
        code=problem.code,
    )

    import litellm

    response = litellm.completion(
        model=model,
        api_base=api_base,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def parse_llm_response(response: str) -> list[dict]:
    """Extract JSON definitions from LLM response text."""
    # Try to find JSON array in the response
    # Look for ```json ... ``` blocks first
    json_blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
    if json_blocks:
        for block in json_blocks:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON array
    match = re.search(r"\[[\s\S]*\]", response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from LLM response")
    return []


# ---------------------------------------------------------------------------
# Definition builder
# ---------------------------------------------------------------------------


def build_definitions(raw_defs: list[dict]) -> list[tuple]:
    """Convert parsed JSON dicts into (Definition, ranges) tuples.

    Returns list of (Definition, workload_ranges) pairs.
    """
    from flashinfer_bench.data import Definition
    from flashinfer_bench.data.definition import AxisConst, AxisVar, TensorSpec

    results = []
    for d in raw_defs:
        try:
            # Build axes
            axes = {}
            for name, spec in d["axes"].items():
                if spec.get("type") == "const":
                    axes[name] = AxisConst(
                        value=spec["value"],
                        description=spec.get("description", ""),
                    )
                else:
                    axes[name] = AxisVar(
                        description=spec.get("description", ""),
                    )

            # Build inputs
            inputs = {}
            for name, spec in d["inputs"].items():
                inputs[name] = TensorSpec(
                    shape=spec["shape"],
                    dtype=spec.get("dtype", "bfloat16"),
                    description=spec.get("description", ""),
                )

            # Build outputs
            outputs = {}
            for name, spec in d["outputs"].items():
                outputs[name] = TensorSpec(
                    shape=spec["shape"],
                    dtype=spec.get("dtype", "bfloat16"),
                    description=spec.get("description", ""),
                )

            defn = Definition(
                name=d["name"],
                op_type=d.get("op_type", "fused"),
                description=d.get("description", ""),
                tags=d.get("tags", []) + ["source:kernelbench"],
                axes=axes,
                inputs=inputs,
                outputs=outputs,
                constraints=d.get("constraints", []),
                reference=d.get("reference", ""),
            )

            ranges = d.get("workload_ranges", {})
            results.append((defn, ranges))

        except Exception as e:
            logger.warning("Failed to build definition %s: %s", d.get("name", "?"), e)

    return results


# ---------------------------------------------------------------------------
# Workload validator
# ---------------------------------------------------------------------------

# Bytes per element for each dtype
_DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "int32": 4, "int8": 1, "bool": 1,
}


def _estimate_tensor_bytes(shape_values: list[int], dtype: str) -> int:
    """Estimate memory for a tensor given concrete shape values."""
    n_elements = 1
    for v in shape_values:
        n_elements *= v
    return n_elements * _DTYPE_BYTES.get(dtype, 2)


def _resolve_shape(
    shape: list[str], axes_values: dict[str, int]
) -> list[int]:
    """Resolve symbolic shape to concrete values using axes."""
    return [axes_values.get(s, 1) for s in shape]


def validate_workload(
    defn_dict: dict,
    axes_values: dict[str, int],
    min_latency_ms: float = 1.0,
    max_latency_ms: float = 15.0,
    max_memory_gb: float = 10.0,
) -> tuple[bool, str]:
    """Validate a single workload (axes assignment) for a definition.

    Checks:
    1. Total memory of all inputs + outputs < max_memory_gb
    2. Estimated compute time is in [min_latency_ms, max_latency_ms]

    Returns (passed, reason).
    """
    # Estimate total memory
    total_bytes = 0
    for name, spec in defn_dict.get("inputs", {}).items():
        shape = _resolve_shape(spec["shape"], axes_values)
        total_bytes += _estimate_tensor_bytes(shape, spec.get("dtype", "bfloat16"))
    for name, spec in defn_dict.get("outputs", {}).items():
        shape = _resolve_shape(spec["shape"], axes_values)
        total_bytes += _estimate_tensor_bytes(shape, spec.get("dtype", "bfloat16"))

    total_gb = total_bytes / (1024**3)
    if total_gb > max_memory_gb:
        return False, f"OOM risk: {total_gb:.1f}GB > {max_memory_gb}GB limit"

    # Estimate compute time based on op_type
    op_type = defn_dict.get("op_type", "")

    # For GEMM: estimate FLOPS = 2*M*N*K, time = FLOPS / peak_throughput
    # B200 bf16 peak: ~2.25 PFLOPS = 2.25e15 FLOPS/s
    # Use conservative estimates (real throughput is lower than peak due to
    # memory access, launch overhead, etc. — typically 30-50% of peak)
    peak_flops = 0.5e15  # ~500 TFLOPS effective (B200 peak is 2.25 PFLOPS)
    # For bandwidth-bound: B200 HBM bandwidth ~8 TB/s, effective ~4 TB/s
    peak_bw = 4e12  # bytes/s effective

    if op_type == "gemm":
        # Try to find M, N, K dimensions
        m = axes_values.get("M", axes_values.get("m", 1))
        n = axes_values.get("N", axes_values.get("n", 1))
        k = axes_values.get("K", axes_values.get("k", 1))
        b = axes_values.get("B", axes_values.get("batch_size", axes_values.get("batch", 1)))
        flops = 2.0 * b * m * n * k
        compute_ms = (flops / peak_flops) * 1000
        # Memory time (read inputs + write output)
        mem_ms = (total_bytes / peak_bw) * 1000
        est_ms = max(compute_ms, mem_ms)
    elif op_type in ("elementwise", "activation"):
        # Pure bandwidth-bound
        est_ms = (total_bytes / peak_bw) * 1000
    elif op_type in ("normalization", "reduction"):
        # Bandwidth-bound with some compute overhead (~2x memory traffic)
        est_ms = (total_bytes * 2 / peak_bw) * 1000
    else:
        # Default: assume bandwidth-bound
        est_ms = (total_bytes / peak_bw) * 1000

    if est_ms < min_latency_ms * 0.1:  # Allow 10x margin for estimation error
        return False, f"Too fast: ~{est_ms:.4f}ms (target {min_latency_ms}-{max_latency_ms}ms)"
    if est_ms > max_latency_ms * 10:  # Allow 10x margin
        return False, f"Too slow: ~{est_ms:.1f}ms (target {min_latency_ms}-{max_latency_ms}ms)"

    return True, f"OK: ~{est_ms:.2f}ms, {total_gb:.2f}GB"


def validate_definition_workloads(
    defn_dict: dict,
    min_valid_fraction: float = 0.5,
) -> tuple[bool, list[str]]:
    """Validate all workload combinations for a definition.

    Samples workloads from workload_ranges and checks each one.
    Returns (passed, list_of_issues).
    """
    ranges = defn_dict.get("workload_ranges", {})
    if not ranges:
        return False, ["No workload_ranges defined"]

    # Get all variable axes
    var_axes = [
        name for name, spec in defn_dict.get("axes", {}).items()
        if spec.get("type") != "const"
    ]

    # Add const axes values
    const_values = {}
    for name, spec in defn_dict.get("axes", {}).items():
        if spec.get("type") == "const":
            const_values[name] = spec["value"]

    issues = []
    n_valid = 0
    n_total = 0

    # Check corner cases: smallest combo, largest combo, and a few samples
    if var_axes:
        # Smallest workload
        smallest = dict(const_values)
        for ax in var_axes:
            if ax in ranges and ranges[ax]:
                smallest[ax] = min(ranges[ax])
        passed, msg = validate_workload(defn_dict, smallest)
        n_total += 1
        if passed:
            n_valid += 1
        else:
            issues.append(f"Smallest workload {smallest}: {msg}")

        # Largest workload
        largest = dict(const_values)
        for ax in var_axes:
            if ax in ranges and ranges[ax]:
                largest[ax] = max(ranges[ax])
        passed, msg = validate_workload(defn_dict, largest)
        n_total += 1
        if passed:
            n_valid += 1
        else:
            issues.append(f"Largest workload {largest}: {msg}")

        # Middle workload
        middle = dict(const_values)
        for ax in var_axes:
            if ax in ranges and ranges[ax]:
                vals = sorted(ranges[ax])
                middle[ax] = vals[len(vals) // 2]
        passed, msg = validate_workload(defn_dict, middle)
        n_total += 1
        if passed:
            n_valid += 1
        else:
            issues.append(f"Middle workload {middle}: {msg}")

    if n_total == 0:
        return False, ["Could not construct any workloads"]

    valid_frac = n_valid / n_total
    overall_pass = valid_frac >= min_valid_fraction
    if not overall_pass:
        issues.insert(0, f"Only {n_valid}/{n_total} workloads valid ({valid_frac:.0%})")

    return overall_pass, issues


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def convert_problem(
    problem: KernelBenchProblem,
    output_dir: Path,
    model: str,
    api_base: str,
    n_workloads: int = 5,
    seed: int = 42,
    verify: bool = False,
) -> list[str]:
    """Convert a single KernelBench problem to Definition(s).

    Returns list of definition names that were written.
    """
    from accrl.pipeline.generate_variants import generate_workloads
    from accrl.pipeline.write_traceset import write_traceset

    logger.info(
        "Converting: Level %d, #%d: %s",
        problem.level,
        problem.problem_id,
        problem.name,
    )

    # Call LLM
    response = call_llm(problem, model=model, api_base=api_base)
    raw_defs = parse_llm_response(response)

    if not raw_defs:
        logger.info("  Skipped (LLM returned empty list)")
        return []

    # Build definitions
    def_pairs = build_definitions(raw_defs)
    if not def_pairs:
        logger.info("  Skipped (no valid definitions)")
        return []

    names = []
    for defn, ranges in def_pairs:
        try:
            # Generate workloads
            if ranges:
                traces = generate_workloads(defn, ranges, n=n_workloads, seed=seed)
            else:
                logger.warning("  No workload ranges for %s, skipping", defn.name)
                continue

            # Write to traceset
            write_traceset(defn, traces, output_dir)
            logger.info("  Wrote %s: %d workloads", defn.name, len(traces))
            names.append(defn.name)

        except Exception as e:
            logger.warning("  Failed to generate workloads for %s: %s", defn.name, e)

    # Optional verification
    if verify and names:
        from accrl.pipeline.verify import verify_definition as _verify

        for defn, ranges in def_pairs:
            if defn.name in names:
                traces = generate_workloads(defn, ranges, n=n_workloads, seed=seed)
                passed, msg = _verify(defn, traces)
                icon = "PASS" if passed else "FAIL"
                logger.info("  [%s] %s: %s", icon, defn.name, msg)

    return names


def main():
    parser = argparse.ArgumentParser(
        description="Convert KernelBench problems to AccRL Definitions"
    )
    parser.add_argument(
        "--kernelbench-root",
        type=str,
        required=True,
        help="Path to KernelBench/KernelBench/ directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/accrl-training-data",
        help="Output directory for traceset (default: ~/accrl-training-data)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:30001/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/Qwen/Qwen3-Coder-Next",
        help="Model name for litellm",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Only convert problems from this level (1-4)",
    )
    parser.add_argument(
        "--problem-ids",
        type=int,
        nargs="+",
        default=None,
        help="Only convert these problem IDs",
    )
    parser.add_argument(
        "--n-workloads",
        type=int,
        default=5,
        help="Number of workloads per definition (default: 5)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify references on GPU after writing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse problems and show info, but don't call LLM",
    )
    parser.add_argument(
        "--save-responses",
        type=str,
        default=None,
        help="Directory to save raw LLM responses for debugging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Parse all problems
    problems = parse_all(args.kernelbench_root)
    logger.info("Parsed %d problems from %s", len(problems), args.kernelbench_root)

    # Filter by level
    if args.level is not None:
        problems = [p for p in problems if p.level == args.level]
        logger.info("Filtered to %d problems at level %d", len(problems), args.level)

    # Filter by problem ID
    if args.problem_ids is not None:
        problems = [p for p in problems if p.problem_id in args.problem_ids]
        logger.info("Filtered to %d problems by ID", len(problems))

    if args.dry_run:
        for p in problems:
            print(f"\n{'='*60}")
            print(f"Level {p.level}, #{p.problem_id}: {p.name}")
            print(f"Globals: {p.global_vars}")
            print(f"Forward:\n{p.forward_source}")
            print(f"Init:\n{p.init_source}")
        return

    output_dir = Path(args.output_dir).expanduser()
    response_dir = Path(args.save_responses) if args.save_responses else None
    if response_dir:
        response_dir.mkdir(parents=True, exist_ok=True)

    all_names = []
    for problem in problems:
        try:
            if response_dir:
                # Save raw LLM response for debugging
                response = call_llm(
                    problem, model=args.model, api_base=args.api_base
                )
                resp_path = response_dir / f"L{problem.level}_{problem.problem_id}_{problem.name.replace(' ', '_')}.md"
                resp_path.write_text(response)

                raw_defs = parse_llm_response(response)
                def_pairs = build_definitions(raw_defs)

                from accrl.pipeline.generate_variants import generate_workloads
                from accrl.pipeline.write_traceset import write_traceset

                for defn, ranges in def_pairs:
                    if ranges:
                        traces = generate_workloads(
                            defn, ranges, n=args.n_workloads, seed=42
                        )
                        write_traceset(defn, traces, output_dir)
                        all_names.append(defn.name)
                        logger.info("  Wrote %s: %d workloads", defn.name, len(traces))
            else:
                names = convert_problem(
                    problem,
                    output_dir,
                    model=args.model,
                    api_base=args.api_base,
                    n_workloads=args.n_workloads,
                    verify=args.verify,
                )
                all_names.extend(names)

        except Exception as e:
            logger.error(
                "Failed on Level %d, #%d %s: %s",
                problem.level,
                problem.problem_id,
                problem.name,
                e,
            )

    print(f"\n{'='*60}")
    print(f"Total definitions written: {len(all_names)}")
    for name in all_names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
