"""Collect TRT-LLM Tier 2 kernel definitions and generate workloads.

Creates 6 definitions (5 kernels, Top-K Last Dim has k=32 and k=128 variants)
with 5 workloads each, written to flashinfer-trace format.

Usage:
    python -m accrl.pipeline.trtllm_tier2 --output-dir ~/accrl-training-data
    python -m accrl.pipeline.trtllm_tier2 --output-dir ~/accrl-training-data --verify
"""

import argparse
import logging
from pathlib import Path

from flashinfer_bench.data import Definition
from flashinfer_bench.data.definition import AxisConst, AxisVar, TensorSpec

from accrl.pipeline.generate_variants import generate_workloads
from accrl.pipeline.trtllm_tier2_solutions import SOLUTION_MAKERS
from accrl.pipeline.write_traceset import write_solution, write_traceset

logger = logging.getLogger(__name__)

# Common axis value ranges
BATCH_RANGE = [16, 64, 128, 256, 512, 1024]
SEQ_LEN_RANGE = [256, 512, 1024, 2048, 4096]
DIM_RANGE = [2048, 3584, 4096, 5120, 8192]
INPUT_LENGTH_RANGE = [4096, 8192, 32000, 65536, 128256]
INPUT_LENGTH_RANGE_SHORT = [512, 1024, 2048, 4096]
NUM_TOKENS_RANGE = [256, 512, 1024, 2048, 4096, 8192]
VOCAB_SIZE_RANGE = [32000, 65536, 128256, 151936]
EMBED_DIM_RANGE = [2048, 3584, 4096, 5120, 7168, 8192]
BATCH_SMALL_RANGE = [16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Definition makers
# ---------------------------------------------------------------------------


def make_causal_conv1d_silu() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_causal_conv1d_silu",
        op_type="conv",
        description=(
            "Mamba-style causal 1D depthwise convolution + SiLU activation. "
            "Left-pads input by kernel_size-1, applies depthwise conv1d, then SiLU."
        ),
        tags=["status:verified", "source:trtllm", "tier:2", "mamba"],
        axes={
            "batch": AxisVar(description="batch size"),
            "seq_len": AxisVar(description="sequence length"),
            "dim": AxisVar(description="number of channels (depthwise)"),
            "kernel_size": AxisConst(value=4, description="convolution kernel size"),
        },
        inputs={
            "input": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="bfloat16"),
            "weight": TensorSpec(shape=["kernel_size", "dim"], dtype="bfloat16"),
            "bias": TensorSpec(shape=["dim"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="bfloat16"),
        },
        reference=(
            "import torch\n"
            "import torch.nn.functional as F\n\n"
            "@torch.no_grad()\n"
            "def run(input, weight, bias):\n"
            "    B, L, D = input.shape\n"
            "    K = weight.shape[0]\n"
            "    x = input.to(torch.float32).permute(0, 2, 1)  # [B, D, L]\n"
            "    x = F.pad(x, (K - 1, 0))  # left-pad by K-1\n"
            "    w = weight.to(torch.float32).permute(1, 0).unsqueeze(1)  # [D, 1, K]\n"
            "    b = bias.to(torch.float32)\n"
            "    y = F.conv1d(x, w, bias=b, groups=D)  # [B, D, L]\n"
            "    y = F.silu(y)\n"
            "    return y.permute(0, 2, 1).to(input.dtype)\n"
        ),
    )
    ranges = {
        "batch": [4, 8, 16, 32, 64],
        "seq_len": SEQ_LEN_RANGE,
        "dim": DIM_RANGE,
    }
    return defn, ranges


def make_topk_last_dim_k32() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_topk_last_dim_k32",
        op_type="selection",
        description=(
            "Top-32 values and indices along the last dimension. "
            "Returns the 32 largest elements and their indices."
        ),
        tags=["status:verified", "source:trtllm", "tier:2"],
        axes={
            "batch": AxisVar(description="batch size"),
            "input_length": AxisVar(description="input dimension to select from"),
            "k": AxisConst(value=32, description="number of top elements"),
        },
        inputs={
            "input": TensorSpec(shape=["batch", "input_length"], dtype="float32"),
        },
        outputs={
            "values": TensorSpec(shape=["batch", "k"], dtype="float32"),
            "indices": TensorSpec(shape=["batch", "k"], dtype="int32"),
        },
        constraints=["input_length >= 32"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    values, indices = torch.topk(input, 32, dim=-1)\n"
            "    return values, indices.to(torch.int32)\n"
        ),
    )
    ranges = {
        "batch": BATCH_RANGE,
        "input_length": INPUT_LENGTH_RANGE,
    }
    return defn, ranges


def make_topk_last_dim_k128() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_topk_last_dim_k128",
        op_type="selection",
        description=(
            "Top-128 values and indices along the last dimension. "
            "Returns the 128 largest elements and their indices."
        ),
        tags=["status:verified", "source:trtllm", "tier:2"],
        axes={
            "batch": AxisVar(description="batch size"),
            "input_length": AxisVar(description="input dimension to select from"),
            "k": AxisConst(value=128, description="number of top elements"),
        },
        inputs={
            "input": TensorSpec(shape=["batch", "input_length"], dtype="float32"),
        },
        outputs={
            "values": TensorSpec(shape=["batch", "k"], dtype="float32"),
            "indices": TensorSpec(shape=["batch", "k"], dtype="int32"),
        },
        constraints=["input_length >= 128"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    values, indices = torch.topk(input, 128, dim=-1)\n"
            "    return values, indices.to(torch.int32)\n"
        ),
    )
    ranges = {
        "batch": BATCH_RANGE,
        "input_length": INPUT_LENGTH_RANGE,
    }
    return defn, ranges


def make_cumsum() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_cumsum",
        op_type="scan",
        description=(
            "Inclusive prefix sum (cumulative sum) along the last dimension."
        ),
        tags=["status:verified", "source:trtllm", "tier:2"],
        axes={
            "batch": AxisVar(description="batch size"),
            "input_length": AxisVar(description="length of input to scan"),
        },
        inputs={
            "input": TensorSpec(shape=["batch", "input_length"], dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["batch", "input_length"], dtype="float32"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    return torch.cumsum(input, dim=-1)\n"
        ),
    )
    ranges = {
        "batch": BATCH_RANGE,
        "input_length": INPUT_LENGTH_RANGE_SHORT,
    }
    return defn, ranges


def make_embedding_lookup() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_embedding_lookup",
        op_type="embedding",
        description=(
            "Token embedding lookup with per-token scaling. "
            "Gathers embedding rows by index and multiplies by per-token scale."
        ),
        tags=["status:verified", "source:trtllm", "tier:2"],
        axes={
            "num_tokens": AxisVar(description="number of tokens to embed"),
            "vocab_size": AxisVar(description="vocabulary size"),
            "embed_dim": AxisVar(description="embedding dimension"),
        },
        inputs={
            "indices": TensorSpec(shape=["num_tokens"], dtype="int32"),
            "weight": TensorSpec(shape=["vocab_size", "embed_dim"], dtype="bfloat16"),
            "per_token_scale": TensorSpec(shape=["vocab_size"], dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["num_tokens", "embed_dim"], dtype="bfloat16"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(indices, weight, per_token_scale):\n"
            "    V = weight.shape[0]\n"
            "    idx = indices.abs() % V\n"
            "    emb = weight[idx].to(torch.float32)\n"
            "    scale = per_token_scale[idx].unsqueeze(-1)\n"
            "    return (emb * scale).to(weight.dtype)\n"
        ),
    )
    ranges = {
        "num_tokens": NUM_TOKENS_RANGE,
        "vocab_size": VOCAB_SIZE_RANGE,
        "embed_dim": EMBED_DIM_RANGE,
    }
    return defn, ranges


def make_penalty_application() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_penalty_application",
        op_type="sampling",
        description=(
            "Apply temperature, repetition, presence, and frequency penalties "
            "to logits for LLM sampling."
        ),
        tags=["status:verified", "source:trtllm", "tier:2", "sampling"],
        axes={
            "batch": AxisVar(description="batch size"),
            "vocab_size": AxisVar(description="vocabulary size"),
        },
        inputs={
            "logits": TensorSpec(shape=["batch", "vocab_size"], dtype="float32"),
            "token_counts": TensorSpec(shape=["batch", "vocab_size"], dtype="int32"),
        },
        outputs={
            "output": TensorSpec(shape=["batch", "vocab_size"], dtype="float32"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(logits, token_counts):\n"
            "    TEMPERATURE = 1.0\n"
            "    REPETITION = 1.2\n"
            "    PRESENCE = 0.6\n"
            "    FREQUENCY = 0.5\n"
            "    counts = token_counts.clamp(min=0).to(torch.float32)\n"
            "    # Temperature\n"
            "    out = logits / TEMPERATURE\n"
            "    # Repetition penalty (conditional on sign)\n"
            "    mask = counts > 0\n"
            "    pos = out > 0\n"
            "    rep = torch.where(pos, out / REPETITION, out * REPETITION)\n"
            "    out = torch.where(mask, rep, out)\n"
            "    # Presence penalty\n"
            "    out = out - PRESENCE * (counts > 0).to(torch.float32)\n"
            "    # Frequency penalty\n"
            "    out = out - FREQUENCY * counts\n"
            "    return out\n"
        ),
    )
    ranges = {
        "batch": BATCH_SMALL_RANGE,
        "vocab_size": VOCAB_SIZE_RANGE,
    }
    return defn, ranges


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------

MAKERS = [
    make_causal_conv1d_silu,
    make_topk_last_dim_k32,
    make_topk_last_dim_k128,
    make_cumsum,
    make_embedding_lookup,
    make_penalty_application,
]


def filter_by_constraints(defn: Definition, traces: list) -> list:
    """Drop workload traces that violate definition constraints."""
    if not defn.constraints:
        return traces

    filtered = []
    for t in traces:
        axes = t.workload.axes
        ok = True
        for constraint in defn.constraints:
            # Evaluate constraint with axis values as locals
            try:
                if not eval(constraint, {"__builtins__": {}}, dict(axes)):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            filtered.append(t)
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate TRT-LLM Tier 2 kernel definitions and workloads"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/accrl-training-data",
        help="Output directory for traceset (default: ~/accrl-training-data)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of workloads per definition (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify references on GPU after writing",
    )
    parser.add_argument(
        "--verify-solutions",
        action="store_true",
        help="Compile and verify baseline CUDA solutions against references",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir).expanduser()

    all_entries: list[tuple[Definition, list]] = []
    for make_fn in MAKERS:
        defn, ranges = make_fn()

        # Generate more candidates for constrained definitions, then filter
        if defn.constraints:
            traces = generate_workloads(defn, ranges, n=args.n * 10, seed=args.seed)
            traces = filter_by_constraints(defn, traces)[: args.n]
        else:
            traces = generate_workloads(defn, ranges, n=args.n, seed=args.seed)

        if len(traces) < args.n:
            logger.warning(
                "%s: only %d workloads after constraint filtering (wanted %d)",
                defn.name,
                len(traces),
                args.n,
            )

        write_traceset(defn, traces, output_dir)
        logger.info(
            "Wrote %s: %d workloads -> %s", defn.name, len(traces), output_dir
        )

        # Write baseline CUDA solution if available
        if defn.name in SOLUTION_MAKERS:
            solution = SOLUTION_MAKERS[defn.name](defn)
            write_solution(defn, solution, output_dir)
            logger.info("Wrote solution %s for %s", solution.name, defn.name)

        all_entries.append((defn, traces))

    if args.verify:
        from accrl.pipeline.verify import verify_traceset

        results = verify_traceset(str(output_dir))
        n_pass = sum(1 for p, _ in results.values() if p)
        n_fail = len(results) - n_pass
        print(f"\nReference verification: {n_pass} passed, {n_fail} failed")
        for name, (passed, msg) in sorted(results.items()):
            icon = "PASS" if passed else "FAIL"
            print(f"  [{icon}] {name}: {msg}")

    if args.verify_solutions:
        from accrl.pipeline.verify import verify_solution

        print("\nSolution verification:")
        sol_pass = 0
        sol_fail = 0
        for defn, traces in all_entries:
            if defn.name not in SOLUTION_MAKERS:
                continue
            solution = SOLUTION_MAKERS[defn.name](defn)
            logger.info("Verifying solution for %s...", defn.name)
            passed, msg = verify_solution(defn, solution, traces)
            icon = "PASS" if passed else "FAIL"
            print(f"  [{icon}] {defn.name}: {msg}")
            if passed:
                sol_pass += 1
            else:
                sol_fail += 1
        print(f"\nSolution verification: {sol_pass} passed, {sol_fail} failed")


if __name__ == "__main__":
    main()
