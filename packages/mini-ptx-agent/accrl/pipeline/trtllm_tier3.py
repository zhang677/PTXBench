"""Collect TRT-LLM Tier 3 kernel definitions and generate workloads.

Creates 6 definitions with workloads each, written to flashinfer-trace format.

Usage:
    python -m accrl.pipeline.trtllm_tier3 --output-dir ~/accrl-training-data
    python -m accrl.pipeline.trtllm_tier3 --output-dir ~/accrl-training-data --verify
"""

import argparse
import logging
from pathlib import Path

from flashinfer_bench.data import Definition
from flashinfer_bench.data.definition import AxisConst, AxisVar, TensorSpec

from accrl.pipeline.generate_variants import generate_workloads
from accrl.pipeline.trtllm_tier3_solutions import SOLUTION_MAKERS
from accrl.pipeline.write_traceset import write_solution, write_traceset

logger = logging.getLogger(__name__)

# Common axis value ranges
BATCH_RANGE = [16, 32, 64, 128]
BATCH_LARGE_RANGE = [64, 128, 256, 512, 1024]
SEQ_LEN_RANGE = [256, 512, 1024, 2048, 4096]
SEQ_LEN_SHORT_RANGE = [256, 512, 1024, 2048]
DIM_RANGE = [2048, 4096, 5120, 8192]
VOCAB_SIZE_RANGE = [32000, 65536, 128256]
ROWS_RANGE = [256, 512, 1024, 2048, 4096]
COLS_RANGE = [2048, 4096, 5120, 8192]
NUM_HEADS_RANGE = [8, 12, 16, 32, 64]


# ---------------------------------------------------------------------------
# Definition makers
# ---------------------------------------------------------------------------


def make_causal_attention_mask() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_causal_attention_mask",
        op_type="attention",
        description=(
            "Generate a causal (lower-triangular) attention mask. "
            "Output[b, i, j] = (j <= i) for all batch elements."
        ),
        tags=["status:pending", "source:trtllm", "tier:3"],
        axes={
            "batch": AxisVar(description="batch size"),
            "seq_len": AxisVar(description="sequence length"),
        },
        inputs={
            "input": TensorSpec(
                shape=["batch", "seq_len", "seq_len"], dtype="bool"
            ),
        },
        outputs={
            "output": TensorSpec(
                shape=["batch", "seq_len", "seq_len"], dtype="bool"
            ),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    B, S, _ = input.shape\n"
            "    mask = torch.tril(\n"
            "        torch.ones(S, S, dtype=torch.bool, device=input.device)\n"
            "    ).expand(B, -1, -1)\n"
            "    return mask\n"
        ),
    )
    ranges = {
        "batch": BATCH_RANGE,
        "seq_len": SEQ_LEN_RANGE,
    }
    return defn, ranges


def make_ban_repeat_ngram() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_ban_repeat_ngram",
        op_type="sampling",
        description=(
            "Ban repeated n-grams during text generation. Scans output_ids for "
            "n-gram matches (ngram_size=3) and sets logits of tokens that would "
            "complete a repeated n-gram to -inf."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "sampling"],
        axes={
            "batch": AxisVar(description="batch size"),
            "vocab_size": AxisVar(description="vocabulary size"),
            "seq_len": AxisVar(description="current sequence length"),
            "ngram_size": AxisConst(value=3, description="n-gram size"),
        },
        inputs={
            "logits": TensorSpec(shape=["batch", "vocab_size"], dtype="float32"),
            "output_ids": TensorSpec(shape=["batch", "seq_len"], dtype="int32"),
        },
        outputs={
            "output": TensorSpec(shape=["batch", "vocab_size"], dtype="float32"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(logits, output_ids):\n"
            "    B, V = logits.shape\n"
            "    _, L = output_ids.shape\n"
            "    NGRAM = 3\n"
            "    out = logits.clone()\n"
            "    if L < NGRAM:\n"
            "        return out\n"
            "    # Current suffix: last (NGRAM-1) tokens\n"
            "    suffix = output_ids[:, -(NGRAM - 1):]  # [B, NGRAM-1]\n"
            "    # All windows of size NGRAM in the history\n"
            "    windows = output_ids.unfold(1, NGRAM, 1)  # [B, L-NGRAM+1, NGRAM]\n"
            "    # Check prefix match: first NGRAM-1 tokens of each window\n"
            "    prefixes = windows[:, :, :NGRAM - 1]  # [B, L-NGRAM+1, NGRAM-1]\n"
            "    match = (prefixes == suffix.unsqueeze(1)).all(dim=2)  # [B, L-NGRAM+1]\n"
            "    # Get the token that follows each matching prefix\n"
            "    next_tokens = windows[:, :, -1]  # [B, L-NGRAM+1]\n"
            "    # Ban those tokens\n"
            "    for b in range(B):\n"
            "        banned = next_tokens[b][match[b]]\n"
            "        banned = banned[banned < V]\n"
            "        if banned.numel() > 0:\n"
            "            out[b, banned.long()] = float('-inf')\n"
            "    return out\n"
        ),
    )
    ranges = {
        "batch": [16, 32, 64],
        "vocab_size": VOCAB_SIZE_RANGE,
        "seq_len": [256, 512, 1024],
    }
    return defn, ranges


def make_group_rmsnorm() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_group_rmsnorm",
        op_type="normalization",
        description=(
            "Grouped RMSNorm: applies independent RMSNorm to two input tensors "
            "with separate weights. Used for parallel attention+MLP normalization."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "normalization"],
        axes={
            "batch": AxisVar(description="batch size (tokens)"),
            "dim1": AxisVar(description="hidden dimension of first input"),
            "dim2": AxisVar(description="hidden dimension of second input"),
        },
        inputs={
            "input1": TensorSpec(shape=["batch", "dim1"], dtype="bfloat16"),
            "input2": TensorSpec(shape=["batch", "dim2"], dtype="bfloat16"),
            "weight1": TensorSpec(shape=["dim1"], dtype="bfloat16"),
            "weight2": TensorSpec(shape=["dim2"], dtype="bfloat16"),
        },
        outputs={
            "output1": TensorSpec(shape=["batch", "dim1"], dtype="bfloat16"),
            "output2": TensorSpec(shape=["batch", "dim2"], dtype="bfloat16"),
        },
        constraints=["dim1 % 32 == 0", "dim2 % 32 == 0"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input1, input2, weight1, weight2):\n"
            "    eps = 1e-6\n"
            "    # RMSNorm for input1\n"
            "    x1 = input1.to(torch.float32)\n"
            "    rms1 = torch.rsqrt(x1.pow(2).mean(dim=-1, keepdim=True) + eps)\n"
            "    out1 = (x1 * rms1 * weight1.to(torch.float32)).to(input1.dtype)\n"
            "    # RMSNorm for input2\n"
            "    x2 = input2.to(torch.float32)\n"
            "    rms2 = torch.rsqrt(x2.pow(2).mean(dim=-1, keepdim=True) + eps)\n"
            "    out2 = (x2 * rms2 * weight2.to(torch.float32)).to(input2.dtype)\n"
            "    return out1, out2\n"
        ),
    )
    ranges = {
        "batch": BATCH_LARGE_RANGE,
        "dim1": DIM_RANGE,
        "dim2": DIM_RANGE,
    }
    return defn, ranges


def make_relative_attention_bias() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_relative_attention_bias",
        op_type="attention",
        description=(
            "T5-style relative position bias with log-bucketed distances. "
            "Computes bidirectional relative position bias from a learned table."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "attention"],
        axes={
            "num_heads": AxisVar(description="number of attention heads"),
            "seq_len": AxisVar(description="sequence length"),
            "num_buckets": AxisConst(
                value=32, description="number of relative position buckets"
            ),
            "max_distance": AxisConst(
                value=128, description="maximum distance for bucket computation"
            ),
        },
        inputs={
            "bias_table": TensorSpec(
                shape=["num_heads", "num_buckets"], dtype="float32"
            ),
            "positions": TensorSpec(shape=["seq_len"], dtype="int32"),
        },
        outputs={
            "output": TensorSpec(
                shape=["num_heads", "seq_len", "seq_len"], dtype="float32"
            ),
        },
        reference=(
            "import torch\n"
            "import math\n\n"
            "@torch.no_grad()\n"
            "def run(bias_table, positions):\n"
            "    H, num_buckets = bias_table.shape\n"
            "    S = positions.shape[0]\n"
            "    max_distance = 128\n"
            "    # Compute relative positions\n"
            "    q_pos = torch.arange(S, device=positions.device)\n"
            "    k_pos = torch.arange(S, device=positions.device)\n"
            "    rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # [S, S]\n"
            "    # Bidirectional bucketing\n"
            "    num_buckets_half = num_buckets // 2\n"
            "    buckets = torch.zeros_like(rel_pos)\n"
            "    neg_mask = rel_pos < 0\n"
            "    rel_pos_abs = rel_pos.abs()\n"
            "    buckets = buckets + (~neg_mask).long() * num_buckets_half\n"
            "    # Small distances: direct mapping\n"
            "    max_exact = num_buckets_half // 2\n"
            "    is_small = rel_pos_abs < max_exact\n"
            "    # Large distances: log-spaced\n"
            "    val_if_large = max_exact + (\n"
            "        torch.log(rel_pos_abs.float() / max_exact)\n"
            "        / math.log(max_distance / max_exact)\n"
            "        * (num_buckets_half - max_exact)\n"
            "    ).long()\n"
            "    val_if_large = val_if_large.clamp(max=num_buckets_half - 1)\n"
            "    bucket_offset = torch.where(is_small, rel_pos_abs, val_if_large)\n"
            "    buckets = buckets + bucket_offset\n"
            "    buckets = buckets.clamp(0, num_buckets - 1)\n"
            "    # Gather bias values\n"
            "    output = bias_table[:, buckets.view(-1)].view(H, S, S)\n"
            "    return output\n"
        ),
    )
    ranges = {
        "num_heads": NUM_HEADS_RANGE,
        "seq_len": SEQ_LEN_SHORT_RANGE,
    }
    return defn, ranges


def make_block_fp8_quant() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_block_fp8_quant",
        op_type="quantization",
        description=(
            "MXFP8 block-scale quantization. Quantizes bfloat16 input to FP8 "
            "(e4m3fn) using per-32-element block scaling."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "quantization"],
        axes={
            "rows": AxisVar(description="number of rows"),
            "cols": AxisVar(description="number of columns"),
        },
        inputs={
            "input": TensorSpec(shape=["rows", "cols"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["rows", "cols"], dtype="float8_e4m3fn"),
        },
        constraints=["cols % 32 == 0"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    BLOCK = 32\n"
            "    FP8_MAX = 448.0\n"
            "    R, C = input.shape\n"
            "    x = input.to(torch.float32).view(R, C // BLOCK, BLOCK)\n"
            "    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)\n"
            "    scale = absmax / FP8_MAX\n"
            "    scaled = x / scale\n"
            "    out = scaled.view(R, C).to(torch.float8_e4m3fn)\n"
            "    return out\n"
        ),
    )
    ranges = {
        "rows": ROWS_RANGE,
        "cols": COLS_RANGE,
    }
    return defn, ranges


def make_fused_relu2_fp8_quant() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_fused_relu2_fp8_quant",
        op_type="quantization",
        description=(
            "Fused ReLU² + MXFP8 block-scale quantization. Applies squared ReLU "
            "activation then quantizes to FP8 (e4m3fn) with per-32-element block scaling."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "quantization"],
        axes={
            "rows": AxisVar(description="number of rows"),
            "cols": AxisVar(description="number of columns"),
        },
        inputs={
            "input": TensorSpec(shape=["rows", "cols"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["rows", "cols"], dtype="float8_e4m3fn"),
        },
        constraints=["cols % 32 == 0"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    BLOCK = 32\n"
            "    FP8_MAX = 448.0\n"
            "    R, C = input.shape\n"
            "    x = input.to(torch.float32)\n"
            "    # ReLU² activation\n"
            "    x = torch.clamp(x, min=0.0).square()\n"
            "    x = x.view(R, C // BLOCK, BLOCK)\n"
            "    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)\n"
            "    scale = absmax / FP8_MAX\n"
            "    scaled = x / scale\n"
            "    out = scaled.view(R, C).to(torch.float8_e4m3fn)\n"
            "    return out\n"
        ),
    )
    ranges = {
        "rows": ROWS_RANGE,
        "cols": COLS_RANGE,
    }
    return defn, ranges


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------

MAKERS = [
    make_causal_attention_mask,
    make_ban_repeat_ngram,
    make_group_rmsnorm,
    make_relative_attention_bias,
    make_block_fp8_quant,
    make_fused_relu2_fp8_quant,
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
        description="Generate TRT-LLM Tier 3 kernel definitions and workloads"
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
