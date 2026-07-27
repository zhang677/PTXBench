"""Collect TRT-LLM Weight-Only GEMV kernel definitions and generate workloads.

Creates 4 definitions (INT8/INT4 x per-channel/groupwise) with workloads,
written to flashinfer-trace format.

Usage:
    python -m accrl.pipeline.trtllm_woq_gemv --output-dir ~/accrl-training-data
    python -m accrl.pipeline.trtllm_woq_gemv --output-dir ~/accrl-training-data --verify
    python -m accrl.pipeline.trtllm_woq_gemv --output-dir ~/accrl-training-data --verify-solutions
"""

import argparse
import logging
from pathlib import Path

from flashinfer_bench.data import Definition
from flashinfer_bench.data.definition import AxisConst, AxisVar, TensorSpec

from accrl.pipeline.generate_variants import generate_workloads
from accrl.pipeline.trtllm_woq_gemv_solutions import SOLUTION_MAKERS
from accrl.pipeline.write_traceset import write_solution, write_traceset

logger = logging.getLogger(__name__)

# Common axis value ranges for GEMV workloads
M_RANGE = [1, 2, 4, 8]  # GEMV regime (small batch)
N_RANGE = [4096, 5120, 8192, 11008, 14336]  # LLM FFN dims
K_RANGE = [4096, 5120, 8192, 11008, 14336]  # All divisible by 128
K_PACKED_RANGE = [2048, 2560, 4096, 5504, 7168]  # K/2 for INT4 packing
NUM_GROUPS_RANGE = [32, 40, 64, 86, 112]  # K/128 for groupwise


# ---------------------------------------------------------------------------
# Definition makers
# ---------------------------------------------------------------------------


def make_w8a16_perchannel() -> tuple[Definition, dict[str, list[int]], int]:
    """INT8 weights, BF16 activations, per-channel scale."""
    defn = Definition(
        name="trtllm_woq_gemv_w8a16_perchannel",
        op_type="gemv",
        description=(
            "Weight-only quantized GEMV with INT8 weights and BF16 activations. "
            "Per-channel dequantization: w_fp = weight * scales[n]. "
            "Output = activation @ w_fp^T."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "quantization", "gemv"],
        axes={
            "m": AxisVar(description="batch/token dimension (GEMV regime)"),
            "n": AxisVar(description="output dimension"),
            "k": AxisVar(description="input/reduction dimension"),
        },
        inputs={
            "activation": TensorSpec(shape=["m", "k"], dtype="bfloat16"),
            "weight": TensorSpec(shape=["n", "k"], dtype="int8"),
            "scales": TensorSpec(shape=["n"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["m", "n"], dtype="bfloat16"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(activation, weight, scales):\n"
            "    w_fp = weight.float() * scales.unsqueeze(1).float()\n"
            "    out = (activation.float() @ w_fp.t()).to(torch.bfloat16)\n"
            "    return out\n"
        ),
    )
    ranges = {"m": M_RANGE, "n": N_RANGE, "k": K_RANGE}
    return defn, ranges, 1


def make_w8a16_groupwise() -> tuple[Definition, dict[str, list[int]], int]:
    """INT8 weights, BF16 activations, group_size=128, with zeros."""
    defn = Definition(
        name="trtllm_woq_gemv_w8a16_groupwise",
        op_type="gemv",
        description=(
            "Weight-only quantized GEMV with INT8 weights, BF16 activations, "
            "and groupwise dequantization (group_size=128). Each group of 128 "
            "elements along K shares a scale and zero point. "
            "w_fp = weight * scale + zero per group."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "quantization", "gemv"],
        axes={
            "m": AxisVar(description="batch/token dimension (GEMV regime)"),
            "n": AxisVar(description="output dimension"),
            "k": AxisVar(description="input/reduction dimension"),
            "num_groups": AxisVar(description="number of groups (k / 128)"),
            "group_size": AxisConst(value=128, description="elements per group"),
        },
        inputs={
            "activation": TensorSpec(shape=["m", "k"], dtype="bfloat16"),
            "weight": TensorSpec(shape=["n", "k"], dtype="int8"),
            "scales": TensorSpec(shape=["num_groups", "n"], dtype="bfloat16"),
            "zeros": TensorSpec(shape=["num_groups", "n"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["m", "n"], dtype="bfloat16"),
        },
        constraints=["num_groups * 128 == k"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(activation, weight, scales, zeros):\n"
            "    N, K = weight.shape\n"
            "    G = scales.shape[0]\n"
            "    GS = 128\n"
            "    # Reshape weight to [N, G, GS] for per-group dequant\n"
            "    w = weight.float().view(N, G, GS)\n"
            "    s = scales.float().unsqueeze(2)   # [G, N, 1] -> need [N, G, 1]\n"
            "    z = zeros.float().unsqueeze(2)\n"
            "    # scales/zeros are [G, N], transpose to [N, G] then unsqueeze\n"
            "    s = scales.float().t().unsqueeze(2)  # [N, G, 1]\n"
            "    z = zeros.float().t().unsqueeze(2)   # [N, G, 1]\n"
            "    w_fp = (w * s + z).view(N, K)\n"
            "    out = (activation.float() @ w_fp.t()).to(torch.bfloat16)\n"
            "    return out\n"
        ),
    )
    ranges = {
        "m": M_RANGE,
        "n": N_RANGE,
        "k": K_RANGE,
        "num_groups": NUM_GROUPS_RANGE,
    }
    return defn, ranges, 10


def make_w4a16_perchannel() -> tuple[Definition, dict[str, list[int]], int]:
    """INT4 packed weights, BF16 activations, per-channel scale."""
    defn = Definition(
        name="trtllm_woq_gemv_w4a16_perchannel",
        op_type="gemv",
        description=(
            "Weight-only quantized GEMV with INT4 packed weights and BF16 activations. "
            "Each int8 byte stores 2 INT4 values (lo nibble, hi nibble). "
            "Unsigned nibble extraction then subtract 8 for signed range [-8,7]. "
            "Per-channel dequantization with single scale per output channel."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "quantization", "gemv"],
        axes={
            "m": AxisVar(description="batch/token dimension (GEMV regime)"),
            "n": AxisVar(description="output dimension"),
            "k": AxisVar(description="logical input dimension"),
            "k_packed": AxisVar(description="physical packed dimension (k/2)"),
        },
        inputs={
            "activation": TensorSpec(shape=["m", "k"], dtype="bfloat16"),
            "weight": TensorSpec(shape=["n", "k_packed"], dtype="int8"),
            "scales": TensorSpec(shape=["n"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["m", "n"], dtype="bfloat16"),
        },
        constraints=["k == k_packed * 2"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(activation, weight, scales):\n"
            "    N, K_packed = weight.shape\n"
            "    # Unpack INT4: each byte -> 2 values\n"
            "    packed = weight.to(torch.int32) & 0xFF\n"
            "    lo = (packed & 0x0F) - 8        # low nibble, signed\n"
            "    hi = ((packed >> 4) & 0x0F) - 8  # high nibble, signed\n"
            "    # Interleave to [N, K]: lo0, hi0, lo1, hi1, ...\n"
            "    w_int = torch.stack([lo, hi], dim=-1).view(N, K_packed * 2)\n"
            "    w_fp = w_int.float() * scales.float().unsqueeze(1)\n"
            "    out = (activation.float() @ w_fp.t()).to(torch.bfloat16)\n"
            "    return out\n"
        ),
    )
    ranges = {
        "m": M_RANGE,
        "n": N_RANGE,
        "k": K_RANGE,
        "k_packed": K_PACKED_RANGE,
    }
    return defn, ranges, 10


def make_w4a16_groupwise() -> tuple[Definition, dict[str, list[int]], int]:
    """INT4 packed weights, BF16 activations, group_size=128, with zeros."""
    defn = Definition(
        name="trtllm_woq_gemv_w4a16_groupwise",
        op_type="gemv",
        description=(
            "Weight-only quantized GEMV with INT4 packed weights, BF16 activations, "
            "and groupwise dequantization (group_size=128). Each int8 byte stores "
            "2 INT4 values. Per-group scale and zero point for dequantization."
        ),
        tags=["status:pending", "source:trtllm", "tier:3", "quantization", "gemv"],
        axes={
            "m": AxisVar(description="batch/token dimension (GEMV regime)"),
            "n": AxisVar(description="output dimension"),
            "k": AxisVar(description="logical input dimension"),
            "k_packed": AxisVar(description="physical packed dimension (k/2)"),
            "num_groups": AxisVar(description="number of groups (k / 128)"),
            "group_size": AxisConst(value=128, description="elements per group"),
        },
        inputs={
            "activation": TensorSpec(shape=["m", "k"], dtype="bfloat16"),
            "weight": TensorSpec(shape=["n", "k_packed"], dtype="int8"),
            "scales": TensorSpec(shape=["num_groups", "n"], dtype="bfloat16"),
            "zeros": TensorSpec(shape=["num_groups", "n"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["m", "n"], dtype="bfloat16"),
        },
        constraints=["k == k_packed * 2", "num_groups * 128 == k"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(activation, weight, scales, zeros):\n"
            "    N, K_packed = weight.shape\n"
            "    K = K_packed * 2\n"
            "    G = scales.shape[0]\n"
            "    GS = 128\n"
            "    # Unpack INT4: each byte -> 2 values\n"
            "    packed = weight.to(torch.int32) & 0xFF\n"
            "    lo = (packed & 0x0F) - 8\n"
            "    hi = ((packed >> 4) & 0x0F) - 8\n"
            "    w_int = torch.stack([lo, hi], dim=-1).view(N, K)\n"
            "    # Per-group dequant: reshape to [N, G, GS]\n"
            "    w = w_int.float().view(N, G, GS)\n"
            "    s = scales.float().t().unsqueeze(2)  # [N, G, 1]\n"
            "    z = zeros.float().t().unsqueeze(2)   # [N, G, 1]\n"
            "    w_fp = (w * s + z).view(N, K)\n"
            "    out = (activation.float() @ w_fp.t()).to(torch.bfloat16)\n"
            "    return out\n"
        ),
    )
    ranges = {
        "m": M_RANGE,
        "n": N_RANGE,
        "k": K_RANGE,
        "k_packed": K_PACKED_RANGE,
        "num_groups": NUM_GROUPS_RANGE,
    }
    return defn, ranges, 50


# ---------------------------------------------------------------------------
# Constraint filtering (reuse from trtllm_tier3)
# ---------------------------------------------------------------------------

MAKERS = [
    make_w8a16_perchannel,
    make_w8a16_groupwise,
    make_w4a16_perchannel,
    make_w4a16_groupwise,
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
        description="Generate TRT-LLM Weight-Only GEMV kernel definitions and workloads"
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
        defn, ranges, multiplier = make_fn()

        # Generate more candidates for constrained definitions, then filter
        if defn.constraints:
            traces = generate_workloads(
                defn, ranges, n=args.n * multiplier, seed=args.seed
            )
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
