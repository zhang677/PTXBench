"""Collect TRT-LLM Tier 1 kernel definitions and generate workloads.

Creates 6 definitions (5 kernels, MoE routing has 2 top_k variants) with
5 workloads each, written to flashinfer-trace format.

Usage:
    python -m accrl.pipeline.trtllm_tier1 --output-dir ~/accrl-training-data
    python -m accrl.pipeline.trtllm_tier1 --output-dir ~/accrl-training-data --verify
"""

import argparse
import logging
from pathlib import Path

from flashinfer_bench.data import Definition
from flashinfer_bench.data.definition import AxisConst, AxisVar, TensorSpec

from accrl.pipeline.generate_variants import generate_workloads
from accrl.pipeline.trtllm_tier1_solutions import SOLUTION_MAKERS
from accrl.pipeline.write_traceset import write_solution, write_traceset

logger = logging.getLogger(__name__)

# Common axis value ranges
NUM_TOKENS_RANGE = [256, 512, 1024, 2048, 4096, 8192, 16384]
HIDDEN_SIZE_RANGE = [2048, 3584, 4096, 5120, 7168, 8192, 14336]
NUM_EXPERTS_RANGE = [8, 16, 32, 64, 128]
HIDDEN_SIZE_RANGE_WIDE = [2048, 3584, 4096, 5120, 7168, 8192, 14336]


# ---------------------------------------------------------------------------
# Definition makers
# ---------------------------------------------------------------------------


def make_layernorm() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_layernorm",
        op_type="normalization",
        description=(
            "LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta. "
            "Core TRT-LLM normalization kernel."
        ),
        tags=["status:verified", "source:trtllm", "tier:1"],
        axes={
            "num_tokens": AxisVar(description="number of tokens (batch * seq_len)"),
            "hidden_size": AxisVar(description="hidden dimension"),
        },
        inputs={
            "input": TensorSpec(shape=["num_tokens", "hidden_size"], dtype="bfloat16"),
            "gamma": TensorSpec(shape=["hidden_size"], dtype="bfloat16"),
            "beta": TensorSpec(shape=["hidden_size"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["num_tokens", "hidden_size"], dtype="bfloat16"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input, gamma, beta):\n"
            "    EPS = 1e-6\n"
            "    x = input.to(torch.float32)\n"
            "    mean = x.mean(dim=-1, keepdim=True)\n"
            "    var = x.var(dim=-1, keepdim=True, unbiased=False)\n"
            "    y = gamma.to(torch.float32) * (x - mean) / torch.sqrt(var + EPS) + beta.to(torch.float32)\n"
            "    return y.to(input.dtype)\n"
        ),
    )
    ranges = {
        "num_tokens": NUM_TOKENS_RANGE,
        "hidden_size": HIDDEN_SIZE_RANGE,
    }
    return defn, ranges


def make_per_token_quant() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_per_token_quant_fp8",
        op_type="quantization",
        description=(
            "Per-token dynamic FP8 quantization. Computes per-row absmax, "
            "scales to FP8 E4M3 range (max 448)."
        ),
        tags=["status:verified", "source:trtllm", "tier:1"],
        axes={
            "num_tokens": AxisVar(description="number of tokens"),
            "hidden_size": AxisVar(description="hidden dimension"),
        },
        inputs={
            "input": TensorSpec(shape=["num_tokens", "hidden_size"], dtype="bfloat16"),
        },
        outputs={
            "output": TensorSpec(shape=["num_tokens", "hidden_size"], dtype="float8_e4m3fn"),
            "scales": TensorSpec(shape=["num_tokens"], dtype="float32"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    FP8_MAX = 448.0\n"
            "    x = input.to(torch.float32)\n"
            "    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)\n"
            "    scales = (amax / FP8_MAX).squeeze(-1)\n"
            "    output = (x / amax * FP8_MAX).to(torch.float8_e4m3fn)\n"
            "    return output, scales\n"
        ),
    )
    ranges = {
        "num_tokens": NUM_TOKENS_RANGE,
        "hidden_size": HIDDEN_SIZE_RANGE,
    }
    return defn, ranges


def make_moe_routing_topk2() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_moe_routing_topk2",
        op_type="routing",
        description=(
            "MoE routing with top_k=2: softmax over expert logits, "
            "select top-2 experts, normalize selected weights."
        ),
        tags=["status:verified", "source:trtllm", "tier:1", "moe"],
        axes={
            "num_tokens": AxisVar(description="number of tokens"),
            "num_experts": AxisVar(description="number of experts"),
            "top_k": AxisConst(value=2, description="number of selected experts"),
        },
        inputs={
            "routing_logits": TensorSpec(
                shape=["num_tokens", "num_experts"], dtype="float32"
            ),
        },
        outputs={
            "expert_weights": TensorSpec(
                shape=["num_tokens", "top_k"], dtype="float32"
            ),
            "expert_indices": TensorSpec(
                shape=["num_tokens", "top_k"], dtype="int32"
            ),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(routing_logits):\n"
            "    probs = torch.softmax(routing_logits, dim=-1)\n"
            "    top_k = 2\n"
            "    weights, indices = torch.topk(probs, top_k, dim=-1)\n"
            "    weights = weights / weights.sum(dim=-1, keepdim=True)\n"
            "    return weights, indices.to(torch.int32)\n"
        ),
    )
    ranges = {
        "num_tokens": NUM_TOKENS_RANGE,
        "num_experts": NUM_EXPERTS_RANGE,
    }
    return defn, ranges


def make_moe_routing_topk8() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_moe_routing_topk8",
        op_type="routing",
        description=(
            "MoE routing with top_k=8: softmax over expert logits, "
            "select top-8 experts, normalize selected weights."
        ),
        tags=["status:verified", "source:trtllm", "tier:1", "moe"],
        axes={
            "num_tokens": AxisVar(description="number of tokens"),
            "num_experts": AxisVar(description="number of experts"),
            "top_k": AxisConst(value=8, description="number of selected experts"),
        },
        inputs={
            "routing_logits": TensorSpec(
                shape=["num_tokens", "num_experts"], dtype="float32"
            ),
        },
        outputs={
            "expert_weights": TensorSpec(
                shape=["num_tokens", "top_k"], dtype="float32"
            ),
            "expert_indices": TensorSpec(
                shape=["num_tokens", "top_k"], dtype="int32"
            ),
        },
        constraints=["num_experts >= 8"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(routing_logits):\n"
            "    probs = torch.softmax(routing_logits, dim=-1)\n"
            "    top_k = 8\n"
            "    weights, indices = torch.topk(probs, top_k, dim=-1)\n"
            "    weights = weights / weights.sum(dim=-1, keepdim=True)\n"
            "    return weights, indices.to(torch.int32)\n"
        ),
    )
    ranges = {
        "num_tokens": NUM_TOKENS_RANGE,
        "num_experts": NUM_EXPERTS_RANGE,
    }
    return defn, ranges


def make_per_channel_scale() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_per_channel_scale",
        op_type="quantization",
        description=(
            "Per-channel scale multiplication: output = input * scale. "
            "Used for FP8 dequantization with per-channel scales."
        ),
        tags=["status:verified", "source:trtllm", "tier:1"],
        axes={
            "num_tokens": AxisVar(description="number of tokens"),
            "hidden_size": AxisVar(description="hidden / channel dimension"),
        },
        inputs={
            "input": TensorSpec(shape=["num_tokens", "hidden_size"], dtype="bfloat16"),
            "scale": TensorSpec(shape=["hidden_size"], dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["num_tokens", "hidden_size"], dtype="bfloat16"),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input, scale):\n"
            "    output = input.to(torch.float32) * scale.unsqueeze(0)\n"
            "    return output.to(input.dtype)\n"
        ),
    )
    ranges = {
        "num_tokens": NUM_TOKENS_RANGE,
        "hidden_size": HIDDEN_SIZE_RANGE_WIDE,
    }
    return defn, ranges


def make_fused_qknorm_rope() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_fused_qknorm_rope",
        op_type="normalization",
        description=(
            "Fused QK-norm + RoPE: RMSNorm each Q/K head, then apply "
            "rotary position embeddings (theta=10000)."
        ),
        tags=["status:verified", "source:trtllm", "tier:1", "fused"],
        axes={
            "num_tokens": AxisVar(description="number of tokens"),
            "num_q_heads": AxisVar(description="number of query heads"),
            "num_kv_heads": AxisVar(description="number of key/value heads"),
            "head_dim": AxisVar(description="head dimension"),
        },
        inputs={
            "query": TensorSpec(
                shape=["num_tokens", "num_q_heads", "head_dim"], dtype="bfloat16"
            ),
            "key": TensorSpec(
                shape=["num_tokens", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
            "q_norm_weight": TensorSpec(shape=["head_dim"], dtype="bfloat16"),
            "k_norm_weight": TensorSpec(shape=["head_dim"], dtype="bfloat16"),
            "positions": TensorSpec(shape=["num_tokens"], dtype="int32"),
        },
        outputs={
            "query_out": TensorSpec(
                shape=["num_tokens", "num_q_heads", "head_dim"], dtype="bfloat16"
            ),
            "key_out": TensorSpec(
                shape=["num_tokens", "num_kv_heads", "head_dim"], dtype="bfloat16"
            ),
        },
        constraints=[
            "num_kv_heads <= num_q_heads",
            "num_q_heads % num_kv_heads == 0",
        ],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(query, key, q_norm_weight, k_norm_weight, positions):\n"
            "    EPS = 1e-6\n"
            "    THETA = 10000.0\n"
            "\n"
            "    def rmsnorm(x, weight):\n"
            "        # x: [num_tokens, num_heads, head_dim]\n"
            "        xf = x.to(torch.float32)\n"
            "        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)\n"
            "        return (xf * inv_rms * weight.to(torch.float32)).to(x.dtype)\n"
            "\n"
            "    def rope(x, positions):\n"
            "        # x: [num_tokens, num_heads, head_dim]\n"
            "        head_dim = x.shape[-1]\n"
            "        half = head_dim // 2\n"
            "        freq_seq = torch.arange(half, device=x.device, dtype=torch.float32)\n"
            "        freqs = 1.0 / (THETA ** (freq_seq / half))\n"
            "        angles = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [T, half]\n"
            "        cos = angles.cos().unsqueeze(1)  # [T, 1, half]\n"
            "        sin = angles.sin().unsqueeze(1)  # [T, 1, half]\n"
            "        xf = x.to(torch.float32)\n"
            "        x1 = xf[..., :half]\n"
            "        x2 = xf[..., half:]\n"
            "        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)\n"
            "        return out.to(x.dtype)\n"
            "\n"
            "    q = rmsnorm(query, q_norm_weight)\n"
            "    k = rmsnorm(key, k_norm_weight)\n"
            "    q_out = rope(q, positions)\n"
            "    k_out = rope(k, positions)\n"
            "    return q_out, k_out\n"
        ),
    )
    ranges = {
        "num_tokens": [256, 512, 1024, 2048, 4096, 8192, 16384],
        "num_q_heads": [8, 16, 32, 64],
        "num_kv_heads": [1, 2, 4, 8],
        "head_dim": [64, 96, 128],
    }
    return defn, ranges


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------

MAKERS = [
    make_layernorm,
    make_per_token_quant,
    make_moe_routing_topk2,
    make_moe_routing_topk8,
    make_per_channel_scale,
    make_fused_qknorm_rope,
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
        description="Generate TRT-LLM Tier 1 kernel definitions and workloads"
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
