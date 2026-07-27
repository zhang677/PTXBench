"""Collect TRT-LLM Tier 4 kernel definitions and generate workloads.

Creates 5 definitions with workloads each, written to flashinfer-trace format.

Usage:
    python -m accrl.pipeline.trtllm_tier4 --output-dir ~/accrl-training-data
    python -m accrl.pipeline.trtllm_tier4 --output-dir ~/accrl-training-data --verify
"""

import argparse
import logging
from pathlib import Path

from flashinfer_bench.data import Definition
from flashinfer_bench.data.definition import AxisConst, AxisVar, TensorSpec

from accrl.pipeline.generate_variants import generate_workloads
from accrl.pipeline.trtllm_tier4_solutions import SOLUTION_MAKERS
from accrl.pipeline.write_traceset import write_solution, write_traceset

logger = logging.getLogger(__name__)

# Common axis value ranges
BATCH_RANGE = [16, 32, 64, 128]
BATCH_SMALL_RANGE = [1, 2, 4, 8]  # for SSM/LRU (decode-path)
SEQ_LEN_RANGE = [256, 512, 1024, 2048]
DIM_RANGE = [768, 1024, 2048, 4096]
NUM_HEADS_RANGE = [8, 16, 32, 64]
HEAD_DIM_RANGE = [64, 80, 128]
BEAM_WIDTH_RANGE = [4, 8, 16, 32]
VOCAB_SIZE_RANGE = [32000, 65536, 128256]


# ---------------------------------------------------------------------------
# Definition makers
# ---------------------------------------------------------------------------


def make_selective_scan() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_selective_scan",
        op_type="ssm",
        description=(
            "Mamba selective scan (S6). Computes discretized state-space recurrence "
            "with input-dependent B, C matrices and D skip connection. "
            "Uses softplus for delta discretization."
        ),
        tags=["status:pending", "source:trtllm", "tier:4", "ssm"],
        axes={
            "batch": AxisVar(description="batch size"),
            "seq_len": AxisVar(description="sequence length"),
            "dim": AxisVar(description="model dimension"),
            "state_dim": AxisConst(value=16, description="SSM state dimension N"),
        },
        inputs={
            "x": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
            "delta": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
            "A": TensorSpec(shape=["dim", "state_dim"], dtype="float32"),
            "B": TensorSpec(shape=["batch", "seq_len", "state_dim"], dtype="float32"),
            "C": TensorSpec(shape=["batch", "seq_len", "state_dim"], dtype="float32"),
            "D_param": TensorSpec(shape=["dim"], dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
        },
        constraints=["dim % 128 == 0"],
        reference=(
            "import torch\n"
            "import torch.nn.functional as F\n\n"
            "@torch.no_grad()\n"
            "def run(x, delta, A, B, C, D_param):\n"
            "    B_sz, L, D = x.shape\n"
            "    N = A.shape[1]\n"
            "    # Discretize: delta_t = softplus(delta)\n"
            "    dt = F.softplus(delta)  # [B, L, D]\n"
            "    # Initialize state\n"
            "    h = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)\n"
            "    outputs = []\n"
            "    for t in range(L):\n"
            "        # Discretized A: exp(dt * A)\n"
            "        dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))  # [B, D, N]\n"
            "        # Discretized B: dt * B\n"
            "        dB = dt[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # [B, D, N]\n"
            "        # State update: h = dA * h + dB * x\n"
            "        h = dA * h + dB * x[:, t, :].unsqueeze(-1)\n"
            "        # Output: y = (h @ C^T) + D * x\n"
            "        y = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # [B, D]\n"
            "        y = y + D_param * x[:, t, :]\n"
            "        outputs.append(y)\n"
            "    return torch.stack(outputs, dim=1)  # [B, L, D]\n"
        ),
    )
    ranges = {
        "batch": BATCH_SMALL_RANGE,
        "seq_len": SEQ_LEN_RANGE,
        "dim": DIM_RANGE,
    }
    return defn, ranges


def make_lru_recurrence() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_lru_recurrence",
        op_type="rnn",
        description=(
            "Linear Recurrent Unit (LRU) recurrence from RecurrentGemma. "
            "Gated recurrence with sigmoid input gate, softplus decay, "
            "and GELU output gating."
        ),
        tags=["status:pending", "source:trtllm", "tier:4", "rnn"],
        axes={
            "batch": AxisVar(description="batch size"),
            "seq_len": AxisVar(description="sequence length"),
            "dim": AxisVar(description="recurrence dimension"),
        },
        inputs={
            "x": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
            "param_a": TensorSpec(shape=["dim"], dtype="float32"),
            "gate_x": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
            "gate_a": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
            "y_param": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
        },
        outputs={
            "output": TensorSpec(shape=["batch", "seq_len", "dim"], dtype="float32"),
        },
        constraints=["dim % 128 == 0"],
        reference=(
            "import torch\n"
            "import torch.nn.functional as F\n\n"
            "@torch.no_grad()\n"
            "def run(x, param_a, gate_x, gate_a, y_param):\n"
            "    B, L, D = x.shape\n"
            "    # Decay rate from param_a\n"
            "    a = -F.softplus(param_a)  # [D], negative for stability\n"
            "    h = torch.zeros(B, D, device=x.device, dtype=x.dtype)\n"
            "    outputs = []\n"
            "    for t in range(L):\n"
            "        # Input gate\n"
            "        gx = torch.sigmoid(gate_x[:, t, :])  # [B, D]\n"
            "        # Recurrence gate (per-step modulation of decay)\n"
            "        ga = torch.sigmoid(gate_a[:, t, :])  # [B, D]\n"
            "        # Effective decay\n"
            "        decay = torch.exp(a.unsqueeze(0) * ga)  # [B, D]\n"
            "        # State update\n"
            "        h = decay * h + gx * x[:, t, :]\n"
            "        # Output gating with GELU\n"
            "        out = h * F.gelu(y_param[:, t, :])  # [B, D]\n"
            "        outputs.append(out)\n"
            "    return torch.stack(outputs, dim=1)  # [B, L, D]\n"
        ),
    )
    ranges = {
        "batch": BATCH_SMALL_RANGE,
        "seq_len": SEQ_LEN_RANGE,
        "dim": DIM_RANGE,
    }
    return defn, ranges


def make_sage_attention_quant() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_sage_attention_quant",
        op_type="quantization",
        description=(
            "Sage Attention block-wise FP8 quantization for 4D attention tensors. "
            "Quantizes [B,H,S,D] bf16 input to FP8 (e4m3fn) with per-block "
            "absmax scaling along the sequence dimension (block_size=64)."
        ),
        tags=["status:pending", "source:trtllm", "tier:4", "quantization"],
        axes={
            "batch": AxisVar(description="batch size"),
            "num_heads": AxisVar(description="number of attention heads"),
            "seq_len": AxisVar(description="sequence length"),
            "head_dim": AxisVar(description="head dimension"),
            "block_size": AxisConst(value=64, description="quantization block size along seq dim"),
        },
        inputs={
            "input": TensorSpec(
                shape=["batch", "num_heads", "seq_len", "head_dim"], dtype="bfloat16"
            ),
        },
        outputs={
            "output": TensorSpec(
                shape=["batch", "num_heads", "seq_len", "head_dim"], dtype="float8_e4m3fn"
            ),
        },
        constraints=["seq_len % 64 == 0", "head_dim % 8 == 0"],
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(input):\n"
            "    BLOCK = 64\n"
            "    FP8_MAX = 448.0\n"
            "    B, H, S, D = input.shape\n"
            "    # Reshape to expose blocks along seq dim\n"
            "    x = input.to(torch.float32).view(B, H, S // BLOCK, BLOCK, D)\n"
            "    # Per-block absmax over (block_elements, head_dim)\n"
            "    absmax = x.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)\n"
            "    scale = absmax / FP8_MAX\n"
            "    scaled = x / scale\n"
            "    out = scaled.view(B, H, S, D).to(torch.float8_e4m3fn)\n"
            "    return out\n"
        ),
    )
    ranges = {
        "batch": BATCH_RANGE,
        "num_heads": NUM_HEADS_RANGE,
        "seq_len": SEQ_LEN_RANGE,
        "head_dim": HEAD_DIM_RANGE,
    }
    return defn, ranges


def make_ring_attention_recovery() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_ring_attention_recovery",
        op_type="attention",
        description=(
            "Ring attention partial output merging via online softmax correction. "
            "Merges accumulated and new partial attention outputs using their "
            "respective max and sum statistics."
        ),
        tags=["status:pending", "source:trtllm", "tier:4", "attention"],
        axes={
            "batch": AxisVar(description="batch size"),
            "num_heads": AxisVar(description="number of attention heads"),
            "seq_len": AxisVar(description="sequence length"),
            "head_dim": AxisVar(description="head dimension"),
        },
        inputs={
            "accu_out": TensorSpec(
                shape=["batch", "num_heads", "seq_len", "head_dim"], dtype="float32"
            ),
            "new_out": TensorSpec(
                shape=["batch", "num_heads", "seq_len", "head_dim"], dtype="float32"
            ),
            "accu_max": TensorSpec(
                shape=["batch", "num_heads", "seq_len"], dtype="float32"
            ),
            "accu_sum": TensorSpec(
                shape=["batch", "num_heads", "seq_len"], dtype="float32"
            ),
            "new_max": TensorSpec(
                shape=["batch", "num_heads", "seq_len"], dtype="float32"
            ),
            "new_sum": TensorSpec(
                shape=["batch", "num_heads", "seq_len"], dtype="float32"
            ),
        },
        outputs={
            "output": TensorSpec(
                shape=["batch", "num_heads", "seq_len", "head_dim"], dtype="float32"
            ),
            "out_max": TensorSpec(
                shape=["batch", "num_heads", "seq_len"], dtype="float32"
            ),
            "out_sum": TensorSpec(
                shape=["batch", "num_heads", "seq_len"], dtype="float32"
            ),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(accu_out, new_out, accu_max, accu_sum, new_max, new_sum):\n"
            "    # Online softmax merging of two partial attention results\n"
            "    # new global max\n"
            "    m = torch.maximum(accu_max, new_max)  # [B, H, S]\n"
            "    # Rescale factors\n"
            "    exp_accu = torch.exp(accu_max - m)  # [B, H, S]\n"
            "    exp_new = torch.exp(new_max - m)    # [B, H, S]\n"
            "    # New global sum\n"
            "    s = exp_accu * accu_sum + exp_new * new_sum  # [B, H, S]\n"
            "    # Merge outputs: weighted combination\n"
            "    out = (\n"
            "        exp_accu.unsqueeze(-1) * accu_sum.unsqueeze(-1) * accu_out\n"
            "        + exp_new.unsqueeze(-1) * new_sum.unsqueeze(-1) * new_out\n"
            "    ) / s.unsqueeze(-1).clamp(min=1e-12)\n"
            "    return out, m, s\n"
        ),
    )
    ranges = {
        "batch": BATCH_RANGE,
        "num_heads": NUM_HEADS_RANGE,
        "seq_len": SEQ_LEN_RANGE,
        "head_dim": HEAD_DIM_RANGE,
    }
    return defn, ranges


def make_beam_score_update() -> tuple[Definition, dict[str, list[int]]]:
    defn = Definition(
        name="trtllm_beam_score_update",
        op_type="sampling",
        description=(
            "Beam search score update. Adds cumulative log probabilities to "
            "current step log probabilities via broadcast addition: "
            "output[b,bw,v] = log_probs[b,bw,v] + cum_log_probs[b,bw]."
        ),
        tags=["status:pending", "source:trtllm", "tier:4", "sampling"],
        axes={
            "batch": AxisVar(description="batch size"),
            "beam_width": AxisVar(description="beam width"),
            "vocab_size": AxisVar(description="vocabulary size"),
        },
        inputs={
            "log_probs": TensorSpec(
                shape=["batch", "beam_width", "vocab_size"], dtype="float32"
            ),
            "cum_log_probs": TensorSpec(
                shape=["batch", "beam_width"], dtype="float32"
            ),
        },
        outputs={
            "output": TensorSpec(
                shape=["batch", "beam_width", "vocab_size"], dtype="float32"
            ),
        },
        reference=(
            "import torch\n\n"
            "@torch.no_grad()\n"
            "def run(log_probs, cum_log_probs):\n"
            "    return log_probs + cum_log_probs.unsqueeze(-1)\n"
        ),
    )
    ranges = {
        "batch": BATCH_RANGE,
        "beam_width": BEAM_WIDTH_RANGE,
        "vocab_size": VOCAB_SIZE_RANGE,
    }
    return defn, ranges


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------

MAKERS = [
    make_selective_scan,
    make_lru_recurrence,
    make_sage_attention_quant,
    make_ring_attention_recovery,
    make_beam_score_update,
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
        description="Generate TRT-LLM Tier 4 kernel definitions and workloads"
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
