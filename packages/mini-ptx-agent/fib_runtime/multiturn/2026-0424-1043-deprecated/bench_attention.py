"""Benchmark the helion reference (from each Definition's `reference`)
against F.scaled_dot_product_attention across the 6 workload shapes.
"""

import importlib.util
import json
import os
import sys
import tempfile
from typing import Dict, Any

import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
B_VAL, H_VAL, D_VAL = 4, 48, 128
S_VALUES = [128, 256, 512, 1024, 2048, 4096]


def load_reference_as_module(defn: Dict[str, Any]):
    tmpdir = tempfile.mkdtemp(prefix="attn_bench_")
    mod_name = f"_attn_bench_{defn['name']}"
    src_path = os.path.join(tmpdir, mod_name + ".py")
    with open(src_path, "w") as f:
        f.write(defn["reference"])
    spec = importlib.util.spec_from_file_location(mod_name, src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def bench_callable(fn, args, warmup: int = 5, iters: int = 50) -> float:
    """Return median latency in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def fmt_us(ms: float) -> str:
    return f"{ms * 1000:>9.1f} us"


def bench_definition(defn: Dict[str, Any]):
    is_causal = "causal:true" in defn["tags"]
    print(f"\n=== {defn['name']}  (is_causal={is_causal}) ===")
    print(f"{'S':>5}  {'helion (med)':>14}  {'F.sdpa bf16 (med)':>20}  "
          f"{'speedup helion/sdpa':>20}")
    print("-" * 70)

    mod = load_reference_as_module(defn)
    helion_run = mod.run

    def sdpa_run(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    for s in S_VALUES:
        torch.manual_seed(0)
        shape = (B_VAL, H_VAL, s, D_VAL)
        Q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        helion_ms = bench_callable(helion_run, (Q, K, V), warmup=3, iters=30)
        sdpa_ms = bench_callable(sdpa_run, (Q, K, V), warmup=3, iters=30)

        ratio = sdpa_ms / helion_ms
        print(f"{s:>5}  {fmt_us(helion_ms):>14}  {fmt_us(sdpa_ms):>20}  "
              f"{ratio:>19.2f}x")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Shapes: B={B_VAL}, H={H_VAL}, D={D_VAL}, dtype=bfloat16")

    for base in ("attention_h48_d128", "attention_h48_d128_causal"):
        with open(os.path.join(HERE, f"{base}.json")) as f:
            defn = json.load(f)
        bench_definition(defn)


if __name__ == "__main__":
    main()
