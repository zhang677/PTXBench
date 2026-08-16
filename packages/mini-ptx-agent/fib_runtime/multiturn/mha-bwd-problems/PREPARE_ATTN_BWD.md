1. In the definition: Q, K, V, O, dO, L, and D are inputs; they are stored as safetensors stored in accrl-training/blob similar to flashinfer-trace/blob
2. To prepare these inputs:
    1. Write the forward pytorch, collect these tensors from torch.normal(0, 1) + torch.normal(0, 100) * torch.bernoulli(0.001) as input Q, K, V; use the same workload shapes as /home/ubuntu/accrl-training/workloads/attention/mha_h48_d128.jsonl
    2. Find cuDNN function that calculates dKdV and dQ because Figure 6 in fa3.pdf compared with cuDNN for the backward pass — we use `cudnn.pygraph.sdpa_backward` directly, the same path FlashAttention-3's `hopper/benchmark_attn.py` uses for Fig. 6.
    3. Match the cuDNN with PyTorch backward
3. Stop here for human inspection 
4. Copy to accrl-training as reference

## Script usage

`scripts/prepare_mha_bwd_h48_d128.py` is now self-contained: it embeds the
definition (`DEFINITION_DICT` + `REFERENCE_SOURCE`) and writes
`definitions/attention/mha_bwd_h48_d128.json` alongside the per-workload
safetensors blobs.

### Common invocations

Materialize all six workloads under the staging directory and skip files that
already exist:

```bash
cd /home/ubuntu/AccRL/fib_runtime/multiturn/mha-bwd-problems
python scripts/prepare_mha_bwd_h48_d128.py --dataset-root .
```

Re-generate everything from scratch (definition JSON + safetensors) and run
cuDNN backward smoke validation without the memory-heavy fp32 PyTorch check:

```bash
python scripts/prepare_mha_bwd.py \
    --dataset-root . --smoke-backward --overwrite
```

Materialize into the published dataset root once verified:

```bash
python scripts/prepare_mha_bwd.py \
    --dataset-root /home/ubuntu/accrl-training --smoke-backward --overwrite
```

### Flags

| flag | default | meaning |
|---|---|---|
| `--dataset-root PATH` | `cwd` | root that contains `workloads/` and where `definitions/` and `blob/` are written. |
| `--check` | off | run cuDNN backward against the fp32 algorithmic reference. |
| `--smoke-backward` | off | run cuDNN backward and validate `dQ/dK/dV` shape, dtype, and device without the fp32 PyTorch comparison. |
| `--overwrite` | off | rewrite the definition JSON and any existing safetensors. |
| `--rtol FLOAT` | `1e-2` | tolerance forwarded to the check metric. |
| `--atol FLOAT` | `1e-2` | tolerance forwarded to the check metric. |
| `--input-distribution {spiky,randn}` | `spiky` | sampler for `Q/K/V/dO`. `spiky` is the FA3-style stress test (`N(0,1)+N(0,100)·Bern(0.001)`); `randn` is well-conditioned and well-suited to pointwise checks. |
| `--metric {allclose,relnorm}` | auto | `allclose`→pointwise `torch.allclose(rtol,atol)`; `relnorm`→`‖err‖₂/‖ref‖₂ ≤ rtol+atol`. Auto picks `allclose` for `randn`, `relnorm` for `spiky`. |
| `--print-schemas` | off | dump the cuDNN forward/backward aten schemas and exit-after-printing. |

### Why two metrics

The reference (`run`) calls `cudnn.pygraph.sdpa_backward` directly — the same
path FlashAttention-3's `hopper/benchmark_attn.py` uses to produce Fig. 6 of
the FA3 paper. The script compares the bf16 cuDNN backward against an
**fp32 algorithmic reference** (explicit softmax + matmul) instead of
`F.scaled_dot_product_attention` autograd: pytorch's SDPA dispatches to cuDNN
by default (CUDNN_ATTENTION is the highest-priority backend), so an autograd
comparison would be cuDNN-vs-cuDNN — informative for kernel parity but not
for catching algorithmic regressions.

| distribution | pointwise `allclose` at 1e-2 | `‖err‖₂/‖ref‖₂` at 1e-2 |
|---|---|---|
| `randn`  | ✅ passes | ✅ ≈ 2e-3 |
| `spiky`  | ❌ fails (single bf16-rounded outliers ~10²) | ✅ ≈ 2e-3 to 9e-3 |

The materialized safetensors deliberately use the spiky distribution (the
PREPARE_ATTN_BWD spec above), so `--metric relnorm` is the meaningful default
for stress data; `--input-distribution randn` is for confirming the bf16
cuDNN backward agrees pointwise with the fp32 reference on well-conditioned
inputs.

### Outputs per run

- `definitions/attention/mha_bwd_h48_d128.json` — written from the embedded
  `DEFINITION_DICT` (skipped if present and `--overwrite` not passed).
- `blob/workloads/attention/mha_bwd_h48_d128/mha_bwd_h48_d128_<uuid>.safetensors`
  — one ~30 MiB file per workload (S=128) up to ~1 GiB (S=4096), each
  containing keys `Q, K, V, O, dO, L, D` with shapes/dtypes matching the
  definition.

### Verify the profiling service
```bash
PYTHONPATH=/home/ubuntu/AccRL python /home/ubuntu/AccRL/fib_runtime/multiturn/mha-bwd-problems/scripts/verify_via_service.py
```

### FlashAttention backward vs cuDNN backward (H100, bf16, non-causal)

The reference (`run` in the definition's `reference`) is now pure
`cudnn.pygraph.sdpa_backward` — same call as FlashAttention-3's
`hopper/benchmark_attn.py::cudnn_spda_bwd_setup`, which is the path used to
produce Fig. 6 of the FA3 paper. The reference receives the stored
`(Q, K, V, O, dO, L, D)` directly and does **not** run a forward; timing is
backward-only and apples-to-apples with the FA3 paper.

Solution under test: `solutions/fa_bwd_main.py` — currently calls
`aten._scaled_dot_product_flash_attention_backward` (FA2 in this PyTorch build),
re-running its forward to recover its own `(philox_seed, philox_offset, cum_seq_*,
max_*)`. The FA solution still times fwd+bwd while the cuDNN reference times
bwd-only, so the comparison is not symmetric — FA pays for its forward pass
but cuDNN does not.

| S | FA bwd latency (fwd+bwd) | cuDNN bwd ref latency (bwd-only) | speedup | max_abs | max_rel |
|---:|---:|---:|---:|---:|---:|
| 128  | 0.126 ms | 0.058 ms | 0.46x | 7.8e-3 | 5.9e3 |
| 256  | 0.193 ms | 0.108 ms | 0.56x | 7.8e-3 | 1.1e4 |
| 512  | 0.486 ms | 0.239 ms | 0.49x | 7.8e-3 | 3.6e4 |
| 1024 | 1.479 ms | 0.635 ms | 0.43x | 3.9e-3 | 1.6e4 |
| 2048 | 5.075 ms | 2.167 ms | 0.43x | 3.9e-3 | 1.4e4 |
| 4096 | 18.417 ms | 7.913 ms | 0.43x | 2.0e-3 | 9.4e3 |

```text
B = 4, H = 48, S = 2048, D = 128
FLOPs estimate = 2.5 * 4 * B * H * S * S * D = 1030792151040
```

#### cuDNN backward TFLOPS via cudnn.pygraph (FA3 Fig. 6 path)

`scripts/profile_cudnn_fa3_fig6.py` runs the FA3 benchmark's exact backward
setup and timing (`triton.testing.do_bench(warmup=3, rep=30)`) on this H100.

| S | latency | TFLOPS |
|---:|---:|---:|
| 128  | 0.057 ms |  70.7 |
| 256  | 0.106 ms | 152.4 |
| 512  | 0.234 ms | 274.8 |
| 1024 | 0.632 ms | 407.6 |
| 2048 | 2.064 ms | **499.5** |
| 4096 | 7.956 ms | 518.2 |

At the user's shape `B=4, H=48, S=2048, D=128`, cuDNN.pygraph backward hits
**499.5 TFLOPS** — within +7% of the FA3 paper's ~465 (cuDNN bwd, non-causal).
The previously reported "318.7 TFLOPS" came from
`aten._scaled_dot_product_flash_attention_backward` (FA2-class), not cuDNN.

Best FA solution to date: `/home/ubuntu/AccRL-exps/eval_runs/2026-0427-1550/success/exp_000/record.json`
124.7 TFLOPS (8.264 ms).

PYTHONPATH=/home/ubuntu/AccRL python /home/ubuntu/AccRL/fib_runtime/multiturn/mha-bwd-problems/scripts/profile_fwd_bwd.py

python scripts/prepare_mha_bwd_h48_d128.py --check --input-distribution randn --dataset-root /home/ubuntu/accrl-training --overwrite
