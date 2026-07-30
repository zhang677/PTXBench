# PTXBench data

This directory is intentionally not populated by the source repository.

- `datasets/` holds mounted or downloaded FIBServe trace datasets.
- `eval_runs/` holds mini-ptx-agent trajectories, plans, logs, and successful kernels.
- `sft_experiments/` holds optional Fixit synthesis/training artifacts.

The two public workflows expect:

```text
sft_experiments/
  test-fixit-qwen36-27b-gemini-glm/             Fixit input/output project
  mha-8def-single-turn-qwen36-27b-gemini-glm/   KernelGen input/output project
eval_runs/                                      referenced trajectories/kernels
```

Authorized users can retrieve the published s0-s6 training parquets from the private
[`Genghan/PTXBench-Qwen3.6-27B-SFT`](https://huggingface.co/datasets/Genghan/PTXBench-Qwen3.6-27B-SFT)
repository. Otherwise, set `PTXBENCH_DATA_ROOT` to an existing compatible data
tree. Source preflight and data-closure validation are intentionally separate.

Release maintainers can export the historical Fixit source inputs without
preserving machine-specific paths:

```bash
python scripts/build_fixit_v6_data_bundle.py \
  --pairs-csv /path/to/fixit-v5-gemini-kernel-pairs.csv \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/fixit-v6-source-data.tar.gz
```

The KernelGen source runs have a separate deterministic exporter:

```bash
python scripts/build_sft_v4_data_bundle.py \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/sft-v4-source-data.tar.gz
```
