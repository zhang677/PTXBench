# KernelGen

This directory preserves the historical `sft-v4` variant of KernelGen. The
supported user entrypoint is `scripts/reproduce_kernelgen.sh`.

KernelGen trains on successful kernels rather than repair pairs:

```text
12 source evaluation runs
  -> 521 correct kernel turns
  -> GLM-5.2 reasoning synthesis
  -> length-filtered three-message SFT parquet
  -> Qwen3.6-27B SFT
  -> serve final checkpoint
  -> expert-guided five-definition evaluation
```

The 12-run input closure is listed in `source-runs.csv`. From an extracted
KernelGen source-data bundle:

```bash
uv sync --all-packages --extra training --group dev

export PTXBENCH_ROOT="$(pwd)"
export MINI_PTX_AGENT_ROOT="$PTXBENCH_ROOT/packages/mini-ptx-agent"
export PTXBENCH_DATA_ROOT=/path/to/ptxbench-data

scripts/reproduce_kernelgen.sh --check
scripts/reproduce_kernelgen.sh --check-data
scripts/reproduce_kernelgen.sh all
```

Run stages `00` through `05` individually when configuring services:

| Stage | Result | External dependency |
| --- | --- | --- |
| `00` | enriched correct-kernel CSV | none |
| `01` | GLM-5.2 reasoning JSONL | OpenRouter |
| `02` | three-message KernelGen parquet | none |
| `03` | trained checkpoint | Tinker API |
| `04` | served final checkpoint | SSH-accessible SGLang host |
| `05` | expert-guided five-definition evaluation roots | model endpoint and FIBServe |

For example:

```bash
scripts/reproduce_kernelgen.sh 00
OPENROUTER_API_KEY=... scripts/reproduce_kernelgen.sh 01
scripts/reproduce_kernelgen.sh 02
TINKER_API_KEY=... scripts/reproduce_kernelgen.sh 03
scripts/reproduce_kernelgen.sh 04
scripts/reproduce_kernelgen.sh 05
```

Release maintainers can package the deterministic 12-run input closure with:

```bash
python scripts/build_sft_v4_data_bundle.py \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/sft-v4-source-data.tar.gz
```

The final parquet uses exactly three messages: the original Gemini system
message, the first user prompt, and `<think>{reasoning}</think>` followed by
the selected Gemini answer. Fresh hosted-model sampling may change the
reasoning text and the number of rows removed by maximum-token filtering.

The serve stage uses `REMOTE_PYTHON` inside the remote SGLang container
(default `/data02/tinker-cookbook/.venv/bin/python`) to download and merge the
Tinker checkpoint. That environment must contain the locked
`tinker-cookbook` dependency.
