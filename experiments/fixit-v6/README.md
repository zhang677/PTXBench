# Fixit

This directory preserves the historical `fixit-v6` variant of the Fixit
recipe. The supported user entrypoint is `scripts/reproduce_fixit.sh`; files
under `packages/` are shared implementation details.

## Pipeline from scratch

```text
Qwen3.6-27B on 8 MHA definitions
  -> failed source kernels
  -> Gemini repairs
  -> successful wrong/fixed kernel pairs
  -> Qwen reasoning synthesis
  -> five-message Fixit SFT parquet
  -> Qwen3.6-27B SFT
  -> serve final checkpoint
  -> expert-guided five-definition evaluation
```

The first four `source-*` stages are the missing upstream data-generation
path. They generate the `qwen36-27b-linfo-mha*` evaluation roots, select failed
kernels, ask Gemini to repair them, and collect successful pairs into the
historical filename `fixit-v5-gemini-kernel-pairs.csv`.

The source Qwen runs intentionally use the unpatched prompt: their purpose is
to mine failures for repair. Final checkpoint evaluation has only the
expert-guided/MHA-patched path used by the paper. There is no second unpatched
evaluation lane.

From the repository root:

```bash
uv sync --all-packages --extra training --group dev

export PTXBENCH_ROOT="$(pwd)"
export MINI_PTX_AGENT_ROOT="$PTXBENCH_ROOT/packages/mini-ptx-agent"
export PTXBENCH_DATA_ROOT=/path/to/ptxbench-data

scripts/reproduce_fixit.sh --check
scripts/reproduce_fixit.sh from-scratch
```

`from-scratch` executes the following stages in order:

| Stage | Result | External dependency |
| --- | --- | --- |
| `source-00` | eight `qwen36-27b-linfo-mha*` source runs | Qwen3.6-27B endpoint and FIBServe |
| `source-01` | balanced failed-kernel set and Gemini config | none |
| `source-02` | Gemini repair trajectories | Gemini API and FIBServe |
| `source-03` | successful wrong/fixed kernel-pair CSV | none |
| `00` | raw Qwen pair-reasoning JSONL | Qwen3.6-27B endpoint |
| `01` | length-filtered/repaired reasoning JSONL | Qwen3.6-27B endpoint |
| `02` | five-message Fixit parquet | none |
| `03` | trained checkpoint | Tinker API |
| `04` | served final checkpoint | SSH-accessible SGLang host |
| `05` | expert-guided five-definition evaluation roots | model endpoint and FIBServe |

The long-running watchers resume interrupted trajectories and exit only after
their output roots pass the final audit. Configure the model and evaluator
endpoints before running:

```bash
MODEL_NAME=Qwen3.6-27B \
ACCRL_MODEL_HOST=localhost:30062 \
SERVICE_URL=http://localhost:10000 \
scripts/reproduce_fixit.sh source-00

GEMINI_API_KEY=... \
SERVICE_URL=http://localhost:10000 \
scripts/reproduce_fixit.sh source-02
```

Each stage can be run separately:

```bash
scripts/reproduce_fixit.sh source-01
scripts/reproduce_fixit.sh source-03
ACCRL_MODEL_HOST=localhost:30022 scripts/reproduce_fixit.sh 00
ACCRL_MODEL_HOST=localhost:30022 scripts/reproduce_fixit.sh 01
scripts/reproduce_fixit.sh 02
TINKER_API_KEY=... scripts/reproduce_fixit.sh 03
scripts/reproduce_fixit.sh 04
scripts/reproduce_fixit.sh 05
```

The serving stage retains historical defaults for its remote host, container,
and ports. Override `REMOTE`, `CONTAINER`, `REMOTE_PORT`, `LOCAL_PORT`, and
`REMOTE_PYTHON` for the target infrastructure. The remote environment must
contain the locked `tinker-cookbook` dependency.

## Starting from the exact historical pairs

For a historical replay rather than regenerating pairs, extract the relocatable
258-pair data bundle and point `PTXBENCH_DATA_ROOT` at its `ptxbench-data`
directory:

```bash
scripts/reproduce_fixit.sh --check-data
scripts/reproduce_fixit.sh all
```

`--check-data` verifies every trajectory, wrong kernel, log, fixed kernel,
plan, and correctness export referenced by the pair CSV.

Release maintainers can build that bundle with:

```bash
python scripts/build_fixit_v6_data_bundle.py \
  --pairs-csv /path/to/fixit-v5-gemini-kernel-pairs.csv \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/fixit-v6-source-data.tar.gz
```

## Layout

| Location | Purpose |
| --- | --- |
| `experiments/fixit-v6/` | Fixit orchestration and run instructions |
| `configs/fixit-v6/` | source-mining and expert-guided evaluation plans |
| `packages/mini-ptx-agent/` | reusable collection, synthesis, training, and agent code |
| `packages/fibserve/` | independent GPU correctness/profiling service |
| `$PTXBENCH_DATA_ROOT/` | source runs and generated artifacts |

For an evaluation-only static preflight, use:

```bash
scripts/smoke_fixit_v6.sh --check
```
