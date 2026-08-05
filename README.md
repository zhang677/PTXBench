# PTXBench

PTXBench is an open-source workspace for building and evaluating PTX/CUDA
kernel agents. It contains:

- **mini-ptx-agent**: the reusable agent, prompt, trajectory inspection, and
  benchmark implementation.
- **FIBServe**: the GPU profiling service derived from FlashInfer Bench.
- **mini-swe-agent 2.4.6**: an external, exactly pinned Python dependency used
  as the agent runtime and Docker environment abstraction.
- **tinker-cookbook**: an optional, locked dependency for reproducing the
  Tinker SFT training and checkpoint-export workflow.

## Get a first result

The smallest real PTXBench run asks one model to optimize one GEMM for three
turns. It needs:

- an NVIDIA Hopper/H100-class FIBServe instance loaded with the
  [`AccRL/accrl-training`](https://huggingface.co/datasets/AccRL/accrl-training)
  FlashInfer Trace dataset;
- the `ptxbench-eval:dev` Docker image; and
- either an OpenAI-compatible Qwen endpoint or credentials for one of the
  hosted models supported by `mini-ptx-agent`.

Set up the Python environment and evaluator image:

```bash
uv sync --all-packages --group dev
docker build -f docker/Dockerfile.eval -t ptxbench-eval:dev .
```

Start FIBServe as described below, then select a model. For a Qwen endpoint,
the served ID must exactly match `MODEL_NAME`:

```bash
export MODEL_NAME=Qwen3.6-27B
export ACCRL_MODEL_HOST=localhost:30062
export SERVICE_URL=http://localhost:10000

uv run ptxbench quickstart --check
uv run ptxbench quickstart --run
```

For example, a hosted OpenAI model can be used without
`ACCRL_MODEL_HOST`:

```bash
export MODEL_NAME=GPT-5.4
export OPENAI_API_KEY=...
uv run ptxbench quickstart --run
```

Every run leaves the full `trajectories/exp_000.json`, evaluator logs, and a
concise `quickstart-result.json` under
`data/eval_runs/quickstart-...-gemm/`. When the model emits a CUDA candidate,
it is saved as `exp_000/kernel.cu`; a correctness-passing kernel is
additionally saved as `success/exp_000/kernel_vN.cu`. The report deliberately
distinguishes “the runner completed” from “the kernel was correct” and “the
1.0x target was achieved.” mini-ptx-agent writes trajectories in a JSON format
compatible with mini-swe-agent and its trajectory tooling. Reprint any run
with:

```bash
uv run ptxbench quickstart --report data/eval_runs/quickstart-...-gemm
```

## Local paths

For local development, configure the project/data paths and the Compose
trace-set collection:

```bash
export PTXBENCH_ROOT=/home/ubuntu/PTXBench
export PTXBENCH_DATA_ROOT="$PTXBENCH_ROOT/data"
export PTXBENCH_TRACESETS_ROOT="$PTXBENCH_ROOT/data/datasets"
export DATASET_ROOTS=/workspace/trace-sets/accrl-training
export MINI_PTX_AGENT_ROOT="$PTXBENCH_ROOT/packages/mini-ptx-agent"
```

`PTXBENCH_DATA_ROOT` may point at an existing directory to hold experiment artifacts.

`PTXBENCH_TRACESETS_ROOT` is a shared parent directory on the host. Each child
is a complete FlashInfer Trace dataset with its own `definitions/` and
`workloads/` directories. For example:

```text
$PTXBENCH_TRACESETS_ROOT/
├── accrl-training/
│   ├── definitions/
│   └── workloads/
└── another-trace-set/
    ├── definitions/
    └── workloads/
```

Compose bind-mounts that parent directory read-only at `/workspace/trace-sets`:

```text
host:      /home/ubuntu/PTXBench/data/datasets
container: /workspace/trace-sets
```

`DATASET_ROOTS` is always a colon-separated list selecting dataset directories
inside the mount; a single dataset is simply a one-item list. Compose uses the
shared parent because it cannot expand one environment variable into a variable
number of bind mounts.

Download the quickstart trace set from
[`AccRL/accrl-training`](https://huggingface.co/datasets/AccRL/accrl-training)
into `$PTXBENCH_TRACESETS_ROOT/accrl-training`. To load another trace set,
place it alongside that directory and extend the list:

```bash
export DATASET_ROOTS=/workspace/trace-sets/accrl-training:/workspace/trace-sets/another-trace-set
```

No files are merged on disk, and no dataset is copied into a Docker image.

## Development setup

```bash
uv sync --all-packages --group dev
```

Install the optional Tinker training stack when reproducing an AccRL training
run:

```bash
uv sync --all-packages --extra training --group dev
uv run python -c "import tinker, tinker_cookbook"
```

The training commands require a valid `TINKER_API_KEY`; keep it in the
environment or a local `.env` file, which is ignored by Git.

Build the isolated kernel evaluator:

```bash
docker build -f docker/Dockerfile.eval -t ptxbench-eval:dev .
```

Build and start FIBServe:

```bash
cp .env.example docker/.env
docker compose --env-file docker/.env -f docker/compose.yaml up --build fibserve
```

The public experiment index starts at [`experiments/README.md`](experiments/README.md).
The source/data boundary and release procedure are documented in
[`RELEASING.md`](RELEASING.md).
For Fixit, use
[`experiments/fixit/README.md`](experiments/fixit/README.md) as the
single start page and `scripts/reproduce_fixit.sh` as the runnable entrypoint.

Use `scripts/smoke_fixit.sh --check` for a non-mutating dependency and
configuration preflight.

Use `scripts/reproduce_fixit.sh --check` to validate the complete source
closure and `from-scratch` to run failure mining, Gemini repair, SFT, and the
paper's expert-guided evaluation in order. KernelGen starts at
[`experiments/kernelgen/README.md`](experiments/kernelgen/README.md) and uses
`scripts/reproduce_kernelgen.sh`. `ptxbench-inspect` remains part of the
supported CLI.

For a live run, select a dedicated OpenAI-compatible model endpoint. PTXBench
does not default to a shared model serve:

```bash
MODEL_NAME=Qwen3.6-27B \
ACCRL_MODEL_HOST=localhost:30062 \
SERVICE_URL=http://localhost:11000 \
scripts/smoke_fixit.sh --run
```

To run the same orchestration from the agent container, set
`PTXBENCH_HOST_ROOT` to the checkout's absolute host path. The path is mounted
at the same location in the agent so sibling eval containers launched through
the Docker socket can mount trajectory workspaces correctly:

```bash
export PTXBENCH_HOST_ROOT="$(pwd)"
docker compose -f docker/compose.yaml run --rm \
  --entrypoint bash agent scripts/smoke_fixit.sh --check
```

## Harbor

PTXBench includes [Harbor-compatible tasks](integrations/harbor/README.md) that
run on an unmodified Harbor checkout while using `ptxbench eval` and FIBServe
for compilation, sanitization, and H100 benchmarking. The agent creates its implementation
from scratch; before launching a run, the checked-in instruction can be
refreshed from the live FIBServe definition to keep its task encoding aligned
with AccRL. The integration guide covers image builds, prompt rendering, Harbor
launch options, and the resulting ATIF trajectory.

## Huggingface
Other example datasets include non-4096 sequence-length attention workload records at [`Genghan/accrl-training-heavy`](https://huggingface.co/datasets/Genghan/accrl-training-heavy) and a more diverse [`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace). The byte-exact historical s0-s6 training parquets are also retained in
[`Genghan/PTXBench-Qwen3.6-27B-SFT`](https://huggingface.co/datasets/Genghan/PTXBench-Qwen3.6-27B-SFT).

## Licensing and provenance

PTXBench and mini-ptx-agent are Apache-2.0. FIBServe is derived from
FlashInfer Bench and preserves its Apache-2.0 license and NOTICE. mini-swe-agent
is consumed under its MIT license and is not vendored into this repository.
tinker-cookbook is consumed under Apache-2.0 and is also not vendored.
