# PTXBench

PTXBench is an open-source workspace for building and evaluating PTX/CUDA
kernel agents. It contains:

- **mini-ptx-agent**: the reusable agent, prompt, distillation-inspection, and
  benchmark implementation ported from AccRL.
- **FIBServe**: the GPU compilation, correctness, profiling, and evaluation
  service derived from FlashInfer Bench.
- **mini-swe-agent 2.4.6**: an external, exactly pinned Python dependency used
  as the agent runtime and Docker environment abstraction.
- **tinker-cookbook**: an optional, locked dependency for reproducing the
  Tinker SFT training and checkpoint-export workflow.

The repository is a modular monorepo. The agent, isolated evaluator, and
FIBServe remain separate runtime images even though users clone one repository.

## Repository layout

```text
packages/mini-ptx-agent/  Agent, prompts, Fixit scripts, inspector, benchmark
packages/fibserve/        GPU evaluation service
configs/fixit-v6/         Portable Fixit-v6 prompt configurations
experiments/              Public experiment index, launchers, and provenance
docker/                   Agent, evaluator, FIBServe, and Compose definitions
data/                     Local datasets and run artifacts (git-ignored)
```

## Get a first result

The smallest real PTXBench run asks one model to optimize one GEMM for three
turns. It needs:

- an NVIDIA Hopper/H100-class FIBServe instance loaded with the two FlashInfer
  Trace datasets;
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
1.0x target was achieved.” Reprint any run with:

```bash
uv run ptxbench quickstart --report data/eval_runs/quickstart-...-gemm
```

## Local paths

All ported scripts accept these environment variables:

```bash
export PTXBENCH_ROOT=/home/ubuntu/PTXBench
export PTXBENCH_DATA_ROOT="$PTXBENCH_ROOT/data"
export PTXBENCH_TRACESET_ROOT=/home/ubuntu/accrl-training
export PTXBENCH_HEAVY_TRACESET_ROOT=/home/ubuntu/accrl-training-heavy
export MINI_PTX_AGENT_ROOT="$PTXBENCH_ROOT/packages/mini-ptx-agent"
```

`PTXBENCH_DATA_ROOT` may temporarily point at an existing AccRL-exps tree while
the public Hugging Face datasets are being prepared.

`PTXBENCH_TRACESET_ROOT` and `PTXBENCH_HEAVY_TRACESET_ROOT` are different:
they must point to FlashInfer Trace datasets containing `definitions/` and
`workloads/`. Compose mounts them read-only at `/workspace/accrl-training` and
`/workspace/accrl-training-heavy` inside FIBServe. The datasets remain outside
the Git checkout and are never copied into a Docker image.

## Development setup

```bash
uv sync --all-packages --group dev
uv run ptxbench doctor
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
test -d "$PTXBENCH_TRACESET_ROOT/definitions"
test -d "$PTXBENCH_TRACESET_ROOT/workloads"
test -d "$PTXBENCH_HEAVY_TRACESET_ROOT/definitions"
test -d "$PTXBENCH_HEAVY_TRACESET_ROOT/workloads"
docker compose --env-file docker/.env -f docker/compose.yaml up --build fibserve
```

The public experiment index starts at [`experiments/README.md`](experiments/README.md).
The source/data boundary and release procedure are documented in
[`RELEASING.md`](RELEASING.md).
For Fixit-v6, use
[`experiments/fixit-v6/README.md`](experiments/fixit-v6/README.md) as the
single start page. Its numbered launchers are the supported user interface;
the files under `packages/` are shared implementation details.

Use `scripts/smoke_fixit_v6.sh --check` for a non-mutating dependency and
configuration preflight.

Use `scripts/reproduce_fixit_v6.sh --check` to validate the complete eight-stage
source closure and `--check-data` to validate the external 258-pair input
bundle before a full run. The SFT-v4 dataset/training lineage is preserved
separately under `experiments/sft-v4` and has the corresponding
`scripts/reproduce_sft_v4.sh` entrypoint. `ptxbench-inspect` remains part of the
supported CLI.

For a live run, select a dedicated OpenAI-compatible model endpoint. PTXBench
does not default to a shared model serve:

```bash
MODEL_NAME=Qwen3.6-27B \
ACCRL_MODEL_HOST=localhost:30062 \
SERVICE_URL=http://localhost:11000 \
scripts/smoke_fixit_v6.sh --run
```

To run the same orchestration from the agent container, set
`PTXBENCH_HOST_ROOT` to the checkout's absolute host path. The path is mounted
at the same location in the agent so sibling eval containers launched through
the Docker socket can mount trajectory workspaces correctly:

```bash
export PTXBENCH_HOST_ROOT="$(pwd)"
docker compose -f docker/compose.yaml run --rm \
  --entrypoint bash agent scripts/smoke_fixit_v6.sh --check
```

## Licensing and provenance

PTXBench and mini-ptx-agent are Apache-2.0. FIBServe is derived from
FlashInfer Bench and preserves its Apache-2.0 license and NOTICE. mini-swe-agent
is consumed under its MIT license and is not vendored into this repository.
tinker-cookbook is consumed under Apache-2.0 and is also not vendored.
