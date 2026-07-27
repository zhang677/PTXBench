# PTXBench

PTXBench is an open-source workspace for building and evaluating PTX/CUDA
kernel agents. It contains:

- **mini-ptx-agent**: the agent, prompt, Fixit, distillation-inspection, and
  benchmark workflows ported from AccRL.
- **FIBServe**: the GPU compilation, correctness, profiling, and evaluation
  service derived from FlashInfer Bench.
- **mini-swe-agent**: an external, pinned Python dependency used as the agent
  runtime and Docker environment abstraction.
- **tinker-cookbook**: an optional, locked dependency for reproducing the
  Tinker SFT training and checkpoint-export workflow.

The repository is a modular monorepo. The agent, isolated evaluator, and
FIBServe remain separate runtime images even though users clone one repository.

## Repository layout

```text
packages/mini-ptx-agent/  Agent, prompts, Fixit scripts, inspector, benchmark
packages/fibserve/        GPU evaluation service
configs/fixit-v6/         Portable Fixit-v6 prompt configurations
docker/                   Agent, evaluator, FIBServe, and Compose definitions
data/                     Local datasets and run artifacts (git-ignored)
```

## Local paths

All ported scripts accept these environment variables:

```bash
export PTXBENCH_ROOT=/home/ubuntu/PTXBench
export PTXBENCH_DATA_ROOT="$PTXBENCH_ROOT/data"
export MINI_PTX_AGENT_ROOT="$PTXBENCH_ROOT/packages/mini-ptx-agent"
```

`PTXBENCH_DATA_ROOT` may temporarily point at an existing AccRL-exps tree while
the public Hugging Face datasets are being prepared.

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
docker compose -f docker/compose.yaml up --build fibserve
```

The Fixit-v6 port lives at:

```text
packages/mini-ptx-agent/fib_runtime/multiturn/construct_eval_scripts/fixit-v6-scripts
```

Use `scripts/smoke_fixit_v6.sh --check` for a non-mutating dependency and
configuration preflight.

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
