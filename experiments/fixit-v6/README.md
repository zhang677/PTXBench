# Fixit-v6

This directory is the single user-facing home for Fixit-v6. Run the numbered
stages here through `scripts/reproduce_fixit_v6.sh`; do not invoke the
implementation files under `packages/` directly.

## The experiment in one picture

```text
258-pair source-data bundle
            |
            v
  00 synthesize reasoning
            |
  01 repair filtered reasoning
            |
  02 build 258-row SFT parquet
            |
  03 train one Qwen3.6-27B checkpoint
            |
            +--> 04 serve checkpoint --> 05 evaluate standard prompts
            |
            +--> 06 serve checkpoint --> 07 evaluate MHA-patched prompts
```

Stages `04` and `06` serve the same trained checkpoint in independent lanes.
“Patched” in stages `06` and `07` refers to the evaluation prompt
configuration, not to a second set of model weights.

## Where everything lives

| Location | Purpose | Should a new user run it directly? |
| --- | --- | --- |
| `experiments/fixit-v6/` | These eight ordered experiment launchers and archival provenance | Yes, through `scripts/reproduce_fixit_v6.sh` |
| `configs/fixit-v6/` | Standard and MHA-patched five-definition evaluation plans | No; the watchers select them |
| `packages/mini-ptx-agent/` | Shared synthesis, training, agent, and evaluation engines | No |
| `packages/fibserve/` | Independent GPU correctness and profiling service | Start it as a service, not as a Fixit stage |
| `$PTXBENCH_DATA_ROOT/` | Downloaded source bundle and newly generated outputs | Data only |

The split is intentional: an experiment owns ordering and parameters, while
packages contain reusable implementations shared with SFT-v4.

## Quick start

From the repository root:

```bash
uv sync --all-packages --extra training --group dev

export PTXBENCH_ROOT="$(pwd)"
export MINI_PTX_AGENT_ROOT="$PTXBENCH_ROOT/packages/mini-ptx-agent"
export PTXBENCH_DATA_ROOT=/path/to/extracted/ptxbench-data

scripts/reproduce_fixit_v6.sh --check
scripts/reproduce_fixit_v6.sh --check-data
```

`--check` validates only source and configuration. `--check-data` additionally
requires the 258-pair source-data bundle under:

```text
$PTXBENCH_DATA_ROOT/sft_experiments/test-fixit-qwen36-27b-gemini-glm/
```

Run one stage at a time:

```bash
ACCRL_MODEL_HOST=localhost:30022 scripts/reproduce_fixit_v6.sh 00
ACCRL_MODEL_HOST=localhost:30022 scripts/reproduce_fixit_v6.sh 01
scripts/reproduce_fixit_v6.sh 02
TINKER_API_KEY=... scripts/reproduce_fixit_v6.sh 03
scripts/reproduce_fixit_v6.sh 04
scripts/reproduce_fixit_v6.sh 05
scripts/reproduce_fixit_v6.sh 06
scripts/reproduce_fixit_v6.sh 07
```

Use `scripts/reproduce_fixit_v6.sh all` only after configuring every external
service used by stages `00` through `07`.

## Stage contract

| Stage | Reads | Produces | External service |
| --- | --- | --- | --- |
| `00` | 258 kernel-repair pairs | raw Qwen reasoning JSONL | Dedicated OpenAI-compatible Qwen3.6-27B endpoint |
| `01` | complete raw reasoning | repaired reasoning JSONL | Same Qwen endpoint |
| `02` | repaired reasoning and source pairs | 258-row, five-message parquet | None |
| `03` | parquet | Tinker training run and checkpoints | Tinker API |
| `04` | final checkpoint | base-lane SGLang endpoint | SSH-accessible SGLang host/container |
| `05` | base model endpoint and five test definitions | standard-prompt evaluation roots | FIBServe and the base-lane model endpoint |
| `06` | the same final checkpoint | independent patched-lane SGLang endpoint | SSH-accessible SGLang host/container |
| `07` | patched-lane endpoint and five test definitions | MHA-patched evaluation roots | FIBServe and the patched-lane model endpoint |

Stage `01` refuses incomplete `00` coverage. Stage `04` waits for a final
training checkpoint. Each watcher resumes interrupted trajectories and exits
only when all five output roots pass the final audit.

The serving scripts have historical remote host, container, and port defaults.
Override `REMOTE`, `CONTAINER`, `REMOTE_PORT`, `LOCAL_PORT`, and
`REMOTE_PYTHON` for your infrastructure. The remote Python environment must
contain the locked `tinker-cookbook` dependency.

## FIBServe is separate

FIBServe does not synthesize data, train the model, or access Google Drive. It
is used only by evaluation stages `05` and `07` through `SERVICE_URL`.
Build and start it using the root README before launching either watcher.

For a small evaluation-only preflight, run:

```bash
scripts/smoke_fixit_v6.sh --check
```

This is not a substitute for the full `--check-data` or eight-stage workflow.

## Reproduction versus historical provenance

[`provenance.json`](provenance.json) records the observed Fixit-v6 run: 258
synthesized records, one repaired record, zero final invalid records, and a
258-row five-message parquet with its historical hashes.

A new rerun uses the same source-data closure, code, prompts, ordering, and
training parameters. Model sampling and hosted-model revisions can still
produce different reasoning, parquet, checkpoints, and evaluation results.
Therefore the hashes in `provenance.json` identify the original run; they are
not equality requirements for a fresh stochastic rerun.

The source-data bundle is not committed. It must contain the 258-row
`fixit-v5-gemini-kernel-pairs.csv` plus every referenced trajectory, wrong
kernel and log, correct kernel, plan, and turn-correctness export. Publishing
that archive remains an external release prerequisite; without it, a new user
can validate the source but cannot run stage `00`.

Generated reasoning, parquet, checkpoints, and evaluation roots are outputs of
the workflow and are not required inputs.

## Maintainer: build the relocatable input bundle

The historical CSV contains machine-specific paths. A release maintainer can
package and rewrite the exact source closure with:

```bash
python scripts/build_fixit_v6_data_bundle.py \
  --pairs-csv /path/to/fixit-v5-gemini-kernel-pairs.csv \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/fixit-v6-source-data.tar.gz
```

Extract the archive and point `PTXBENCH_DATA_ROOT` at its `ptxbench-data`
directory. `scripts/reproduce_fixit_v6.sh --check-data` expands the portable
path variables and validates the full 258-pair closure before any expensive
work starts.

## Internal implementation map

For maintainers tracing behavior:

```text
00..04  -> construct_eval_scripts/fixit_downstream_process.py
05, 07  -> construct_eval_scripts/watch_eval_common.sh
        -> fib_runtime/multiturn/run_parallel_v2.py
        -> fib_runtime/multiturn/run_v2.py
        -> fib_runtime/multiturn/common.py
```

These internal locations may change without changing the supported numbered
experiment interface.
