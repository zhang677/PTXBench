# Fixit SFT

Fixit trains a Qwen3.6-27B LoRA adapter from failed base-model CUDA kernels
repaired by Gemini 3.1 Pro. It uses the shared multiturn runner for both
source generation and final evaluation. The checked-in process example covers
four bundled H100 MHA definitions and five-turn configs. It intentionally does
not reproduce the original paper's dataset size or results.

## Prerequisites

Follow the root [setup](../../README.md#setup), including the profiler's two
backward-attention blobs and the multiturn Docker image. A base Qwen3.6-27B
OpenAI-compatible endpoint must serve model ID `Qwen/Qwen3.6-27B`; set
`PTXBENCH_MODEL_HOST=host:port`. Export `GEMINI_API_KEY` for the repair stage,
`TINKER_API_KEY` for training, and use a Qwen endpoint again for reasoning
synthesis. A separate SGLang GPU host is used to serve the trained checkpoint.

```bash
export PTXBENCH_DATA_ROOT="$PWD/data"   # or persistent storage
export SERVICE_URL=http://localhost:10000
export PTXBENCH_MODEL_HOST=localhost:30002
bash scripts/reproduce_fixit.sh --check
```

The dispatcher accepts a stage number (`00` through `09`) or `all`. Run one
stage at a time when checking output and choosing resource limits. `all` is
provided for an environment with every external service configured; it does
not wait for the asynchronous Tinker training job to finish before stage 08.
Stage 08 does wait for the final checkpoint.

| Stage | Command | Output or action |
| --- | --- | --- |
| 00 | `bash scripts/reproduce_fixit.sh 00` | Base Qwen source trajectories from `source-runs.csv` |
| 01 | `bash scripts/reproduce_fixit.sh 01` | Failed kernels, first five turns, repair config and per-kernel prompt files under `data/fixit/` |
| 02 | `bash scripts/reproduce_fixit.sh 02` | Gemini repair trajectories under `data/eval_runs/fixit-repairs/` |
| 03 | `bash scripts/reproduce_fixit.sh 03` | Correct repair pairs with dynamic SASS verification |
| 04 | `bash scripts/reproduce_fixit.sh 04` | Qwen reasoning JSONL for repair pairs |
| 05 | `bash scripts/reproduce_fixit.sh 05` | Resynthesized rows that failed the reasoning quality or token filters |
| 06 | `bash scripts/reproduce_fixit.sh 06` | SFT parquet in `data/fixit/data/` |
| 07 | `bash scripts/reproduce_fixit.sh 07` | Tinker LoRA training in a local tmux session |
| 08 | `bash scripts/reproduce_fixit.sh 08` | Download, merge, and serve final checkpoint through SGLang and SSH tunnel |
| 09 | `bash scripts/reproduce_fixit.sh 09` | Five-workload SFT evaluation from `eval-runs.csv` |

Stage 01 keeps only the MHA prompt families `hopper-07/08` and
`hopper-012/013`, then selects source turns 0–4. It keeps all selected rows by
default; set `PER_DEFINITION_CAP` to downsample each definition. If no failed
kernel is found, run more source trajectories or use another source manifest.
Stage 03 requires the profiling service's runtime SASS path; errors in this
step should be resolved before treating the pairs as training data. Run the
collector on a host with CUDA `nvcc` and `cuobjdump` available.

Stages 04 and 05 use the base Qwen endpoint via `OPENAI_BASE_URL`, which the
wrappers set from `PTXBENCH_MODEL_HOST`. Stage 04 can take many passes; set
`MAX_PASSES` and `MAX_CONCURRENT` to match the endpoint. Stage 05 uses the
`Qwen/Qwen3.6-27B` tokenizer and can download it from Hugging Face.

For stage 08, set `REMOTE` to the SSH host and `CONTAINER` to an existing
SGLang container on that host. The container must have `tmux`, Tinker Cookbook,
a writable `/data02` directory, and its Tinker key in `/data02/TINKER_API_KEY`.
`REMOTE_PYTHON` and `PTXBENCH_REMOTE_DATA_ROOT` change those defaults. The
serving stage copies the checkpoint manifest and shared download helper to the
container, downloads and merges weights, starts SGLang, and creates the local
SSH tunnel. It serves ID `Qwen/Qwen3.6-27B`; for stage 09 set
`PTXBENCH_MODEL_HOST` to the tunnel (`localhost:30052` by default).

Source and evaluation manifests can be edited to use more workloads or
trajectory configs; `experiments/shared/run_manifest.py --check` validates
paths and prompt tags before running. Output roots resume from their
`plan.json`; choose fresh roots for a different experiment. Keep source and
SFT evaluation roots separate.
