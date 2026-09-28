# KernelGen SFT

KernelGen trains a Qwen3.6-27B LoRA adapter from correct Gemini-generated CUDA
kernels plus synthesized reasoning. It shares the profiler, prompt registry,
multiturn runner, trainer, and serving helper with Fixit. The checked-in source
manifest covers four bundled H100 d128 MHA definitions; final evaluation adds
GEMM. This is a runnable process example, not the full original experiment.

## Prerequisites

Complete the root [setup](../../README.md#setup). Export `GEMINI_API_KEY` for
source generation, `OPENROUTER_API_KEY` for GLM-5.2 reasoning synthesis, and
`TINKER_API_KEY` for training. Runtime SASS collection requires the profiling
service. Training uses Tinker; serving requires a remote SGLang GPU container.

```bash
export PTXBENCH_DATA_ROOT="$PWD/data"   # or persistent storage
export SERVICE_URL=http://localhost:10000
bash scripts/reproduce_kernelgen.sh --check
```

| Stage | Command | Output or action |
| --- | --- | --- |
| 00 | `bash scripts/reproduce_kernelgen.sh 00` | Gemini source trajectories from `source-runs.csv` |
| 01 | `bash scripts/reproduce_kernelgen.sh 01` | Turn correctness, kernel files, dynamically tagged correct kernels, enriched source CSV |
| 02 | `bash scripts/reproduce_kernelgen.sh 02` | GLM-5.2 reasoning JSONL using the shared prompt hub and source kernel |
| 03 | `bash scripts/reproduce_kernelgen.sh 03` | Filtered Qwen chat-template SFT parquet |
| 04 | `bash scripts/reproduce_kernelgen.sh 04` | Tinker LoRA training in a local tmux session |
| 05 | `bash scripts/reproduce_kernelgen.sh 05` | Download, merge, and serve final checkpoint via SGLang |
| 06 | `bash scripts/reproduce_kernelgen.sh 06` | Five-workload SFT evaluation from `eval-runs.csv` |

Stage 01 exports `turn_correctness_arch.csv` with runtime profiling and extracts
kernel files. Run it on a host with CUDA `nvcc` and `cuobjdump`. The collector
requires correctness, speedup above zero, and the
appropriate dynamic SASS architecture tag. If no kernel qualifies, generate
more source runs or change the example config. Stage 02 reads
`synthesize.yaml`; its paths expand `KERNELGEN_PROJECT` from
`experiments/kernelgen/paths.sh`. Stage 03 tokenizes with
`Qwen/Qwen3.6-27B` and writes a parquet under `data/kernelgen/data/`.

Stages 04 and 05 use the shared Tinker and SGLang process. For stage 05, set
`REMOTE` to an SSH host and `CONTAINER` to an existing SGLang container. That
container needs Tinker Cookbook, `tmux`, a writable `/data02`, and its Tinker
key at `/data02/TINKER_API_KEY`; override `REMOTE_PYTHON` and
`PTXBENCH_REMOTE_DATA_ROOT` for other layouts. Stage 05 waits for training's
final checkpoint, serves ID `Qwen/Qwen3.6-27B`, and creates a local tunnel.
Set `PTXBENCH_MODEL_HOST=localhost:30012` for stage 06, or use your chosen
`LOCAL_PORT`.

Each manifest row selects a definition, workload, CUDA test script, prompt
config, and output root. Change manifests and configs for a broader experiment.
The runner resumes a root with `plan.json`, so use new roots when changing the
setup. The dispatcher also accepts `all`, but the staged commands make it
easier to inspect generated artifacts between external calls.
