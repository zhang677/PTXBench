# Public release boundary

PTXBench has two independent runtime products:

- `mini-ptx-agent` runs the Fixit and KernelGen experiment pipelines.
- `FIBServe` evaluates CUDA kernels against locally mounted FlashInfer Trace
  datasets.

FIBServe does not read experiment reasoning, SFT artifacts, cloud-drive
mounts, or either experiment data bundle.

## Source release

Publish source from a reviewed, clean Git checkout. Before tagging a release,
validate both experiment entrypoints:

```bash
scripts/reproduce_fixit.sh --check
scripts/reproduce_kernelgen.sh --check
```

## Input-data releases

Large source inputs are published separately from Git source:

```bash
python scripts/build_fixit_data_bundle.py \
  --pairs-csv /path/to/fixit-v5-gemini-kernel-pairs.csv \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/fixit-source-data.tar.gz

python scripts/build_kernelgen_data_bundle.py \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/kernelgen-source-data.tar.gz
```

These deterministic archives contain source trajectories, selected kernels,
plans, logs, and manifests. They do not contain generated reasoning, training
parquets, checkpoints, or evaluation outputs.

After extraction, point `PTXBENCH_DATA_ROOT` at the archive's
`ptxbench-data` directory and run the corresponding `--check-data` command.

## Private historical SFT artifacts

The byte-exact Qwen3.6-27B s0-s6 training parquets are stored in the private
[`Genghan/PTXBench-Qwen3.6-27B-SFT`](https://huggingface.co/datasets/Genghan/PTXBench-Qwen3.6-27B-SFT)
dataset repository. The corresponding final PEFT adapters are private model
repositories named `Genghan/PTXBench-Qwen3.6-27B-s0` through
`Genghan/PTXBench-Qwen3.6-27B-s6`. Hugging Face authentication and explicit
repository access are required. All eight repositories are indexed in the
private
[`PTXBench Qwen3.6-27B SFT Series`](https://huggingface.co/collections/Genghan/ptxbench-qwen36-27b-sft-series-6a6bd1594f2f23d31e25ca1f)
collection.

Release maintainers can reproduce the private staging trees and upload them
with:

```bash
python scripts/publish_qwen36_sft_series.py stage \
  --adapter-root /path/to/qwen36-adapters \
  --stage-root /path/to/staging

python scripts/publish_qwen36_sft_series.py upload \
  --adapter-root /path/to/qwen36-adapters \
  --stage-root /path/to/staging
```

The uploader creates private repositories, reapplies private visibility before
every upload, verifies visibility afterward, and adds all eight repositories
to a private Hugging Face collection.

## Reproduction contract

A public rerun executes the numbered stages from the source inputs. Generated
reasoning, parquets, and checkpoints may differ because model sampling and
hosted-model revisions can change generated text.

The external services needed for a live rerun are:

- the configured reasoning-model endpoint and credentials;
- Tinker credentials for SFT;
- an SGLang serving host;
- FIBServe with the required local trace datasets and GPUs.

None of these services requires Google Drive.
