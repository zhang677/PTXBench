# Public release boundary

PTXBench has two independent runtime products:

- `mini-ptx-agent` runs the Fixit-v6 and SFT-v4 experiment pipelines.
- `FIBServe` evaluates CUDA kernels against locally mounted FlashInfer Trace
  datasets.

FIBServe does not read experiment reasoning, SFT artifacts, cloud-drive
mounts, or either experiment data bundle.

## Source release

Build the audited source distribution with:

```bash
python scripts/build_public_release.py \
  --output dist/ptxbench-source.tar.gz
```

The archive is the union of the source closures declared by
`experiments/fixit-v6/provenance.json` and
`experiments/sft-v4/provenance.json`, plus FIBServe, package metadata,
containers, tests, and public entrypoints. It intentionally excludes unrelated
AccRL analyses, historical output plots, debug experiments, and obsolete SFT
pipelines. `accrl/distill/inspector.py` is explicitly retained.

Every archive contains `RELEASE-MANIFEST.sha256`. Verify it after extraction:

```bash
sha256sum -c RELEASE-MANIFEST.sha256
scripts/reproduce_fixit_v6.sh --check
scripts/reproduce_sft_v4.sh --check
```

## Input-data releases

Large source inputs are published separately from Git source:

```bash
python scripts/build_fixit_v6_data_bundle.py \
  --pairs-csv /path/to/fixit-v5-gemini-kernel-pairs.csv \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/fixit-v6-source-data.tar.gz

python scripts/build_sft_v4_data_bundle.py \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/sft-v4-source-data.tar.gz
```

These deterministic archives contain source trajectories, selected kernels,
plans, logs, and manifests. They do not contain generated reasoning, training
parquets, checkpoints, or evaluation outputs.

After extraction, point `PTXBENCH_DATA_ROOT` at the archive's
`ptxbench-data` directory and run the corresponding `--check-data` command.

## Reproduction contract

A public rerun executes the numbered stages from the source inputs. Generated
reasoning and downstream artifacts are expected to satisfy the recorded schema,
row-count, filtering, and stage-order contracts. They are not required to be
byte-identical to the historical artifacts: model sampling and hosted model
revisions can change generated text.

Historical hashes in each experiment's `provenance.json` identify the original
reported run. They are reference evidence, not downloads or runtime gates.

The external services needed for a live rerun are:

- the configured reasoning-model endpoint and credentials;
- Tinker credentials for SFT;
- an SGLang serving host;
- FIBServe with the required local trace datasets and GPUs.

None of these services requires Google Drive.
