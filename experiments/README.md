# Reproducible experiments

This directory is the public experiment index. Runtime implementations live in
the two packages, while each experiment directory contains only its ordered
launchers, provenance, artifact contract, and reproduction notes.

| Experiment | Scope |
| --- | --- |
| [`fixit-v6`](fixit-v6/README.md) | reasoning synthesis, filtered repair, parquet build, SFT, two serving lanes, and base/patched five-definition evaluation |
| [`sft-v4`](sft-v4/README.md) | correct-kernel collection enrichment, GLM-5.2 reasoning synthesis, the exact three-message parquet contract, SFT, serving, and evaluation |

Large trajectories, kernels, reasoning JSONL, parquet, and checkpoints are data
artifacts rather than source files. Their expected locations and hashes are
recorded with each experiment.

For a narrow public source bundle, run
`python scripts/build_public_release.py`. The archive includes FIBServe,
mini-ptx-agent's CLI and inspector, and the union of the two audited experiment
closures; legacy analysis outputs and unrelated AccRL scripts are excluded.
Use `python scripts/build_fixit_v6_data_bundle.py` separately to export the
large, relocatable Fixit-v6 source-data archive.
Use `python scripts/build_sft_v4_data_bundle.py` for the corresponding
12-run/521-row SFT-v4 source closure.
