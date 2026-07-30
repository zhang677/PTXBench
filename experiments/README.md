# Reproducible experiments

This directory is the public experiment index. Runtime implementations live in
the two packages, while each experiment directory contains only its ordered
launchers and runnable instructions.

| Experiment | Scope |
| --- | --- |
| [Fixit](fixit/README.md) | base-Qwen failure mining, Gemini repair, pair-reasoning synthesis, SFT, and expert-guided evaluation |
| [KernelGen](kernelgen/README.md) | correct-kernel collection, GLM-5.2 reasoning synthesis, the three-message SFT contract, and expert-guided evaluation |

Large trajectories, kernels, reasoning JSONL, parquet, and checkpoints are
published data artifacts rather than source files.

For a narrow public source bundle, run
`python scripts/build_public_release.py`. The archive includes FIBServe,
mini-ptx-agent's runnable pipeline sources, and both experiment directories.
Use `python scripts/build_fixit_data_bundle.py` separately to export the
large, relocatable historical Fixit source-data archive.
Use `python scripts/build_kernelgen_data_bundle.py` for the corresponding
12-run/521-row KernelGen source closure.
