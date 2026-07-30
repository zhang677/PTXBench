# Reproducible experiments

This directory is the public experiment index. Runtime implementations live in
the two packages, while each experiment directory contains only its ordered
launchers and runnable instructions.

| Experiment | Scope |
| --- | --- |
| [Fixit](fixit-v6/README.md) | base-Qwen failure mining, Gemini repair, pair-reasoning synthesis, SFT, and expert-guided evaluation; `fixit-v6` is the retained historical variant name |
| [KernelGen](sft-v4/README.md) | correct-kernel collection, GLM-5.2 reasoning synthesis, the three-message SFT contract, and expert-guided evaluation; `sft-v4` is the retained historical variant name |

Large trajectories, kernels, reasoning JSONL, parquet, and checkpoints are
published data artifacts rather than source files.

For a narrow public source bundle, run
`python scripts/build_public_release.py`. The archive includes FIBServe,
mini-ptx-agent's runnable pipeline sources, and both experiment directories.
Use `python scripts/build_fixit_v6_data_bundle.py` separately to export the
large, relocatable historical Fixit source-data archive.
Use `python scripts/build_sft_v4_data_bundle.py` for the corresponding
12-run/521-row KernelGen source closure.
