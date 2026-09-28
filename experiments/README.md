# Experiments

All experiment paths use the same multiturn runner and profiling service.
`prepared/configs/example.json` is a small model-loop example. The five configs
in `configs/` provide H100 MHA and GEMM source and SFT-evaluation prompt arms.

Fixit and KernelGen read their `source-runs.csv` and `eval-runs.csv` manifests
through `shared/run_manifest.py`. Each row names a model, GPU architecture,
workload definition, output root, evaluator test, and trajectory config. Paths
may use `PTXBENCH_ROOT` and `PTXBENCH_DATA_ROOT`. A fresh root records a plan;
a later invocation resumes that root. Never point a manifest at an unrelated
existing directory.

- [Fixit stages](fixit/README.md)
- [KernelGen stages](kernelgen/README.md)
- [Shared profiling service](../fib-profile/scripts/README.md)

The example manifests exercise the bundled profiler subset. They are process
examples, not a claim to reproduce the full paper populations or results.
