# Shared evaluation implementation

This directory contains reusable implementation used by the public experiment
launchers. It is not an experiment entrypoint.

New users should start with:

- `experiments/fixit/README.md` for Fixit.
- `experiments/kernelgen/README.md` for KernelGen.

The retained files have one shared responsibility each:

| File | Responsibility |
| --- | --- |
| `fixit_downstream_process.py` | Synthesis, parquet, training, checkpoint, and remote-serving orchestration |
| `ptxbench_paths.sh` | Portable repository, package, configuration, and data roots |
| `watch_eval_common.sh` | Resumable watcher and FIBServe lifecycle functions |
| `watch_eval_audit.py` | Read-only trajectory, output-root, and restart-state audits |

Experiment-specific ordering, model identities, ports, and configurations
belong under `experiments/`, not here.
