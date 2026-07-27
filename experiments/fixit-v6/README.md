# Fixit-v6

The authoritative launchers are:

```text
packages/mini-ptx-agent/fib_runtime/multiturn/construct_eval_scripts/fixit-v6-scripts/
  00_synthesize_qwen36-27b_reasoning.sh
  01_resynthesize_filtered_reasonings.sh
  02_build_full_parquet.sh
  03_train_sft_full.sh
  04_serve_remote_full.sh
  05_watch_v6_full_5defs_eval.sh
  06_serve_patched_remote_full.sh
  07_watch_v6_full_5defs_eval.sh
```

The observed completed dataset lineage and hashes are recorded in
[`provenance.json`](provenance.json): 258 synthesized records, one repaired
record, zero final invalid records, and a 258-row five-message parquet.

Run a static source/configuration audit:

```bash
scripts/reproduce_fixit_v6.sh --check
```

After placing the input bundle under
`$PTXBENCH_DATA_ROOT/sft_experiments/test-fixit-qwen36-27b-gemini-glm`,
validate it with:

```bash
scripts/reproduce_fixit_v6.sh --check-data
```

Run one numbered stage, or the ordered pipeline:

```bash
ACCRL_MODEL_HOST=localhost:30022 scripts/reproduce_fixit_v6.sh 00
scripts/reproduce_fixit_v6.sh 03
scripts/reproduce_fixit_v6.sh all
```

`all` preserves the original strict ordering. Stage `01` refuses incomplete
synthesis coverage, stage `04` waits for a final training checkpoint, and each
watcher exits only after its five output roots pass the final audit.

The large input bundle is not committed. It must include the 258-row
`fixit-v5-gemini-kernel-pairs.csv` and every trajectory, wrong kernel/log, and
correct kernel referenced by that CSV. Publishing this bundle is still an
external release prerequisite.

The historical CSV contains machine-specific absolute paths. A release
maintainer can turn the original tree into a relocatable archive with:

```bash
python scripts/build_fixit_v6_data_bundle.py \
  --pairs-csv /path/to/fixit-v5-gemini-kernel-pairs.csv \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/fixit-v6-source-data.tar.gz
```

Extract the archive and set `PTXBENCH_DATA_ROOT` to its `ptxbench-data`
directory. The rewritten CSV uses `${PTXBENCH_DATA_ROOT}` and
`${MINI_PTX_AGENT_ROOT}` references, and `--check-data` expands and validates
them. The archive includes the synthesis trajectories and success records in
addition to every path explicitly named by the CSV. Generated reasoning,
parquet, checkpoints, and evaluation runs are outputs of the public workflow,
not required inputs; their historical hashes are reference provenance only.

Serving requires a Python environment inside the remote SGLang container with
the locked `tinker-cookbook` dependency. Its default path is
`/data02/tinker-cookbook/.venv/bin/python`; set `REMOTE_PYTHON` when the remote
image installs it elsewhere. No separate local tinker-cookbook checkout is
required.
