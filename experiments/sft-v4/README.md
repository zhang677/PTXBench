# SFT-v4

This directory preserves the complete script lineage for the
`Qwen3.6-27B-sft-v4` result without importing the unrelated AccRL-exps task
archive.

The authoritative artifact trail is:

1. The 12 source runs in `source-runs.csv` are collected at `speedup > 0` into
   `correct-kernels.csv` (521 selected correct turns), then enriched with
   prompt, trajectory, turn, and evaluation feedback.
2. GLM-5.2 synthesizes one reasoning record per row using
   `v1_correct_kernel_adopted`.
3. The builder emits exactly three messages: the original Gemini system
   message, the first user prompt, and
   `<think>{reasoning}</think> + selected Gemini answer`.
4. Qwen3.6-27B is trained for 5 epochs at `4.65e-4`, batch size 2, maximum
   length 65,536, and LoRA rank 32.
5. The final checkpoint is served and evaluated on the same five public
   Fixit-v6 definitions.

The historical manifest proves 521 rows before filtering, 27 rows filtered
only for exceeding 65,536 tokens, and 494 final rows. The final parquet SHA-256
is `5416899bf9f8312e1e5361dc12f20613246ef597f073852224c81eb314c10ff8`.
See [`provenance.json`](provenance.json) for the retained script closure and
artifact hashes.

Prepare the data project:

```bash
export PTXBENCH_DATA_ROOT=/path/to/ptxbench-data
mkdir -p "$PTXBENCH_DATA_ROOT/sft_experiments/mha-8def-single-turn-qwen36-27b-gemini-glm"
```

Then run the numbered scripts in order. Stage `01` generates a new
`reasoning_pairs.jsonl`; it does not require the historical archived copy.
Because model sampling and hosted model revisions can vary, a fresh reasoning
JSONL and parquet need not have the historical hashes recorded in
`provenance.json`.

Use `scripts/reproduce_sft_v4.sh --check` for source closure,
`--check-data` for the 12-run input closure, a number from `00` through `05`
for one stage, or `all` for the full ordered workflow.

Release maintainers can package the exact 12-run, 521-row source closure into a
relocatable archive:

```bash
python scripts/build_sft_v4_data_bundle.py \
  --data-root /path/to/AccRL-exps \
  --mini-agent-root /path/to/AccRL \
  --output dist/sft-v4-source-data.tar.gz
```

The bundle contains only the selected kernels and trajectories plus the
per-run plans and turn-correctness CSVs. It also includes portable regenerated
`correct-kernels.csv` and `correct-kernels.enriched.csv`. Generated reasoning,
parquet, checkpoints, and evaluation runs are outputs of the public workflow,
not required inputs. Historical hashes in `provenance.json` are references for
the original run rather than gates on a new stochastic rerun.

The serve stage uses `REMOTE_PYTHON` inside the remote SGLang container
(default `/data02/tinker-cookbook/.venv/bin/python`) to download and merge the
Tinker checkpoint. That environment must have the locked `tinker-cookbook`
dependency; no separate local checkout is needed.

`packages/mini-ptx-agent/accrl/distill/inspector.py` is an explicit retained
part of this release and remains available as `ptxbench-inspect`.
