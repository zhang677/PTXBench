# Qwen3.5-35B-A3B Allgather CP NaN Repro

This is the smallest currently known AccRL/Miles repro for the
Qwen3.5-35B-A3B long-sequence SFT NaN.

## Local Prerequisites

The repro assumes the same single-node 8-GPU setup used by the AccRL SFT
harness:

```text
/home/chengze/work/AccRL
/home/chengze/work/AccRL/accrl/distill/sft/miles
/data/local/models/qwen35-35B-A3B
/data/local/models/qwen35-35B-A3B_torch_dist
/home/chengze/work/tmp/repro_nan_35b_cp4/first6_rollouts_exact_shuffle_seed42.parquet
```

The parquet has 12 SFT rows, about 803 KB. It was arranged so Miles'
`shuffle(seed=42)` serves the same early long samples that reproduced the
failure. It is not checked into Git.

## Failing Repro

From the AccRL repo:

```bash
cd /home/chengze/work/AccRL
./accrl/distill/sft/scripts/repro_qwen35_35b_allgather_cp_nan.sh fail
```

Equivalent key settings:

```text
MODEL_PRESET=qwen35-35b-a3b
SFT_DATA=/home/chengze/work/tmp/repro_nan_35b_cp4/first6_rollouts_exact_shuffle_seed42.parquet
TP_SIZE=1
CP_SIZE=4
EP_SIZE=8
SFT_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=2
ALLGATHER_CP=1
DEBUG_DISABLE_OPTIMIZER=1
SFT_NAN_GRAD_DIAGNOSTICS=1
SKIP_NAN_STEPS=0
LR=1e-8
NUM_EPOCH=1
SFT_ENABLE_EVAL=0
SAVE_CHECKPOINTS=0
```

Expected failure:

```text
RuntimeError: ... Unexpected result nan
(message='found NaN in local grad norm for bucket #0 in backward pass before data-parallel communication collective')
```

The important isolation is `DEBUG_DISABLE_OPTIMIZER=1`: optimizer updates are
disabled, so the non-finite gradients are produced by the model backward path,
not by Adam state updates.

## Passing Control

Run the same repro with allgather CP disabled:

```bash
cd /home/chengze/work/AccRL
./accrl/distill/sft/scripts/repro_qwen35_35b_allgather_cp_nan.sh pass
```

This changes only:

```text
ALLGATHER_CP=0
```

Expected result: the same data completes the 1-epoch repro with finite
`train/loss` and finite `train/grad_norm`.

## Interpretation

Context parallelism itself is required for these long SFT sequences. The issue
appears specific to the allgather CP implementation path for this model/data
shape. The non-allgather CP path is the current fixed recipe used for full SFT
runs.
