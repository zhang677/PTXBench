# Fixit v2-clean + v5-full training experiment

## Decision

Use one epoch at the existing learning rate of `4.65e-4` as a conservative
first experiment, not as a loss-derived optimum. Keeping the learning rate
fixed controls that variable: relative to the previous runs, the intended
changes are the d128 training mixture and the number of epochs. The existing
loss curves do not show that one epoch is sufficient; they show that lower
training loss is not enough to select a checkpoint that generalizes better.

The proposed run continues from the final fixit-v2 Tinker training-state
checkpoint and trains for one additional epoch on the combined parquet. Its
`model_name` remains `Qwen/Qwen3.6-27B` for the model preset and tokenizer, while
`load_checkpoint_path` initializes the LoRA from
`tinker://62e73b90-5995-56a8-98d9-f31536036be5:train:0/weights/final`. The v2
examples also act as replay data intended to preserve the behavior that was
lost in v5.

## Evidence from the existing runs

Both checkpoints used five epochs, learning rate `4.65e-4`, LoRA rank 32, and
loss only on the final assistant message. Their epoch-average training NLL was:

| Epoch | fixit-v2-glm | fixit-v5 |
| ---: | ---: | ---: |
| 1 | 0.2658 | 0.2702 |
| 2 | 0.1995 | 0.2053 |
| 3 | 0.1320 | 0.1403 |
| 4 | 0.0714 | 0.0810 |
| 5 | 0.0346 | 0.0429 |

The mean of the last ten updates in each epoch tells the same story:

| Epoch | fixit-v2-glm | fixit-v5 |
| ---: | ---: | ---: |
| 1 | 0.2587 | 0.2543 |
| 2 | 0.2078 | 0.2009 |
| 3 | 0.1257 | 0.1508 |
| 4 | 0.0647 | 0.1050 |
| 5 | 0.0382 | 0.0416 |

V5 therefore reached nearly the same low training NLL as v2 while performing
substantially worse on the matched d128 evaluation: 27 versus 70 correct turns,
19 versus 38 ever-correct trajectories, and 2 versus 7 turn-0 successes. More
optimization clearly improved in-sample fit, but it did not establish better
repair behavior.

This is why one epoch is reasonable as a risk-controlled probe:

1. It exposes every combined example once without repeatedly fitting a small,
   specialized corpus.
2. The combined corpus has 274 unique d128 rows, rather than 153 v2-clean rows
   or 121 selected d128 rows from v5-full alone, so one pass covers both replay
   and the expanded v5 examples for the same four target problems.
3. Keeping `4.65e-4` matches both previous runs, so the effect is not confounded
   by simultaneously changing the learning rate.
4. The v5 result demonstrates that driving training NLL from about 0.25 after
   epoch 1 toward about 0.04 after epoch 5 is not, by itself, a reason to prefer
   the later checkpoint.

The important caveat is that the epoch-1 loss had not plateaued in either old
run, so a one-epoch run could underfit. Loss alone therefore supports trying one
epoch as the safest first point, not claiming it is optimal. The next controlled
comparison should be one versus two epochs at the same `4.65e-4`, evaluated with
the same five-definition benchmark. The helper
now accepts both settings as command-line arguments, so that follow-up does not
require another code change.

## Relation to the difficulty/generalization paper

“Data Difficulty and the Generalization-Extrapolation Tradeoff in LLM
Fine-Tuning” predicts that hard examples can hurt in-distribution generalization
when the dataset is small, even while training loss converges. Harder examples
become more useful as data scale increases. This is consistent with v5 adding a
broader d64/d128 and later-turn failure distribution while losing d128
reliability, and it motivates mixing v2-clean replay with v5-full rather than
training on v5 alone.

The paper does not prove that one epoch is optimal here. Its task, dataset size,
and training setup differ from this 274-row CUDA LoRA experiment. Its mechanism
supports reducing repeated pressure on a small difficult corpus and measuring
generalization directly.

Paper: <https://arxiv.org/abs/2605.12906>

## Combined parquet contract

Script 19 combines:

- `glm-5.2-mha-d128-4def-full-no-myreasoning.parquet`: 153 selected rows
- `glm-5.2-fixit-v5-full.parquet`: 121 selected d128 rows from 258 input rows

The sources have no overlapping IDs. The output has 274 unique rows and contains
only the four d128 definitions; all 137 d64 v5-full rows are excluded. Script 19
validates the exact five-message role sequence
`(system, user, assistant, user, assistant)`, loss masks
`(0, 0, 0, 0, 1)`, absence of `<my_reasoning>`, unique IDs, deterministic
shuffling, and readback. It also writes a hash/count manifest beside the output.

## Workflow

```bash
cd /home/ubuntu/AccRL/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts
./19_combine_v2_clean_v5_full_parquet.sh
./20_train_v2_clean_v5_full_e1_lr4.65e-4.sh
```

After training, script 21 merges and serves the final checkpoint as
`qwen36-27b-SFT-$BASE_RUN_DATE-v2-clean-v5-full-d128-from-v2-final` through
`localhost:30032`:

```bash
./21_serve_remote_v2_clean_v5_full.sh
```

Script 22 deliberately keeps the same five definitions, configs, parallelism,
profiling host, and watcher behavior as `17_watch_v5_full_5defs_eval.sh`; it
targets the model served by script 21 and changes the output suffix. Run:

```bash
./22_watch_v2_clean_v5_full_5defs_eval.sh
```

Alternatively, after script 20 has launched the asynchronous tmux training job,
script 23 waits for the final checkpoint and performs both handoffs in order:

```bash
./23_wait_train_then_serve_and_watch.sh
```

It runs script 21 in the foreground until the model endpoint is ready and then
replaces itself with script 22. It stops on a missing final checkpoint, serving
failure, or a 48-hour training timeout.

The primary comparison should include correct turns, ever-correct trajectories,
turn-0 correctness, failure-to-correct recovery, compile errors, and timeouts.
Do not select the epoch using training NLL alone.
