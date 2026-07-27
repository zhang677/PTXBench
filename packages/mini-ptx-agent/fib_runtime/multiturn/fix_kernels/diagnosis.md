# Diagnosis: Qwen3.6-27B Fixit v0/v1/v2

Last updated: 2026-06-19.

## Scope

This note diagnoses the three benchmarked fixit models registered in `/home/ubuntu/AccRL/benchmark/experiments.csv`:

- `Qwen3.6-27B-fixit-v0`
- `Qwen3.6-27B-fixit-v1`
- `Qwen3.6-27B-fixit-v2`

The training source of truth is `/home/ubuntu/AccRL/benchmark/sft_mapping.csv`:

| tag | training parquet | Tinker run |
| --- | --- | --- |
| fixit-v0 | `data/kimi-k2.7-code.parquet` | `...2026-06-16-01-30/checkpoints.jsonl` |
| fixit-v1 | `data/kimi-k2.7-code-mha-d128-4def.parquet` | `...2026-06-17-10-01/checkpoints.jsonl` |
| fixit-v2 | `data/kimi-k2.7-code-mha-d128-4def-full.parquet` | `...2026-06-17-19-12/checkpoints.jsonl` |

All three Tinker configs use the same training hyperparameters: `Qwen/Qwen3.6-27B`, renderer `qwen3_5`, max length `65536`, batch size `2`, train-on `last_assistant_message`, LoRA rank `32`, learning rate `4.65e-4`, linear schedule, and `5` epochs. The only intentional config difference is the dataset path.

## Current Eval Outcomes

From `/home/ubuntu/AccRL/benchmark/figures/correctness_rate_by_release_date.csv`:

| model | eval | turn 1 | turn 4 | turn 8 | notes |
| --- | --- | ---: | ---: | ---: | --- |
| v0 | `mha_with_lse_d128` | 1/20 | 2/20 | 6/20 | full 8-turn local artifacts |
| v1 | `mha_with_lse_d128` | 0/20 | 2/20 | 2/20 | local artifacts only contain 4 turns |
| v2 | `mha_with_lse_d128` | 0/20 | 0/20 | 0/20 | full 8-turn local artifacts |
| v0 | `gemm_n7168_k5120` | 0/20 | 2/20 | 5/20 | full 8-turn local artifacts |
| v1 | `gemm_n7168_k5120` | 0/20 | 1/20 | 3/20 | full 8-turn local artifacts |
| v2 | `gemm_n7168_k5120` | 0/10 | 1/10 | 1/10 | incomplete/stale local artifact; see below |
| v1 | `mha_with_lse_d128_causal` | 0/20 | 0/20 | 0/20 | local artifacts only contain 4 turns |
| v2 | `mha_with_lse_d128_causal` | 0/20 | 1/20 | 2/20 | full 8-turn local artifacts |
| v1 | `mha_bwd_d128` | 0/20 | 0/20 | 1/20 | mostly full 8-turn artifacts |
| v2 | `mha_bwd_d128` | 0/20 | 0/20 | 0/20 | full 8-turn local artifacts |

Two artifact caveats matter:

- `2026-0617-1001-mha-d128` and `2026-0617-1001-mha-d128-causal` have only 80 turn CSV rows each, 20 trajectories times 4 turns. Their plotted turn-8 correctness is therefore not a real 8-turn result; it is the same ceiling as turn 4.
- `2026-0617-1912-gemm` has 18 trajectory JSONs but no local `figures/turn_correctness_arch.csv` in the inspected checkout. Treat the current plotted `n=10` v2 GEMM row as incomplete until regenerated or rerun.

The cleanest comparison is forward MHA d128: v0 reaches 6/20 by 8 turns, v1 reaches 2/20 with only 4 turns available, and v2 reaches 0/20 with 8 turns available.

## Dataset Composition

The trainer filter is implemented in `tinker_sft_train.py`: it drops samples whose sum of tokenizer-encoded message content lengths exceeds `max_length=65536`, then shuffles with seed 0.

Reproducing that filter with the local `Qwen/Qwen3.6-27B` tokenizer gives:

| tag | raw rows | kept rows | dropped | training steps | direct `mha_with_lse_d128` rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 | 53 | 38 | 15 | 95 | 38 |
| v1 | 74 | 74 | 0 | 185 | 27 |
| v2 | 140 | 140 | 0 | 350 | 44 |

Definition mix among kept rows:

| tag | `mha_with_lse_d128` | `mha_with_lse_d128_causal` | `mha_bwd_d128` | `mha_bwd_d128_causal` |
| --- | ---: | ---: | ---: | ---: |
| v0 | 38 | 0 | 0 | 0 |
| v1 | 27 | 26 | 10 | 11 |
| v2 | 44 | 41 | 25 | 30 |

This explains much of v1: it has more total data than v0, but fewer examples for the exact forward non-causal MHA d128 task that was used for the headline comparison.

## v2 Adds Mostly Low-Value Rows

`v2` contains all 74 kept v1 rows plus 66 extra rows.

The 66 added rows have very different quality:

| slice | rows | speedup mean | speedup p50 | speedup p90 | speedup max | rows with speedup < 0.15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 rows | 74 | 0.305 | 0.288 | 0.463 | 0.690 | 0/74 |
| v2-only rows | 66 | 0.052 | 0.047 | 0.109 | 0.139 | 66/66 |

The v2-only rows all pass, but every one of them is below the 0.15 speedup threshold that was used elsewhere as a useful repair target. Five are marked `improved=False`. The v2-only rows also skew later in the teacher repair process: `correct_turn=4` is the most common bucket for v2-only rows.

This is the strongest data-quality explanation for v2: it has more data, but the additional data is mostly low-margin repair behavior. It can teach the model to produce passing-but-weak or meandering repair patterns rather than stronger fixes.

## Training Exposure and Overfitting Risk

Because all runs use 5 epochs and batch size 2, larger datasets receive many more optimizer updates at the same peak LR:

| tag | kept rows | batches per epoch | total steps | first-10 mean loss | last-10 mean loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 | 38 | 19 | 95 | 0.4024 | 0.0751 |
| v1 | 74 | 37 | 185 | 0.4477 | 0.1411 |
| v2 | 140 | 70 | 350 | 0.4507 | 0.1099 |

There is no validation split and no evaluator during training. The final checkpoint is used by default. This makes it impossible to know whether v1/v2 had an earlier checkpoint that preserved more base-model capability.

The fixit SFT recipe therefore changes two variables at once:

- More and different data.
- More optimizer steps at the same LR and same epoch count.

This can produce capability drift or catastrophic forgetting, especially on GEMM, which is not present in these fixit parquets.

## Eval Failure-Mode Shifts

Forward MHA d128 turn-level failure counts:

| tag | correct turns | compilation errors | runtime errors | numerical errors | timeouts | other/extraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 9 | 50 | 55 | 32 | 11 | 3 |
| v1 | 3 | 46 | 12 | 3 | 15 | 1 |
| v2 | 0 | 65 | 48 | 24 | 19 | 4 |

v1 has fewer total turns for this eval, so its lower count of runtime/numerical errors is partly a shorter-run artifact. v2 has a full 8-turn run and still has zero correct turns, with more compilation errors than v0. That suggests the v2 final checkpoint is not merely slower to improve; it is producing less viable code on this eval.

Output-format checks did not show a simple markdown/code-fence regression. The median final assistant response still contains one `cpp` fenced block pattern in these runs. The problem looks more semantic/algorithmic than extraction-only.

## Main Diagnosis

The worse v1/v2 results are not paradoxical once "more data" is decomposed:

1. v1 has fewer direct examples for the main evaluated task.
   - v0 trains on 38 kept forward MHA d128 rows.
   - v1 trains on 74 rows total, but only 27 are forward MHA d128.
   - v1's forward MHA eval also only ran 4 turns locally.

2. v2 adds many low-value examples.
   - v2 adds 66 rows beyond v1.
   - All 66 added rows have speedup below 0.15.
   - The added rows have mean speedup about 0.052 and median about 0.047.
   - This is more data, but not more high-quality repair signal.

3. v1/v2 train for many more update steps at the same LR.
   - v0: 95 steps.
   - v1: 185 steps.
   - v2: 350 steps.
   - With no validation/eval during training, the final checkpoint may be over-adapted to teacher repair traces and worse at fresh kernel synthesis.

4. The added task mixture can interfere with the target behavior.
   - v1/v2 mix forward, causal, backward, and backward-causal MHA d128.
   - The exact forward non-causal eval is no longer the dominant training slice.
   - GEMM degradation is consistent with forgetting or base capability drift, since GEMM is outside the fixit data.

5. The teacher traces are imitation data, not guaranteed capability transfer.
   - The target contains long Kimi repair reasoning plus final code.
   - Low-margin or inconsistent reasoning can teach style and local patches without teaching robust CUDA design.

## Recommended Next Checks

1. Evaluate intermediate checkpoints.
   - v1 has checkpoints around steps 50, 100, 150, and final.
   - v2 has checkpoints around steps 50, 100, 150, 200, 250, 300, and final.
   - If earlier checkpoints beat final, the main issue is update budget/overfitting.

2. Train equal-step ablations.
   - Train v1/v2 with `max_steps=95`, matching v0.
   - Alternatively train all variants for one epoch only.

3. Filter v2 by repair value.
   - Rebuild v2 with `speedup >= 0.15`.
   - Optionally require `improved=True`.
   - Compare against the current v2 full parquet.

4. Train a task-matched forward-MHA-only ablation.
   - Use only `definition == mha_with_lse_d128`.
   - Compare v0's 38 rows, v1's 27 rows, and v2's 44 rows under equal steps.

5. Add validation during SFT.
   - Hold out a few forward MHA and GEMM tasks.
   - Select checkpoints by eval signal instead of always serving final.

6. Fix or regenerate questionable eval artifacts.
   - Rerun or regenerate `2026-0617-1001-mha-d128` if an 8-turn v1 comparison is needed.
   - Regenerate `2026-0617-1912-gemm/figures/turn_correctness_arch.csv` or rerun v2 GEMM before treating it as a clean benchmark.

7. Consider weighting or sampling by task.
   - Keep forward MHA d128 sufficiently represented when that is the target eval.
   - Avoid letting many low-speedup backward/causal rows dominate the update budget.

## Bottom Line

v1 and v2 were trained on more total rows, but not on more high-quality, target-matched signal. v1 diluted the target task and has a shorter forward-MHA eval artifact. v2 doubled the dataset with low-speedup fixes and trained 350 steps at the same LR, which likely caused over-adaptation and capability drift. The next experiment should control update count and filter by speedup/improvement before concluding that fixit data scaling itself is harmful.
