# Train/Eval Split Notes

This note captures the current experiment-design decisions for MHA-centered
training and prompt selection. The main goal is to keep retraining minimal while
preserving interpretable conclusions about supervision channels, shape transfer,
cross-precision transfer, and task-family transfer.

## Current Training Split

Use non-FP8 MHA as the source domain:

- `mha_with_lse_d64`
- `mha_with_lse_d64_causal`
- `mha_with_lse_d128`
- `mha_with_lse_d128_causal`
- `mha_bwd_d64`
- `mha_bwd_d64_causal`
- `mha_bwd_d128`
- `mha_bwd_d128_causal`

The cleanest first comparison is still supervision-channel controlled:

- Demo SFT: task plus algorithm prompt -> correct kernel.
- Fixit SFT: task plus algorithm prompt plus wrong kernel plus execution feedback -> repair reasoning plus corrected kernel.
- Notes/experience prompting: task plus retrieved distilled notes -> complete kernel.
- Hybrid: repair trajectories plus distilled experiences, only after the simpler comparisons are informative.

Forward and backward should use task-appropriate algorithm prompts. FlashAttention
forward and backward are different algorithms, so the prompt is part of the task
interface rather than a neutral wrapper.

## Evaluation Tiers

Use the same first set of trained checkpoints across the tiers below. Do not train
separate models for FP8, GQA, MLA, GDN, or contrast ops until a result creates a
specific follow-up question.

1. In-domain MHA heldout:
   - Held-out seeds, prompt tags, or sequence lengths from the same d64/d128
     MHA definitions.

2. Shape-heldout MHA:
   - Non-FP8 d96 and/or d256 MHA if task files are stable.

3. Cross-precision MHA:
   - FP8 MHA d64/d128 as precision-OOD with seen shapes.
   - FP8 MHA d96/d256 as precision-OOD plus shape-OOD.
   - This tests whether non-FP8 MHA repair supervision transfers to FP8-specific
     instruction and numerical regimes.

4. Near-OOD attention:
   - GQA decode.
   - GQA paged prefill.
   - GQA ragged prefill.
   - MLA paged or ragged.
   - Run both unassisted and algorithm-assisted prompt conditions when possible.

5. Primitive transfer:
   - GEMM and stable FP8 GEMM variants.

6. Far-OOD serving/generalization:
   - GDN decode/prefill/MTP.
   - RMSNorm or sampling.
   - Treat these as boundary tests, not the main result.

For fast iteration, use a small sentinel eval first and only expand after a
checkpoint is worth full analysis.

## Prompt-Tag Analysis

The following analysis uses turn-level counting, not trajectory-level counting.
Each turn is one sample.

Definitions:

- `turn_success_rate`: `Correct` turns divided by total turns.
- `best_speedup`: maximum speedup among correct turns.
- `avg_speedup_correct_turns`: average speedup over correct turns only.
- `avg_speedup_all_turns`: average speedup over all turns, with failed turns
  counted as `0`. This is the best single utility metric for prompt selection.

Included runs:

```text
gemini-3.1-pro-preview,hopper,mha_with_lse_d128,bc38b351-d595-451b-9153-8e225702e53b,/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340
gemini-3.1-pro-preview,hopper,mha_with_lse_d128_causal,6d2f67a7-225a-4af5-87d3-cbb99b496325,/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140
gemini-3.1-pro-preview,hopper,mha_bwd_d128,38c3b07c-f006-5f5e-9860-ba214c805a6b,/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240
gemini-3.1-pro-preview,hopper,mha_bwd_d128_causal,c119b3f0-c051-5e96-9c2a-2268d992fe1a,/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340
gemini-3.1-pro-preview,hopper,mha_with_lse_d64,7d2575a0-bcc2-42a0-812f-6a7e9a57d97f,/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1120
gemini-3.1-pro-preview,hopper,mha_with_lse_d64_causal,b69f7675-568f-40f2-9a4b-8bbe374b4a59,/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1220
gemini-3.1-pro-preview,hopper,mha_bwd_d64,d3bcb902-6a13-5ada-9251-fa841b10cd0b,/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1320
gemini-3.1-pro-preview,hopper,mha_bwd_d64_causal,5799ea50-77aa-56cb-9f62-a4c1f5473770,/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1420
```

Source fields:

- Prompt tags are read from each run's `plan.json`.
- Correctness and speedup are read from each run's
  `figures/turn_correctness_arch.csv`.
- Blank speedup cells are treated as `0`.

### Forward MHA Prompt Tags

The added `mha_with_lse_d128` run has only one trajectory per prompt tag, so it is
noisier than the later four-trajectory-per-tag runs.

| prompt_tag | total_turns | correct_turns | turn_success_rate | best_speedup | avg_speedup_correct_turns | avg_speedup_all_turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hopper-08` | 104 | 56 | 53.8% | 0.731 | 0.429 | 0.231 |
| `hopper-014` | 104 | 55 | 52.9% | 0.649 | 0.366 | 0.193 |
| `hopper-09` | 104 | 50 | 48.1% | 0.713 | 0.390 | 0.188 |
| `hopper-07` | 104 | 48 | 46.2% | 0.720 | 0.432 | 0.199 |
| `hopper-no-hint` | 104 | 43 | 41.3% | 0.788 | 0.464 | 0.192 |

Forward interpretation:

- `hopper-08` has the best turn success rate and best `avg_speedup_all_turns`.
- `hopper-no-hint` has the highest best speedup and highest average speedup among
  correct turns, but it is less reliable.
- The added `mha_with_lse_d128` run is a warning for `hopper-08`: it had `0/8`
  correct turns on that specific non-causal d128 slice.

Per-prompt result on `mha_with_lse_d128` only:

| prompt_tag | correct_turns / total_turns | avg_speedup_all_turns |
| --- | ---: | ---: |
| `hopper-09` | 4 / 8 | 0.247 |
| `hopper-07` | 4 / 8 | 0.237 |
| `hopper-014` | 3 / 8 | 0.213 |
| `hopper-08` | 0 / 8 | 0.000 |
| `hopper-no-hint` | 0 / 8 | 0.000 |

### Legacy h48/d128 Forward Evidence

Candidate replacement/additional run:

```text
gemini-3.1-pro-preview,hopper,mha_with_lse_h48_d128,f4a505f6-a887-44ae-9000-88553af1e433,/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313
```

Use this as additional evidence rather than a silent replacement for
`mha_with_lse_d128`:

- Pros: the run has 4 trajectories per prompt for `hopper-no-hint`,
  `hopper-07`, `hopper-08`, and `hopper-09`, so it is much less noisy than
  `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340`, which has only 1
  trajectory per prompt.
- Cons: it uses the older prompt config
  `/home/ubuntu/AccRL-exps/prompt_configs/2026-0503-fa3-fwd.json`, lacks
  `hopper-014`, and uses the legacy `_h48` definition name. Treat it as
  legacy alias evidence for canonical d128 only when that alias mapping is
  acceptable for the analysis being performed.

Turn-level result for `mha_with_lse_h48_d128` only:

| prompt_tag | total_turns | correct_turns | turn_success_rate | best_speedup | avg_speedup_correct_turns | avg_speedup_all_turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hopper-07` | 32 | 16 | 50.0% | 0.856 | 0.517 | 0.259 |
| `hopper-08` | 32 | 11 | 34.4% | 0.562 | 0.442 | 0.152 |
| `hopper-no-hint` | 32 | 10 | 31.2% | 0.721 | 0.491 | 0.154 |
| `hopper-09` | 32 | 7 | 21.9% | 0.469 | 0.244 | 0.053 |

If the underpowered canonical `mha_with_lse_d128` run is replaced by the legacy
`mha_with_lse_h48_d128` run in the forward aggregate, the prompt ranking becomes:

| prompt_tag | total_turns | correct_turns | turn_success_rate | best_speedup | avg_speedup_correct_turns | avg_speedup_all_turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hopper-014` | 96 | 52 | 54.2% | 0.649 | 0.354 | 0.192 |
| `hopper-08` | 128 | 67 | 52.3% | 0.731 | 0.431 | 0.226 |
| `hopper-07` | 128 | 60 | 46.9% | 0.856 | 0.452 | 0.212 |
| `hopper-no-hint` | 128 | 53 | 41.4% | 0.788 | 0.469 | 0.194 |
| `hopper-09` | 128 | 53 | 41.4% | 0.713 | 0.363 | 0.150 |

Replacement interpretation:

- `hopper-08` remains the strongest primary prompt by reliability/utility
  aggregate.
- `hopper-07` becomes the strongest forward backup prompt because the legacy
  h48/d128 run strongly favors it.
- `hopper-no-hint` remains a speed/diversity control, not the main collection
  prompt.

### Backward MHA Prompt Tags

| prompt_tag | total_turns | correct_turns | turn_success_rate | best_speedup | avg_speedup_correct_turns | avg_speedup_all_turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hopper-012` | 128 | 42 | 32.8% | 0.537 | 0.298 | 0.098 |
| `hopper-013` | 128 | 38 | 29.7% | 0.578 | 0.292 | 0.087 |
| `hopper-010` | 128 | 37 | 28.9% | 0.607 | 0.297 | 0.086 |
| `hopper-011` | 128 | 34 | 26.6% | 0.399 | 0.162 | 0.043 |
| `hopper-no-hint` | 128 | 28 | 21.9% | 0.746 | 0.327 | 0.071 |

Backward interpretation:

- `hopper-012` is the best default by turn success rate and
  `avg_speedup_all_turns`.
- `hopper-013` and `hopper-010` are close secondary prompts.
- `hopper-no-hint` has the fastest outlier and the highest average speedup among
  correct turns, but its turn success rate is worst.

## Prompt Recommendation

Use task-specific algorithm prompts instead of one shared prompt for all MHA tasks:

- Forward primary: `hopper-08`.
- Forward backup/diversity: `hopper-07` if using the legacy h48/d128 evidence;
  otherwise `hopper-09` or `hopper-07` are both defensible.
- Backward primary: `hopper-012`.
- Backward backup/diversity: `hopper-013`.

For data collection, avoid using only `hopper-08` for all forward MHA because the
single-trajectory `mha_with_lse_d128` run failed under `hopper-08`. If using only
the canonical d128 evidence, a practical low-cost forward mixture is
`hopper-08 + hopper-09`. If accepting the legacy h48/d128 run as alias evidence,
prefer `hopper-08 + hopper-07`: `hopper-08` supplies the best aggregate
reliability, while `hopper-07` covers the stronger legacy d128 signal and has the
best h48/d128 utility.

Keep `hopper-no-hint` as a control and possible diversity source, not as the main
collection prompt. It can produce faster correct turns, but it is less reliable in
turn-level success.
