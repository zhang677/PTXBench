# Why linfo underperforms

## Conclusion

These folders do **not** provide a clean “short feedback versus long feedback” ablation. The unpatched `regular` and `linfo` arms are matched stochastic repeats: they use identical prompts, checkpoint, sampling settings, and structured diagnostics. The patched pair is closer to a feedback comparison, but its system-reference text also changed. The checkpoint comparison is `fixit-v2-glm` versus `fixit-v5`; it changes the checkpoint, system prompt, inference-time diagnostics, and SFT data distribution together.

Across all three views, the most consistent mechanism is **repair thrashing around a brittle initial architecture**. Detailed diagnostics are read and often localized correctly, but the next full-code rewrite preserves inconsistent TMA descriptor, WGMMA layout, mbarrier, shared-memory, or launch assumptions—or introduces a new defect. Because part of the performance gap already exists on turn 0, feedback alone cannot explain it.

## Outcome and behavior shift

| group | correct turns | ever-correct trajectories | turn-0 correct | compile | runtime | numeric | timeout | mean initial code chars | mean initial reasoning chars | mean feedback chars | feedback with primary diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regular | 9/256 | 4/32 | 0/32 | 66 | 147 | 17 | 6 | 19,893 | 90,932 | 2,363 | 59.8% |
| linfo | 3/256 | 2/32 | 0/32 | 76 | 128 | 33 | 5 | 22,142 | 85,546 | 2,663 | 51.2% |
| regular-patched | 25/256 | 14/32 | 2/32 | 60 | 71 | 73 | 23 | 20,586 | 63,768 | 1,654 | 0.0% |
| linfo-patched | 21/256 | 9/32 | 0/32 | 82 | 53 | 61 | 31 | 22,943 | 69,801 | 2,310 | 32.8% |
| fixit-v2-glm | 70/797 | 38/100 | 7/100 | 209 | 250 | 159 | 82 | 17,410 | 83,236 | 1,628 | 0.0% |
| fixit-v5 | 27/797 | 19/100 | 2/100 | 264 | 215 | 168 | 108 | 21,245 | 100,446 | 2,512 | 39.9% |

The v2-to-v5 drop is large on the matched 797 turns: correct turns fall from 70 to 27, and ever-correct trajectories from 38/100 to 19/100. V5 has 55 more compile errors and 26 more timeouts. It also starts from longer samples: mean initial code grows by 3,835 characters and initial reasoning by 17,209.

Some of both gaps exists before evaluator feedback is shown. `regular-patched` has 2 turn-0 successes versus 0 for `linfo-patched`; `fixit-v2-glm` has 7 versus 2 for v5. Feedback affects recovery, not those initial-sample differences.

## Condition audit: what is actually being compared

- Unpatched `regular` versus `linfo`: 32/32 matched trajectories have the same system prompt, 32/32 the same task prompt, and every pair uses the same checkpoint and sampling settings. Both arms contain structured CUDA diagnostics. This is principally a stochastic repeat, not a feedback-detail ablation.
- `regular-patched` versus `linfo-patched`: all 32 pairs use the same checkpoint, task prompt, and sampling settings, but 0/32 system prompts are byte-identical because the Hopper reference text changed. The old regular run has no `Primary diagnostics` blocks, while linfo-patched has them on 84/256 turns. This is the closest feedback comparison, though it is still not perfectly isolated.
- `fixit-v2-glm` versus `fixit-v5`: all 100/100 pairs have the same task prompt and all 100/100 use the same sampling settings, but 0/100 use the same checkpoint or byte-identical system prompt. Structured `Primary diagnostics` rise from 0 to 318/797 turns. This measures the combined v5 checkpoint and evaluation stack, not feedback length alone.

## What the model does with the information

| group | turns with TMA | turns with WGMMA | turns with mbarrier | turns with extended launch | major rewrites | mean adjacent-code Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| regular | 92.6% | 95.7% | 94.1% | 18.8% | 43 | 0.675 |
| linfo | 92.6% | 94.9% | 93.8% | 12.5% | 44 | 0.692 |
| regular-patched | 98.4% | 98.4% | 98.4% | 18.4% | 18 | 0.765 |
| linfo-patched | 96.5% | 96.9% | 97.3% | 6.6% | 21 | 0.753 |
| fixit-v2-glm | 96.4% | 96.2% | 96.4% | 21.6% | 120 | 0.693 |
| fixit-v5 | 97.9% | 98.5% | 98.2% | 3.6% | 158 | 0.670 |

All arms already use TMA, WGMMA, and mbarrier on nearly every turn. V5 does not merely use these mechanisms more often; it begins with larger code and performs 158 major rewrites versus 120 for v2. The reasoning can be specific to the latest reported fault while the overall descriptor/layout/barrier/launch contract remains inconsistent.

## Representative trajectories

### Same checkpoint, patched forward attention

| group | definition | tag | replica | trajectory | turn outcomes | initial code chars | initial reasoning chars |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regular-patched | mha_with_lse_d128 | hopper-07 | 1 | exp_005 | compile -> numeric -> OK -> OK -> compile -> numeric -> OK -> OK | 15,999 | 24,438 |
| linfo-patched | mha_with_lse_d128 | hopper-07 | 1 | exp_001 | timeout -> compile -> numeric -> numeric -> numeric -> numeric -> numeric -> numeric | 20,740 | 148,199 |

`regular-patched/exp_005` fixes a compile-time shared-memory pointer problem, then notices an unused `kv_offset` after a numerical failure and reaches correctness at turn 2. `linfo-patched/exp_001` times out before receiving feedback. Its coredump correctly localizes a V-buffer mbarrier hang, but later rewrites move through compile failure and persistent NaN/Inf. The model proposes several locally plausible fixes without making the whole double-buffered TMA/WGMMA kernel coherent.

### Same checkpoint, unpatched backward attention

| group | definition | tag | replica | trajectory | turn outcomes | initial code chars | initial reasoning chars |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regular | mha_bwd_d128 | hopper-012 | 0 | exp_012 | compile -> numeric -> runtime -> runtime -> numeric -> OK -> numeric -> numeric | 23,323 | 29,581 |
| linfo | mha_bwd_d128 | hopper-012 | 0 | exp_000 | compile -> compile -> runtime -> compile -> runtime -> compile -> numeric -> numeric | 30,765 | 75,597 |

`regular/exp_012` progresses from a nonexistent BF16 intrinsic through layout failures and passes at turn 5. `linfo/exp_000` samples a larger two-kernel design with extended cluster launch, TMA, WGMMA, and explicit register allocation. It bounces among host API, launch-argument, runtime, and numerical failures without passing. Since the condition is otherwise identical, this pair demonstrates sensitivity to the initial sampled architecture.

### Fixit v2 versus v5

| group | definition | tag | replica | trajectory | turn outcomes | initial code chars | initial reasoning chars |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixit-v2-glm | mha_with_lse_d128 | hopper-07 | 0 | exp_004 | OK -> compile -> compile -> numeric -> runtime -> OK -> numeric -> runtime | 15,497 | 96,111 |
| fixit-v5 | mha_with_lse_d128 | hopper-07 | 0 | exp_004 | compile -> runtime -> numeric -> compile -> numeric -> numeric -> compile -> numeric | 19,019 | 108,363 |

For the same forward-attention prompt, tag, and replica, `fixit-v2-glm/exp_004` is correct on turn 0 and recovers once again at turn 5 after an optimization regression. `fixit-v5/exp_004` starts with a declaration-order compile error. It fixes that issue, then moves through an illegal shared-memory read, a numerical failure, an ambiguous BF16 conversion, NaN/Inf, a shared-memory alias, and finally a newly introduced undefined identifier. Several diagnoses are correct locally, but the eight rewrites never produce a correct kernel. This is the clearest v5 example of symptom-by-symptom repair without stable global invariants.

## What changed in v5 training

The two training configs use the same base model, learning rate (0.000465), epochs (5), LoRA rank (32), and `train_on_what=last_assistant_message`. The parquets are different lineages with 0 shared IDs:

- V2 uses 158 rows, all d128, with wrong-turn distribution {0: 31, 1: 40, 2: 52, 3: 35}.
- V5 uses 170 rows: 76 d128 and 94 d64. All 170 wrong kernels come from linfo trajectories. Its wrong-turn distribution is {0: 27, 1: 31, 2: 36, 3: 33, 4: 36, 5: 6, 7: 1}; 43 rows are from turns 4 or later.
- Mean evaluator-feedback length rises from 1,608 to 2,155 characters, and mean target length from 47,467 to 50,170.

For this d128 evaluation, v5 therefore has fewer than half as many in-domain d128 SFT rows as v2 (76 versus 158), while adding d64 and linfo-failure coverage. That distribution shift is a plausible contributor to the lower d128 result. It also means the v5 checkpoint comparison cannot identify a simple “more feedback is worse” effect: the checkpoint was trained on a different task mix and failure-source distribution.

## Recovery evidence

| group | previous failure | next-turn recoveries | rate |
| --- | --- | --- | --- |
| regular | Compilation error | 1/62 | 1.6% |
| regular | Runtime error | 0/126 | 0.0% |
| regular | Numerical error | 3/14 | 21.4% |
| linfo | Compilation error | 1/72 | 1.4% |
| linfo | Runtime error | 1/109 | 0.9% |
| linfo | Numerical error | 1/26 | 3.8% |
| regular-patched | Compilation error | 1/57 | 1.8% |
| regular-patched | Runtime error | 7/61 | 11.5% |
| regular-patched | Numerical error | 9/64 | 14.1% |
| linfo-patched | Compilation error | 5/74 | 6.8% |
| linfo-patched | Runtime error | 2/49 | 4.1% |
| linfo-patched | Numerical error | 3/51 | 5.9% |
| fixit-v2-glm | Compilation error | 8/195 | 4.1% |
| fixit-v2-glm | Runtime error | 18/207 | 8.7% |
| fixit-v2-glm | Numerical error | 21/138 | 15.2% |
| fixit-v5 | Compilation error | 4/244 | 1.6% |
| fixit-v5 | Runtime error | 11/192 | 5.7% |
| fixit-v5 | Numerical error | 5/131 | 3.8% |

The low absolute recovery rates support the trajectory reading. Detailed logs expose a real defect, but these coupled kernels commonly contain several defects at once. Fixing one does not make the next kernel correct, and full-file reconstruction can reintroduce already-solved errors.

## Bottom line

1. The corrected checkpoint comparison is `fixit-v2-glm` versus `fixit-v5`.
2. V5 is materially worse on these matched d128 trajectories: 27 versus 70 correct turns and 19 versus 38 ever-correct trajectories.
3. The gap is not attributable to feedback alone. V5 changes checkpoint, system prompt, diagnostic stack, and SFT distribution; it also has only 76 d128 training rows versus v2's 158.
4. Across representative failures, structured diagnostics improve local fault localization but do not enforce rollback or whole-kernel invariants. The model keeps repairing a brittle design and often creates the next error.

The practical implication is to pair detailed diagnostics with control rules: establish a compiling minimal kernel first, add one optimization pattern at a time, and after repeated compile/runtime failures force a revert to the last correct kernel or a restart from a simpler architecture.

## Reproduction

```bash
python analyze_linfo_underperformance.py
```

The script reads the two existing `holistic_turns.csv` files, joins matched trajectory IDs to source JSONs, reads the v2 and v5 SFT parquets and training configs, and writes this report plus `trajectory_metrics.csv`.
