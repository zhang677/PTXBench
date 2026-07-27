# GLM 5.2 vs Gemini Fix-It Analysis

Generated: 2026-07-14 06:37:42 UTC

## Scope

This compares the direct fix-it runs that use the same planned d128 Qwen failed kernels:

- GLM 5.2: `/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-glm52`
- Gemini: `/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-gemini`

The analysis reads current `trajectories/*.json` and extracts per-turn correctness and speedup with `/home/ubuntu/AccRL/benchmark/export_turn_correctness_arch.py`. Success directories are used only as a materialized-kernel cross-check.

## Overall

| metric | GLM 5.2 | Gemini |
| --- | --- | --- |
| planned prompts | 222 | 222 |
| trajectory files | 222 | 222 |
| turn rows | 1093 | 992 |
| correct turns | 64 | 145 |
| correct turns using `H` instruction | 36 | 142 |
| any correct | 46 (20.7%) | 117 (52.7%) |
| any correct using `H` instruction | 27 (12.2%) | 114 (51.4%) |
| target hits >= 0.15 | 15 (6.8%) | 74 (33.3%) |
| `H` instruction target hits >= 0.15 | 15 (6.8%) | 74 (33.3%) |
| best-correct median speedup | 0.0558 | 0.1697 |
| best-correct mean speedup | 0.1113 | 0.1911 |
| best `H`-instruction-correct median speedup | 0.1589 | 0.1740 |
| materialized success trajectories | 46 | 117 |

## Paired Outcomes

| outcome | count |
| --- | --- |
| both_fixed | 29 |
| glm_only | 17 |
| gemini_only | 88 |
| neither | 88 |
| glm_better_when_both_fixed | 8 |
| gemini_better_when_both_fixed | 21 |
| mean_glm_minus_gemini_best_speedup_when_both_fixed | -0.10334827716785461 |
| median_glm_minus_gemini_best_speedup_when_both_fixed | -0.12733578810474766 |
| both_instruction_fixed | 17 |
| glm_instruction_only | 10 |
| gemini_instruction_only | 97 |
| neither_instruction_fixed | 98 |
| glm_better_when_both_instruction_fixed | 5 |
| gemini_better_when_both_instruction_fixed | 12 |
| mean_glm_minus_gemini_best_instruction_speedup_when_both_instruction_fixed | -0.08247295209610211 |
| median_glm_minus_gemini_best_instruction_speedup_when_both_instruction_fixed | -0.09413742277160243 |

## By Definition

| definition | GLM fixed | Gemini fixed | GLM instr fixed | Gemini instr fixed | GLM target hits | Gemini target hits | GLM instr target hits | Gemini instr target hits | GLM median speedup | Gemini median speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mha_bwd_d128 | 8/72 | 32/72 | 4/72 | 32/72 | 2 | 18 | 2 | 18 | 0.0526 | 0.1734 |
| mha_bwd_d128_causal | 15/74 | 40/74 | 4/74 | 38/74 | 1 | 20 | 1 | 20 | 0.0328 | 0.1330 |
| mha_with_lse_d128_causal | 23/76 | 45/76 | 19/76 | 44/76 | 12 | 36 | 12 | 36 | 0.1553 | 0.2227 |

## By Turn

| turn | GLM attempts | GLM correct | GLM instr correct | GLM target hits | GLM instr target hits | GLM median correct speedup | Gemini attempts | Gemini correct | Gemini instr correct | Gemini target hits | Gemini instr target hits | Gemini median correct speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 222 | 3 | 2 | 0 | 0 | 0.0670 | 222 | 5 | 3 | 2 | 2 | 0.0141 |
| 1 | 222 | 11 | 5 | 1 | 1 | 0.0322 | 220 | 24 | 23 | 13 | 13 | 0.2032 |
| 2 | 221 | 20 | 10 | 4 | 4 | 0.0360 | 206 | 40 | 40 | 23 | 23 | 0.1573 |
| 3 | 217 | 15 | 11 | 6 | 6 | 0.0673 | 183 | 40 | 40 | 22 | 22 | 0.1610 |
| 4 | 211 | 15 | 8 | 4 | 4 | 0.0587 | 161 | 36 | 36 | 14 | 14 | 0.0834 |

## Notable GLM Wins

| trajectory | definition | prompt tag | GLM best | Gemini best | outcome |
| --- | --- | --- | --- | --- | --- |
| exp_199 | mha_with_lse_d128_causal | hopper-09 | 0.4758 |  | glm_only |
| exp_152 | mha_with_lse_d128_causal | hopper-no-hint | 0.3310 |  | glm_only |
| exp_182 | mha_with_lse_d128_causal | hopper-08 | 0.3055 | 0.2907 | both_fixed |
| exp_178 | mha_with_lse_d128_causal | hopper-08 | 0.3054 | 0.1581 | both_fixed |
| exp_163 | mha_with_lse_d128_causal | hopper-07 | 0.2941 | 0.0014 | both_fixed |
| exp_179 | mha_with_lse_d128_causal | hopper-08 | 0.2842 | 0.1975 | both_fixed |
| exp_022 | mha_bwd_d128 | hopper-010 | 0.2739 |  | glm_only |
| exp_148 | mha_with_lse_d128_causal | hopper-no-hint | 0.2080 |  | glm_only |
| exp_186 | mha_with_lse_d128_causal | hopper-08 | 0.2032 |  | glm_only |
| exp_046 | mha_bwd_d128 | hopper-012 | 0.1827 | 0.0515 | both_fixed |

## Notable Gemini-Only Wins

| trajectory | definition | prompt tag | Gemini best |
| --- | --- | --- | --- |
| exp_160 | mha_with_lse_d128_causal | hopper-no-hint | 0.6176 |
| exp_042 | mha_bwd_d128 | hopper-012 | 0.6097 |
| exp_041 | mha_bwd_d128 | hopper-012 | 0.5470 |
| exp_207 | mha_with_lse_d128_causal | hopper-014 | 0.5231 |
| exp_150 | mha_with_lse_d128_causal | hopper-no-hint | 0.4828 |
| exp_157 | mha_with_lse_d128_causal | hopper-no-hint | 0.4785 |
| exp_203 | mha_with_lse_d128_causal | hopper-09 | 0.4553 |
| exp_201 | mha_with_lse_d128_causal | hopper-09 | 0.4001 |
| exp_002 | mha_bwd_d128 | hopper-no-hint | 0.3730 |
| exp_151 | mha_with_lse_d128_causal | hopper-no-hint | 0.3648 |

## Outputs

- `summary.md`
- `prompt_summary.csv`: one paired row per prompt, including ordinary correctness and instruction-correctness fields

## Interpretation

Gemini fixes substantially more prompts than GLM 5.2 on this paired set: `117` vs `46`.
It also has many more target-speedup hits at `0.15`: `74` vs `15`.
Using the Hopper instruction tag `H`, Gemini also leads on instruction-correct prompts: `114` vs `27`, and instruction-correct target hits: `74` vs `15`.
GLM 5.2 still has isolated useful wins, so it is worth mining GLM-only successes, but Gemini is the stronger primary fixed-kernel source for this run pair.

Regenerate with:

```bash
python /home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/compare_glm52_gemini_fixes.py
```
