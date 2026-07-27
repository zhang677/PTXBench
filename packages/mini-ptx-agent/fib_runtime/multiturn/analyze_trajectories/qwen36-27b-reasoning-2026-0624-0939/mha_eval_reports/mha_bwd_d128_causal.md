# mha_bwd_d128_causal Reasoning Comparison

- baseline: `Qwen3.6-27B`
- sft: `Qwen3.6-27B-fixit-v2-glm`
- sft run family: `2026-0624-0939`
- turn_limit: `all`
- workload: `c119b3f0-c051-5e96-9c2a-2268d992fe1a`
- baseline exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128-causal`
- sft exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128-causal`

## Overall

| metric | baseline | sft | delta |
| --- | ---: | ---: | ---: |
| turn rows | 160 | 160 | 0 |
| correct rate | 0.00% | 3.75% | 3.75% |
| rows with reasoning-token counters | 160 | 160 | 0 |
| mean reasoning tokens | 5971.3 | 24581.5 | 18610.2 |
| mean completion tokens | 15153.1 | 34140.7 | 18987.6 |
| mean provider reasoning chars | 18751.3 | 69738.3 | 50986.9 |
| mean visible content chars | 26850.7 | 24144.8 | -2705.8 |
| mean pre-code chars | 218.7 | 1.9 | -216.8 |
| mean code chars | 24880.7 | 23992.3 | -888.4 |

## Correctness Buckets

| correctness | baseline count | baseline pct | sft count | sft pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 6 | 3.75% | 3.75% |
| Compilation error | 84 | 52.50% | 49 | 30.63% | -21.88% |
| Extraction error | 5 | 3.12% | 4 | 2.50% | -0.62% |
| Kernel Execution Timeout | 7 | 4.38% | 12 | 7.50% | 3.12% |
| Numerical error | 10 | 6.25% | 31 | 19.38% | 13.12% |
| Other error | 44 | 27.50% | 0 | 0.00% | -27.50% |
| Runtime error | 9 | 5.62% | 58 | 36.25% | 30.62% |
| Sanitize Timeout | 1 | 0.62% | 0 | 0.00% | -0.62% |

## By Turn

| turn | baseline correct | sft correct | delta | baseline mean reasoning tokens | sft mean reasoning tokens | baseline mean provider reasoning chars | sft mean provider reasoning chars |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00% | 0.00% | 0.00% | 22209.1 | 37762.3 | 67837.1 | 108064.9 |
| 1 | 0.00% | 0.00% | 0.00% | 3307.4 | 23482.8 | 11020.1 | 65952.4 |
| 2 | 0.00% | 5.00% | 5.00% | 7216.8 | 23848.5 | 22388.2 | 67023.4 |
| 3 | 0.00% | 5.00% | 5.00% | 6494.6 | 21566.5 | 19502.0 | 61827.2 |
| 4 | 0.00% | 5.00% | 5.00% | 4093.6 | 23937.7 | 12855.0 | 67580.0 |
| 5 | 0.00% | 5.00% | 5.00% | 1700.1 | 25761.8 | 6514.8 | 71833.1 |
| 6 | 0.00% | 0.00% | 0.00% | 1242.8 | 20484.4 | 4649.2 | 58190.3 |
| 7 | 0.00% | 10.00% | 10.00% | 1506.2 | 19808.2 | 5244.1 | 57434.8 |

## Aligned Turns

- aligned baseline/SFT turns: `160`
- SFT improved correctness on aligned turns: `6`
- SFT regressed correctness on aligned turns: `0`

## Most Changed Aligned Turns

| trajectory | turn | baseline correctness | sft correctness | delta reasoning tokens | delta completion tokens | delta code chars |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `exp_010` | 0 | Compilation error | Extraction error | 80326.0 | 0.0 | -222487.0 |
| `exp_011` | 2 | Compilation error | Compilation error | 76201.0 | 79007.0 | -8593.0 |
| `exp_008` | 0 | Compilation error | Compilation error | 72544.0 | 22657.0 | -154480.0 |
| `exp_004` | 0 | Compilation error | Compilation error | 67518.0 | 41574.0 | -81708.0 |
| `exp_003` | 0 | Other error | Compilation error | 63426.0 | 64755.0 | 479.0 |
| `exp_011` | 4 | Other error | Runtime error | 59003.0 | 68245.0 | 23608.0 |
| `exp_000` | 0 | Extraction error | Compilation error | -54698.0 | -46015.0 | 21149.0 |
| `exp_016` | 7 | Other error | Runtime error | 52591.0 | 55116.0 | 3152.0 |
| `exp_019` | 6 | Compilation error | Kernel Execution Timeout | 51595.0 | 56150.0 | 10727.0 |
| `exp_012` | 0 | Extraction error | Compilation error | -51039.0 | -41443.0 | 23552.0 |
