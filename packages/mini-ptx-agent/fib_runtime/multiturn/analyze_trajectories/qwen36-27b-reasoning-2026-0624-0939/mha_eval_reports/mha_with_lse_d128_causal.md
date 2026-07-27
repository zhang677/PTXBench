# mha_with_lse_d128_causal Reasoning Comparison

- baseline: `Qwen3.6-27B`
- sft: `Qwen3.6-27B-fixit-v2-glm`
- sft run family: `2026-0624-0939`
- turn_limit: `all`
- workload: `6d2f67a7-225a-4af5-87d3-cbb99b496325`
- baseline exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128-causal`
- sft exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128-causal`

## Overall

| metric | baseline | sft | delta |
| --- | ---: | ---: | ---: |
| turn rows | 160 | 160 | 0 |
| correct rate | 0.00% | 5.00% | 5.00% |
| rows with reasoning-token counters | 160 | 160 | 0 |
| mean reasoning tokens | 5799.3 | 24301.1 | 18501.8 |
| mean completion tokens | 12930.9 | 31067.0 | 18136.1 |
| mean provider reasoning chars | 18846.9 | 68712.9 | 49866.0 |
| mean visible content chars | 21141.6 | 17239.6 | -3902.0 |
| mean pre-code chars | 153.3 | 2.0 | -151.3 |
| mean code chars | 18071.6 | 17227.8 | -843.7 |

## Correctness Buckets

| correctness | baseline count | baseline pct | sft count | sft pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 8 | 5.00% | 5.00% |
| Compilation error | 99 | 61.88% | 46 | 28.75% | -33.13% |
| Extraction error | 6 | 3.75% | 3 | 1.88% | -1.88% |
| Kernel Execution Timeout | 15 | 9.38% | 34 | 21.25% | 11.88% |
| Numerical error | 15 | 9.38% | 43 | 26.88% | 17.50% |
| Other error | 1 | 0.62% | 0 | 0.00% | -0.62% |
| Runtime error | 24 | 15.00% | 26 | 16.25% | 1.25% |

## By Turn

| turn | baseline correct | sft correct | delta | baseline mean reasoning tokens | sft mean reasoning tokens | baseline mean provider reasoning chars | sft mean provider reasoning chars |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00% | 0.00% | 0.00% | 13193.4 | 20870.0 | 44809.2 | 59486.2 |
| 1 | 0.00% | 0.00% | 0.00% | 12877.2 | 31495.3 | 38951.9 | 89149.2 |
| 2 | 0.00% | 0.00% | 0.00% | 2691.8 | 27840.8 | 8931.6 | 77972.6 |
| 3 | 0.00% | 0.00% | 0.00% | 4574.4 | 26557.5 | 15521.7 | 74919.9 |
| 4 | 0.00% | 10.00% | 10.00% | 4942.7 | 23648.6 | 16041.2 | 67301.1 |
| 5 | 0.00% | 5.00% | 5.00% | 2593.3 | 21906.2 | 8413.2 | 61728.1 |
| 6 | 0.00% | 10.00% | 10.00% | 3382.0 | 20920.2 | 11059.9 | 59757.7 |
| 7 | 0.00% | 15.00% | 15.00% | 2139.3 | 21170.2 | 7046.7 | 59388.3 |

## Aligned Turns

- aligned baseline/SFT turns: `160`
- SFT improved correctness on aligned turns: `8`
- SFT regressed correctness on aligned turns: `0`

## Most Changed Aligned Turns

| trajectory | turn | baseline correctness | sft correctness | delta reasoning tokens | delta completion tokens | delta code chars |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `exp_016` | 0 | Extraction error | Kernel Execution Timeout | -73455.0 | -66890.0 | 16130.0 |
| `exp_008` | 3 | Compilation error | Compilation error | 64632.0 | 67935.0 | 5429.0 |
| `exp_010` | 1 | Numerical error | Numerical error | -60856.0 | -61237.0 | 15417.0 |
| `exp_008` | 1 | Compilation error | Compilation error | 59420.0 | 57432.0 | -7574.0 |
| `exp_001` | 2 | Extraction error | Extraction error | 57955.0 | 57955.0 | 0.0 |
| `exp_014` | 2 | Compilation error | Compilation error | 55974.0 | 60409.0 | 9287.0 |
| `exp_016` | 1 | Compilation error | Extraction error | -54995.0 | -56444.0 | 0.0 |
| `exp_003` | 6 | Kernel Execution Timeout | Runtime error | 53351.0 | 57346.0 | 9215.0 |
| `exp_016` | 3 | Compilation error | Numerical error | 48471.0 | 48554.0 | -3009.0 |
| `exp_013` | 7 | Kernel Execution Timeout | Kernel Execution Timeout | 48066.0 | 48624.0 | -1678.0 |
