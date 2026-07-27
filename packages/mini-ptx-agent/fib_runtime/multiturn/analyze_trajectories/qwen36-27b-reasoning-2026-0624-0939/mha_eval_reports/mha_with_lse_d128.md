# mha_with_lse_d128 Reasoning Comparison

- baseline: `Qwen3.6-27B`
- sft: `Qwen3.6-27B-fixit-v2-glm`
- sft run family: `2026-0624-0939`
- turn_limit: `all`
- workload: `bc38b351-d595-451b-9153-8e225702e53b`
- baseline exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128`
- sft exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128`

## Overall

| metric | baseline | sft | delta |
| --- | ---: | ---: | ---: |
| turn rows | 160 | 160 | 0 |
| correct rate | 0.00% | 16.25% | 16.25% |
| rows with reasoning-token counters | 160 | 160 | 0 |
| mean reasoning tokens | 6076.9 | 22687.9 | 16610.9 |
| mean completion tokens | 14475.0 | 29211.8 | 14736.9 |
| mean provider reasoning chars | 19755.4 | 64078.1 | 44322.7 |
| mean visible content chars | 24474.6 | 16422.6 | -8052.0 |
| mean pre-code chars | 116.5 | 1.9 | -114.6 |
| mean code chars | 21271.9 | 16411.1 | -4860.8 |

## Correctness Buckets

| correctness | baseline count | baseline pct | sft count | sft pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 26 | 16.25% | 16.25% |
| Compilation error | 90 | 56.25% | 41 | 25.62% | -30.63% |
| Extraction error | 3 | 1.88% | 6 | 3.75% | 1.88% |
| Kernel Execution Timeout | 23 | 14.37% | 7 | 4.38% | -10.00% |
| Numerical error | 11 | 6.88% | 43 | 26.88% | 20.00% |
| Other error | 1 | 0.62% | 0 | 0.00% | -0.62% |
| Profiling Service Timeout | 1 | 0.62% | 0 | 0.00% | -0.62% |
| Runtime error | 31 | 19.38% | 37 | 23.12% | 3.75% |

## By Turn

| turn | baseline correct | sft correct | delta | baseline mean reasoning tokens | sft mean reasoning tokens | baseline mean provider reasoning chars | sft mean provider reasoning chars |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00% | 10.00% | 10.00% | 17086.0 | 27372.7 | 54947.0 | 76070.4 |
| 1 | 0.00% | 5.00% | 5.00% | 7675.7 | 27050.8 | 23858.8 | 76796.6 |
| 2 | 0.00% | 25.00% | 25.00% | 4128.4 | 20740.8 | 14682.1 | 58777.7 |
| 3 | 0.00% | 10.00% | 10.00% | 4532.1 | 24511.2 | 14818.8 | 69980.6 |
| 4 | 0.00% | 20.00% | 20.00% | 4419.3 | 24581.3 | 14209.0 | 69358.1 |
| 5 | 0.00% | 15.00% | 15.00% | 4096.4 | 16019.9 | 13148.6 | 44990.4 |
| 6 | 0.00% | 20.00% | 20.00% | 2555.6 | 21305.3 | 8550.9 | 60673.2 |
| 7 | 0.00% | 25.00% | 25.00% | 4122.1 | 19920.9 | 13827.5 | 55977.7 |

## Aligned Turns

- aligned baseline/SFT turns: `160`
- SFT improved correctness on aligned turns: `26`
- SFT regressed correctness on aligned turns: `0`

## Most Changed Aligned Turns

| trajectory | turn | baseline correctness | sft correctness | delta reasoning tokens | delta completion tokens | delta code chars |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `exp_019` | 1 | Extraction error | Compilation error | -73411.0 | -66897.0 | 15676.0 |
| `exp_002` | 3 | Compilation error | Extraction error | 66925.0 | 61267.0 | -16833.0 |
| `exp_014` | 3 | Runtime error | Compilation error | 66426.0 | 66361.0 | -3675.0 |
| `exp_003` | 4 | Numerical error | Runtime error | 60500.0 | 63568.0 | 5535.0 |
| `exp_018` | 1 | Compilation error | Numerical error | 60072.0 | 20051.0 | -88221.0 |
| `exp_016` | 6 | Compilation error | Runtime error | 58762.0 | 64215.0 | 11518.0 |
| `exp_019` | 0 | Extraction error | Extraction error | -55949.0 | -55949.0 | 0.0 |
| `exp_004` | 1 | Compilation error | Compilation error | 55448.0 | 57178.0 | 2118.0 |
| `exp_016` | 0 | Extraction error | Compilation error | -54302.0 | -47504.0 | 16523.0 |
| `exp_018` | 2 | Compilation error | Numerical error | 53633.0 | 54101.0 | -1564.0 |
