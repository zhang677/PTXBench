# mha_bwd_d128 Reasoning Comparison

- baseline: `Qwen3.6-27B`
- sft: `Qwen3.6-27B-fixit-v2-glm`
- sft run family: `2026-0624-0939`
- turn_limit: `all`
- workload: `38c3b07c-f006-5f5e-9860-ba214c805a6b`
- baseline exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128`
- sft exp_dir: `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128`

## Overall

| metric | baseline | sft | delta |
| --- | ---: | ---: | ---: |
| turn rows | 160 | 160 | 0 |
| correct rate | 0.00% | 2.50% | 2.50% |
| rows with reasoning-token counters | 160 | 160 | 0 |
| mean reasoning tokens | 7102.4 | 23952.4 | 16850.0 |
| mean completion tokens | 16914.4 | 33975.8 | 17061.4 |
| mean provider reasoning chars | 22878.8 | 67017.2 | 44138.4 |
| mean visible content chars | 28587.6 | 24849.8 | -3737.8 |
| mean pre-code chars | 226.1 | 1.9 | -224.2 |
| mean code chars | 26650.2 | 24787.4 | -1862.8 |

## Correctness Buckets

| correctness | baseline count | baseline pct | sft count | sft pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 4 | 2.50% | 2.50% |
| Compilation error | 79 | 49.38% | 50 | 31.25% | -18.13% |
| Extraction error | 5 | 3.12% | 8 | 5.00% | 1.88% |
| Kernel Execution Timeout | 20 | 12.50% | 24 | 15.00% | 2.50% |
| Numerical error | 12 | 7.50% | 35 | 21.88% | 14.37% |
| Other error | 3 | 1.88% | 0 | 0.00% | -1.88% |
| Runtime error | 41 | 25.62% | 39 | 24.38% | -1.25% |

## By Turn

| turn | baseline correct | sft correct | delta | baseline mean reasoning tokens | sft mean reasoning tokens | baseline mean provider reasoning chars | sft mean provider reasoning chars |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00% | 0.00% | 0.00% | 33576.9 | 27262.8 | 104739.6 | 76042.1 |
| 1 | 0.00% | 0.00% | 0.00% | 3885.6 | 25538.8 | 14694.8 | 72253.8 |
| 2 | 0.00% | 5.00% | 5.00% | 4031.6 | 26005.7 | 13157.4 | 73070.6 |
| 3 | 0.00% | 0.00% | 0.00% | 1950.5 | 25567.5 | 7106.1 | 72001.5 |
| 4 | 0.00% | 0.00% | 0.00% | 5717.4 | 25222.0 | 18925.1 | 70300.6 |
| 5 | 0.00% | 5.00% | 5.00% | 4533.6 | 18821.8 | 13879.1 | 51895.9 |
| 6 | 0.00% | 5.00% | 5.00% | 1447.8 | 23079.5 | 4958.6 | 64687.8 |
| 7 | 0.00% | 5.00% | 5.00% | 1675.5 | 20120.8 | 5570.0 | 55885.6 |

## Aligned Turns

- aligned baseline/SFT turns: `160`
- SFT improved correctness on aligned turns: `4`
- SFT regressed correctness on aligned turns: `0`

## Most Changed Aligned Turns

| trajectory | turn | baseline correctness | sft correctness | delta reasoning tokens | delta completion tokens | delta code chars |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `exp_002` | 2 | Runtime error | Extraction error | 81848.0 | 75398.0 | -17543.0 |
| `exp_009` | 0 | Extraction error | Compilation error | -76090.0 | -68590.0 | 18978.0 |
| `exp_005` | 1 | Compilation error | Extraction error | 74105.0 | 71486.0 | -6935.0 |
| `exp_018` | 0 | Compilation error | Compilation error | 71811.0 | 0.0 | -194320.0 |
| `exp_005` | 0 | Extraction error | Compilation error | -67849.0 | -57228.0 | 26756.0 |
| `exp_004` | 0 | Extraction error | Compilation error | -66765.0 | -54165.0 | 30524.0 |
| `exp_015` | 4 | Runtime error | Extraction error | 65795.0 | 55837.0 | -28296.0 |
| `exp_012` | 2 | Compilation error | Compilation error | 65269.0 | 65262.0 | -5393.0 |
| `exp_001` | 0 | Compilation error | Compilation error | -65251.0 | -56874.0 | 21916.0 |
| `exp_008` | 4 | Runtime error | Extraction error | 56941.0 | 49976.0 | -20509.0 |
