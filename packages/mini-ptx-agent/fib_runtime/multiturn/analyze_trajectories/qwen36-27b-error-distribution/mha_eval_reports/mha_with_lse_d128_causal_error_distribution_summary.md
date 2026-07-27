# mha_with_lse_d128_causal Error Distribution Comparison

- left: `Qwen3.6-27B`
- right: `Qwen3.6-27B-fixit-v2-glm`
- turn_limit: `all`
- paired evals: `1`

## Paired Runs

| definition | left exp_dir | right exp_dir |
| --- | --- | --- |
| `mha_with_lse_d128_causal` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128-causal` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128-causal` |

## Overall Distribution

| correctness | left count | left pct | right count | right pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 8 | 5.00% | +5.00% |
| Compilation error | 99 | 61.88% | 46 | 28.75% | -33.13% |
| Extraction error | 6 | 3.75% | 3 | 1.88% | -1.88% |
| Kernel Execution Timeout | 15 | 9.38% | 34 | 21.25% | +11.88% |
| Numerical error | 15 | 9.38% | 43 | 26.88% | +17.50% |
| Other error | 1 | 0.62% | 0 | 0.00% | -0.62% |
| Runtime error | 24 | 15.00% | 26 | 16.25% | +1.25% |
