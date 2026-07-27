# mha_bwd_d128_causal Error Distribution Comparison

- left: `Qwen3.6-27B`
- right: `Qwen3.6-27B-fixit-v2-glm`
- turn_limit: `all`
- paired evals: `1`

## Paired Runs

| definition | left exp_dir | right exp_dir |
| --- | --- | --- |
| `mha_bwd_d128_causal` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128-causal` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128-causal` |

## Overall Distribution

| correctness | left count | left pct | right count | right pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 6 | 3.75% | +3.75% |
| Compilation error | 98 | 61.25% | 49 | 30.63% | -30.63% |
| Extraction error | 7 | 4.38% | 4 | 2.50% | -1.87% |
| Kernel Execution Timeout | 16 | 10.00% | 12 | 7.50% | -2.50% |
| Numerical error | 14 | 8.75% | 31 | 19.38% | +10.63% |
| Other error | 3 | 1.88% | 0 | 0.00% | -1.88% |
| Runtime error | 21 | 13.12% | 58 | 36.25% | +23.12% |
| Sanitize Timeout | 1 | 0.62% | 0 | 0.00% | -0.62% |
