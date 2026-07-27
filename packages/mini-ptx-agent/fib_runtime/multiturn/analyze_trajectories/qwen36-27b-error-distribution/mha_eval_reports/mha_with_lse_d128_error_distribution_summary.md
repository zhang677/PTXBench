# mha_with_lse_d128 Error Distribution Comparison

- left: `Qwen3.6-27B`
- right: `Qwen3.6-27B-fixit-v2-glm`
- turn_limit: `all`
- paired evals: `1`

## Paired Runs

| definition | left exp_dir | right exp_dir |
| --- | --- | --- |
| `mha_with_lse_d128` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128` |

## Overall Distribution

| correctness | left count | left pct | right count | right pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 26 | 16.25% | +16.25% |
| Compilation error | 90 | 56.25% | 41 | 25.62% | -30.63% |
| Extraction error | 3 | 1.88% | 6 | 3.75% | +1.88% |
| Kernel Execution Timeout | 23 | 14.37% | 7 | 4.38% | -10.00% |
| Numerical error | 11 | 6.88% | 43 | 26.88% | +20.00% |
| Other error | 1 | 0.62% | 0 | 0.00% | -0.62% |
| Profiling Service Timeout | 1 | 0.62% | 0 | 0.00% | -0.62% |
| Runtime error | 31 | 19.38% | 37 | 23.12% | +3.75% |
