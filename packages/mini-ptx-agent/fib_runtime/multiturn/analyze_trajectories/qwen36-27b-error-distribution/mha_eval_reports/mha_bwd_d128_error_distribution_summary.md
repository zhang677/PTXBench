# mha_bwd_d128 Error Distribution Comparison

- left: `Qwen3.6-27B`
- right: `Qwen3.6-27B-fixit-v2-glm`
- turn_limit: `all`
- paired evals: `1`

## Paired Runs

| definition | left exp_dir | right exp_dir |
| --- | --- | --- |
| `mha_bwd_d128` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128` |

## Overall Distribution

| correctness | left count | left pct | right count | right pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 0 | 0.00% | 4 | 2.50% | +2.50% |
| Compilation error | 79 | 49.38% | 50 | 31.25% | -18.13% |
| Extraction error | 5 | 3.12% | 8 | 5.00% | +1.88% |
| Kernel Execution Timeout | 20 | 12.50% | 24 | 15.00% | +2.50% |
| Numerical error | 12 | 7.50% | 35 | 21.88% | +14.37% |
| Other error | 3 | 1.88% | 0 | 0.00% | -1.88% |
| Runtime error | 41 | 25.62% | 39 | 24.38% | -1.25% |
