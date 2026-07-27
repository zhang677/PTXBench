# Qwen3.6-27B Error Distribution Comparison

- left: `Qwen3.6-27B`
- right: `Qwen3.6-27B-fixit-v2-glm`
- turn_limit: `all`
- paired evals: `5`

## Paired Runs

| definition | left exp_dir | right exp_dir |
| --- | --- | --- |
| `gemm_n7168_k5120` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0524-1345` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-gemm` |
| `mha_bwd_d128` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128` |
| `mha_bwd_d128_causal` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128-causal` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128-causal` |
| `mha_with_lse_d128` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128` |
| `mha_with_lse_d128_causal` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128-causal` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128-causal` |

## Overall Distribution

| correctness | left count | left pct | right count | right pct | delta pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct | 19 | 1.58% | 70 | 8.75% | +7.17% |
| Compilation error | 475 | 39.58% | 210 | 26.25% | -13.33% |
| Extraction error | 22 | 1.83% | 27 | 3.38% | +1.54% |
| Kernel Execution Timeout | 117 | 9.75% | 84 | 10.50% | +0.75% |
| Numerical error | 179 | 14.92% | 159 | 19.88% | +4.96% |
| Other error | 9 | 0.75% | 0 | 0.00% | -0.75% |
| Profiling Service Timeout | 1 | 0.08% | 0 | 0.00% | -0.08% |
| Runtime error | 377 | 31.42% | 250 | 31.25% | -0.17% |
| Sanitize Timeout | 1 | 0.08% | 0 | 0.00% | -0.08% |

## Outputs

- distribution_csv: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-error-distribution/error_distribution.csv`
- error_rows_csv: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-error-distribution/error_rows.csv`
- summary_md: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-error-distribution/error_distribution_summary.md`
- manifest_json: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-error-distribution/manifest.json`
