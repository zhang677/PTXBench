# Qwen3.6-27B linfo vs qwen36 MHA error types

Generated: 2026-07-06 03:14:37 UTC

Rows are read from each run's `figures/turn_correctness_arch.csv`.

Plot: `qwen36_linfo_vs_qwen36_mha_error_types.png`

Long CSV: `error_distribution_by_problem.csv`

Wide CSV: `error_distribution_summary.csv`

## Aggregate over the four MHA problems

| Condition | Total | Compilation error | Runtime error | Kernel Execution Timeout | Numerical error | Extraction error | Profiling Service Timeout | Sanitize Timeout | Other error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen36-27b-linfo | 640 | 340 (53.1%) | 268 (41.9%) | 14 (2.2%) | 9 (1.4%) | 6 (0.9%) | 0 (0.0%) | 0 (0.0%) | 3 (0.5%) |
| qwen36-27b | 640 | 366 (57.2%) | 117 (18.3%) | 74 (11.6%) | 52 (8.1%) | 21 (3.3%) | 1 (0.2%) | 1 (0.2%) | 8 (1.2%) |

## linfo minus qwen36-27b delta

| Correctness | linfo | qwen36-27b | count_delta | fraction_delta |
| --- | --- | --- | --- | --- |
| Compilation error | 340 | 366 | -26 | -4.1 pp |
| Runtime error | 268 | 117 | 151 | +23.6 pp |
| Kernel Execution Timeout | 14 | 74 | -60 | -9.4 pp |
| Numerical error | 9 | 52 | -43 | -6.7 pp |
| Extraction error | 6 | 21 | -15 | -2.3 pp |
| Profiling Service Timeout | 0 | 1 | -1 | -0.2 pp |
| Sanitize Timeout | 0 | 1 | -1 | -0.2 pp |
| Other error | 3 | 8 | -5 | -0.8 pp |

## Per-problem key counts

| Problem | Condition | Total | Compilation | Runtime | Kernel Timeout | Numerical | Correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MHA d128 | qwen36-27b-linfo | 160 | 80 | 74 | 2 | 3 | 0 |
| MHA d128 | qwen36-27b | 160 | 90 | 31 | 23 | 11 | 0 |
| MHA d128 causal | qwen36-27b-linfo | 160 | 89 | 64 | 2 | 2 | 0 |
| MHA d128 causal | qwen36-27b | 160 | 99 | 24 | 15 | 15 | 0 |
| MHA bwd d128 | qwen36-27b-linfo | 160 | 90 | 61 | 3 | 2 | 0 |
| MHA bwd d128 | qwen36-27b | 160 | 79 | 41 | 20 | 12 | 0 |
| MHA bwd d128 causal | qwen36-27b-linfo | 160 | 81 | 69 | 7 | 2 | 0 |
| MHA bwd d128 causal | qwen36-27b | 160 | 98 | 21 | 16 | 14 | 0 |

## Source CSVs

| Problem | Condition | CSV |
| --- | --- | --- |
| MHA d128 | qwen36-27b-linfo | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d128/figures/turn_correctness_arch.csv |
| MHA d128 | qwen36-27b | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128/figures/turn_correctness_arch.csv |
| MHA d128 causal | qwen36-27b-linfo | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d128-causal/figures/turn_correctness_arch.csv |
| MHA d128 causal | qwen36-27b | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128-causal/figures/turn_correctness_arch.csv |
| MHA bwd d128 | qwen36-27b-linfo | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d128/figures/turn_correctness_arch.csv |
| MHA bwd d128 | qwen36-27b | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128/figures/turn_correctness_arch.csv |
| MHA bwd d128 causal | qwen36-27b-linfo | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d128-causal/figures/turn_correctness_arch.csv |
| MHA bwd d128 causal | qwen36-27b | /home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128-causal/figures/turn_correctness_arch.csv |

