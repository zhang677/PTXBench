# Qwen3.6-27B Reasoning Comparison

- baseline: `Qwen3.6-27B`
- sft: `Qwen3.6-27B-fixit-v2-glm`
- sft run family: `2026-0624-0939`
- turn_limit: `all`
- registry-matched rows: `5`

## Paired Registry Rows

| definition | workload | baseline exp_dir | sft exp_dir |
| --- | --- | --- | --- |
| `gemm_n7168_k5120` | `94920358-01a8-4c5b-9209-3103fd490e94` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0524-1345` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-gemm` |
| `mha_bwd_d128` | `38c3b07c-f006-5f5e-9860-ba214c805a6b` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128` |
| `mha_bwd_d128_causal` | `c119b3f0-c051-5e96-9c2a-2268d992fe1a` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128-causal` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128-causal` |
| `mha_with_lse_d128` | `bc38b351-d595-451b-9153-8e225702e53b` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128` |
| `mha_with_lse_d128_causal` | `6d2f67a7-225a-4af5-87d3-cbb99b496325` | `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128-causal` | `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-d128-causal` |

## Overall

| metric | baseline | sft | delta |
| --- | ---: | ---: | ---: |
| turn rows | 1200 | 800 | -400 |
| correct rate | 1.58% | 8.75% | 7.17% |
| rows with reasoning-token counters | 640 | 800 | 160 |
| mean reasoning tokens | 6237.5 | 24035.7 | 17798.2 |
| mean completion tokens | 12053.6 | 31499.4 | 19445.8 |
| mean provider reasoning chars | 17837.6 | 67911.7 | 50074.1 |
| mean visible content chars | 18224.8 | 18827.9 | 603.1 |
| mean pre-code chars | 167.3 | 1.9 | -165.4 |
| mean code chars | 16790.6 | 18778.0 | 1987.4 |

## Aligned Turns

- aligned baseline/SFT turns: `800`
- SFT improved correctness on aligned turns: `69`
- SFT regressed correctness on aligned turns: `2`

## Outputs

- per_turn_csv: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-reasoning-2026-0624-0939/reasoning_turns.csv`
- aggregate_csv: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-reasoning-2026-0624-0939/reasoning_aggregates.csv`
- aligned_delta_csv: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-reasoning-2026-0624-0939/aligned_turn_deltas.csv`
- paired_runs_csv: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-reasoning-2026-0624-0939/paired_registry_rows.csv`
- mha_reports_dir: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-reasoning-2026-0624-0939/mha_eval_reports`
- summary_md: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-reasoning-2026-0624-0939/reasoning_summary.md`
- manifest_json: `/home/ubuntu/AccRL/fib_runtime/multiturn/analyze_trajectories/qwen36-27b-reasoning-2026-0624-0939/manifest.json`
