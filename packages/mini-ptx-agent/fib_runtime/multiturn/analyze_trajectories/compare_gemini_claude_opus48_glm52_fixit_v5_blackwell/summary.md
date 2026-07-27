# Blackwell: Gemini vs Claude Opus 4.8 vs GLM-5.2 vs Fixit v5

Source of truth: each run's `figures/turn_correctness_arch.csv`, joined to `plan.json` for definition and prompt-tag metadata.
Prompt tags are kept exactly as recorded in `plan.json`. For cross-group matching only, a trailing `-mha-patched` suffix is ignored.
Rows retain their original turn numbers; turns are not intersected or trimmed across groups.

Stages:

- `gemini-3.1-pro-preview`
- `claude-opus-4.8-xhigh`
- `glm-5.2`
- `fixit-v5`

## Blackwell Prompt-Tag Rules

Comparison rows are selected from these configs:

- `/home/ubuntu/AccRL-exps/prompt_configs/b200-gemm-3-r8-p4.json`
- `/home/ubuntu/AccRL-exps/prompt_configs/b200-mha-3-r8-p4.json`
- `/home/ubuntu/AccRL-exps/prompt_configs/b200-mha-bwd-3-r8-p4.json`

GLM-5.2 and Fixit v5 must cover all configured B200 tags. Gemini and Claude use the same raw, unpatched tag names and may omit configured tags; their omissions do not reduce GLM-5.2 or Fixit v5 coverage.

## Data Notes

- No Claude Opus 4.7 Blackwell artifacts exist in the repository; the available registered family is Claude Opus 4.8 xhigh (`anthropic/claude-opus-4-8`).
- The 15 comparison pairs come from the B200 GEMM, forward-MHA, and backward-MHA prompt configs. Gemini and Claude may omit configured tags without reducing GLM-5.2 or Fixit v5 coverage.
- Rows retain their original turn numbers and trajectory budgets are not equalized across groups.

## Run Roots

### gemini-3.1-pro-preview
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-1900-complete`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-1040`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0040`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0240`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0513-1125`

### claude-opus-4.8-xhigh
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-1540-complete`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-1740`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-1940`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-2140`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0601-2320`

### glm-5.2
- `/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-mha-bwd-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/glm52-b200-gemm`

### fixit-v5
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-mha-bwd-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0712-0058-b200-gemm`

All source rows loaded: 1357
Configured definitions: 5 (`gemm_n7168_k5120`, `mha_bwd_d128`, `mha_bwd_d128_causal`, `mha_with_lse_d128`, `mha_with_lse_d128_causal`)
Configured `(definition, prompt_tag)` pairs: 15
Configured rows after filtering: 1277
Comparison rows at original turn numbers: 1277

CSV output:

- `holistic_turns.csv`

Turn-transition alluvial outputs:

- `turn_transition_alluvial_index.md`
- `turn_transition_alluvial_by_definition/`

## Source Coverage

| group | n_runs | n_definitions | n_prompt_pairs | n_trajectories | n_turns | definitions |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 5 | 5 | 17 | 29 | 232 | gemm_n7168_k5120, mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |
| claude-opus-4.8-xhigh | 5 | 5 | 19 | 22 | 176 | gemm_n7168_k5120, mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |
| glm-5.2 | 5 | 5 | 15 | 60 | 480 | gemm_n7168_k5120, mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |
| fixit-v5 | 5 | 5 | 15 | 60 | 469 | gemm_n7168_k5120, mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |

## Configured-Prompt Overall Metrics

All metric triplets are ordered `≤1 / ≤4 / ≤8` turns.

| group | parquet_rows | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | N/A | 25 | 200 | 2 / 16 / 47 | 0.080000 / 0.160000 / 0.235000 | 0.040000 / 0.150000 / 0.230000 | 0.080000 / 0.360000 / 0.600000 | 0.040000 / 0.320000 / 0.560000 | 0.055324 / 0.668624 / 0.824753 | 0.020792 / 0.181872 / 0.231884 |
| claude-opus-4.8-xhigh | N/A | 16 | 128 | 5 / 32 / 91 | 0.312500 / 0.500000 / 0.710938 | 0.312500 / 0.437500 / 0.554688 | 0.312500 / 0.875000 / 1.000000 | 0.312500 / 0.687500 / 0.750000 | 0.067305 / 0.815204 / 0.946941 | 0.015832 / 0.100509 / 0.152039 |
| glm-5.2 | N/A | 60 | 480 | 15 / 61 / 133 | 0.250000 / 0.254167 / 0.277083 | 0.050000 / 0.058333 / 0.060417 | 0.250000 / 0.650000 / 0.800000 | 0.050000 / 0.116667 / 0.183333 | 0.162367 / 0.632160 / 0.632160 | 0.021244 / 0.032274 / 0.033382 |
| fixit-v5 | 170 | 60 | 469 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

## Configured-Definition Collective Metrics

Each table pools all configured prompt tags available for one problem and group. Metric triplets remain ordered `≤1 / ≤4 / ≤8` turns.

### `gemm_n7168_k5120`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 4 | 32 | 0 / 6 / 16 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.750000 / 1.000000 | 0.0 / 0.750000 / 1.000000 | 0.0 / 0.668624 / 0.824753 | 0.0 / 0.446163 / 0.540524 |
| claude-opus-4.8-xhigh | 4 | 32 | 0 / 8 / 24 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.500000 / 0.750000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.815204 / 0.946941 | 0.0 / 0.447741 / 0.644038 |
| glm-5.2 | 12 | 96 | 4 / 18 / 33 | 0.333333 / 0.375000 / 0.343750 | 0.166667 / 0.270833 / 0.281250 | 0.333333 / 0.833333 / 0.916667 | 0.166667 / 0.500000 / 0.750000 | 0.162367 / 0.632160 / 0.632160 | 0.066442 / 0.151729 / 0.189143 |
| fixit-v5 | 12 | 96 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### `mha_bwd_d128`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 3 | 24 | 0 / 0 / 2 | 0.0 / 0.0 / 0.083333 | 0.0 / 0.0 / 0.083333 | 0.0 / 0.0 / 0.333333 | 0.0 / 0.0 / 0.333333 | 0.0 / 0.0 / 0.280429 | 0.0 / 0.0 / 0.271434 |
| claude-opus-4.8-xhigh | 3 | 24 | 0 / 2 / 11 | 0.0 / 0.166667 / 0.458333 | 0.0 / 0.083333 / 0.250000 | 0.0 / 0.666667 / 1.000000 | 0.0 / 0.333333 / 0.666667 | 0.0 / 0.075695 / 0.181726 | 0.0 / 0.042292 / 0.099531 |
| glm-5.2 | 12 | 96 | 1 / 12 / 29 | 0.083333 / 0.250000 / 0.302083 | 0.0 / 0.0 / 0.0 | 0.083333 / 0.500000 / 0.666667 | 0.0 / 0.0 / 0.0 | 0.018537 / 0.029868 / 0.039573 | 0.018537 / 0.022385 / 0.022857 |
| fixit-v5 | 12 | 93 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### `mha_bwd_d128_causal`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 3 | 24 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| claude-opus-4.8-xhigh | 3 | 24 | 0 / 3 / 15 | 0.0 / 0.250000 / 0.625000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.666667 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.025753 / 0.103088 | 0.0 / 0.020622 / 0.034287 |
| glm-5.2 | 12 | 96 | 2 / 9 / 22 | 0.166667 / 0.187500 / 0.229167 | 0.0 / 0.0 / 0.0 | 0.166667 / 0.500000 / 0.750000 | 0.0 / 0.0 / 0.0 | 0.018471 / 0.034209 / 0.034209 | 0.015313 / 0.016339 / 0.017059 |
| fixit-v5 | 12 | 96 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### `mha_with_lse_d128`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 12 | 96 | 1 / 8 / 25 | 0.083333 / 0.166667 / 0.260417 | 0.0 / 0.145833 / 0.250000 | 0.083333 / 0.416667 / 0.666667 | 0.0 / 0.333333 / 0.583333 | 0.055324 / 0.318255 / 0.377801 | 0.055324 / 0.146838 / 0.160560 |
| claude-opus-4.8-xhigh | 3 | 24 | 2 / 8 / 19 | 0.666667 / 0.666667 / 0.791667 | 0.666667 / 0.666667 / 0.791667 | 0.666667 / 1.000000 / 1.000000 | 0.666667 / 1.000000 / 1.000000 | 0.014745 / 0.228556 / 0.339425 | 0.011159 / 0.073287 / 0.143046 |
| glm-5.2 | 12 | 96 | 4 / 11 / 25 | 0.333333 / 0.229167 / 0.260417 | 0.0 / 0.0 / 0.010417 | 0.333333 / 0.750000 / 0.916667 | 0.0 / 0.0 / 0.083333 | 0.024435 / 0.024435 / 0.034888 | 0.013724 / 0.013629 / 0.016207 |
| fixit-v5 | 12 | 93 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### `mha_with_lse_d128_causal`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.1-pro-preview | 3 | 24 | 1 / 2 / 4 | 0.333333 / 0.166667 / 0.166667 | 0.333333 / 0.166667 / 0.166667 | 0.333333 / 0.333333 / 0.666667 | 0.333333 / 0.333333 / 0.666667 | 0.007814 / 0.107582 / 0.190412 | 0.007814 / 0.028994 / 0.072209 |
| claude-opus-4.8-xhigh | 3 | 24 | 3 / 11 / 22 | 1.000000 / 0.916667 / 0.916667 | 1.000000 / 0.916667 / 0.916667 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 0.067305 / 0.249984 / 0.296217 | 0.019991 / 0.076923 / 0.113212 |
| glm-5.2 | 12 | 96 | 4 / 11 / 24 | 0.333333 / 0.229167 / 0.250000 | 0.083333 / 0.020833 / 0.010417 | 0.333333 / 0.666667 / 0.750000 | 0.083333 / 0.083333 / 0.083333 | 0.097977 / 0.097977 / 0.097977 | 0.012814 / 0.015793 / 0.019085 |
| fixit-v5 | 12 | 91 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

## Configured Definition / Prompt-Tag Rows

| definition | prompt_tag | group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemm_n7168_k5120 | b200-bf16 | claude-opus-4.8-xhigh | 4 | 32 | 0 / 8 / 24 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.500000 / 0.750000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.815204 / 0.946941 | 0.0 / 0.447741 / 0.644038 |
| gemm_n7168_k5120 | b200-bf16 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| gemm_n7168_k5120 | b200-bf16 | gemini-3.1-pro-preview | 4 | 32 | 0 / 6 / 16 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.750000 / 1.000000 | 0.0 / 0.750000 / 1.000000 | 0.0 / 0.668624 / 0.824753 | 0.0 / 0.446163 / 0.540524 |
| gemm_n7168_k5120 | b200-bf16 | glm-5.2 | 4 | 32 | 4 / 9 / 13 | 1.000000 / 0.562500 / 0.406250 | 0.500000 / 0.437500 / 0.312500 | 1.000000 / 1.000000 / 1.000000 | 0.500000 / 0.500000 / 0.500000 | 0.162367 / 0.632160 / 0.632160 | 0.066442 / 0.190712 / 0.219257 |
| gemm_n7168_k5120 | b200-bf16-00 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| gemm_n7168_k5120 | b200-bf16-00 | glm-5.2 | 4 | 32 | 0 / 4 / 9 | 0.0 / 0.250000 / 0.281250 | 0.0 / 0.187500 / 0.250000 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.358236 / 0.548841 | 0.0 / 0.107887 / 0.183904 |
| gemm_n7168_k5120 | b200-bf16-04 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| gemm_n7168_k5120 | b200-bf16-04 | glm-5.2 | 4 | 32 | 0 / 5 / 11 | 0.0 / 0.312500 / 0.343750 | 0.0 / 0.187500 / 0.281250 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.500000 / 1.000000 | 0.0 / 0.385884 / 0.570779 | 0.0 / 0.132067 / 0.162532 |
| mha_bwd_d128 | b200-bf16 | claude-opus-4.8-xhigh | 1 | 8 | 0 / 1 / 3 | 0.0 / 0.250000 / 0.375000 | 0.0 / 0.250000 / 0.375000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.023629 / 0.181726 | 0.0 / 0.023629 / 0.089669 |
| mha_bwd_d128 | b200-bf16 | fixit-v5 | 4 | 29 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | b200-bf16 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | b200-bf16 | glm-5.2 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | b200-bf16-012 | claude-opus-4.8-xhigh | 1 | 8 | 0 / 0 / 3 | 0.0 / 0.0 / 0.375000 | 0.0 / 0.0 / 0.375000 | 0.0 / 0.0 / 1.000000 | 0.0 / 0.0 / 1.000000 | 0.0 / 0.0 / 0.086374 | 0.0 / 0.0 / 0.080365 |
| mha_bwd_d128 | b200-bf16-012 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | b200-bf16-012 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 2 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 1.000000 | 0.0 / 0.0 / 1.000000 | 0.0 / 0.0 / 0.280429 | 0.0 / 0.0 / 0.271434 |
| mha_bwd_d128 | b200-bf16-012 | glm-5.2 | 4 | 32 | 1 / 4 / 12 | 0.250000 / 0.250000 / 0.375000 | 0.0 / 0.0 / 0.0 | 0.250000 / 0.500000 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.018537 / 0.026479 / 0.034733 | 0.018537 / 0.020198 / 0.020750 |
| mha_bwd_d128 | b200-bf16-013 | claude-opus-4.8-xhigh | 1 | 8 | 0 / 1 / 5 | 0.0 / 0.250000 / 0.625000 | 0.0 / 0.0 / 0.0 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.075695 / 0.157344 | 0.0 / 0.075695 / 0.120471 |
| mha_bwd_d128 | b200-bf16-013 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | b200-bf16-013 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | b200-bf16-013 | glm-5.2 | 4 | 32 | 0 / 8 / 17 | 0.0 / 0.500000 / 0.531250 | 0.0 / 0.0 / 0.0 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.029868 / 0.039573 | 0.0 / 0.023565 / 0.024472 |
| mha_bwd_d128_causal | b200-bf16 | claude-opus-4.8-xhigh | 1 | 8 | 0 / 1 / 5 | 0.0 / 0.250000 / 0.625000 | 0.0 / 0.0 / 0.0 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.025753 / 0.069507 | 0.0 / 0.025753 / 0.028917 |
| mha_bwd_d128_causal | b200-bf16 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | b200-bf16 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | b200-bf16 | glm-5.2 | 4 | 32 | 0 / 2 / 4 | 0.0 / 0.125000 / 0.125000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.014806 / 0.017734 | 0.0 / 0.009383 / 0.010199 |
| mha_bwd_d128_causal | b200-bf16-012 | claude-opus-4.8-xhigh | 1 | 8 | 0 / 0 / 4 | 0.0 / 0.0 / 0.500000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.030896 | 0.0 / 0.0 / 0.029100 |
| mha_bwd_d128_causal | b200-bf16-012 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | b200-bf16-012 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | b200-bf16-012 | glm-5.2 | 4 | 32 | 1 / 6 / 11 | 0.250000 / 0.375000 / 0.343750 | 0.0 / 0.0 / 0.0 | 0.250000 / 0.750000 / 0.750000 | 0.0 / 0.0 / 0.0 | 0.018471 / 0.034209 / 0.034209 | 0.018471 / 0.020502 / 0.020731 |
| mha_bwd_d128_causal | b200-bf16-013 | claude-opus-4.8-xhigh | 1 | 8 | 0 / 2 / 6 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.0 / 0.0 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.019603 / 0.103088 | 0.0 / 0.018454 / 0.044082 |
| mha_bwd_d128_causal | b200-bf16-013 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | b200-bf16-013 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | b200-bf16-013 | glm-5.2 | 4 | 32 | 1 / 1 / 7 | 0.250000 / 0.062500 / 0.218750 | 0.0 / 0.0 / 0.0 | 0.250000 / 0.250000 / 0.750000 | 0.0 / 0.0 / 0.0 | 0.012695 / 0.012695 / 0.033038 | 0.012695 / 0.012695 / 0.016847 |
| mha_with_lse_d128 | b200-bf16 | claude-opus-4.8-xhigh | 1 | 8 | 1 / 3 / 6 | 1.000000 / 0.750000 / 0.750000 | 1.000000 / 0.750000 / 0.750000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 0.014745 / 0.228556 / 0.254516 | 0.014745 / 0.080864 / 0.142460 |
| mha_with_lse_d128 | b200-bf16 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | b200-bf16 | gemini-3.1-pro-preview | 4 | 32 | 1 / 2 / 9 | 0.250000 / 0.125000 / 0.281250 | 0.0 / 0.062500 / 0.250000 | 0.250000 / 0.500000 / 0.750000 | 0.0 / 0.250000 / 0.500000 | 0.055324 / 0.171151 / 0.260724 | 0.055324 / 0.097307 / 0.162775 |
| mha_with_lse_d128 | b200-bf16 | glm-5.2 | 4 | 32 | 1 / 4 / 6 | 0.250000 / 0.250000 / 0.187500 | 0.0 / 0.0 / 0.0 | 0.250000 / 0.750000 / 0.750000 | 0.0 / 0.0 / 0.0 | 0.010218 / 0.022003 / 0.034766 | 0.010218 / 0.012529 / 0.014809 |
| mha_with_lse_d128 | b200-bf16-010 | claude-opus-4.8-xhigh | 1 | 8 | 0 / 2 / 6 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.500000 / 0.750000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.198464 / 0.339425 | 0.0 / 0.085510 / 0.181602 |
| mha_with_lse_d128 | b200-bf16-010 | fixit-v5 | 4 | 29 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | b200-bf16-010 | gemini-3.1-pro-preview | 4 | 32 | 0 / 4 / 10 | 0.0 / 0.250000 / 0.312500 | 0.0 / 0.250000 / 0.312500 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.318255 / 0.377801 | 0.0 / 0.155658 / 0.165575 |
| mha_with_lse_d128 | b200-bf16-010 | glm-5.2 | 4 | 32 | 2 / 5 / 9 | 0.500000 / 0.312500 / 0.281250 | 0.0 / 0.0 / 0.031250 | 0.500000 / 1.000000 / 1.000000 | 0.0 / 0.0 / 0.250000 | 0.012893 / 0.012893 / 0.026690 | 0.011920 / 0.011659 / 0.012690 |
| mha_with_lse_d128 | b200-bf16-011 | claude-opus-4.8-xhigh | 1 | 8 | 1 / 3 / 7 | 1.000000 / 0.750000 / 0.875000 | 1.000000 / 0.750000 / 0.875000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 0.008444 / 0.164446 / 0.216606 | 0.008444 / 0.059929 / 0.116994 |
| mha_with_lse_d128 | b200-bf16-011 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | b200-bf16-011 | gemini-3.1-pro-preview | 4 | 32 | 0 / 2 / 6 | 0.0 / 0.125000 / 0.187500 | 0.0 / 0.125000 / 0.187500 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.197951 / 0.217261 | 0.0 / 0.197179 / 0.149435 |
| mha_with_lse_d128 | b200-bf16-011 | glm-5.2 | 4 | 32 | 1 / 2 / 10 | 0.250000 / 0.125000 / 0.312500 | 0.0 / 0.0 / 0.0 | 0.250000 / 0.500000 / 1.000000 | 0.0 / 0.0 / 0.0 | 0.024435 / 0.024435 / 0.034888 | 0.024435 / 0.023830 / 0.021322 |
| mha_with_lse_d128_causal | b200-bf16 | claude-opus-4.8-xhigh | 1 | 8 | 1 / 4 / 8 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 0.008620 / 0.174809 / 0.188857 | 0.008620 / 0.063271 / 0.099811 |
| mha_with_lse_d128_causal | b200-bf16 | fixit-v5 | 4 | 27 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | b200-bf16 | gemini-3.1-pro-preview | 1 | 8 | 1 / 2 / 3 | 1.000000 / 0.500000 / 0.375000 | 1.000000 / 0.500000 / 0.375000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 0.007814 / 0.107582 / 0.169851 | 0.007814 / 0.028994 / 0.052267 |
| mha_with_lse_d128_causal | b200-bf16 | glm-5.2 | 4 | 32 | 2 / 2 / 6 | 0.500000 / 0.125000 / 0.187500 | 0.0 / 0.0 / 0.0 | 0.500000 / 0.500000 / 0.500000 | 0.0 / 0.0 / 0.0 | 0.006695 / 0.006695 / 0.069202 | 0.006247 / 0.006247 / 0.015608 |
| mha_with_lse_d128_causal | b200-bf16-010 | claude-opus-4.8-xhigh | 1 | 8 | 1 / 4 / 8 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 0.067305 / 0.249984 / 0.296217 | 0.067305 / 0.159397 / 0.207915 |
| mha_with_lse_d128_causal | b200-bf16-010 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | b200-bf16-010 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | b200-bf16-010 | glm-5.2 | 4 | 32 | 1 / 2 / 8 | 0.250000 / 0.125000 / 0.250000 | 0.0 / 0.0 / 0.0 | 0.250000 / 0.500000 / 0.750000 | 0.0 / 0.0 / 0.0 | 0.007051 / 0.011246 / 0.036735 | 0.007051 / 0.008905 / 0.018934 |
| mha_with_lse_d128_causal | b200-bf16-011 | claude-opus-4.8-xhigh | 1 | 8 | 1 / 3 / 6 | 1.000000 / 0.750000 / 0.750000 | 1.000000 / 0.750000 / 0.750000 | 1.000000 / 1.000000 / 1.000000 | 1.000000 / 1.000000 / 1.000000 | 0.013771 / 0.066229 / 0.111522 | 0.013771 / 0.037783 / 0.059545 |
| mha_with_lse_d128_causal | b200-bf16-011 | fixit-v5 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | b200-bf16-011 | gemini-3.1-pro-preview | 1 | 8 | 0 / 0 / 1 | 0.0 / 0.0 / 0.125000 | 0.0 / 0.0 / 0.125000 | 0.0 / 0.0 / 1.000000 | 0.0 / 0.0 / 1.000000 | 0.0 / 0.0 / 0.190412 | 0.0 / 0.0 / 0.190412 |
| mha_with_lse_d128_causal | b200-bf16-011 | glm-5.2 | 4 | 32 | 1 / 7 / 10 | 0.250000 / 0.437500 / 0.312500 | 0.250000 / 0.062500 / 0.031250 | 0.250000 / 1.000000 / 1.000000 | 0.250000 / 0.250000 / 0.250000 | 0.097977 / 0.097977 / 0.097977 | 0.097977 / 0.024247 / 0.021669 |

## Interpretation

- Configured comparison coverage is 15 definition/prompt-tag pairs across 5 definitions.
- At ≤8 turns, correctness ranks: `claude-opus-4.8-xhigh` 0.710938, `glm-5.2` 0.277083, `gemini-3.1-pro-preview` 0.235000, `fixit-v5` 0.0.
- At ≤8 turns, correct-and-Blackwell-instruction use ranks: `claude-opus-4.8-xhigh` 0.554688, `gemini-3.1-pro-preview` 0.230000, `glm-5.2` 0.060417, `fixit-v5` 0.0.
- At ≤8 turns, trajectory correctness ranks: `claude-opus-4.8-xhigh` 1.000000, `glm-5.2` 0.800000, `gemini-3.1-pro-preview` 0.600000, `fixit-v5` 0.0.
- At ≤8 turns, trajectory correct-and-Blackwell-instruction use ranks: `claude-opus-4.8-xhigh` 0.750000, `gemini-3.1-pro-preview` 0.560000, `glm-5.2` 0.183333, `fixit-v5` 0.0.
- Treat these rankings as descriptive rather than fully paired because Gemini and Claude use smaller trajectory budgets.
