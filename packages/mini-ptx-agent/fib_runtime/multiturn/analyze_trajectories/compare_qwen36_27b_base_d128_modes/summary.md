# Qwen3.6-27B Base-Model Shared d128 MHA Comparison

Source of truth: each run's `figures/turn_correctness_arch.csv`, joined to `plan.json` for definition and prompt-tag metadata.

The plain `qwen36-27b-mha-*d128` roots use `*-mha-patched` prompt tags. The base-model non-patched d128 MHA roots with matching CSVs are the `qwen36-27b-linfo-*` and `qwen36-27b-linfo-singleuser-*` roots.

Modes:

- `regular-patched`
- `linfo`
- `singleuser`

## Run Roots

### regular-patched
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-mha-bwd-d128-causal`

### linfo
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d128-causal`

### singleuser
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-singleuser-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-singleuser-mha-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-singleuser-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-singleuser-mha-bwd-d128-causal`

All source rows loaded: 1536
Shared `(definition, prompt_tag)` pairs: 8
Shared rows after filtering: 768
Matched cross-mode turn rows: 256

Turn-transition alluvial outputs:

- `turn_transition_alluvial_index.md`
- `turn_transition_alluvial_by_definition/`

## Matched Overall Metrics

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_success_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regular-patched | 32 | 256 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| linfo | 32 | 256 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| singleuser | 32 | 256 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Shared Definition / Prompt-Tag Pairs

| definition | prompt_tag | group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_success_rate | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mha_bwd_d128 | hopper-012 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128 | hopper-012 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128 | hopper-012 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128 | hopper-013 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128 | hopper-013 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128 | hopper-013 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128_causal | hopper-012 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128_causal | hopper-012 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128_causal | hopper-012 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128_causal | hopper-013 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128_causal | hopper-013 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_bwd_d128_causal | hopper-013 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128 | hopper-07 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128 | hopper-07 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128 | hopper-07 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128 | hopper-08 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128 | hopper-08 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128 | hopper-08 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128_causal | hopper-07 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128_causal | hopper-07 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128_causal | hopper-07 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128_causal | hopper-08 | linfo | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128_causal | hopper-08 | regular-patched | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| mha_with_lse_d128_causal | hopper-08 | singleuser | 4 | 32 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
