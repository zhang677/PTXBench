# Gemini 3.1 Pro MHA Backward d128: Linfo vs June Baseline vs May NCU

Source of truth: each run's `figures/turn_correctness_arch.csv`, joined to `plan.json` for definition and prompt-tag metadata.
Prompt tags are kept exactly as recorded in `plan.json`. For cross-group matching only, a trailing `-mha-patched` suffix is ignored.
Rows retain their original turn numbers; turns are not intersected or trimmed across groups.

Stages:

- `gemini-31-pro-linfo`
- `gemini-31-pro-june`
- `gemini-31-pro-may-ncu`

## Shared Prompt-Tag Rules

Comparison rows are selected from these configs:

- `/home/ubuntu/AccRL-exps/prompt_configs/2026-0504-fa3-bwd.json`

All three conditions must cover all four tags in the selected config. Matching is exact by definition, prompt tag, replica, and turn.

## Data Notes

- All three runs use `gemini/gemini-3.1-pro-preview`, four replicas per selected prompt tag, and eight turns per trajectory.
- The paired slice is the four tags shared by all runs: `hopper-010`, `hopper-011`, `hopper-012`, and `hopper-013` (16 trajectories and 128 turns per condition).
- The June run's additional `hopper-no-hint` trajectories remain in `holistic_turns.csv` with `is_comparison_pair=0` and are excluded from comparison metrics and diagrams.
- The May task name is the legacy alias `mha_bwd_h48_d128`; its tensor shapes and operation match the canonical `mha_bwd_d128` task used by the other two runs.

## Run Roots

### gemini-31-pro-linfo
- `/home/ubuntu/AccRL-exps/eval_runs/gemini-31-pro-linfo-mha-bwd-d128`

### gemini-31-pro-june
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240`

### gemini-31-pro-may-ncu
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-1130`

All source rows loaded: 416
Configured definitions: 1 (`mha_bwd_d128`)
Configured `(definition, prompt_tag)` pairs: 4
Configured rows after filtering: 384
Comparison rows at original turn numbers: 384

CSV output:

- `holistic_turns.csv`

Turn-transition alluvial outputs:

- `turn_transition_alluvial_index.md`
- `turn_transition_alluvial_by_definition/`

## Source Coverage

| group | n_runs | n_definitions | n_prompt_pairs | n_trajectories | n_turns | definitions |
| --- | --- | --- | --- | --- | --- | --- |
| gemini-31-pro-linfo | 1 | 1 | 4 | 16 | 128 | mha_bwd_d128 |
| gemini-31-pro-june | 1 | 1 | 5 | 20 | 160 | mha_bwd_d128 |
| gemini-31-pro-may-ncu | 1 | 1 | 4 | 16 | 128 | mha_bwd_d128 |

## Configured-Prompt Overall Metrics

All metric triplets are ordered `≤1 / ≤4 / ≤8` turns.

| group | parquet_rows | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-31-pro-linfo | N/A | 16 | 128 | 1 / 15 / 32 | 0.062500 / 0.234375 / 0.250000 | 0.062500 / 0.234375 / 0.250000 | 0.062500 / 0.500000 / 0.687500 | 0.062500 / 0.500000 / 0.687500 | 0.169822 / 0.602755 / 0.602755 | 0.169822 / 0.276571 / 0.226266 |
| gemini-31-pro-june | N/A | 16 | 128 | 0 / 12 / 27 | 0.0 / 0.187500 / 0.210938 | 0.0 / 0.187500 / 0.210938 | 0.0 / 0.500000 / 0.812500 | 0.0 / 0.500000 / 0.812500 | 0.0 / 0.509056 / 0.606557 | 0.0 / 0.271241 / 0.238335 |
| gemini-31-pro-may-ncu | N/A | 16 | 128 | 1 / 15 / 29 | 0.062500 / 0.234375 / 0.226562 | 0.062500 / 0.187500 / 0.203125 | 0.062500 / 0.750000 / 0.750000 | 0.062500 / 0.625000 / 0.687500 | 0.402743 / 0.602896 / 0.638722 | 0.402743 / 0.137957 / 0.188080 |

## Configured-Definition Collective Metrics

Each table pools all configured prompt tags available for one problem and group. Metric triplets remain ordered `≤1 / ≤4 / ≤8` turns.

### `mha_bwd_d128`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-31-pro-linfo | 16 | 128 | 1 / 15 / 32 | 0.062500 / 0.234375 / 0.250000 | 0.062500 / 0.234375 / 0.250000 | 0.062500 / 0.500000 / 0.687500 | 0.062500 / 0.500000 / 0.687500 | 0.169822 / 0.602755 / 0.602755 | 0.169822 / 0.276571 / 0.226266 |
| gemini-31-pro-june | 16 | 128 | 0 / 12 / 27 | 0.0 / 0.187500 / 0.210938 | 0.0 / 0.187500 / 0.210938 | 0.0 / 0.500000 / 0.812500 | 0.0 / 0.500000 / 0.812500 | 0.0 / 0.509056 / 0.606557 | 0.0 / 0.271241 / 0.238335 |
| gemini-31-pro-may-ncu | 16 | 128 | 1 / 15 / 29 | 0.062500 / 0.234375 / 0.226562 | 0.062500 / 0.187500 / 0.203125 | 0.062500 / 0.750000 / 0.750000 | 0.062500 / 0.625000 / 0.687500 | 0.402743 / 0.602896 / 0.638722 | 0.402743 / 0.137957 / 0.188080 |

## Configured Definition / Prompt-Tag Rows

| definition | prompt_tag | group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mha_bwd_d128 | hopper-010 | gemini-31-pro-june | 4 | 32 | 0 / 2 / 7 | 0.0 / 0.125000 / 0.218750 | 0.0 / 0.125000 / 0.218750 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.465304 / 0.606557 | 0.0 / 0.434224 / 0.348515 |
| mha_bwd_d128 | hopper-010 | gemini-31-pro-linfo | 4 | 32 | 0 / 3 / 7 | 0.0 / 0.187500 / 0.218750 | 0.0 / 0.187500 / 0.218750 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.500000 / 0.750000 | 0.0 / 0.279085 / 0.279085 | 0.0 / 0.244266 / 0.178996 |
| mha_bwd_d128 | hopper-010 | gemini-31-pro-may-ncu | 4 | 32 | 0 / 4 / 6 | 0.0 / 0.250000 / 0.187500 | 0.0 / 0.250000 / 0.187500 | 0.0 / 1.000000 / 1.000000 | 0.0 / 1.000000 / 1.000000 | 0.0 / 0.228747 / 0.228747 | 0.0 / 0.142024 / 0.151083 |
| mha_bwd_d128 | hopper-011 | gemini-31-pro-june | 4 | 32 | 0 / 3 / 8 | 0.0 / 0.187500 / 0.250000 | 0.0 / 0.187500 / 0.250000 | 0.0 / 0.500000 / 1.000000 | 0.0 / 0.500000 / 1.000000 | 0.0 / 0.326325 / 0.326325 | 0.0 / 0.212154 / 0.182032 |
| mha_bwd_d128 | hopper-011 | gemini-31-pro-linfo | 4 | 32 | 0 / 1 / 4 | 0.0 / 0.062500 / 0.125000 | 0.0 / 0.062500 / 0.125000 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.132668 / 0.132668 | 0.0 / 0.132668 / 0.076519 |
| mha_bwd_d128 | hopper-011 | gemini-31-pro-may-ncu | 4 | 32 | 0 / 3 / 7 | 0.0 / 0.187500 / 0.218750 | 0.0 / 0.187500 / 0.218750 | 0.0 / 0.500000 / 0.500000 | 0.0 / 0.500000 / 0.500000 | 0.0 / 0.423316 / 0.423316 | 0.0 / 0.307651 / 0.203846 |
| mha_bwd_d128 | hopper-012 | gemini-31-pro-june | 4 | 32 | 0 / 3 / 5 | 0.0 / 0.187500 / 0.156250 | 0.0 / 0.187500 / 0.156250 | 0.0 / 0.500000 / 1.000000 | 0.0 / 0.500000 / 1.000000 | 0.0 / 0.509056 / 0.509056 | 0.0 / 0.293717 / 0.299732 |
| mha_bwd_d128 | hopper-012 | gemini-31-pro-linfo | 4 | 32 | 0 / 5 / 11 | 0.0 / 0.312500 / 0.343750 | 0.0 / 0.312500 / 0.343750 | 0.0 / 0.750000 / 0.750000 | 0.0 / 0.750000 / 0.750000 | 0.0 / 0.602755 / 0.602755 | 0.0 / 0.368364 / 0.335336 |
| mha_bwd_d128 | hopper-012 | gemini-31-pro-may-ncu | 4 | 32 | 1 / 5 / 12 | 0.250000 / 0.312500 / 0.375000 | 0.250000 / 0.250000 / 0.343750 | 0.250000 / 0.750000 / 0.750000 | 0.250000 / 0.750000 / 0.750000 | 0.402743 / 0.602896 / 0.638722 | 0.402743 / 0.217890 / 0.306666 |
| mha_bwd_d128 | hopper-013 | gemini-31-pro-june | 4 | 32 | 0 / 4 / 7 | 0.0 / 0.250000 / 0.218750 | 0.0 / 0.250000 / 0.218750 | 0.0 / 0.750000 / 0.750000 | 0.0 / 0.750000 / 0.750000 | 0.0 / 0.343991 / 0.382339 | 0.0 / 0.242813 / 0.188282 |
| mha_bwd_d128 | hopper-013 | gemini-31-pro-linfo | 4 | 32 | 1 / 6 / 10 | 0.250000 / 0.375000 / 0.312500 | 0.250000 / 0.375000 / 0.312500 | 0.250000 / 0.500000 / 0.750000 | 0.250000 / 0.500000 / 0.750000 | 0.169822 / 0.354916 / 0.435254 | 0.169822 / 0.261954 / 0.266842 |
| mha_bwd_d128 | hopper-013 | gemini-31-pro-may-ncu | 4 | 32 | 0 / 3 / 4 | 0.0 / 0.187500 / 0.125000 | 0.0 / 0.062500 / 0.062500 | 0.0 / 0.750000 / 0.750000 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.140472 / 0.350145 | 0.0 / 0.027783 / 0.052347 |

## Interpretation

- The paired comparison contains 128 exact tag/replica/turn positions per condition.
- At ≤8 turns, turn correctness ranks: `gemini-31-pro-linfo` 0.250000, `gemini-31-pro-may-ncu` 0.226562, `gemini-31-pro-june` 0.210938.
- At ≤8 turns, trajectory correctness ranks: `gemini-31-pro-june` 0.812500, `gemini-31-pro-may-ncu` 0.750000, `gemini-31-pro-linfo` 0.687500.

### Pairwise correctness disagreements

| left | right | both correct | left only | right only | neither |
| --- | --- | ---: | ---: | ---: | ---: |
| gemini-31-pro-linfo | gemini-31-pro-june | 7 | 25 | 20 | 76 |
| gemini-31-pro-linfo | gemini-31-pro-may-ncu | 8 | 24 | 21 | 75 |
| gemini-31-pro-june | gemini-31-pro-may-ncu | 7 | 20 | 22 | 79 |

### Condition audit

| group | model | mean system chars | distinct system prompts | mean feedback chars | primary-diagnostic turns | NCU-feedback turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gemini-31-pro-linfo | `gemini/gemini-3.1-pro-preview` | 88692 | 4 | 2262 | 32/128 | 0/128 |
| gemini-31-pro-june | `gemini/gemini-3.1-pro-preview` | 89044 | 4 | 1664 | 0/128 | 0/128 |
| gemini-31-pro-may-ncu | `gemini/gemini-3.1-pro-preview` | 91904 | 4 | 3773 | 0/128 | 29/128 |

All three conditions use the same model ID, but their system prompts and feedback stacks differ. The result is therefore a condition-level comparison, not a clean single-variable ablation.
