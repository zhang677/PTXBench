# Qwen3.6-27B Retrieved Notes / Note Feedback2 Analysis

Source of truth: each run's `figures/turn_correctness_arch.csv`, joined to `plan.json` for definition and prompt metadata.
Prompt tags are kept as recorded in `plan.json`; patched tags keep their `-mha-patched` suffix.
No cross-group prompt-tag alignment or intersection is applied. Retrieved-notes and feedback2 groups include every available prompt tag; the fixit-v2-glm groups use the exact applicable prompt-tag allowlists below.
Missing planned turns are retained in `holistic_turns.csv` with `has_source_turn=0` and count against planned-turn rates.

Fixit-v2-GLM prompt configurations:

- `/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-2-r8-p4.json`
- `/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-bwd-2-r8-p4.json`
- `/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-2-r8-p4-patched.json`
- `/home/ubuntu/AccRL-exps/prompt_configs/hopper-mha-bwd-2-r8-p4-patched.json`

Watchers:

- `retrieved_notes_d128`: `/home/ubuntu/AccRL-exps/tasks/collect_notes/watch_qwen36_27b_retrieved_notes_d128_4defs.sh`
- `note_feedback2_d128_d96`: `/home/ubuntu/AccRL-exps/tasks/collect_notes/watch_qwen36_27b_note_feedback2_mha_d128_d96_8defs.sh`

Groups:

- `fixit_v2_glm_nopatched_d128`: fixit-v2-glm non-patched baseline over the four d128 definitions and two config-selected prompt tags per definition.
- `fixit_v2_glm_d128`: fixit-v2-glm patched baseline over the four d128 definitions and two config-selected `-mha-patched` prompt tags per definition.
- `fixit_v2_glm_d96`: fixit-v2-glm patched baseline over the four d96 definitions and two config-selected `-mha-patched` prompt tags per definition.
- `retrieved_notes_d128`: retrieved-note mode over the four d128 definitions.
- `note_feedback2_d128`: feedback2 mode over the four d128 definitions.
- `note_feedback2_d96`: feedback2 mode over the four d96 definitions.

Holistic CSV: `holistic_turns.csv` (1536 planned turn rows)

## Source Coverage

| group | n_runs | n_definitions | n_prompt_pairs | n_trajectories | planned_turns | observed_turns | missing_turns | definitions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_nopatched_d128 | 4 | 4 | 8 | 32 | 256 | 256 | 0 | mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |
| fixit_v2_glm_d128 | 4 | 4 | 8 | 32 | 256 | 256 | 0 | mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |
| fixit_v2_glm_d96 | 4 | 4 | 8 | 32 | 256 | 256 | 0 | mha_bwd_d96, mha_bwd_d96_causal, mha_with_lse_d96, mha_with_lse_d96_causal |
| retrieved_notes_d128 | 4 | 4 | 8 | 32 | 256 | 256 | 0 | mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |
| note_feedback2_d128 | 4 | 4 | 4 | 32 | 256 | 252 | 4 | mha_bwd_d128, mha_bwd_d128_causal, mha_with_lse_d128, mha_with_lse_d128_causal |
| note_feedback2_d96 | 4 | 4 | 4 | 32 | 256 | 246 | 10 | mha_bwd_d96, mha_bwd_d96_causal, mha_with_lse_d96, mha_with_lse_d96_causal |

## Included-Prompt Overall Metrics

All metric triplets are ordered `≤1 / ≤4 / ≤8` turns. Missing planned turns remain in the denominator.

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_nopatched_d128 | 32 | 256 | 0 / 2 / 9 | 0.0 / 0.015625 / 0.035156 | 0.0 / 0.015625 / 0.035156 | 0.0 / 0.031250 / 0.125000 | 0.0 / 0.031250 / 0.125000 | 0.0 / 0.499509 / 0.499509 | 0.0 / 0.497641 / 0.347231 |
| fixit_v2_glm_d128 | 32 | 256 | 2 / 8 / 25 | 0.062500 / 0.062500 / 0.097656 | 0.062500 / 0.062500 / 0.097656 | 0.062500 / 0.218750 / 0.437500 | 0.062500 / 0.218750 / 0.437500 | 0.555985 / 0.555985 / 0.565364 | 0.526248 / 0.391706 / 0.337186 |
| fixit_v2_glm_d96 | 32 | 256 | 0 / 2 / 5 | 0.0 / 0.015625 / 0.019531 | 0.0 / 0.015625 / 0.019531 | 0.0 / 0.062500 / 0.062500 | 0.0 / 0.062500 / 0.062500 | 0.0 / 0.239972 / 0.291815 | 0.0 / 0.165870 / 0.156310 |
| retrieved_notes_d128 | 32 | 256 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| note_feedback2_d128 | 32 | 256 | 0 / 29 / 63 | 0.0 / 0.226562 / 0.246094 | 0.0 / 0.226562 / 0.246094 | 0.0 / 0.593750 / 0.656250 | 0.0 / 0.593750 / 0.656250 | 0.0 / 0.646881 / 0.679044 | 0.0 / 0.191098 / 0.182573 |
| note_feedback2_d96 | 32 | 256 | 0 / 3 / 8 | 0.0 / 0.023438 / 0.031250 | 0.0 / 0.015625 / 0.027344 | 0.0 / 0.062500 / 0.125000 | 0.0 / 0.031250 / 0.093750 | 0.0 / 0.172923 / 0.477097 | 0.0 / 0.067432 / 0.119646 |

## Included-Prompt-Definition Collective Metrics

Each table pools every included prompt tag for one problem and group without cross-group prompt-tag matching. Metric triplets remain ordered `≤1 / ≤4 / ≤8` turns.

### `mha_bwd_d128`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_nopatched_d128 | 8 | 64 | 0 / 0 / 4 | 0.0 / 0.0 / 0.062500 | 0.0 / 0.0 / 0.062500 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.413403 | 0.0 / 0.0 / 0.288846 |
| fixit_v2_glm_d128 | 8 | 64 | 0 / 1 / 4 | 0.0 / 0.031250 / 0.062500 | 0.0 / 0.031250 / 0.062500 | 0.0 / 0.125000 / 0.250000 | 0.0 / 0.125000 / 0.250000 | 0.0 / 0.379511 / 0.388544 | 0.0 / 0.379511 / 0.336132 |
| retrieved_notes_d128 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| note_feedback2_d128 | 8 | 64 | 0 / 5 / 14 | 0.0 / 0.156250 / 0.218750 | 0.0 / 0.156250 / 0.218750 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.548649 / 0.548649 | 0.0 / 0.300507 / 0.233307 |

### `mha_bwd_d128_causal`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_nopatched_d128 | 8 | 64 | 0 / 2 / 3 | 0.0 / 0.062500 / 0.046875 | 0.0 / 0.062500 / 0.046875 | 0.0 / 0.125000 / 0.125000 | 0.0 / 0.125000 / 0.125000 | 0.0 / 0.499509 / 0.499509 | 0.0 / 0.497641 / 0.478680 |
| fixit_v2_glm_d128 | 8 | 64 | 0 / 1 / 5 | 0.0 / 0.031250 / 0.078125 | 0.0 / 0.031250 / 0.078125 | 0.0 / 0.125000 / 0.500000 | 0.0 / 0.125000 / 0.500000 | 0.0 / 0.192159 / 0.199286 | 0.0 / 0.192159 / 0.190274 |
| retrieved_notes_d128 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| note_feedback2_d128 | 8 | 64 | 0 / 6 / 8 | 0.0 / 0.187500 / 0.125000 | 0.0 / 0.187500 / 0.125000 | 0.0 / 0.500000 / 0.500000 | 0.0 / 0.500000 / 0.500000 | 0.0 / 0.173243 / 0.173459 | 0.0 / 0.113979 / 0.112317 |

### `mha_with_lse_d128`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_nopatched_d128 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| fixit_v2_glm_d128 | 8 | 64 | 2 / 6 / 13 | 0.250000 / 0.187500 / 0.203125 | 0.250000 / 0.187500 / 0.203125 | 0.250000 / 0.625000 / 0.750000 | 0.250000 / 0.625000 / 0.750000 | 0.555985 / 0.555985 / 0.565364 | 0.526248 / 0.443403 / 0.459877 |
| retrieved_notes_d128 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| note_feedback2_d128 | 8 | 64 | 0 / 6 / 15 | 0.0 / 0.187500 / 0.234375 | 0.0 / 0.187500 / 0.234375 | 0.0 / 0.625000 / 0.625000 | 0.0 / 0.625000 / 0.625000 | 0.0 / 0.502671 / 0.502671 | 0.0 / 0.176377 / 0.156937 |

### `mha_with_lse_d128_causal`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_nopatched_d128 | 8 | 64 | 0 / 0 / 2 | 0.0 / 0.0 / 0.031250 | 0.0 / 0.0 / 0.031250 | 0.0 / 0.0 / 0.125000 | 0.0 / 0.0 / 0.125000 | 0.0 / 0.0 / 0.313978 | 0.0 / 0.0 / 0.310016 |
| fixit_v2_glm_d128 | 8 | 64 | 0 / 0 / 3 | 0.0 / 0.0 / 0.046875 | 0.0 / 0.0 / 0.046875 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.394664 | 0.0 / 0.0 / 0.228988 |
| retrieved_notes_d128 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| note_feedback2_d128 | 8 | 64 | 0 / 12 / 26 | 0.0 / 0.375000 / 0.406250 | 0.0 / 0.375000 / 0.406250 | 0.0 / 0.875000 / 1.000000 | 0.0 / 0.875000 / 1.000000 | 0.0 / 0.646881 / 0.679044 | 0.0 / 0.213288 / 0.202734 |

### `mha_bwd_d96`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| note_feedback2_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### `mha_bwd_d96_causal`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| note_feedback2_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### `mha_with_lse_d96`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_d96 | 8 | 64 | 0 / 1 / 2 | 0.0 / 0.031250 / 0.031250 | 0.0 / 0.031250 / 0.031250 | 0.0 / 0.125000 / 0.125000 | 0.0 / 0.125000 / 0.125000 | 0.0 / 0.239972 / 0.291815 | 0.0 / 0.239972 / 0.264627 |
| note_feedback2_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### `mha_with_lse_d96_causal`

| group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixit_v2_glm_d96 | 8 | 64 | 0 / 1 / 3 | 0.0 / 0.031250 / 0.046875 | 0.0 / 0.031250 / 0.046875 | 0.0 / 0.125000 / 0.125000 | 0.0 / 0.125000 / 0.125000 | 0.0 / 0.114650 / 0.114650 | 0.0 / 0.114650 / 0.110041 |
| note_feedback2_d96 | 8 | 64 | 0 / 3 / 8 | 0.0 / 0.093750 / 0.125000 | 0.0 / 0.062500 / 0.109375 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.125000 / 0.375000 | 0.0 / 0.172923 / 0.477097 | 0.0 / 0.067432 / 0.119646 |

## Included Definition / Prompt-Tag Rows

| definition | prompt_tag | group | n_trajectories | n_turns | correct_turns | correctness_rate | correct_and_use_instruction_rate | trajectory_correctness_rate | trajectory_correct_and_use_instruction_rate | best_speedup | correct_turn_speedup_geomean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mha_bwd_d128 | hopper-012 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 0 / 4 | 0.0 / 0.0 / 0.125000 | 0.0 / 0.0 / 0.125000 | 0.0 / 0.0 / 0.500000 | 0.0 / 0.0 / 0.500000 | 0.0 / 0.0 / 0.413403 | 0.0 / 0.0 / 0.288846 |
| mha_bwd_d128 | hopper-012 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | hopper-012-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | hopper-013 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | hopper-013 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128 | hopper-013-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 0 / 1 / 4 | 0.0 / 0.062500 / 0.125000 | 0.0 / 0.062500 / 0.125000 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.379511 / 0.388544 | 0.0 / 0.379511 / 0.336132 |
| mha_bwd_d128 | hopper-wo-knowledge | note_feedback2_d128 | 8 | 64 | 0 / 5 / 14 | 0.0 / 0.156250 / 0.218750 | 0.0 / 0.156250 / 0.218750 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.375000 / 0.500000 | 0.0 / 0.548649 / 0.548649 | 0.0 / 0.300507 / 0.233307 |
| mha_bwd_d128_causal | hopper-012 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | hopper-012 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | hopper-012-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 0 / 1 / 4 | 0.0 / 0.062500 / 0.125000 | 0.0 / 0.062500 / 0.125000 | 0.0 / 0.250000 / 0.750000 | 0.0 / 0.250000 / 0.750000 | 0.0 / 0.192159 / 0.194705 | 0.0 / 0.192159 / 0.188086 |
| mha_bwd_d128_causal | hopper-013 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 2 / 3 | 0.0 / 0.125000 / 0.093750 | 0.0 / 0.125000 / 0.093750 | 0.0 / 0.250000 / 0.250000 | 0.0 / 0.250000 / 0.250000 | 0.0 / 0.499509 / 0.499509 | 0.0 / 0.497641 / 0.478680 |
| mha_bwd_d128_causal | hopper-013 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d128_causal | hopper-013-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 0 / 0 / 1 | 0.0 / 0.0 / 0.031250 | 0.0 / 0.0 / 0.031250 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.199286 | 0.0 / 0.0 / 0.199286 |
| mha_bwd_d128_causal | hopper-wo-knowledge | note_feedback2_d128 | 8 | 64 | 0 / 6 / 8 | 0.0 / 0.187500 / 0.125000 | 0.0 / 0.187500 / 0.125000 | 0.0 / 0.500000 / 0.500000 | 0.0 / 0.500000 / 0.500000 | 0.0 / 0.173243 / 0.173459 | 0.0 / 0.113979 / 0.112317 |
| mha_bwd_d96 | hopper-012-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d96 | hopper-013-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d96 | hopper-wo-knowledge | note_feedback2_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d96_causal | hopper-012-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d96_causal | hopper-013-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_bwd_d96_causal | hopper-wo-knowledge | note_feedback2_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | hopper-07 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | hopper-07 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | hopper-07-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 1 / 3 / 7 | 0.250000 / 0.187500 / 0.218750 | 0.250000 / 0.187500 / 0.218750 | 0.250000 / 0.500000 / 0.750000 | 0.250000 / 0.500000 / 0.750000 | 0.555985 / 0.555985 / 0.565364 | 0.555985 / 0.439799 / 0.470586 |
| mha_with_lse_d128 | hopper-08 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | hopper-08 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128 | hopper-08-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 1 / 3 / 6 | 0.250000 / 0.187500 / 0.187500 | 0.250000 / 0.187500 / 0.187500 | 0.250000 / 0.750000 / 0.750000 | 0.250000 / 0.750000 / 0.750000 | 0.498101 / 0.498101 / 0.510909 | 0.498101 / 0.447037 / 0.447691 |
| mha_with_lse_d128 | hopper-wo-knowledge | note_feedback2_d128 | 8 | 64 | 0 / 6 / 15 | 0.0 / 0.187500 / 0.234375 | 0.0 / 0.187500 / 0.234375 | 0.0 / 0.625000 / 0.625000 | 0.0 / 0.625000 / 0.625000 | 0.0 / 0.502671 / 0.502671 | 0.0 / 0.176377 / 0.156937 |
| mha_with_lse_d128_causal | hopper-07 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 0 / 2 | 0.0 / 0.0 / 0.062500 | 0.0 / 0.0 / 0.062500 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.250000 | 0.0 / 0.0 / 0.313978 | 0.0 / 0.0 / 0.310016 |
| mha_with_lse_d128_causal | hopper-07 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | hopper-07-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 0 / 0 / 3 | 0.0 / 0.0 / 0.093750 | 0.0 / 0.0 / 0.093750 | 0.0 / 0.0 / 0.500000 | 0.0 / 0.0 / 0.500000 | 0.0 / 0.0 / 0.394664 | 0.0 / 0.0 / 0.228988 |
| mha_with_lse_d128_causal | hopper-08 | fixit_v2_glm_nopatched_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | hopper-08 | retrieved_notes_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | hopper-08-mha-patched | fixit_v2_glm_d128 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d128_causal | hopper-wo-knowledge | note_feedback2_d128 | 8 | 64 | 0 / 12 / 26 | 0.0 / 0.375000 / 0.406250 | 0.0 / 0.375000 / 0.406250 | 0.0 / 0.875000 / 1.000000 | 0.0 / 0.875000 / 1.000000 | 0.0 / 0.646881 / 0.679044 | 0.0 / 0.213288 / 0.202734 |
| mha_with_lse_d96 | hopper-07-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 1 / 2 | 0.0 / 0.062500 / 0.062500 | 0.0 / 0.062500 / 0.062500 | 0.0 / 0.250000 / 0.250000 | 0.0 / 0.250000 / 0.250000 | 0.0 / 0.239972 / 0.291815 | 0.0 / 0.239972 / 0.264627 |
| mha_with_lse_d96 | hopper-08-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d96 | hopper-wo-knowledge | note_feedback2_d96 | 8 | 64 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d96_causal | hopper-07-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 0 / 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| mha_with_lse_d96_causal | hopper-08-mha-patched | fixit_v2_glm_d96 | 4 | 32 | 0 / 1 / 3 | 0.0 / 0.062500 / 0.093750 | 0.0 / 0.062500 / 0.093750 | 0.0 / 0.250000 / 0.250000 | 0.0 / 0.250000 / 0.250000 | 0.0 / 0.114650 / 0.114650 | 0.0 / 0.114650 / 0.110041 |
| mha_with_lse_d96_causal | hopper-wo-knowledge | note_feedback2_d96 | 8 | 64 | 0 / 3 / 8 | 0.0 / 0.093750 / 0.125000 | 0.0 / 0.062500 / 0.109375 | 0.0 / 0.250000 / 0.500000 | 0.0 / 0.125000 / 0.375000 | 0.0 / 0.172923 / 0.477097 | 0.0 / 0.067432 / 0.119646 |

## State Counts

| group | correctness | count | fraction |
| --- | --- | --- | --- |
| fixit_v2_glm_d128 | Correct | 25 | 0.097656 |
| fixit_v2_glm_d128 | Compilation error | 60 | 0.234375 |
| fixit_v2_glm_d128 | Runtime error | 71 | 0.277344 |
| fixit_v2_glm_d128 | Kernel Execution Timeout | 23 | 0.089844 |
| fixit_v2_glm_d128 | Numerical error | 73 | 0.285156 |
| fixit_v2_glm_d128 | Extraction error | 4 | 0.015625 |
| fixit_v2_glm_d96 | Correct | 5 | 0.019531 |
| fixit_v2_glm_d96 | Compilation error | 61 | 0.238281 |
| fixit_v2_glm_d96 | Runtime error | 57 | 0.222656 |
| fixit_v2_glm_d96 | Kernel Execution Timeout | 21 | 0.082031 |
| fixit_v2_glm_d96 | Numerical error | 57 | 0.222656 |
| fixit_v2_glm_d96 | Extraction error | 16 | 0.062500 |
| fixit_v2_glm_d96 | Other error | 39 | 0.152344 |
| fixit_v2_glm_nopatched_d128 | Correct | 9 | 0.035156 |
| fixit_v2_glm_nopatched_d128 | Compilation error | 66 | 0.257812 |
| fixit_v2_glm_nopatched_d128 | Runtime error | 147 | 0.574219 |
| fixit_v2_glm_nopatched_d128 | Kernel Execution Timeout | 6 | 0.023438 |
| fixit_v2_glm_nopatched_d128 | Numerical error | 17 | 0.066406 |
| fixit_v2_glm_nopatched_d128 | Extraction error | 11 | 0.042969 |
| note_feedback2_d128 | Correct | 63 | 0.246094 |
| note_feedback2_d128 | Compilation error | 60 | 0.234375 |
| note_feedback2_d128 | Runtime error | 30 | 0.117188 |
| note_feedback2_d128 | Kernel Execution Timeout | 35 | 0.136719 |
| note_feedback2_d128 | Numerical error | 62 | 0.242188 |
| note_feedback2_d128 | Extraction error | 1 | 0.003906 |
| note_feedback2_d128 | Other error | 1 | 0.003906 |
| note_feedback2_d128 | Missing turn | 4 | 0.015625 |
| note_feedback2_d96 | Correct | 8 | 0.031250 |
| note_feedback2_d96 | Compilation error | 69 | 0.269531 |
| note_feedback2_d96 | Runtime error | 59 | 0.230469 |
| note_feedback2_d96 | Kernel Execution Timeout | 33 | 0.128906 |
| note_feedback2_d96 | Numerical error | 76 | 0.296875 |
| note_feedback2_d96 | Extraction error | 1 | 0.003906 |
| note_feedback2_d96 | Missing turn | 10 | 0.039062 |
| retrieved_notes_d128 | Compilation error | 101 | 0.394531 |
| retrieved_notes_d128 | Runtime error | 79 | 0.308594 |
| retrieved_notes_d128 | Kernel Execution Timeout | 55 | 0.214844 |
| retrieved_notes_d128 | Numerical error | 17 | 0.066406 |
| retrieved_notes_d128 | Extraction error | 3 | 0.011719 |
| retrieved_notes_d128 | Other error | 1 | 0.003906 |
