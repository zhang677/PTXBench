# `2026-0524-1345` vs `inkling-gemm`

## Question

Compare the two GEMM evaluation runs under prompt tags present in both runs, using `figures/turn_correctness_arch.csv` as the source of truth for per-turn correctness, speedup, and architecture tags.

Runs:

- `/home/ubuntu/AccRL-exps/eval_runs/2026-0524-1345`
- `/home/ubuntu/AccRL-exps/eval_runs/inkling-gemm`

## Source artifacts

Performance fields came only from:

- `/home/ubuntu/AccRL-exps/eval_runs/2026-0524-1345/figures/turn_correctness_arch.csv`
- `/home/ubuntu/AccRL-exps/eval_runs/inkling-gemm/figures/turn_correctness_arch.csv`

The two `plan.json` files were used only to map each `exp_NNN` trajectory to its `prompt_tag`:

- `/home/ubuntu/AccRL-exps/eval_runs/2026-0524-1345/plan.json`
- `/home/ubuntu/AccRL-exps/eval_runs/inkling-gemm/plan.json`

CSV snapshot checksums at analysis time:

```text
715657961e252fdd4f9dce3655143deb744d45eac2d810fa7a4f0ce8a1efe9f9  2026-0524-1345/figures/turn_correctness_arch.csv
fe65b300a281c813d7a114d80aaf8c02e42483345be1267a807c52719bd62435  inkling-gemm/figures/turn_correctness_arch.csv
```

No result in this report is taken from `summary.json`, `success/record.json`, or an ad hoc reclassification of trajectory messages.

## CSV generation

`inkling-gemm` initially had no `turn_correctness_arch.csv`. It was generated with the standard repository exporter. The temporary experiments manifest contained:

```csv
arch,exp_dir
hopper,/home/ubuntu/AccRL-exps/eval_runs/inkling-gemm
```

The export command was:

```bash
python /home/ubuntu/AccRL/benchmark/export_turn_correctness_arch.py \
  --experiments-csv <temporary-experiments.csv> \
  --force
```

The exporter wrote 96 rows to `inkling-gemm/figures/turn_correctness_arch.csv`.

## Comparison process

1. Load each `plan.json` and construct `exp_NNN -> prompt_tag` from `exp_index`.
2. Load each CSV and validate the required schema: `trajectory_id`, `turn`, `correctness`, `speedup`, and `arch_tag`.
3. Verify that `(trajectory_id, turn)` keys are unique and every CSV trajectory maps to a plan entry.
4. Take the intersection of prompt tags. It is `hopper-no-hint`, `hopper-00`, and `hopper-05`.
5. Exclude the older run's non-shared `hopper-01`, `hopper-02`, `hopper-03`, and `hopper-04` trajectories.
6. Restrict both runs to turns 0 through 7. `inkling-gemm` has 8 turns per trajectory, while `2026-0524-1345` has 20; truncating the older run gives both sides the same turn budget.
7. Aggregate 4 trajectories and 32 turns per prompt tag, or 12 trajectories and 96 turns per run overall.

Metrics:

- **Turn success rate:** rows whose `correctness` is `Correct`, divided by all included rows.
- **Trajectory solve rate:** distinct trajectories with at least one `Correct` row, divided by all included trajectories.
- **Average speedup over all turns:** sum of `speedup`, treating a blank value as zero, divided by all included rows.
- **Best speedup:** maximum numeric `speedup` in the included rows.
- **Target hit:** a row with speedup at least the configured target of `1.2`.

Turn-level rows within a trajectory are correlated. Therefore, trajectory solve rate is the better measure of independent solves, while turn success rate measures how often a run emitted a correct kernel.

## Fair comparison: turns 0-7

### Per-prompt results

| Prompt tag | Run | Correct turns | Turn success rate | Solved trajectories | Average speedup, all turns | Best speedup | First correct | Correct arch tags |
|---|---|---:|---:|---:|---:|---:|---|---|
| `hopper-no-hint` | `2026-0524-1345` | 0/32 | 0.000% | 0/4 | 0 | 0 | - | - |
| `hopper-no-hint` | `inkling-gemm` | 0/32 | 0.000% | 0/4 | 0 | 0 | - | - |
| `hopper-00` | `2026-0524-1345` | 0/32 | 0.000% | 0/4 | 0 | 0 | - | - |
| `hopper-00` | `inkling-gemm` | 1/32 | 3.125% | 1/4 | 0.000251 | 0.008043 | `exp_005`, turn 1 | `G`: 1 |
| `hopper-05` | `2026-0524-1345` | 4/32 | 12.500% | 1/4 | 0.017036 | 0.269778 | `exp_025`, turn 2 | `H`: 4 |
| `hopper-05` | `inkling-gemm` | 0/32 | 0.000% | 0/4 | 0 | 0 | - | - |

### Overall shared-tag result

| Metric | `2026-0524-1345` | `inkling-gemm` |
|---|---:|---:|
| Included trajectories | 12 | 12 |
| Included turns | 96 | 96 |
| Correct turns | 4 | 1 |
| Turn success rate | 4.167% | 1.042% |
| Solved trajectories | 1/12 | 1/12 |
| Trajectory solve rate | 8.333% | 8.333% |
| Average speedup over all turns | 0.005679 | 0.000084 |
| Best speedup | 0.269778 | 0.008043 |
| Rows reaching speedup 1.2 | 0 | 0 |

The correct turns by zero-based turn index are:

| Turn | `2026-0524-1345` | `inkling-gemm` |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 0 | 0 |
| 4 | 1 | 0 |
| 5 | 1 | 0 |
| 6 | 0 | 0 |
| 7 | 1 | 0 |

All four correct rows in `2026-0524-1345` belong to the same `hopper-05` trajectory, `exp_025`, at turns 2, 4, 5, and 7. The one correct Inkling row belongs to `hopper-00` trajectory `exp_005` at turn 1.

### Outcome distribution

| Outcome | `2026-0524-1345` | `inkling-gemm` |
|---|---:|---:|
| `Correct` | 4 (4.167%) | 1 (1.042%) |
| `Runtime error` | 62 (64.583%) | 48 (50.000%) |
| `Compilation error` | 20 (20.833%) | 29 (30.208%) |
| `Numerical error` | 2 (2.083%) | 9 (9.375%) |
| `Kernel Execution Timeout` | 6 (6.250%) | 9 (9.375%) |
| `Extraction error` | 1 (1.042%) | 0 |
| `Other error` | 1 (1.042%) | 0 |

Inkling has fewer runtime-error rows, but this is offset by more compilation errors, numerical errors, and kernel timeouts. Its lower runtime-error count does not translate into more independent solved trajectories.

## Interpretation

At the equal 8-turn budget, neither run is a clear winner on independent solves: both solve exactly 1 of 12 trajectories. The prompt tag associated with that solve differs:

- Inkling's only solve is under `hopper-00`.
- `2026-0524-1345`'s only solve is under `hopper-05`.
- Neither run solves a `hopper-no-hint` trajectory.

`2026-0524-1345` is stronger on turn-level correctness and measured performance. It has four correct rows rather than one, a 4.167% rather than 1.042% turn success rate, and a best speedup of 0.269778 rather than 0.008043. However, the four correct rows are repeated successes from one trajectory, so they must not be described as four independent solves.

The architecture evidence is also sparse: the older run's four correct rows are tagged `H`, while Inkling's single correct row is tagged `G`. This is descriptive evidence from five correct turns, not enough to estimate a robust architecture-usage tendency.

Neither run produces a practically competitive kernel in this sample. Both best speedups are below 1.0 and far below the configured 1.2 target.

With only four trajectories per prompt tag, the tag-specific reversal (`hopper-00` for Inkling and `hopper-05` for the older run) should be treated as directional evidence, not a stable prompt ranking.

## Older run's additional turns

This section is not part of the fair head-to-head comparison. It records what happens when all 20 turns of `2026-0524-1345` are retained for the three shared tags.

| Prompt tag | Correct turns | Solved trajectories | Average speedup, all turns | Best speedup | First correct |
|---|---:|---:|---:|---:|---|
| `hopper-no-hint` | 0/80 | 0/4 | 0 | 0 | - |
| `hopper-00` | 1/80 | 1/4 | 0.000320 | 0.025561 | `exp_005`, turn 15 |
| `hopper-05` | 11/80 | 1/4 | 0.030192 | 0.270461 | `exp_025`, turn 2 |
| **Total** | **12/240** | **2/12** | **0.010170** | **0.270461** | - |

The extra budget allows one `hopper-00` trajectory to become correct only at turn 15. This raises the older run to 2/12 solved trajectories over 20 turns, but it cannot be used as evidence against an Inkling run that stopped after turn 7.

## Bottom line

- **Independent solve rate at equal budget:** tied at 1/12.
- **Correct-turn frequency:** favors `2026-0524-1345`, 4/96 versus 1/96.
- **Best valid-kernel performance:** favors `2026-0524-1345`, 0.269778 versus 0.008043.
- **Target-speedup success:** neither run reaches 1.2.
- **Prompt evidence:** `hopper-00` yields Inkling's only solve; `hopper-05` yields the older run's only fair-window solve; `hopper-no-hint` yields none.
- **Confidence:** low for prompt ranking because each tag has only four trajectories.
