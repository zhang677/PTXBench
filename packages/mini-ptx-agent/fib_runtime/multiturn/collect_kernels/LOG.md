# collect_kernels Run Log

This log inventories the `/home/ubuntu/AccRL-exps` artifacts that are tied to
`/home/ubuntu/AccRL/fib_runtime/multiturn/collect_kernels`.

Scope rule: a run or artifact is listed here when local provenance references one
of the `collect_kernels` scripts, or when it is a direct input/output artifact for
that toolchain. Generic eval dirs that merely contain `success/` or `kernels/`
are not included unless there is such provenance.

Real success counts below mean files matching `success/exp_*/kernel_v*.cu`.

## Script Roles

| Script | Purpose |
| --- | --- |
| `collect_correct_kernels.py` | Reads selected eval-run metadata and per-turn correctness CSVs, then emits a CSV of correct kernels above a speedup threshold. |
| `mask_kernels.py` | Masks selected CUDA kernels, currently by replacing descriptor-call arguments with `?`, and writes masked kernels plus a manifest CSV. |
| `masked_csv_to_prompt_config.py` | Converts masked-kernel CSV rows into prompt configs consumed by the masked launcher. |
| `run_parallel_masked_v2.py` | Launches masked reconstruction evals. Each experiment asks the model to fill the missing `?` region in a kernel. |
| `collect_success_trajectories.py` | Collects trajectory paths for eval experiments with real success kernels. Used by loop-SFT reconstruction. |
| `generate_success_diff_report.py` / `investigate_success_diffs.py` | Summarize or inspect success-set differences between eval runs. |

## Prepared Kernel Sets

These are not eval runs; they are input manifests and masked-kernel artifacts
used to build masked reconstruction prompts.

| Cohort | Selected runs | Selected kernels | Masked kernels | Prompt config | What it does |
| --- | ---: | ---: | ---: | --- | --- |
| `2026-0608-1500` | 6 | 68 | 68 | `/home/ubuntu/AccRL-exps/prompt_configs/2026-0608-1500-masked.json` | Selected correct kernels from six source runs, masked them, and prepared fill-in prompts. |
| `2026-0610-0900` | 3 | 118 | 118 | `/home/ubuntu/AccRL-exps/prompt_configs/2026-0610-0900-masked.json` and `2026-0610-0900-masked-t4.json` | Prepared masked prompts later used by the `2026-0610-1100` masked reconstruction run. |
| `2026-0610-1600` | 8 | 205 | 205 | `/home/ubuntu/AccRL-exps/prompt_configs/2026-0610-1600-masked.json` | Prepared the 205-item masked corpus used by the loop-SFT reconstruction eval sequence. |

Related files:

- `/home/ubuntu/AccRL-exps/tasks/selected-runs-*.csv`
- `/home/ubuntu/AccRL-exps/tasks/selected-kernels-*.csv`
- `/home/ubuntu/AccRL-exps/tasks/masked-kernels-*.csv`
- `/home/ubuntu/AccRL-exps/selected-kernels/{2026-0608-1500,2026-0610-0900,2026-0610-1600}/`

## Masked Reconstruction Eval Runs

| Eval dir | Config/model provenance | Real successes | What it does |
| --- | --- | ---: | --- |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1100` | `run_parallel_masked_v2.py` with `2026-0610-0900-masked-t4.json`, `Qwen3.6-27B`; original command recorded in `tasks/restore-2026-0610-1100.sh` | 97 | Initial masked fill-in run; interrupted before completing the tail of the expanded config. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1100-rerun` | Tail restore launched by `tasks/restore-2026-0610-1100.sh` | 12 | Reruns the missing tail range from the interrupted `2026-0610-1100` run. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1100-complete` | Merge of base plus rerun from `restore-2026-0610-1100.sh` | 109 | Completed version of the `2026-0610-1100` masked reconstruction eval. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0613-2014-eval` | `run_parallel_masked_v2.py` with `2026-0610-1600-masked.json`, SFT model `qwen36-27b-SFT-2026-0613-2014`; recorded in loop audit commands | 24 | First loop-SFT reconstruction eval from the `2026-0611-1826` loop. Later found incomplete. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0613-2014-eval-rerun` | Tail restore launched by `tasks/restore-2026-0613-2014.sh` | 38 | Reruns the missing tail for `2026-0613-2014-eval`. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0613-2014-eval-complete` | Merge of `2026-0613-2014-eval` plus rerun | 62 | Completed version of the `2026-0613-2014` masked reconstruction eval. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0614-2257-eval` | `run_parallel_masked_v2.py` with `2026-0610-1600-masked.json`, model `qwen36-27b-SFT-2026-0614-2257` | 68 | Next loop-SFT reconstruction eval after training on the prior success trajectories. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0615-0823-eval` | `run_parallel_masked_v2.py` with `2026-0610-1600-masked.json`, model `qwen36-27b-SFT-2026-0615-0823` | 58 | Third loop-SFT reconstruction eval. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0615-1652-eval` | `run_parallel_masked_v2.py` with `2026-0610-1600-masked.json`, model `qwen36-27b-SFT-2026-0615-1652` | 56 | Fourth loop-SFT reconstruction eval. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0616-0048-eval` | Launch recorded in `loop-sft-reconstruct-2026-0613-2014-20260615-004822/commands.jsonl`, model `qwen36-27b-SFT-2026-0616-0048` | 29 | Partial next eval; not present in that audit dir's `success_counts.csv` snapshot. |

Comparison points used by the reconstruction plot:

| Eval dir | Real successes | Notes |
| --- | ---: | --- |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0610-2332-eval` | 48 | Qwen 27B eval comparison point. Its `plan.json` hash matches `2026-0611-1826-eval`. |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0611-1826-eval` | 54 | Starting point for the loop-SFT reconstruction analysis. |

## Loop-SFT Reconstruction Audit Dirs

These dirs are provenance logs written by
`/home/ubuntu/AccRL-exps/tasks/loop_sft_reconstruct.py`. They are not themselves
eval outputs; they record commands, round logs, and success-count snapshots.

| Audit dir | Success-count snapshot | What it does |
| --- | --- | --- |
| `eval_runs/loop-sft-reconstruct-2026-0611-1826-20260613-201425` | round 0: `2026-0611-1826-eval` = 54 | Starts from the `2026-0611-1826` eval and collects success trajectories for SFT. |
| `eval_runs/loop-sft-reconstruct-2026-0611-1826-20260613-214347` | round 0: 54; round 1: `2026-0613-2014-eval` = 24 | Launches the first SFT eval from the masked corpus and collects its success trajectories. |
| `eval_runs/loop-sft-reconstruct-2026-0611-1826-20260614-024844` | round 1: `2026-0611-1826-eval` = 54 | Resume/provenance breadcrumb with no direct masked-launch command in `commands.jsonl`. |
| `eval_runs/loop-sft-reconstruct-2026-0611-1826-20260614-025411` | round 1: `2026-0611-1826-eval` = 54 | Records a launch attempt for `2026-0614-0221-eval`; that eval dir is no longer present locally. |
| `eval_runs/loop-sft-reconstruct-2026-0613-2014-20260614-225502` | round 0: `2026-0613-2014-eval` = 24 | Starts from the incomplete `2026-0613-2014-eval` and collects success trajectories. |
| `eval_runs/loop-sft-reconstruct-2026-0613-2014-20260614-225723` | round 0: `2026-0613-2014-eval-complete` = 62 | Corrected start from the completed `2026-0613-2014` eval. |
| `eval_runs/loop-sft-reconstruct-2026-0613-2014-20260615-002835` | round 0: `2026-0613-2014-eval-complete` = 62 | Resume/provenance breadcrumb with the same corrected starting count. |
| `eval_runs/loop-sft-reconstruct-2026-0613-2014-20260615-004822` | rounds 0-3: 62, 68, 58, 56 | Main four-round reconstruction sequence through `2026-0615-1652-eval`; also records a later partial `2026-0616-0048-eval` launch. |

The corresponding SFT trajectory CSV outputs are under:

- `/home/ubuntu/AccRL-exps/sft_experiments/2026-0609-1535_wgmma_desc_masked/data/*-success-trajectories.csv`

## Analysis Artifacts

| Artifact | What it does |
| --- | --- |
| `/home/ubuntu/AccRL-exps/eval_runs/*/success-diffs.md` | Per-run success-difference reports generated around the selected/masked cohorts. |
| `/home/ubuntu/AccRL-exps/success-overlap-by-problem-2026-0609-2100-plus-0610-0900-vs-2026-0610-1600.png` | Overlap plot comparing success sets across pre-reconstruction runs. |
| `/home/ubuntu/AccRL-exps/success-overlap-by-problem-2026-0610-1100-complete-vs-0900.png` | Overlap plot comparing the completed masked run against the `2026-0610-0900` source cohort. |
| `analysis/reconstruction-cases-vs-runs.csv` | Plotted data generated in this workspace for qwen eval, starting eval, and loop-SFT reconstruction rounds. |
| `analysis/reconstruction-cases-vs-runs.png` | Plot of real reconstruction success cases vs run sequence. |

## Current Reconstruction Plot Data

| Sequence | Run | Real successes |
| ---: | --- | ---: |
| 0 | qwen-27b eval `2026-0610-2332-eval` | 48 |
| 1 | starting eval `2026-0611-1826-eval` | 54 |
| 2 | reconstruct round 0 `2026-0613-2014-eval-complete` | 62 |
| 3 | reconstruct round 1 `2026-0614-2257-eval` | 68 |
| 4 | reconstruct round 2 `2026-0615-0823-eval` | 58 |
| 5 | reconstruct round 3 `2026-0615-1652-eval` | 56 |
