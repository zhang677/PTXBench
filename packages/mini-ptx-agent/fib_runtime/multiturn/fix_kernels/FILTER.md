# Fix-It SFT Data Filtering

This note tracks the filters applied when building fix-it SFT parquet data from original Qwen3.6-27B eval runs. The filtering happens across several stages; the parquet builder is only the last stage.

## 1. Select Failed Source Kernels

Script: `select_failed_kernels.py`

Source of truth:

```text
<exp_dir>/figures/turn_correctness_arch.csv
```

By default, only failed turns with these correctness labels are selected:

```text
Runtime error
Kernel Execution Timeout
Numerical error
Compilation error
```

Rows are skipped when:

- `trajectory_id` or `turn` is missing.
- the assistant turn does not end with a closing code fence, unless `--allow-non-fenced-turns` is passed.
- the extracted per-turn kernel is missing at `kernels/<trajectory_id>/kernel_t<turn>.cu`.
- the extracted per-turn log is missing at `kernels/<trajectory_id>/log_t<turn>.txt`.

If `figures/turn_correctness_arch.csv` is missing, the script generates it with `benchmark/export_turn_correctness_arch.py`. If `kernels/` is missing, it generates per-turn kernels/logs with `mini_swe_agent_docker/plots/analyze_kernel_per_turn.py`.

## 2. Collect Successful Fixes

Script: `collect_success_kernel_pairs.py`

Only successful fix-it outputs under `success/exp_*` are considered. The collector pairs each successful correct kernel with the original wrong kernel from the run's `plan.json`.

Common current workflow flags:

```text
--correct-kernel-mode all
--arch-tag H
--min-speedup <threshold>
```

Filters at this stage:

- only `success/exp_*` directories are scanned.
- a matching `plan.json` entry must exist for the success `exp_id`.
- the plan entry must provide a wrong/error kernel path.
- if `--arch-tag H` is used, the turn/run arch tags must include `H`.
- if `--min-speedup` is used, a correct kernel version is kept only when its minimum recorded `evaluation.performance.speedup_factor` is at least the threshold.
- `--correct-kernel-mode` controls whether one or multiple success kernels are kept:
  - `best`: best recorded speedup, falling back to latest.
  - `latest`: latest `kernel_v*.cu`.
  - `all`: every eligible `kernel_v*.cu`.

For the d128 full data-quantity workflow, `--min-speedup 0.0` was used to keep any nonnegative-speedup successful fixed kernel rather than the older stricter `0.15` threshold.

## 3. Synthesize Teacher Reasoning

Script: `synthesize_pair_reasoning_openrouter.py`

Input is the kernel-pairs CSV from `collect_success_kernel_pairs.py`. Output is reasoning JSONL used by the parquet builder.

Filters/skips at this stage:

- rows already present in the output JSONL are skipped unless `overwrite: true` is set in the YAML config.
- if `limit` is configured, only the first `limit` rows are considered.
- rows are skipped if prompt construction cannot read the required wrong/correct kernel or log files.
- failed LLM calls produce no record and must be retried by rerunning synthesis.
- reasoning shorter than `min_reasoning_chars` is dropped.
- reasoning longer than `max_reasoning_chars` is dropped.

The synthesis run should be repeated until every expected pair key in the kernel-pairs CSV has a reasoning record, not just until the process exits cleanly.

## 4. Build Fix-It Parquet

Script: `build_sft_dataset_fixit.py`

Required inputs:

```text
--pairs / --pairs-jsonl      reasoning JSONL
--kernel-pairs-csv           collected wrong/correct kernel-pair CSV
--output                     .parquet or .jsonl
```

Current Qwen3.6-27B workflows typically pass:

```text
--tokenizer Qwen/Qwen3.6-27B
--max-tokens 65536
```

Chat-template normalization is enabled by default in `build_sft_dataset_fixit.py`.
Do not pass the old `--normalize-with-chat-template` flag; it is no longer part
of this builder's CLI. Use `--no-normalize-with-chat-template` only when a run
explicitly needs raw constructed messages.

The builder drops a reasoning record when:

- `metadata.correct_kernel_path` is missing.
- the reasoning record's correct kernel path does not match any row in the kernel-pairs CSV.
- the `reasoning` field is empty.
- `wrong_trajectory_path` is missing or unreadable.
- the original trajectory is not a JSON object.
- the original trajectory lacks a system message or first user message.
- `wrong_turn` is blank or invalid.
- the correct kernel file is missing.
- any output message contains forbidden `<my_reasoning>` or `</my_reasoning>` tokens.
- `--max-tokens` is provided and the final composed chat exceeds that token budget.

By default, each row is rendered through the tokenizer chat template and parsed back into messages before token filtering and writing. This changes message normalization, not row inclusion by itself, except that tokenizer loading must succeed.

With `--shuffle`, output rows are shuffled after all filtering and before writing. This is not enabled by default. The Tinker trainer separately shuffles the loaded parquet dataset by default.

## Effective Filter Summary

For the usual Qwen3.6-27B fix-it SFT construction path, rows survive only if they are:

1. failed original Qwen turns with selected failure labels;
2. fenced and extractable into per-turn kernel/log artifacts;
3. successfully repaired by the fix-it collection model;
4. optionally above the configured speedup threshold;
5. optionally matching the required arch tag, typically `H`;
6. covered by accepted teacher reasoning of configured length;
7. reconstructable into the wrong-kernel/log-to-reasoned-correct-kernel chat format;
8. free of forbidden reasoning wrapper tokens in output messages;
9. within the Qwen3.6-27B token budget when `--max-tokens 65536` is used.

The builder writes a sidecar manifest next to the output parquet recording input counts, dropped rows, max-token filtered rows, shuffle setting, and input/output checksums.
