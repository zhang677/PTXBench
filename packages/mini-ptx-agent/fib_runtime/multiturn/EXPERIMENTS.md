# Experiments Log

Distilled from `fib_runtime/mini_swe_agent_docker/experiments.jsonl` (commands logged by
`run_experiment.sh`). Plot / analysis / scan / merge commands are omitted — this table
only tracks the *launch* commands that produced a new `eval_runs/<id>` directory.

Stages below group runs by the dominant change driving them: harness rev, model swap,
problem swap, prompt-ensemble feature, etc. Each entry links a run id to the model,
problem, and what's notably different from the previous run.

Conventions:
- "**Problem**" — the kernel definition under test
- "**Test**" — abbreviated test path (under `mini_swe_agent_docker/envs/` unless noted)
- "**Config**" — prompt-ensemble config (`run_parallel_v2.py` only)
- All runs use 14400s timeout and `gemini-3.1-pro-preview` unless noted

---

## Stage 1 — Out-of-the-box mini-swe-agent on a prepared workspace

Goal: characterize raw LLM ability on a fixed GEMM, comparing models. The harness
(`launch_eval_v8.py` / `launch_eval_v9.py` and their `_parallel` wrappers) drives
mini-swe-agent's stock `DefaultAgent` + `LitellmTextbasedModel` — the agent loop itself
is unchanged from upstream; only the docker environment is subclassed
(`CheckedDockerEnvironmentCUDA`) for command guardrails. v9 additionally restricts the
workspace to a fixed allow-list of filenames.

Before launching the agent, the harness materializes a per-experiment tmpdir, copies the
test file in, creates empty mount points for the tutorial repos, then runs the container
with `-v {tmpdir}:/workspace` plus read-only bind mounts for the tutorials.

Pre-run host folder structure (`tmpdir`, mounted as `/workspace`):

```
<tmpdir>/                              # auto-generated /tmp/swe_eval_* or --output-dir
├── test.py                            # copied from --test-path
└── tutorials/
    ├── ptx_doc/                       # empty mountpoint dir (host-side)
    └── examples/                      # empty mountpoint dir (host-side)
```

Bind mounts at container start:

```
{tmpdir}                                       → /workspace             (rw)
.../mini_swe_agent_docker/envs/ptx_isa         → /workspace/tutorials/ptx_doc   (ro)
.../mini_swe_agent_docker/remote_tutorials     → /workspace/tutorials/examples  (ro)
```

Inside the container during the agent loop, the agent itself produces:

```
/workspace/
├── test.py                            # provided
├── kernel.cu                          # written by the agent
├── compile.sh                         # written by the agent (or copied tutorial)
├── kernel.so                          # produced by `bash compile.sh`
├── traces.json                        # produced by `python test.py`
├── .container_id                      # container bookkeeping
└── tutorials/                         # ro mounts (ptx_doc, examples)
```

v9's `ALLOWED_WORKSPACE_FILES` is exactly:
`{tutorials, .container_id, compile.sh, test.py, kernel.cu, kernel.so, traces.json, __pycache__}`
— anything outside this set is rejected by `CheckedDockerEnvironmentCUDA`.

Problem throughout: **`gemm_n6144_k4096`**. Later runs switch the test from
`test_measure_cuda_*` to `test_profile_cuda_*` so the remote flashinfer-bench profiling
service is invoked instead of an in-container measurement.

| eval_run | Harness | Model | Test | Notes |
|---|---|---|---|---|
| 2026-0401-1033 | v8 | Qwen3.5-35B-A3B | test_measure_cuda_gemm | `--exclude-ncu-metrics`, n=14 |
| 2026-0401-1225 | v8 | gemini-3.1-pro-preview | test_measure_cuda_gemm | n=14 |
| 2026-0401-1807 | v8 | Qwen3.5-397B-A17B | test_measure_cuda_gemm | larger MoE |
| 2026-0401-2234 | v8 | Qwen3.5-397B-A17B-FP8 | test_measure_cuda_gemm | FP8 quant variant |
| 2026-0402-1308 | **v9** | Qwen3.5-35B-A3B | test_measure_cuda_gemm | v9: workspace restricted to allowed files |
| 2026-0402-2000 | v9 | Qwen3.5-397B-A17B-FP8 | **test_profile_cuda_gemm** | switch to remote profiling service |
| 2026-0403-0815 | v9 | Qwen3.5-35B-A3B | test_profile_cuda_gemm | timeout 1800 |
| 2026-0403-1115 | v9 | gemini-3.1-pro-preview | test_profile_cuda_gemm | gemini + profile path |
| 2026-0405-0922 | v9 | Qwen3.5-35B-A3B | test_profile_cuda_gemm | repeat for variance |

---

## Stage 2 — Multi-turn introduction (retired pre-v2 launcher)

Goal: switch from free tool calls to controlled multiturn. Adds `--max-turns` and
`--target-speedup`. Same problem as Stage 1.

| eval_run | Model | max-turns | Notes |
|---|---|---|---|
| 2026-0407-1815 | Qwen3.5-35B-A3B | 120 | first multi-turn run, test_profile_cuda |
| 2026-0407-2252 | gemini-3.1-pro-preview | 120 | switched to test_measure_cuda |
| 2026-0407-2352 | Qwen3.5-397B-A17B-FP8 | 120 | timeout 3600 |
| 2026-0408-1324 | GLM-5.1 | 10 | single-experiment smoke test (n=1) |
| 2026-0408-1936 | GLM-5.1 | 10 | n=1 retry |
| 2026-0408-2233 | Qwen3.6-plus | 10 | n=1, new model preview |
| 2026-0408-2312 | GLM-5.1 | 10 | full sweep, timeout 3600, n=14 |

---

## Stage 3 — Remote-only compile/measure (`--without-local-gpu`)

Environment update: drop the host-GPU path entirely; container only compiles and the
flashinfer-bench service measures. Test path changes to `compile_measure_*`.

| eval_run | Model | Test | Notes |
|---|---|---|---|
| 2026-0410-1325 | Qwen3.5-35B-A3B | compile_measure_sm90_cuda_gemm | first remote-only, max-turns 100 |
| 2026-0411-1350 | gemini-3.1-pro-preview | test_measure_cuda_gemm | alternate pre-v2 driver, max-turns 20 |
| 2026-0412-1320 | Qwen3.5-35B-A3B | compile_measure_cuda_gemm | drop `_sm90_` from test, max-turns 100 |

---

## Stage 4 — New GEMM problem `gemm_n7168_k5120` + custom test harness

Problem swap. Test path now lives under `multiturn/2026-0413-1611/` (per-problem
directory containing the test, NCU template, and verification script). All runs in this
stage use `gemini-3.1-pro-preview`, `--max-turns 20`, `--without-local-gpu`, and the
custom `gemm_n7168_k5120_94920358-...py` test.

| eval_run | n | Notes |
|---|---|---|
| 2026-0413-1715 | 1 | n=1 dry-run on the new problem |
| 2026-0413-2130 | 16 | scale-up |
| 2026-0413-2223, 2336 | 8 | back-to-back prompt-iteration cycles |
| 2026-0414-0013 | 8 | post `093f219 Add examples` |
| 2026-0414-1214 | 8 | post prompt fixes (`0521961`, `efb5f28`) |
| 2026-0414-1755 | 8 | post `a46c3e0 Minor prompt update` |
| 2026-0414-2046 | 8 | post `bb14efc Finalize prompts` |
| 2026-0415-0026 | 8 | post `4a5df4b Another prompt update` |
| 2026-0415-1243 | 8 | post `694c575 Remove redundant prompt` |
| 2026-0415-1305 | 8 | Qwen3.5-35B-A3B re-baseline, max-turns 100 |
| 2026-0415-1335 | 8 | gemini, post-prompt-cleanup |
| 2026-0417-1107 | 8 | post inspector tooling (`6224733`, `f77cbcd`) |
| 2026-0417-1244 | 8 | Qwen3.5-35B-A3B repeat |

---

## Stage 5 — Architecture-hint ablation (`--gpu-arch hopper-no-hint`)

Prompt-optimization stage: `--gpu-arch` is added so the system prompt can be selected
between "hopper" (with explicit Hopper instruction hints) and "hopper-no-hint" (no
arch-specific guidance). Also introduces **Kimi-K2.6** as a model.

| eval_run | Model | max-turns | Notes |
|---|---|---|---|
| 2026-0420-0930 | gemini-3.1-pro-no-reasoning | 20 | n=1 dry-run, no-reasoning variant |
| 2026-0420-1127 | gemini-3.1-pro-preview | 20 | hopper-no-hint full run |
| 2026-0420-1440 | gemini-3.1-pro-preview | 20 | repeat post `c4cba88 max attempts a variable` |
| 2026-0420-2114 | **Kimi-K2.6** | 8 | first Kimi-K2.6 run, timeout 7200 |
| 2026-0420-2305 | gemini-3.1-pro-preview | 8 | matched 8-turn budget for comparison |

---

## Stage 6 — Prompt ensemble (`run_parallel_v2.py` with `--config`)

Major harness change: the retired launcher was replaced by **`run_parallel_v2.py`**. Instead of
`-n / --gpus / --max-turns / --target-speedup` flags, runs are driven by a JSON config
listing prompt tags (`hopper-00`…`hopper-13`, `hopper-no-hint`) with per-tag
trajectory/turn budgets — i.e., the parallel agents now sweep multiple system prompts in
one experiment. Configs live under `AccRL-exps/prompt_configs/`.

Problem: still `gemm_n7168_k5120`.

| eval_run | Model | Config | Notes |
|---|---|---|---|
| 2026-0421-2352 | Qwen3.5-35B-A3B | 2026-0421-2352.json | first v2 run, broad prompt sweep (hopper-00..04 + no-hint) |
| 2026-0422-0844 | gemini-3.1-pro-preview | 2026-0422-0844.json | gemini on same config |
| 2026-0422-1002 | gemini-3.1-pro-preview | 2026-0422-0844.json | repeat, `--max-parallel 7` |
| 2026-0422-1538 | gemini-3.1-pro-preview | 2026-0422-1538.json | revised prompt mix |
| 2026-0422-1834 | gemini-3.1-pro-preview | 2026-0422-1834.json | further prompt revision, max-parallel 6 |
| 2026-0422-2049 | gemini-3.1-pro-preview | 2026-0422-2049.json | post `cd066e1 Add more prompts` |
| 2026-0422-2352 | (Kimi/gemini sweep) | (task script) | followed by a legacy scan/rerun/merge recovery pass into `2026-0422-2352_complete` |
| 2026-0424-1020 | **DeepSeek-V4:pro** | 2026-0424-0940.json | first DeepSeek run |

---

## Stage 7 — Attention problems (`mha_h48_d128`)

Problem-collection stage: switch from GEMM to attention. Custom test harness lives in
`multiturn/2026-0424-1043/` with `make_attention_problem.py`, `bench_attention.py`,
`profile_fa2.py`, `verify_via_service.py`.

Definition `helion_mha_h48_d128` (Helion variant) is used briefly, then replaced with
`mha_h48_d128` (the canonical name) once `a52ebd7 Update verify via service` lands.

| eval_run | Model | Definition | Config | Notes |
|---|---|---|---|---|
| 2026-0424-1603 | gemini-3.1-pro-preview | helion_mha_h48_d128 | 2026-0424-1603.json | first attention run |
| 2026-0424-1632 | Kimi-K2.6 | helion_mha_h48_d128 | 2026-0424-1603.json | Kimi on attention |
| 2026-0424-1728 | gemini-3.1-pro-preview | mha_h48_d128 | 2026-0424-1603.json | renamed definition |
| 2026-0424-2032 | gemini-3.1-pro-preview | mha_h48_d128 | 2026-0424-1935.json | revised prompt mix |

---

## Stage 8 — Attention + LSE output (`mha_with_lse_h48_d128`)

Reference update: cuDNN-based reference now also returns `log-sum-exp` (post
`1ae100a Update reference to cudnn for lse output`, `c6f9fbc Add mha with lse`). Custom
test harness in `multiturn/2026-0426-1410/`.

| eval_run | Config | Notes |
|---|---|---|
| 2026-0426-1446 | 2026-0426-1446.json | first lse-aware run |
| 2026-0426-1646 | 2026-0426-1646.json | post `8eb4f05 Add prompts` (hopper-07..09) |

---

## Stage 9 — Attention backward (`mha_bwd_h48_d128`)

Backward-pass stage. New problem dir `multiturn/2026-0427-1308/` with bwd test, NCU
template, scripts/, and reference solutions/. Driven by commits
`039d9b7 Add bwd` → `48f507a Prepare S=2048 bwd` → `824d010 Add flops calculation`.

| eval_run | Config | Notes |
|---|---|---|
| 2026-0427-1550 | 2026-0427-1550.json | first bwd run |
| 2026-0427-2022 | 2026-0427-2022.json | revised prompts post `5267814 Add exp2 func` / `9ef0f6c Add fa2 solution for mha` |

---

## Stage 10 — NCU-aware tests + per-turn timeout

Environment update: tests now use the **`ncu_*`** prefix (NCU profiling integrated into
the measurement loop) and `--turn-timeout 750` is added so a hung agent turn does not
consume the whole experiment budget. Config narrows to `hopper-no-hint` only.

| eval_run | Definition | Test | turn-timeout | Notes |
|---|---|---|---|---|
| 2026-0428-1642 | mha_bwd_h48_d128 | ncu_mha_bwd_h48_d128 | 750 | first NCU + per-turn timeout |
| 2026-0428-1801 | mha_bwd_h48_d128 | ncu_mha_bwd_h48_d128 | 750 | rerun on same config (`d88b0e7 Remove truncation` lands here; agent stops truncating its observation buffer.) |
| 2026-0428-1910 | mha_with_lse_h48_d128 | ncu_mha_with_lse_h48_d128 | 750 | NCU on fwd-with-lse |



---

## Stage 11 — Flash-Attention-3-style prompts

Prompt-optimization stage: new "fa3" prompt configs (`hopper-010`, `-011`, etc.) seeded
with FA3 docs (`bfd1dec Add fa3 docs`, `b994dc5 Update tma doc`). Both forward (lse) and
backward use FA3-flavored prompts.

| eval_run | Definition | Config | turn-timeout | Notes |
|---|---|---|---|---|
| 2026-0503-2311 | mha_bwd_h48_d128 | 2026-0503-fa3-bwd.json | 360 | first FA3-bwd run |
| 2026-0503-2313 | mha_with_lse_h48_d128 | 2026-0503-fa3-fwd.json | 360 | FA3-fwd run |
| 2026-0504-0944 | mha_bwd_h48_d128 | 2026-0503-fa3-bwd.json | 960 | NCU test, longer per-turn budget |
| 2026-0504-1206 | mha_bwd_h48_d128 | **2026-0504-fa3-bwd.json** | 360 | revised FA3-bwd config (post `dd0c05b Update prompts`) |
| 2026-0504-1455 | mha_bwd_h48_d128 | 2026-0504-fa3-bwd.json | 960 | NCU + longer per-turn |

# Stage 12 - Remove D from MHA Bwd
Rename previous mha_bwd_h48_d128 to mha_bwd_wd_h48_d128. From now on, mha_bwd_h48_d128 has no D input, and mha_bwd_wd_h48_d128 has D input.
| eval_run | Definition | Config | turn-timeout | Notes |
| 2026-0504-1900 | mha_bwd_h48_d128 | 2026-0504-fa3-bwd.json | 360 | post `25171a1 Remove D` |
| 2026-0505-0017 | mha_bwd_h48_d128 | 2026-0504-fa3-bwd.json | 360 | NCU+ post `49a8b9f Use cc to fix numerical error`, timeout 28800 |
| 2026-0505-0018 | mha_bwd_h48_d128 | 2026-0504-fa3-bwd.json | 360 | no NCU of 2026-0505-0017 |

---

## Auxiliary commands (not eval runs)

For reference, `experiments.jsonl` also logs:

- `plots/plot_success_trajectories.py`, `plots/plot_token_breakdown.py`,
  `plots/plot_turn_categories.py` — post-run analysis
- `analyze_patterns_batch.py` (`2026-0420-2305`, `2026-0422-1002`,
  `2026-0422-2352_complete`) — per-run AST pattern analysis
- `analyze_patterns_cross_run.py` (`2026-0422-2352_complete` vs `2026-0422-1002`,
  Kimi-K2.6 vs gemini-3.1-pro-preview) — cross-model diversity comparison
- `draw_pattern_trees_batch.py` (`2026-0420-2305`) — pattern-tree visualization
- The retired scan/rerun/merge helpers recovered container-killed experiments
  in `2026-0422-2352`; current launchers use integrated resume handling in
  `resume_utils.py`.
- `bash …/tasks/launch_*.sh` and `…/tasks/plot_*.sh` — orchestration scripts that wrap
  the launchers/plotters

These are excluded from the stage tables above because they do not produce new
`eval_runs/<id>` directories (or, in the rerun case, produce an `<id>_rerun` /
`<id>_complete` derivative tied to an existing run).

# Launch flashinfer-bench server:
```
TVM_FFI_CUDA_ARCH_LIST="9.0a" flashinfer-bench serve --local /home/ubuntu/accrl-training --port 10000 --timeout 20 --atol 1e-2 --rtol 1e-2
```

# Experiments usable for SFT 
## /home/ubuntu/accrl-shared/drive/glm51_full4def_reverse_reasoning_no_thinking_fixed_history_20260527/reasoning_pairs.jsonl
gemm_n7168_k5120:
- 2026-0422-1002
- 2026-0422-1538
- 2026-0422-1834

mha_with_lse_h48_d128:
- 2026-0503-2313
- 2026-0504-0944 (+ncu)

mha_bwd_h48_d128:
- 2026-0506-0930
- 2026-0506-1130 (+ncu)

fp8_gemm_nt_1d2d_n4096_k7168:
- 2026-0507-1828
- 2026-0507-2120

## mha focused
mha_with_lse_h48_d128:
- 2026-0503-2313

mha_with_lse_d128
- 2026-0529-2340

mha_with_lse_d128_causal:
- 2026-0530-0140
- 2026-0609-2140

mha_bwd_h48_d128:
- 2026-0506-0930

mha_bwd_d128:
- 2026-0530-0340
- 2026-0609-2240

mha_bwd_d128_causal:
- 2026-0609-2340

mha_with_lse_d64:
- 2026-0610-1120

mha_with_lse_d64_causal:
- 2026-0610-1220

mha_bwd_d64:
- 2026-0610-1320

mha_bwd_d64_causal:
- 2026-0610-1420


# Experiments with potentially low quality
## Lanuch more jobs than GPUs
mha_bwd_h48_d128:
- 2026-0504-1206
- 2026-0504-1455 (+ncu)
