# Profiling Service Improvements Since `f7b4d8d185625ab2d609233a1a06e99ee18a0c6b`

Baseline: `f7b4d8d185625ab2d609233a1a06e99ee18a0c6b` (`feat: add top_k_top_p_sampling_from_probs_v202048 definition`).

Current `flashinfer-bench` head inspected: `80dffabaaa3fe74c6c2b119b0e2f968336f152fd`.

This note is for `flashinfer-bench` profiling-service designers. It focuses on the service API, worker lifecycle, memory behavior, launch/restart design, and deployment-wrapper lessons. Workflow-specific run bookkeeping is intentionally omitted.

## High-Level Summary

The profiling service changed from a mostly single-backend evaluation server into a multi-task profiling/debug service with:

- explicit `/evaluate`, `/profile`, `/sanitize`, and `/debug` task types;
- a dispatcher that fronts multiple per-GPU backends and routes work by queue health;
- non-invasive health checks that avoid perturbing busy workers;
- managed subprocess/process-group cleanup for NCU, compute-sanitizer, debug, timeouts, and shutdown;
- memory-aware baseline-cache admission using real CUDA memory readings;
- per-request evaluation overrides for timeout, tolerances, and baseline behavior;
- stored-output workloads so deterministic traces can skip live baseline execution;
- direct and dispatcher launch scripts for tmux-managed service processes;
- external restart wrappers that validate CUDA/container state, filter hot GPUs, clear stale GPU holders, and relaunch only the service layer;
- supervisor scripts that wait for real service readiness and clean orphaned `_solution_runner` jobs before restarting the service.

## Service API Improvements

The server now exposes task-specific endpoints instead of only basic evaluation:

- `/evaluate` accepts per-request overrides: `profile_baseline`, `run_baseline`, `timeout`, `atol`, and `rtol`.
- `/profile` runs NCU against submitted solutions, with controls for NCU set, sections, kernel name, page, path, timeout, and line truncation.
- `/sanitize` runs compute-sanitizer with selectable sanitizer tools, path, timeout, line limit, and print limit.
- `/debug` runs source-line-focused CUDA debugging, including sanitizer output, coredump support, source context, and structured metadata.
- `/tasks/{task_id}` and `/tasks/batch` now return task `kind`, evaluation traces, or profiling/debug `logs` depending on task type.
- `/` reports server version plus git commit/upstream metadata, which makes live service identity easier to verify.

Relevant files:

- `flashinfer_bench/serve/app.py`
- `flashinfer_bench/serve/task_store.py`
- `flashinfer_bench/__init__.py`

## Dispatcher and Multi-Backend Service

A new HTTP dispatcher was added at `flashinfer_bench/serve/dispatcher.py`.

Key behavior:

- Fronts multiple backend `flashinfer-bench serve` processes behind one URL.
- Routes submit endpoints (`/evaluate`, `/profile`, `/sanitize`, `/debug`) to the least-loaded healthy backend by queue size.
- Tracks `task_id -> backend URL` so task polling goes back to the backend that owns the task.
- Splits `/tasks/batch` by backend and returns results in the original request order.
- Aggregates backend health into dispatcher `/health`, including per-backend queue size and healthy state.
- Uses a separate health client from the long-timeout request client, so health probes are not blocked by long-running profile/debug calls.

Launch support:

- `scripts/launch_fib_serve_dispatcher.sh` starts one backend per GPU in tmux and fronts them with the dispatcher.
- `scripts/launch_fib_serve_direct.sh` starts a single direct backend without the dispatcher.
- Both launchers validate dataset roots and config paths, support one or more local trace roots, and restart backend tmux loops after service crashes.

## Worker Lifecycle and Health

Worker handling is much more defensive now.

Improvements:

- Health checks use `is_available`, which is non-invasive: it reports process liveness without forcing expensive worker commands while the worker is busy.
- Dead idle workers can be restarted by health checks.
- If a worker cannot start or restart, the backend exits so the tmux launcher/supervisor can restart the backend cleanly.
- Evaluation `TIMEOUT` now forces a worker restart because the CUDA state is unknown.
- Runtime failures, sanitizer runs, and debug runs trigger worker-health checks and restart unhealthy workers.
- Service shutdown calls `kill_all_tracked_subprocesses()` before joining workers, so in-flight NCU/sanitizer/debug subprocess groups do not continue holding GPUs after the service exits.
- `PersistentSubprocessWorker` has runner-owned baseline clearing, process-group handling, worker restart hardening, and parent-side timeout handling.
- `_solution_runner` calls `set_parent_death_signal()` so it is killed if its direct profiling parent (`ncu` or `compute-sanitizer`) dies.

Relevant files:

- `flashinfer_bench/serve/scheduler.py`
- `flashinfer_bench/bench/runner/persistent_runner.py`
- `flashinfer_bench/utils.py`
- `flashinfer_bench/agents/_solution_runner.py`

## Memory and Baseline Cache Improvements

The serve scheduler now includes per-worker memory admission for cached baselines.

Behavior:

- Estimates cached baseline bytes from workload schema and `num_trials`.
- Estimates live evaluation reserve bytes separately, including the subprocess copy and float32 correctness-check working set.
- Reads actual device occupancy via `torch.cuda.mem_get_info()`.
- Checks `required_bytes + real_used_bytes` against `max_mem_ratio * total_hbm`.
- If the next workload will not fit, clears all cached baselines for that worker, clears CUDA allocator cache, re-reads real memory, and only then fails.
- Clears both scheduler tracking and runner-owned baseline handles, so drift between parent and worker state does not leave memory pinned.
- Cache keys now include `profile_baseline` and `run_baseline`, preventing incompatible baseline reuse.

This replaced earlier fragile behavior where cached baselines could accumulate until a restart was needed to make profiling succeed.

Relevant files:

- `flashinfer_bench/serve/scheduler.py`
- `flashinfer_bench/bench/config.py`
- `flashinfer_bench/bench/runner/persistent_runner.py`
- `tests/serve/test_mem_admission.py`

## Evaluation Override and Stored-Output Support

Evaluation can now be adjusted per request without restarting the server.

Important changes:

- `timeout` on `/evaluate` maps to `BenchmarkConfig.timeout_seconds` for that task only.
- `atol` and `rtol` can be overridden per request.
- `profile_baseline=False` keeps correctness evaluation but reports zero reference latency.
- `run_baseline=False` skips live reference execution and loads reference outputs from `workload.outputs`.
- `run_baseline=False` validates that all selected workloads have stored outputs and deterministic safetensors/scalar inputs.
- Workloads now support optional `outputs` safetensors mappings.
- `TraceSet.from_paths(...)` can merge multiple trace roots and absolutize safetensors paths from secondary roots.
- `flashinfer-bench serve --local` now accepts multiple roots.
- Config loading through `--config` is supported for serve, and CLI overrides only override explicitly provided values.

Relevant files:

- `flashinfer_bench/data/workload.py`
- `flashinfer_bench/data/trace_set.py`
- `flashinfer_bench/bench/evaluators/default.py`
- `flashinfer_bench/bench/config.py`
- `flashinfer_bench/cli/main.py`
- `tests/serve/test_evaluate_overrides.py`
- `tests/data/test_workload_outputs.py`
- `tests/data/test_trace_set.py`

## Profiling and Debug Tooling

The service now has a source-oriented CUDA debug path in addition to raw profiling.

Improvements:

- NCU execution is exposed through `/profile`.
- Compute-sanitizer execution is exposed through `/sanitize`.
- `/debug` combines sanitizer output, source-location extraction, coredump triggering, optional `cuda-gdb` backtraces, and structured metadata sections.
- Debug output is condensed into higher-signal sections such as precise CUDA fault, coredump analysis, primary diagnostics, source locations, and retained coredump paths.
- NCU and sanitizer invocations use managed subprocesses, so timeouts kill the whole process group rather than only the immediate wrapper process.
- Debug/NCU/sanitizer helpers write temporary solution/workload artifacts and can load trace-set context for safetensors data.
- `nvidia-cudnn-frontend<1.22` is pinned to avoid cuDNN backend mismatch failures with the local cuDNN 9.20 stack.
- Non-finite checks no longer fail on low-bit dtypes where `torch.isinf` / `torch.isnan` are unsupported.

Relevant files:

- `flashinfer_bench/agents/debug.py`
- `flashinfer_bench/agents/ncu.py`
- `flashinfer_bench/agents/sanitizer.py`
- `flashinfer_bench/utils.py`
- `flashinfer_bench/bench/evaluators/default.py`
- `pyproject.toml`
- `tests/agent/test_debug.py`

## Profile-Service Launch and Restart Scripts

Current launch flow in this repo:

- `scripts/launch_fib_serve_dispatcher.sh`
  - starts backend tmux sessions named like `${TMUX_PREFIX}-backend-N`;
  - starts `${TMUX_PREFIX}-dispatcher`;
  - serves one or more dataset roots via `--local`;
  - passes `--config`, `--devices`, `--timeout`, host, and ports;
  - loops each backend so it restarts after a crash.

- `scripts/launch_fib_serve_direct.sh`
  - starts one backend on one device;
  - checks that the target port is free before launching;
  - supports one or more dataset roots;
  - loops the backend under tmux.

External deployment-wrapper behavior:

- [`envs/profile_service/restart_profiling.sh`](https://github.com/zhang677/AccRL/blob/62e5a814f4971bdb6ffd47b8036898678c98ade0/envs/profile_service/restart_profiling.sh)
  - removes all tmux sessions for `TMUX_PREFIX` instead of rerunning full setup;
  - verifies dataset roots, config, venv, `flashinfer-bench`, and dispatcher dependencies;
  - runs CUDA/NVML preflight before killing/restarting the service when CUDA is broken;
  - filters service devices through thermal preflight;
  - kills stale TVM-FFI compiler process groups;
  - terminates leftover GPU processes visible inside `fib-profile`;
  - refuses to continue if the dispatcher port remains held by a non-tmux process;
  - relaunches the dispatcher launcher and verifies that the dispatcher tmux session and port come up.

GPU temperature handling is in the external wrapper, not in `flashinfer-bench` core:

- [`envs/profile_service/gpu_preflight.sh`](https://github.com/zhang677/AccRL/blob/62e5a814f4971bdb6ffd47b8036898678c98ade0/envs/profile_service/gpu_preflight.sh)
  - queries `nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits`;
  - keeps only requested GPUs whose temperature is below `FIB_GPU_MAX_TEMP_C`;
  - defaults `FIB_GPU_MAX_TEMP_C=40`;
  - excludes GPUs with unavailable temperature readings;
  - can be bypassed with `FIB_SKIP_GPU_THERMAL_PREFLIGHT=1`.

- [`envs/profile_service/run_container.sh`](https://github.com/zhang677/AccRL/blob/62e5a814f4971bdb6ffd47b8036898678c98ade0/envs/profile_service/run_container.sh)
  - filters `FIB_DEVICES` before creating the long-lived container.

- [`envs/profile_service/restart_profiling.sh`](https://github.com/zhang677/AccRL/blob/62e5a814f4971bdb6ffd47b8036898678c98ade0/envs/profile_service/restart_profiling.sh)
  - removes leftover GPU processes first, then filters `SERVICE_DEVICES`, then relaunches the dispatcher/backends.

## Supervisor Readiness Lessons

The latest external supervisors inspected:

- [`fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/01_watch_gemini_fixit.sh`](https://github.com/zhang677/AccRL/blob/62e5a814f4971bdb6ffd47b8036898678c98ade0/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/01_watch_gemini_fixit.sh)
- [`fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/07_watch_v5_12defs_eval.sh`](https://github.com/zhang677/AccRL/blob/62e5a814f4971bdb6ffd47b8036898678c98ade0/fib_runtime/multiturn/construct_eval_scripts/fixit-v5-scripts/07_watch_v5_12defs_eval.sh)

Service-design lessons from those wrappers:

- Supervisors should avoid launching duplicate clients for the same service lane.
- Restart should be gated on service-level need, not workflow summary files.
- Readiness should require `/health` status `ok`, a non-empty backend/worker list, all backends/workers healthy, `queue_size == 0`, and at least one definition lookup succeeding.
- Container CUDA should be checked before relaunching the service layer.
- Orphaned `flashinfer_bench.agents._solution_runner` processes should be terminated before restarting the service, because they can keep GPUs busy after NCU/sanitizer timeouts.
- Non-interactive container restarts should use `bash -lc`, avoiding interactive shell job-control warnings.

## Tests Added Around the Service

Focused tests now cover the new service behavior:

- `tests/serve/test_dispatcher.py`: dispatcher forwarding and task tracking, including `/debug`.
- `tests/serve/test_evaluate_overrides.py`: timeout, `profile_baseline`, `run_baseline`, stored-output validation, and correctness with stored outputs.
- `tests/serve/test_mem_admission.py`: baseline byte estimates, eval reserve estimates, cache clearing, and early failure behavior without requiring CUDA.
- `tests/serve/test_serve_api.py`: expanded task response/API behavior.
- `tests/agent/test_debug.py`: debug report/metadata behavior.
- `tests/bench/test_persistent_runner.py`: persistent worker lifecycle and baseline handling.
- `tests/data/test_trace_set.py` and `tests/data/test_workload_outputs.py`: multiple roots and stored-output workload support.

## Practical Lessons

- Treat `/health` plus `queue_size == 0` and a definition probe as the readiness gate, not just an open port.
- For dispatcher-backed services, health may report `backends`; for direct services, it may report `workers`. Watchers correctly accept either shape.
- Restarting the service should clear tmux sessions, stale compiler groups, stale GPU processes, and orphaned `_solution_runner` jobs before relaunching backends.
- Supervisor restart decisions should be based on service state and whether client work remains, not stale workflow summary files.
- Memory failures should first be debugged against real GPU occupancy and baseline-cache state. The current service clears owned cached baselines under pressure before failing.
- Use stored outputs plus `run_baseline=False` for deterministic safetensors workloads when live baseline cost is unnecessary.
- Use `bash -lc` for non-interactive container restarts unless interactive job control is truly needed.
- GPU-temperature filtering currently belongs to the external deployment wrapper: filter requested devices before service launch/restart, and make the temperature threshold explicit.
