# `/evaluate` Pseudocode After the GPU Memory Gate

This is Python-like pseudocode for the current `/evaluate` call path after
`_GPUWorkerThread._admit_baseline()` accepts a workload. Process boundaries are
explicit because the reference baseline and candidate kernel execute in different
processes.

## 1. Top-level workload loop

Source: `flashinfer_bench/serve/scheduler.py:453`

```python
# ============================================================
# PROCESS A: FastAPI/backend process
# THREAD: gpu-worker-cuda:N
# Actual class: _GPUWorkerThread
# ============================================================

def _evaluate_task(task):
    definition = trace_set.definitions[task.definition_name]
    workloads = resolve_workloads(task)
    cfg = apply_request_overrides(
        base_config,
        timeout=task.timeout,
        atol=task.atol,
        rtol=task.rtol,
    )

    traces = []

    for workload in workloads:
        ref_handle = _get_or_build_ref(
            definition,
            workload,
            cfg=cfg,
            profile_baseline=task.profile_baseline,
            run_baseline=task.run_baseline,
        )

        evaluation = gpu_worker.run_solution(
            solution=task.solution,
            baseline=ref_handle,
            cfg=cfg,
        )

        traces.append(
            Trace(
                definition=definition.name,
                workload=workload,
                solution=task.solution.name,
                evaluation=evaluation,
            )
        )

        # A timeout means the child may still be executing CUDA.
        if evaluation.status == TIMEOUT:
            _restart_worker(force=True)

        # Other failures get a CUDA health check first.
        elif evaluation.status != PASSED:
            if not gpu_worker.is_healthy():
                _restart_worker(force=True)

    return traces
```

## 2. Baseline construction after the memory gate

Source: `flashinfer_bench/serve/scheduler.py:609`

```python
# PROCESS A: backend process
# THREAD: gpu-worker-cuda:N

def _get_or_build_ref(definition, workload, cfg, ...):
    cache_key = (
        definition.name,
        workload.uuid,
        profile_baseline,
        run_baseline,
    )

    if cache_key in baseline_cache:
        return baseline_cache[cache_key]

    resolved_cfg = cfg.resolve_eval_config(definition, ...)
    estimated_baseline_bytes = estimate_baseline_bytes(...)
    estimated_eval_bytes = estimate_eval_reserve_bytes(...)

    # GPU MEMORY GATE
    _admit_baseline(
        definition,
        workload,
        estimated_baseline_bytes,
        estimated_eval_bytes,
    )

    # Everything below occurs only after admission succeeds.
    handle = gpu_worker.run_ref(
        definition,
        workload,
        cfg,
        trace_set_root,
        profile_baseline=profile_baseline,
        run_baseline=run_baseline,
    )

    baseline_cache[cache_key] = handle
    return handle
```

`gpu_worker` is a `PersistentSubprocessWorker` object, but `run_ref()` executes
in the backend process. It does not send baseline construction to the candidate
worker subprocess.

Source: `flashinfer_bench/bench/runner/persistent_runner.py:316`

```python
# PROCESS A: backend process

class PersistentSubprocessWorker:
    def run_ref(definition, workload, cfg, ...):
        evaluator_class = resolve_evaluator(definition)
        eval_cfg = cfg.resolve_eval_config(definition, ...)

        baseline = evaluator_class.build_baseline(
            definition=definition,
            workload=workload,
            cfg=eval_cfg,
            device=self.device,
            trace_set_root=trace_set_root,
        )

        # Baseline owns inputs, reference outputs, and reference latency.
        self.baselines[baseline.handle] = baseline
        return baseline.handle
```

Default baseline construction is approximately:

Source: `flashinfer_bench/bench/evaluators/default.py:41`

```python
# PROCESS A: backend process
# CUDA context and reference tensors live here.

def DefaultEvaluator.build_baseline(definition, workload, cfg, device, root):
    safe_tensors = load_required_safetensors(workload, root)
    inputs = []
    reference_outputs = []

    if not cfg.run_baseline:
        stored_outputs = load_safetensors_outputs(...)
        stored_outputs = move_to_gpu(stored_outputs, device)

        for trial in range(cfg.num_trials):
            trial_inputs = gen_inputs(
                definition, workload, device, safe_tensors
            )
            inputs.append(trial_inputs)
            reference_outputs.append(clone(stored_outputs))

        reference_latency_ms = 0.0

    else:
        # definition.reference becomes a Python pseudo-solution.
        reference_runnable = BuilderRegistry.build_reference(definition)

        for trial in range(cfg.num_trials):
            trial_inputs = gen_inputs(
                definition, workload, device, safe_tensors
            )
            inputs.append(trial_inputs)

            with torch.no_grad():
                result = reference_runnable(*trial_inputs)

            torch.cuda.synchronize(device)
            reference_outputs.append(normalize_result(result))

        if cfg.profile_baseline:
            latencies = []

            for trial_inputs in inputs:
                latency = time_runnable(
                    reference_runnable,
                    trial_inputs,
                    warmup=cfg.warmup_runs,
                    iters=cfg.iterations,
                    device=device,
                )
                latencies.append(latency)

            reference_latency_ms = mean(latencies)
        else:
            reference_latency_ms = 0.0

    return DeviceBaseline(
        handle=new_uuid(),
        definition=definition,
        device=device,
        inputs=inputs,
        outputs=reference_outputs,
        mean_latency_ms=reference_latency_ms,
    )
```

## 3. Send the candidate evaluation to the child process

Source: `flashinfer_bench/bench/runner/persistent_runner.py:340`

```python
# PROCESS A: backend process
# Object: PersistentSubprocessWorker

def run_solution(solution, baseline_handle, cfg):
    baseline = self.baselines[baseline_handle]

    failure = should_skip_solution(solution.name)
    if failure:
        return Evaluation(
            status=failure.last_status,
            log="Solution skipped after repeated failures",
        )

    # The parent creates the log so it can recover output if the child dies.
    log_path = tempfile.mkstemp(prefix="fib_", suffix=".log")

    message = {
        "cmd": "run_solution",
        "definition": baseline.definition,
        "solution": solution,

        # GPU tensors cross the multiprocessing connection.
        "inputs": baseline.inputs,
        "ref_outputs": baseline.outputs,

        "ref_mean_latency_ms": baseline.mean_latency_ms,
        "config": cfg,
        "log_path": log_path,
    }

    parent_pipe.send(message)

    # This timeout covers IPC receipt, compilation/cache lookup, input
    # cloning, correctness runs, and performance runs.
    if parent_pipe.poll(timeout=cfg.timeout_seconds):
        response = parent_pipe.recv()

        if response.cmd == "evaluation":
            evaluation = response.evaluation

            if evaluation.status == PASSED:
                clear_failure_record(solution.name)
            elif evaluation.status in {
                RUNTIME_ERROR,
                INCORRECT_SHAPE,
                INCORRECT_DTYPE,
                COMPILE_ERROR,
            }:
                record_failure(solution.name, evaluation)

            remove_log_file_if_still_present(log_path)
            return evaluation

        if response.cmd == "error":
            record_runtime_failure(solution.name)
            return make_eval(
                status=RUNTIME_ERROR,
                log_path=log_path,
                extra_msg=response.error,
            )

        return make_eval(
            status=RUNTIME_ERROR,
            log_path=log_path,
            extra_msg="Unexpected worker response",
        )

    # poll() expired. The child has not necessarily stopped.
    record_failure(solution.name, status=TIMEOUT)

    return make_eval(
        status=TIMEOUT,
        log_path=log_path,
        extra_msg=f"Evaluation timed out after {cfg.timeout_seconds}s",
    )
```

The caller sees `TIMEOUT` and force-restarts the worker process group before
accepting another task on this GPU lane.

## 4. Child receives, compiles, and evaluates the candidate

Source: `flashinfer_bench/bench/runner/persistent_runner.py:737`

```python
# ============================================================
# PROCESS B: long-lived spawned Python process for cuda:N
# Function: _persistent_worker_main
# ============================================================

def persistent_worker_main(pipe, device):
    os.setsid()                 # Own process group
    set_parent_death_signal()   # Die if backend parent dies
    torch.cuda.set_device(device)

    registry = BuilderRegistry.get_instance()
    pipe.send({"cmd": "ready"})

    while True:
        message = pipe.recv()

        if message.cmd == "run_solution":
            definition = message.definition
            solution = message.solution
            baseline_inputs = message.inputs
            reference_outputs = message.ref_outputs
            reference_latency = message.ref_mean_latency_ms
            cfg = message.config

            # Redirect file descriptors 1 and 2 to the parent-created file.
            log_path = redirect_stdio_to_tempfile(message.log_path)

            try:
                runnable = registry.build(definition, solution)

                # Clone received inputs for this evaluation's working set.
                inputs = [
                    [
                        value.clone()
                        if isinstance(value, torch.Tensor)
                        else value
                        for value in trial
                    ]
                    for trial in baseline_inputs
                ]

                evaluator_class = resolve_evaluator(definition)
                eval_cfg = cfg.resolve_eval_config(definition)

                evaluation = evaluator_class.evaluate(
                    definition=definition,
                    sol_runnable=runnable,
                    inputs=inputs,
                    ref_outputs=reference_outputs,
                    ref_mean_latency_ms=reference_latency,
                    cfg=eval_cfg,
                    log_path=log_path,
                    device=device,
                )

                pipe.send({
                    "cmd": "evaluation",
                    "evaluation": evaluation,
                })

            except BuildError:
                traceback.print_exc()

                evaluation = make_eval(
                    status=COMPILE_ERROR,
                    device=device,
                    log_path=log_path,
                )
                pipe.send({"cmd": "evaluation", "evaluation": evaluation})

            except Exception:
                traceback.print_exc()

                evaluation = make_eval(
                    status=RUNTIME_ERROR,
                    device=device,
                    log_path=log_path,
                )
                pipe.send({"cmd": "evaluation", "evaluation": evaluation})

            finally:
                # Drop IPC and cloned tensor references.
                inputs = None
                baseline_inputs = None
                reference_outputs = None
                message = None

                # Return unused PyTorch allocator blocks to the driver.
                torch.cuda.empty_cache()
```

## 5. Compilation and cache lookup inside the child

Source: `flashinfer_bench/compile/registry.py:93`

```python
# PROCESS B: candidate worker subprocess

def BuilderRegistry.build(definition, solution):
    solution_hash = solution.hash()

    # Process-local Runnable or BuildError cache.
    if solution_hash in runnable_cache:
        cached = runnable_cache[solution_hash]

        if isinstance(cached, BuildError):
            raise cached

        return cached

    # Cross-process compilation lock.
    with FileLock(
        FIB_CACHE_PATH / "build_locks" / f"{solution_hash}.lock"
    ):
        builder = first_builder_that_can_build(solution)

        try:
            runnable = builder.build(definition, solution)
        except BuildError as error:
            runnable_cache[solution_hash] = error
            raise

        runnable_cache[solution_hash] = runnable
        return runnable
```

For normal CUDA with the default TVM-FFI binding:

Source: `flashinfer_bench/compile/builders/tvm_ffi_builder.py:234`

```python
def TVMFFIBuilder.build(definition, solution):
    package_name, build_path = derive_cache_path(solution)
    entry_symbol = solution.spec.entry_point.symbol

    if cached_sources_and_shared_library_match(build_path, solution):
        shared_library = build_path / f"{package_name}.so"

    else:
        write_sources_to_path(build_path, solution.sources)
        cpp_files, cuda_files = classify_sources(solution.sources)

        shared_library = tvm_ffi.cpp.build(
            name=package_name,
            cpp_files=cpp_files,
            cuda_files=cuda_files,
            extra_include_paths=[build_path],
            extra_cuda_cflags=["-lineinfo"],
            extra_ldflags=[
                "-L<cuda-lib-path>",
                "-lcuda",
                "-lcublas",
            ],
            build_directory=build_path,
        )

        # tvm_ffi.cpp.build internally:
        #   generate build.ninja
        #   ninja -v
        #     -> nvcc kernel.cu -c -o cuda_0.o
        #     -> c++ cuda_0.o -shared ... -o package.so

    # Fail immediately on unresolved dynamic symbols.
    ctypes.CDLL(shared_library, mode=RTLD_NOW)

    module = tvm_ffi.load_module(shared_library)
    exported_function = getattr(module, entry_symbol)

    return Runnable(
        callable=exported_function,
        metadata={
            "build_type": "tvm_ffi",
            "destination_passing_style":
                solution.spec.destination_passing_style,
            "binary": shared_library,
        },
    )
```

## 6. Correctness followed by performance

Source: `flashinfer_bench/bench/evaluators/evaluator.py:67`

```python
# PROCESS B: candidate worker subprocess

def Evaluator.evaluate(...):
    correctness, failure = check_correctness(...)

    if failure is not None:
        return failure

    performance, failure = eval_performance(...)

    if failure is not None:
        return failure

    return make_eval(
        status=PASSED,
        correctness=correctness,
        performance=performance,
        log_path=log_path,
    )
```

Default correctness evaluation:

Source: `flashinfer_bench/bench/evaluators/default.py:115`

```python
def check_correctness(runnable, inputs, reference_outputs, cfg, device):
    max_absolute_error = 0
    max_relative_error = 0
    numerically_incorrect = False

    for trial, trial_inputs in enumerate(inputs):
        try:
            if runnable.destination_passing_style:
                outputs = allocate_outputs(
                    definition, trial_inputs, device
                )

                with torch.no_grad():
                    # TVM-FFI host wrapper launches the CUDA kernel here.
                    runnable(*trial_inputs, *outputs)

            else:
                with torch.no_grad():
                    result = runnable(*trial_inputs)

                outputs = normalize_result(result)

            # Surface asynchronous CUDA launch errors on the host.
            torch.cuda.synchronize(device)

        except Exception:
            return None, make_eval(status=RUNTIME_ERROR)

        for candidate, reference in zip(
            outputs, reference_outputs[trial]
        ):
            if candidate.shape != reference.shape:
                return None, make_eval(status=INCORRECT_SHAPE)

            if candidate.dtype != reference.dtype:
                return None, make_eval(status=INCORRECT_DTYPE)

            if contains_nan_or_inf(candidate):
                return errors, make_eval(status=INCORRECT_NUMERICAL)

            abs_error, rel_error, exceeds_tolerance = (
                compute_error_stats(candidate, reference, cfg)
            )

            max_absolute_error = max(max_absolute_error, abs_error)
            max_relative_error = max(max_relative_error, rel_error)
            numerically_incorrect |= exceeds_tolerance

    correctness = Correctness(
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
    )

    if numerically_incorrect:
        return correctness, make_eval(status=INCORRECT_NUMERICAL)

    return correctness, None
```

Default performance evaluation:

Source: `flashinfer_bench/bench/evaluators/default.py:217`

```python
def eval_performance(runnable, inputs, reference_latency, cfg, device):
    candidate_latencies = []

    for trial_inputs in inputs:
        if runnable.destination_passing_style:
            outputs = allocate_outputs(
                definition, trial_inputs, device
            )
            arguments = [*trial_inputs, *outputs]
        else:
            arguments = trial_inputs

        latency = time_runnable(
            runnable,
            arguments,
            warmup=cfg.warmup_runs,
            iters=cfg.iterations,
            device=device,
        )
        candidate_latencies.append(latency)

    candidate_latency = mean(candidate_latencies)

    return Performance(
        latency_ms=candidate_latency,
        reference_latency_ms=reference_latency,
        speedup_factor=reference_latency / candidate_latency,
    ), None
```

Source: `flashinfer_bench/bench/timing.py:47`

```python
def time_runnable(runnable, args, warmup, iterations, device):
    with per_device_multiprocessing_lock(device):
        with torch.cuda.device(device):
            samples = bench_gpu_time_with_cupti(
                fn=runnable,
                dry_run_iters=warmup,
                repeat_iters=iterations,
                input_args=tuple(args),
                cold_l2_cache=True,
                use_cuda_graph=False,
            )

    return median(samples)
```

## Condensed process boundary

```text
Backend process:
    GPU memory gate
    -> build or load reference baseline
    -> retain baseline GPU tensors
    -> send tensors and solution through multiprocessing Pipe
    -> enforce the wall-clock timeout

Persistent child process:
    compile or load candidate
    -> clone inputs
    -> launch candidate for correctness
    -> launch candidate repeatedly for timing
    -> return Evaluation through multiprocessing Pipe

Backend GPU worker thread:
    wrap Evaluation in Trace
    -> on timeout, kill and replace the child process group
    -> on another failure, health-check and replace it if unhealthy
```
