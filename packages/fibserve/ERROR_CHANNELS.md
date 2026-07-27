# Error Channels And Process Ownership

This note summarizes how `flashinfer-bench serve` launches work, how failures are
reported to clients, and which layer is responsible for killing child processes.

## Execution Layers

### Serve Backend

`flashinfer_bench/cli/main.py` starts a FastAPI/uvicorn backend. The backend owns:

- one in-memory `TaskStore`
- one shared FIFO task queue
- one `_GPUWorkerThread` Python thread per configured CUDA device

Each `_GPUWorkerThread` is a Python thread backed by an OS thread. It pulls task
IDs from the queue, marks the task running, executes the endpoint-specific path,
then marks the task completed or failed.

Code:

- `flashinfer_bench/serve/scheduler.py::_GPUWorkerThread.run`
- `flashinfer_bench/serve/task_store.py`

### `/evaluate`

`/evaluate` uses a long-lived GPU-owning OS subprocess:

```text
uvicorn backend OS process
  -> Python/OS thread: gpu-worker-cuda:N
     -> PersistentSubprocessWorker OS process
        -> CUDA context on cuda:N
        -> benchmark/evaluator code
           -> CUDA kernels
              -> GPU threads / warps / blocks
```

The persistent worker is created with `torch.multiprocessing` using `spawn`. The
child calls `os.setsid()` so it has its own process group, installs a parent
death signal, sets the CUDA device, then waits for pipe commands.

Normal build/runtime exceptions inside the worker are caught and returned over
the pipe as `Evaluation` objects. If the worker process dies or the pipe breaks,
the parent maps the communication failure to an `EvaluationStatus.RUNTIME_ERROR`
where possible.

Code:

- `flashinfer_bench/bench/runner/persistent_runner.py::PersistentSubprocessWorker`
- `flashinfer_bench/bench/runner/persistent_runner.py::_persistent_worker_main`

### `/profile`

`/profile` runs NVIDIA Nsight Compute (`ncu`) as a managed subprocess. `ncu`
then launches `_solution_runner`:

```text
uvicorn backend OS process
  -> Python/OS thread: gpu-worker-cuda:N
     -> ncu OS process
        -> python -m flashinfer_bench.agents._solution_runner
           -> CUDA context on cuda:N
           -> CUDA kernels
```

This path does not primarily use the persistent evaluation subprocess. It uses a
separate tool process tree for each workload.

Code:

- `flashinfer_bench/serve/scheduler.py::_profile_task`
- `flashinfer_bench/agents/ncu.py`
- `flashinfer_bench/agents/_solution_runner.py`

### `/sanitize`

`/sanitize` runs `compute-sanitizer` as a managed subprocess. The sanitizer then
launches `_solution_runner`:

```text
uvicorn backend OS process
  -> Python/OS thread: gpu-worker-cuda:N
     -> compute-sanitizer OS process
        -> python -m flashinfer_bench.agents._solution_runner
           -> CUDA context on cuda:N
           -> CUDA kernels
```

After each workload, the scheduler checks the persistent worker health and may
restart it, but the sanitizer run itself is a separate process tree.

Code:

- `flashinfer_bench/serve/scheduler.py::_sanitize_task`
- `flashinfer_bench/agents/sanitizer.py`

### `/debug`

`/debug` uses debug tooling around `_solution_runner`. The fast path uses
`compute-sanitizer`; timeout/coredump paths may also run a direct runner pass and
then inspect coredumps with `cuda-gdb`.

```text
uvicorn backend OS process
  -> Python/OS thread: gpu-worker-cuda:N
     -> compute-sanitizer or direct runner process group
        -> _solution_runner
     -> optionally cuda-gdb
```

Code:

- `flashinfer_bench/serve/scheduler.py::_debug_task`
- `flashinfer_bench/agents/debug.py`

## Kill Ownership

### Persistent `/evaluate` Worker

The parent tracks the worker process group ID. On shutdown or forced restart, it
tries to kill the entire worker process group with `os.killpg(..., SIGKILL)`.

This usually kills the persistent worker and any descendants in that process
group. It is not an absolute guarantee:

- if `os.setsid()` failed, the parent may not have a separate process group ID
- if the worker created grandchildren in a different process group, they may not
  be killed by the worker group kill
- `SIGKILL` kills Linux processes, but CUDA driver cleanup and GPU memory release
  can lag until the driver notices the context is gone
- if the backend itself is killed before its cleanup code runs, the worker should
  receive the parent-death signal, but only for processes that installed it

Code:

- `PersistentSubprocessWorker._kill_worker_process_group`
- `PersistentSubprocessWorker._shutdown_worker`
- `_persistent_worker_main` calls `os.setsid()` and `set_parent_death_signal()`

### Tool Process Groups (`/profile`, `/sanitize`, `/debug`)

Tool subprocesses use `run_managed_subprocess()`.

That helper starts the direct child in a new session/process group and registers
the `Popen` object in a weak set. On timeout, exception, or final cleanup, it
kills the whole process group with `SIGKILL`. Server shutdown also calls
`kill_all_tracked_subprocesses()` to kill in-flight tool process groups.

This usually kills `ncu`/`compute-sanitizer` and `_solution_runner`. It is not
absolute:

- a grandchild can escape if it starts its own process group/session
- kill can fail with permissions or process-race errors
- GPU memory cleanup is asynchronous from the service's point of view
- if a process becomes uninterruptible in kernel/driver state, the Linux process
  may not disappear immediately

Code:

- `flashinfer_bench/utils.py::run_managed_subprocess`
- `flashinfer_bench/utils.py::kill_all_tracked_subprocesses`

### Tmux Dispatcher Deployment

In AccRL profile-service usage, tmux is an external supervisor. It launches:

```text
fib-serve-dispatcher
fib-serve-backend-0 -> one GPU
fib-serve-backend-1 -> one GPU
...
fib-serve-backend-7 -> one GPU
```

The tmux restart script can kill and recreate these sessions, but it is outside
the in-memory `TaskStore`. Killing a backend or dispatcher discards its task map.
Clients with old task IDs can see connection resets, 5xx responses, timeouts, or
`404 Task not found` after a dispatcher restart.

## Error Channels

The current API has multiple, non-uniform error channels.

| Source | Mechanism | Task status | Client-visible field |
| --- | --- | --- | --- |
| `/evaluate` workload failure | worker returns `Evaluation` | `completed` | `traces[*].evaluation.status` |
| `/evaluate` timeout | parent times out waiting for worker pipe; worker is restarted | `completed` | `traces[*].evaluation.status = TIMEOUT` |
| `/evaluate` memory admission failure | scheduler raises before producing a trace | `failed` | top-level `error` |
| `/evaluate` worker pipe break | parent converts to runtime evaluation when possible | usually `completed` | `traces[*].evaluation.status = RUNTIME_ERROR` |
| Worker restart failure | backend calls `os._exit(1)` | no stable task result | connection reset / 5xx / timeout |
| `/profile` NCU timeout | tool helper returns error text | `completed` | `logs[*].log` |
| `/profile` NCU nonzero exit | tool helper returns error text | `completed` | `logs[*].log` |
| `/sanitize` timeout | tool helper appends error text | `completed` | `logs[*].log` |
| `/sanitize` detected issue | tool helper appends warning/log output | `completed` | `logs[*].log` |
| `/debug` timeout/coredump path | debug helper returns metadata | `completed` | `logs[*].metadata` |
| Bad request / missing definition before enqueue | FastAPI exception | no task | HTTP 400 |
| Unknown task after restart/TTL | task store lookup fails | no task | HTTP 404 |

## Abnormal GPU Subprocess Exit

### Persistent Worker Exits While Idle

Health checks use non-invasive liveness first. If the worker process is dead
while idle, `is_available` calls `_restart_worker("dead idle worker detected by
health check")`.

If restart succeeds:

- the worker process is replaced
- scheduler baseline cache is cleared
- future tasks can run

If restart fails:

- `_exit_backend()` calls `os._exit(1)`
- the uvicorn backend dies immediately
- all in-memory tasks in that backend are lost
- tmux or another supervisor may restart the backend, but task IDs from the old
  process no longer exist

### Persistent Worker Exits During `/evaluate`

If the worker dies while a solution is running, the parent usually detects one of
these:

- no pipe response before timeout
- `EOFError`
- `BrokenPipeError`
- `ConnectionResetError`
- decode/unpickle failure

The parent generally returns an `Evaluation` with `RUNTIME_ERROR` or `TIMEOUT`.
The scheduler appends that trace, then restarts the worker if the status is
`TIMEOUT` or if a health check fails after a non-passed status.

If the restart succeeds:

- the task can still be marked `completed`
- the failed workload is visible inside `traces`
- later workloads in the same `/evaluate` request may continue on a fresh worker

If the restart fails:

- `_exit_backend()` terminates the backend with `os._exit(1)`
- clients may see connection reset, 5xx, or timeout
- in-memory task state is lost

### Tool Subprocess Exits Abnormally

For `/profile`, `/sanitize`, and `/debug`, an abnormal `ncu`,
`compute-sanitizer`, direct runner, or `cuda-gdb` exit is usually converted into
text or metadata. The task is still marked `completed` unless an exception
escapes the scheduler method.

Examples:

- NCU timeout becomes `ERROR: NCU profiling timed out after ...`
- sanitizer timeout becomes `ERROR: memcheck timed out after ...`
- sanitizer nonzero exit becomes a warning/log payload
- debug timeout may trigger a direct coredump pass and return metadata

This means clients cannot rely on top-level `TaskStatus` to know whether the GPU
work succeeded.

## Current Issues

1. `TaskStatus` mixes transport success with benchmark success.

   A task can be `completed` even if every workload timed out or a tool returned
   `ERROR: ...`. Conversely, memory admission is `failed` because it raises as a
   scheduler exception before a trace/log object exists.

2. Endpoint result schemas are not uniform.

   Clients must inspect different fields:

   - `/evaluate`: `traces[*].evaluation.status`
   - `/profile`: `logs[*].log`
   - `/sanitize`: `logs[*].log`
   - `/debug`: `logs[*].metadata`
   - scheduler exceptions: top-level `error`

3. Partial results can be lost on task failure.

   If `_evaluate_task` produces some traces and then raises later, the outer
   loop calls `fail_task(task_id, error)` and does not attach the partial traces.

4. Backend death loses the task store.

   The task store is in memory. Any backend `os._exit(1)`, tmux restart, or
   dispatcher restart can invalidate task IDs.

5. Kill semantics are best-effort, not guaranteed.

   The service uses process groups and parent-death signals, which is the right
   shape, but it cannot guarantee that escaped grandchildren, driver-stuck
   processes, or externally supervised tmux processes are cleaned up instantly.

6. Persistent-worker and tool-worker health are coupled only indirectly.

   `/sanitize` and `/debug` check persistent-worker health after tool runs even
   though their GPU work happened in separate tool process trees. `/profile`
   does not do the same post-run persistent-worker health check.

7. Restart success is hidden from the task result.

   A workload may time out, trigger a worker restart, and the task may still
   complete. The client sees the failed trace but does not get a structured
   `worker_restarted=true` event.

## Suggested Direction

Introduce one uniform per-workload result envelope for all endpoints:

```text
task.status: pending | running | completed | failed
task.error: scheduler/backend-level error, if any

workloads[*].status:
  passed | compile_error | runtime_error | timeout | tool_error |
  tool_timeout | memory_admission_failed | worker_lost | infra_error

workloads[*].endpoint: evaluate | profile | sanitize | debug
workloads[*].attempts: integer
workloads[*].worker_restarted: boolean
workloads[*].killed_process_group: boolean
workloads[*].error_message: string
workloads[*].log: string or metadata
```

Then keep `TaskStatus` for request lifecycle only, and use the per-workload
status for benchmark/tool outcome. This would remove the current need for
clients to parse different fields and error strings per endpoint.
