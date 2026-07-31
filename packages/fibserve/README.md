# FIBServe

FIBServe is the GPU-side service used by PTXBench's `mini-ptx-agent`. It is a
focused derivative of
[FlashInfer Bench](https://github.com/flashinfer-ai/flashinfer-bench), retaining
only the runtime needed to load FlashInfer trace sets and compile, evaluate,
sanitize, debug, and profile generated CUDA kernels.

The supported HTTP contract is:

- trace discovery: `/definitions`, `/definitions/{name}`,
  `/definitions/{name}/workloads`, and `/workloads/{uuid}`
- kernel operations: `/evaluate`, `/sanitize`, `/debug`, and `/profile`
- asynchronous operation: `/tasks/batch` and `/tasks/{id}`
- service operation: `/health` and `/shutdown`

For one process, start FIBServe from the PTXBench workspace:

```bash
uv run --package fibserve --extra serve fibserve serve \
  --local /path/to/accrl-training \
  --devices cuda:0,cuda:1 \
  --port 10000
```

`--local` accepts one or more trace-set paths. Multiple paths are merged in
memory; identical overlaps are de-duplicated and conflicting objects are
rejected.

The production-compatible topology starts one backend tmux session per GPU and
one dispatcher tmux session that exposes a single API:

```bash
cd packages/fibserve
DATASET_ROOTS=/path/to/accrl-training:/path/to/accrl-training-heavy \
DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
DISPATCH_PORT=10000 \
./scripts/launch_fib_serve_dispatcher.sh
```

`DATASET_ROOTS` is optional for a single dataset (`DATASET_ROOT` is also
accepted) and is a colon-separated list when multiple trace sets are desired.

This creates `fib-serve-backend-0`, `fib-serve-backend-1`, and so on, plus
`fib-serve-dispatcher`. Clients use only
`PROFILE_BASE_URL=http://localhost:10000`. The dispatcher selects a healthy
least-loaded backend and retains task-to-backend ownership for `/tasks` polling.
See [`SERVER.md`](SERVER.md) for the complete request, scheduler, subprocess,
memory-admission, and dispatcher lifecycle.

The reproducible `docker/compose.yaml` setup uses this same multi-tmux topology
inside the FIBServe container. It publishes only the dispatcher:

```bash
docker compose --env-file docker/.env -f docker/compose.yaml up -d fibserve
docker exec ptxbench-fibserve tmux list-sessions
curl http://localhost:10000/health
```

Operational helpers are retained alongside the launcher:

```bash
# Restart only the tmux-managed service inside an existing container.
docker exec ptxbench-fibserve \
  /workspace/PTXBench/packages/fibserve/scripts/restart_profiling.sh

# Verify a local public dispatcher, or set REMOTE and REMOTE_PORT for SSH.
PROFILE_BASE_URL=http://localhost:10000 \
  ./packages/fibserve/scripts/run_verify.sh
```

`gpu_preflight.sh` supplies the CUDA temperature filter used by the restart
path. `run_verify.sh` calls the retained mini-ptx-agent service verifier.

FIBServe is licensed under Apache-2.0. See `LICENCE` and `NOTICE` for upstream
attribution.
