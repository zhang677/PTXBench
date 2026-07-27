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

Start it from the PTXBench workspace:

```bash
uv run --package fibserve --extra serve fibserve serve \
  --local /path/to/accrl-training /path/to/accrl-training-heavy \
  --devices cuda:0,cuda:1 \
  --port 10000
```

For the reproducible container setup, use `docker/compose.yaml` at the PTXBench
repository root.

FIBServe is licensed under Apache-2.0. See `LICENCE` and `NOTICE` for upstream
attribution.
