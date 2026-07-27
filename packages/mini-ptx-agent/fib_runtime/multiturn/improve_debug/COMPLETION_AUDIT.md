# Debug Endpoint Completion Audit

## Goal Artifacts

- Selected kernels: `selected_error_kernels.json`
- Selector: `select_representative_error_kernels.py`
- Source eval root: `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128`
- Kernel source root: `/home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128/kernels`

`selected_error_kernels.json` contains 20 kernels:

- 10 `Runtime error`
- 10 `Kernel Execution Timeout`

## Local FlashInfer Bench Changes

The local checkout at `/home/ubuntu/flashinfer-bench` was changed in place.
The debug implementation lives in:

- `/home/ubuntu/flashinfer-bench/flashinfer_bench/agents/debug.py`
- `/home/ubuntu/flashinfer-bench/flashinfer_bench/serve/app.py`
- `/home/ubuntu/flashinfer-bench/flashinfer_bench/serve/scheduler.py`
- `/home/ubuntu/flashinfer-bench/flashinfer_bench/serve/task_store.py`
- `/home/ubuntu/flashinfer-bench/flashinfer_bench/utils.py`
- `/home/ubuntu/flashinfer-bench/scripts/launch_fib_serve_direct.sh`
- `/home/ubuntu/flashinfer-bench/tests/agent/test_debug.py`

## Container And Serve Contract

Launch from this directory:

```bash
./launch_local_fib_profile_debug.sh
```

The launcher recreates `fib-profile` with:

- `/home/ubuntu/flashinfer-bench` mounted at `/workspace/flashinfer-bench-private`
- one visible GPU, default host GPU `0`
- one direct `flashinfer-bench serve` backend on port `10000`
- no dispatcher
- tmux session `fib-serve-direct`

The direct serve command is owned by:

```bash
/home/ubuntu/flashinfer-bench/scripts/launch_fib_serve_direct.sh
```

## Debug Wrapper

Use the host-side client to read a kernel, submit its source content directly to
`/debug`, poll the task, and print the returned debug metadata as a readable
sectioned report:

```bash
./debug_error_kernel.py \
  /home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128/kernels/exp_001/kernel_t5.cu \
  --timeout 120 --wait-timeout 180
```

To preserve the raw API response, including `logs[0].metadata`, add
`--dump-result result.json`.

The compatibility wrapper calls the same direct-submit client. It does not copy
files into `fib-profile`:

```bash
./debug_error_kernel_in_fib_profile.sh \
  /home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128/kernels/exp_001/kernel_t5.cu \
  --timeout 120 --wait-timeout 180
```

For timeout kernels, allow more time for the coredump pass:

```bash
./debug_error_kernel_in_fib_profile.sh \
  /home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128/kernels/exp_000/kernel_t0.cu \
  --timeout 45 --wait-timeout 240 --coredump-grace-seconds 30
```

## Verified Returned Messages

Runtime misaligned address:

- Kernel: `exp_001/kernel_t5.cu`
- Returned fault type: `Misaligned shared or local address`
- Returned faulting instruction: `kernel.cu:61`
- Returned device frame: `kernel.cu:574`
- Returned async check site: `kernel.cu:666`

Runtime illegal memory access:

- Kernel: `exp_003/kernel_t2.cu`
- Returned CUDA API error: `an illegal memory access was encountered`
- Returned likely failing launch: `kernel.cu:730`
- Returned check site: `kernel.cu:731`

Runtime invalid argument:

- Kernel: `exp_012/kernel_t1.cu`
- Returned CUDA API error: `invalid argument`
- Returned likely failing launch: `kernel.cu:629`
- Returned check site: `kernel.cu:630`

Kernel execution timeout:

- Kernel: `exp_000/kernel_t0.cu`
- First pass: compute-sanitizer timeout
- Fallback pass: direct run with CUDA coredump env
- Returned coredump path and automatic `cuda-gdb` backtrace
- Returned source locations: `kernel.cu:60` and inlined `kernel.cu:239`
- Full raw output saved in the task debug directory as `debug_raw.log`

## Notes

The `/debug` API receives kernel source in the JSON payload; it does not require
the kernel file to exist inside `fib-profile`.
