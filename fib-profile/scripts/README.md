# PTXBench profiling container

The host directory `fib-profile/` contains the mounted service inputs:

- `flashinfer-bench-private/`: GPU profiling service
- `dataset/`: the PTXBench workload subset and the script to generate its large blobs.
- `scripts/`: local image, container, service, and verification scripts.

The image contains CUDA and uv. The bench and dataset are bind mounted at `/workspace/flashinfer-bench-private` and `/workspace/dataset`; the scripts are mounted read only at `/workspace/scripts`.

## Build and start

From this directory on a four GPU Hopper host:

```bash
./scripts/build.sh
./scripts/run_container.sh
docker exec fib-profile bash -lc 'bash /workspace/scripts/entrypoint.sh'
```

`entrypoint.sh` creates `/workspace/acc`, installs the mounted bench and its service dependencies, then starts four single GPU backends plus a dispatcher. It pins Torch `2.12.1+cu130` and Triton `3.7.1` so clean installs use the same profiling runtime. It requires network access to install public packages. Later service restarts reuse that environment:

```bash
docker exec fib-profile bash -lc 'bash /workspace/scripts/restart_profiling.sh'
```

To stop only the service sessions:

```bash
docker exec fib-profile bash -lc 'bash /workspace/scripts/stop_profiling.sh'
```

`run_container.sh` replaces an existing container named `fib-profile`. By default it exposes host port 10000, selects CUDA devices 0 through 3, and requires GPU temperature below 45 C. The container runs with Docker `--init`.

For the B200 runs, use these same scripts on a B200 host and set
`TVM_FFI_CUDA_ARCH_LIST=10.0a` when running `run_container.sh`. The default
`9.0a` targets H100; set `GPUS` and `FIB_DEVICES` to the available devices on
either host.
`GPUS` selects host devices, while `FIB_DEVICES` uses indices visible inside
the container. For example, host GPUs 4–7 appear as container GPUs 0–3:

```bash
GPUS='"device=4,5,6,7"' FIB_DEVICES='cuda:0,cuda:1,cuda:2,cuda:3' ./scripts/run_container.sh
```

## Verify

```bash
docker exec fib-profile bash -lc 'python3 /workspace/scripts/verify_service.py'
```

The verifier checks four healthy backends, confirms that each serves all five mounted definitions and workloads, and evaluates the included GEMM workload on each backend plus the dispatcher using its reference Python function. `--all` evaluates all five workloads through the dispatcher. The short shell wrappers `run_verify.sh` and `run_complete_verify.sh` invoke these two modes.

Key overrides: `IMAGE`, `NAME`, `GPUS`, `FIB_DEVICES`, `FIB_GPU_MAX_TEMP_C`, `PROFILE_PORT`, `HOST_PROFILE_PORT`, and `TVM_FFI_CUDA_ARCH_LIST` for container creation; `FIB_GPU_COOLDOWN_SLEEP_S`, `PROFILE_MAX_GPUS`, and `FIB_TIMEOUT` for restarts.
