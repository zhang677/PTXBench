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

`entrypoint.sh` creates `/workspace/acc`, installs the mounted bench with its `serve` and `profile-cu130` extras, then starts four single GPU backends plus a dispatcher. The profile extra pins Torch `2.12.1+cu130`, Triton `3.7.1`, FlashInfer `0.7.2`, CUTLASS DSL `4.6.2`, cuDNN Python frontend `1.31.0`, and CUDA Python `13.0.3` so clean installs use the same tested core GPU stack. The CUDA Python pin keeps its bindings compatible with CUDA 13.0 drivers during sanitizer checks. It also clones [FlashKDA](https://github.com/MoonshotAI/FlashKDA) into `/workspace/flash-kda`, initializes its submodules, and builds it with `uv pip install -v --no-build-isolation .`. Fresh clones use the repository's default branch without a commit pin; existing checkouts are reused without a pull or checkout. The source URL is fixed; `FLASH_KDA_DIR` overrides the checkout directory. Startup verifies `import flash_kda` and the pinned Torch/Triton versions before serving. It requires network access to install public packages. Later service restarts reuse that environment:

```bash
docker exec fib-profile bash -lc 'bash /workspace/scripts/restart_profiling.sh'
```

### Dataset modes

`FIB_DATASET_MODE=existing` is the default: the entrypoint serves the mounted
dataset without generating or verifying blobs. Supply its complete inputs,
including any blobs referenced by its workloads. `HOST_DATASET_DIR` defaults
to `fib-profile/dataset`; override it to mount another dataset:

```bash
HOST_DATASET_DIR=/absolute/path/to/dataset FIB_DATASET_MODE=existing ./scripts/run_container.sh
docker exec fib-profile bash /workspace/scripts/entrypoint.sh
```

For PTXBench's bundled evaluator subset, opt into generation at startup:

```bash
FIB_DATASET_MODE=bundled ./scripts/run_container.sh
docker exec fib-profile bash /workspace/scripts/entrypoint.sh
```

`bundled` mode requires the mounted dataset's `generate_blobs.py` and its two
specific backward workload rows. After installation, it selects a cool GPU
from `FIB_DEVICES` using the UUID-based thermal filter and generates or verifies
the two blobs there. A missing generator, failed temperature check, or checksum
mismatch stops startup. The dataset mount must be writable for missing blobs.
See [generation requirements and verification status](../dataset/README.md).
Service restarts reuse prepared inputs and do not rerun generation.

To stop only the service sessions:

```bash
docker exec fib-profile bash -lc 'bash /workspace/scripts/stop_profiling.sh'
```

`run_container.sh` replaces an existing container named `fib-profile`. By default it exposes host port 10000, selects CUDA devices 0 through 3, and requires GPU temperature below 45 C. The container runs with Docker `--init`.

The profiling port is published only on `127.0.0.1`, for access through a
local client or SSH tunnel.

For the B200 runs, use these same scripts on a B200 host and set
`TVM_FFI_CUDA_ARCH_LIST=10.0a` when running `run_container.sh`. The default
`9.0a` targets H100; set `GPUS` and `FIB_DEVICES` to the available devices on
either host.
`GPUS` selects host devices, while `FIB_DEVICES` uses indices visible inside
the container. For example, host GPUs 4–7 appear as container GPUs 0–3:

```bash
GPUS='"device=4,5,6,7"' FIB_DEVICES='cuda:0,cuda:1,cuda:2,cuda:3' ./scripts/run_container.sh
```

The host temperature check uses the devices selected by `GPUS` and leaves
`FIB_DEVICES` unchanged. Docker GPU count requests defer temperature checks to
the container because the selected host identities are not known in advance.
During service startup, each CUDA device is matched by its UUID to the
`nvidia-smi` temperature record; CUDA and `nvidia-smi` indices need not agree.
Hot devices are excluded. Missing UUIDs or temperature readings stop startup.
The host check uses `python3`; the container check uses its installed Torch runtime.

## Verify

```bash
docker exec fib-profile bash -lc 'python3 /workspace/scripts/verify_service.py'
```

The verifier checks four healthy backends, confirms that each serves all five mounted definitions and workloads, and evaluates the included GEMM workload on each backend plus the dispatcher using its reference Python function. `--all` evaluates all five workloads through the dispatcher. The short shell wrappers `run_verify.sh` and `run_complete_verify.sh` invoke these two modes.

This verifier targets the bundled evaluator subset; an external dataset needs
checks using its own definitions and workloads.

Key overrides: `IMAGE`, `NAME`, `HOST_DATASET_DIR`, `FIB_DATASET_MODE`, `GPUS`, `FIB_DEVICES`, `FIB_GPU_MAX_TEMP_C`, `PROFILE_PORT`, `HOST_PROFILE_PORT`, and `TVM_FFI_CUDA_ARCH_LIST` for container creation; `FIB_GPU_COOLDOWN_SLEEP_S`, `PROFILE_MAX_GPUS`, and `FIB_TIMEOUT` for restarts.
