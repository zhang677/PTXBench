# Coding-agent workflow

## GPU-owning Python launchers

The shared AccRL launchers under `fib_runtime/multiturn/construct_eval_scripts`
accept an experiment JSON and own GPU acquisition and cleanup:

```bash
/scratch/zgh23/miniconda3/envs/acc/bin/python ../run_radix_coding_agent.py CONFIG.json
/scratch/zgh23/miniconda3/envs/acc/bin/python ../run_modal_coding_agent.py CONFIG.json
```

`run_radix_coding_agent.py` supports `agent: "codex"`, `"codex-persistent"`,
and `"agy"`. It selects one available GPU of the configured class, acquires a
Radix assignment, provisions `fib-profile`, starts its SSH tunnel and gateway,
renews the assignment while needed, and releases only its owned assignment on
exit. It refuses acquisition while another assignment or active PTXBench
container remains. The legacy `run_radix_codex.sh` and `run_radix_agy.sh` remain
available during deprecation; they require an already-acquired lease.

Add these fields to a normal experiment config for Radix (omit `modal`):

```json
{
  "radix": {
    "gpu": "H100",
    "ssh_key": "/home/zgh23/.ssh/id_rsa",
    "duration": "1h",
    "extend_by": "1h",
    "acquire_timeout_seconds": 3600,
    "ready_timeout_seconds": 3600
  }
}
```

`radix.gpu` must match `target_hardware` and the CUDA architecture, when used.
Supported classes are H100, H200, and B200. GPU count is one. The default local
profiling endpoint is `http://127.0.0.1:18085`; set `profile_base_url` to another
free loopback port if needed. Keep selected-workload Hugging Face settings under
`profile_service.dataset`, as with the leased shell launchers. SSH host, port,
and host-key files are obtained from the assignment. The input JSON is never
rewritten; resolved configs and ownership/cleanup evidence go under
`experiment_root/attempts/`.

## Persistent Codex capacity recovery

```json
{
  "agent": "codex-persistent",
  "build_images": true,
  "codex_persistent": {
    "gpu_policy": "release",
    "initial_delay_seconds": 30,
    "max_delay_seconds": 300,
    "max_wait_seconds": null,
    "probe_timeout_seconds": 120,
    "drain_timeout_seconds": 1000
  }
}
```

Use `gpu_policy: "retain"` (the default) to keep the current lease and gateway
while waiting, or `"release"` to stop them and reacquire a GPU after model
availability returns. Both policies work with the Python Modal and Radix
launchers. Modal's configured sandbox lifetime still applies to retained GPUs.
Rebuild the Codex image once to install `ptxbench-codex-persistent`.

The new `src/ptxbench_eval/codex_persistent_runner.py` stores non-ephemeral
Codex threads. On terminal `serverOverloaded` it checkpoints the conversation,
pauses active and queued runs, and lets submitted evaluations drain before
releasing a GPU. One supervisor probes the same model, account, and service tier
using short inference requests with shell, web, and MCP tools disabled. Retry
delays grow exponentially with jitter up to the configured maximum; a supplied
retry-after delay is respected. `max_wait_seconds: null` waits until recovery
or manual cancellation. Other errors stop recovery without changing models.

After a successful probe, the first unfinished run resumes its stored thread;
other runs start after it produces model output. Completed runs, evaluation
counts, token usage, and artifacts are preserved. Feedback committed while
agents were stopping is supplied through `.ptxbench-resume.json`. New GPU
acquisitions revalidate workload/prompt hashes and dataset provenance. Missing
or stale session checkpoints stop automatic replay. Credentials stay in the
container tmpfs and are excluded from the durable Codex snapshots.

`capacity.json` records the waiting state and next probe time;
`capacity-probes.jsonl` records probe results. Persistent sessions live under
`codex_home/<run_id>/persistent/`. The same source JSON can resume a persistent
experiment after a graceful stop. Configuration changes and concurrent
supervisors are rejected. Ordinary `codex` retains terminal-capacity fail-fast
behavior; `agy` retains its existing runner and authentication.

The low-level `ptxbench_eval.launcher launch` returns exit code 75 when a
persistent batch is ready for recovery. Use the GPU-owning Python entry points
above for automatic waiting and GPU lifecycle management. Existing runs made
with ephemeral `codex` threads cannot be converted into persistent sessions.

## Manually managed profiling service

The gateway and launcher code in this directory are the coding-agent path.
They reuse `../multiturn/prompts/hub.json` and the same profiling service as
the model-loop experiments. The launcher materializes prompt fragments and a
language-specific integration example inside each agent workspace.

1. Start `fib-profile` as described in `../fib-profile/scripts/README.md` and
   verify `http://127.0.0.1:10000/health`.
2. Install the host package with `pip install -e ./ptxbench-eval`.
3. Choose `examples/gemini_gemm` or `examples/gpt56_gemm`. Review its
   `experiment.json` for the task, prompt tags, model, limits, and output root.
4. Export `GEMINI_API_KEY` for Gemini, or provide the Codex auth file named in
   the Codex config. Run that example's `run_experiment.sh` from the repo root.

The scripts build the gateway and agent Docker images, prepare a new run,
start the gateway, and persist trajectory and profiling artifacts under the
experiment root. `SKIP_BUILD=1` reuses already built images. Set
`EXPERIMENT_ROOT` to a fresh absolute path to keep runs separate;
`PROFILE_BASE_URL` and `GATEWAY_PORT` override the default endpoints. The
bundled examples target H100 GEMM. Copy a config and update its workload UUID,
definition, target hardware, and prompt tags for another workload.

The gateway accepts `--max-profiles N` to run at most N complete `/v1/evaluate`
operations concurrently. Each operation holds its slot through compilation,
sanitization, profiling, debugging, and retries; excess requests wait in the
gateway while callers remain synchronous. The limit is per gateway and separate
from the launcher's `max_parallel` agent limit. Omitting it preserves the default
thread-pool concurrency. `/health` reports the configured `max_profiles` value.

## Passing kernel exports

The gateway automatically exports fully passing Codex and AGY evaluations to
`success/<run_id>/kernel_v0.cu`, `kernel_v1.cu`, and so on (`.py` for Triton
and CuTeDSL). `record.json` uses the same list format as the multiturn Gemini
runner: `version`, `turn`, `timestamp`, and complete `traces`. Versions count
successful submissions from zero; `turn` is the original zero-based evaluation
turn, including failures. Passing kernels are saved even below the target
speedup, and repeated submissions remain separate versions.

The shared implementation is `src/ptxbench_eval/export_success.py`. Backfill an
existing experiment with the same code, without running any GPU evaluations:

```bash
PYTHONPATH=ptxbench-eval/src python -m ptxbench_eval.export_success \
  --run-dir /path/to/experiment
```

Use `--run-id exp_000` to select a run, `--dry-run` to inspect the result, or
`--backup-existing` to move incompatible artifacts intact into
`success-backups/` before replacing them. Matching exports are left unchanged;
later successes append versions. Source bytes are checked against saved hashes.
For older trajectories, recorded profile submissions restore any trailing
whitespace stripped by the old serializer; when those events are unavailable,
the canonical source block is used. Backfills use saved evaluation timestamps
(an empty timestamp means the old evidence did not contain one).

Rebuild the gateway image to enable automatic exports in an existing image;
the normal launch scripts do this when image building is enabled. The CLI works
directly from the checkout for previously completed runs.

## CuTeDSL

The gateway also supports CuTeDSL kernels. Use `language: "cutedsl"` in
the experiment config and omit `binding`, `nvcc_gencode`, and
`tvm_ffi_example`. Set `cutedsl_example` to
`../cutedsl/integration_example.py` when copying a bundled GEMM config,
and use the `hopper-arch-only` prompt tag to avoid C++ helper headers.
All config paths are relative to the config file.

Set `cutedsl_source` to a directory containing an exported snapshot of the
profiling service's installed CuTeDSL sources. The directory must include
`cutlass/cute/__init__.py` and `source_manifest.json` with this schema:

```json
{
  "schema_version": 1,
  "version": "4.6.0",
  "packages": {"nvidia-cutlass-dsl": "4.6.0"},
  "files": {"cutlass/cute/__init__.py": "<SHA-256 of the file>"}
}
```

Include every exported file's relative path and SHA-256 in `files`; omit
compiled libraries and symlinks. The launcher validates the snapshot,
records its version and hashes, and mounts it read-only at
`/workspace/cutedsl` for source inspection. Preparation and launch reject
changes to the source snapshot or integration example in an existing run.

The agent writes a self-contained `kernel.py` with a destination-passing
`run(...)`: definition inputs followed by preallocated outputs, in definition
order. Submit it with:

```bash
ptxbench-eval /workspace/kernel.py /opt/ptxbench/workload.json
```

The fixed MCP evaluator selects the same Python file automatically. The
gateway checks Python syntax without executing it, then submits
`language: "python"`, `kernel.py::run`, and the CuTeDSL dependencies to the
profiling service for JIT compilation, sanitization, and benchmarking.
The service must have `nvidia-cutlass-dsl` and `cuda-python` installed;
`fib-profile/scripts/entrypoint.sh` installs them on a fresh setup.

## Triton

Use `language: "triton"` and omit `binding`, `nvcc_gencode`, and
`tvm_ffi_example`. When copying a bundled GEMM config, set `triton_example`
to `../../../multiturn/prompts/examples/triton.py` and select the
`triton-hopper` prompt tag (or `triton-blackwell` for Blackwell). Paths are
relative to the config file. No CuTeDSL source snapshot is needed.

The agent writes a self-contained `/workspace/kernel.py` with a Python
`run(...)` wrapper that receives definition inputs followed by preallocated
outputs in definition order, and launches `@triton.jit` kernels to fill those
outputs. PyTorch may provide tensor views, but must not compute the result.
Use the same `ptxbench-eval /workspace/kernel.py /opt/ptxbench/workload.json`
command as CuTeDSL; the MCP evaluator also selects `kernel.py` automatically.

The gateway checks Python syntax without importing the submission and sends
`language: "triton"`, `kernel.py::run`, and the `triton` dependency to the
profiling service's Triton builder. JIT compilation, sanitization, correctness
checks, and timing happen in that service. The standard profiling setup already
installs Triton via `fib-profile/scripts/entrypoint.sh`. Preparation records the
integration example's hash, and preparation/launch reject example drift for an
existing run.


## Error diagnostics

Runtime-error and timeout feedback preserves the complete original evaluation
log before appending additional CUDA diagnostics. This applies to both feedback
styles; `multiturn` omits structured traces from the response, not the error text.

Diagnostic selection uses the solution's existing backend language. CuTeDSL
remains `python`, and Triton remains `triton`; both keep their original Python/JIT
tracebacks and only append the CUDA debug report when it contains device-fault
or core-dump evidence. CUDA submissions retain the additional CUDA debug report.
No separate CuTeDSL diagnostics tag or configuration field is required.
Submitted-source excerpts are matched to traceback paths, so library frames are
not relabeled as lines in `kernel.py` or `kernel.cu`. Sanitizer checks and FLA
backend selection are unchanged, including `FLA_TILELANG=0` in the LA references.
