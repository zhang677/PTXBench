# Coding-agent workflow

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
