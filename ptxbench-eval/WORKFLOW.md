# Coding-agent workflow

The gateway and launcher code in this directory are the coding-agent path.
They reuse `../multiturn/prompts/hub.json` and the same profiling service as
the model-loop experiments. The launcher materializes prompt fragments and a
TVM-FFI example inside each agent workspace.

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
