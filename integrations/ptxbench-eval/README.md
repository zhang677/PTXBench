# ptxbench-eval

This integration provides the `ptxbench-eval` kernel feedback CLI, the
`ptxbench-eval-server` gateway, and the `ptxbench-run` coding-agent launcher.
The launcher supports Codex and Antigravity agents and sends evaluation
requests through the gateway to FIBServe.

Install the host-side Python package from the PTXBench repository root:

```bash
python -m pip install -e integrations/ptxbench-eval
```

The runnable GEMM examples are in [`examples/gpt56_gemm`](examples/gpt56_gemm)
and [`examples/gemini_gemm`](examples/gemini_gemm). Their scripts build the
agent and gateway images with `integrations/` as the Docker build context.
See the [repository README](../../README.md#coding-agent-evaluation) for a
launch command and required service setup.
