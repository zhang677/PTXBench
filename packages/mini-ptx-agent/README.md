# mini-ptx-agent

mini-ptx-agent is the PTXBench agent component ported from AccRL. The initial
port intentionally preserves the proven `fib_runtime/multiturn` experiment
layout while replacing machine-specific roots with PTXBench environment
variables.

The stable CLI entry points are:

```bash
ptxbench paths
ptxbench-inspect DATA.jsonl
```

## Licensing

mini-ptx-agent is Apache-2.0. Code adapted from Helion and FlashAttention
retains its BSD-3-Clause terms under [`licenses/`](licenses/). The repository's
top-level [`NOTICE`](../../NOTICE) records the exact affected files and the
provenance of NVIDIA reference material and hosted-model research workflows.
