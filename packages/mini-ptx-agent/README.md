# mini-ptx-agent

mini-ptx-agent is the PTXBench agent component ported from AccRL. The initial
port intentionally preserves the proven `fib_runtime/multiturn` experiment
layout while replacing machine-specific roots with PTXBench environment
variables.

The stable CLI entry points are:

```bash
ptxbench doctor
ptxbench paths
ptxbench-inspect DATA.jsonl
```

