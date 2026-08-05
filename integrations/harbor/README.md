# Harbor integration

This directory contains PTXBench tasks that run on an unmodified Harbor
checkout. Harbor's agent adapter owns the ATIF trajectory; `ptxbench eval`
provides structured kernel feedback as ordinary tool output.

The Harbor agent image extends the same `ptxbench-eval:dev` CPU evaluation
image used by mini-ptx-agent multiturn runs. It adds Harbor's shell-agent
dependencies and the `ptxbench` command, but does not expose a GPU to the agent.
Compilation runs in the agent container; sanitizer and benchmark requests run
on the H100 workers behind FIBServe.

Each evaluation also emits a virtual-architecture PTX artifact locally. The
structured result's `ptx` field reports the artifact hash and instruction-family
counts extracted from that PTX; it does not infer architecture usage from CUDA
C++ source text.

Build the local images from the PTXBench repository root:

```bash
docker build -f docker/Dockerfile.eval -t ptxbench-eval:dev .
docker build -f docker/Dockerfile.harbor -t ptxbench/harbor:dev .
```

Start FIBServe with the trace set containing the task definition and workload,
then render the task instruction from the definition returned by that service:

```bash
python integrations/harbor/render_instruction.py \
  integrations/harbor/tasks/gemm_n7168_k5120 \
  --service-url http://localhost:11000
```

The renderer mirrors AccRL prompt synthesis: it fetches
`/definitions/<definition>`, removes the top-level `tags` field, formats the
remaining object with two-space JSON indentation, and substitutes it for
`{task_content}` in `instruction.template.md`. The definition name comes from
the task's `environment/task.json`.

Run the local dataset from the Harbor checkout:

```bash
PTXBENCH_HARBOR_SERVICE_URL=http://host.docker.internal:11000 \
harbor run \
  -p /path/to/PTXBench/integrations/harbor/tasks \
  -a mini-swe-agent \
  -m openai/gpt-5.4-mini \
  --ak 'config={"agent":{"step_limit":30},"environment":{"timeout":600}}'
```

The environment timeout override is required because mini-swe-agent defaults
individual shell calls to 30 seconds, while sanitizer and benchmark queues can
take longer. Harbor's task-level timeout remains the outer limit.

The ATIF trajectory is written under the Harbor job directory as
`<trial>/agent/trajectory.json`.
