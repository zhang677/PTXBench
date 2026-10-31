# Modal launchers

These scripts run PTXBench experiments with one Modal profiling GPU. The sandbox
clones the default branch of PTXBench on startup and records the revision in
`modal-runtime.json`. The host does not need a second PTXBench checkout or a
local profiling GPU.

Install the host requirements in the existing `acc` environment, including
Modal and the PTXBench evaluator package from Git. Docker must be available:

```bash
python -m pip install modal "git+https://github.com/zhang677/PTXBench.git#subdirectory=ptxbench-eval"
```

The coding-agent launcher uses
the installed `ptxbench_eval` package on the host. When `build_images` is true,
it builds the agent and gateway images from the PTXBench Git URL as a Docker
build context. When false, those images must already exist locally. The
multiturn launcher uses `multiturn/run_parallel_v2.py` and its prompt files from
this repository; its GPU profiling service runs in the Modal sandbox.

From a PTXBench checkout, run:

```bash
python multiturn/modal/run_modal_coding_agent.py CONFIG.json
python multiturn/modal/modal_profile_runtime.py CONFIG.json --validate-only
python multiturn/modal/modal_profile_runtime.py CONFIG.json
```

Use your own path to the `acc` interpreter if it differs. Coding-agent configs
use the `ptxbench_eval.launcher` JSON format plus a `modal` object. Multiturn
configs use the `modal_profile_runtime.py` JSON format without a
`modal.ptxbench_root` field. `run_modal_profile.py` is the lower level supervisor
used by that runtime. Config paths resolve
relative to the JSON file. Set `modal.gpu` to `B200`, `H100`, or `H200`; for
coding agents, it must match `target_hardware`. The optional
`modal.lock_gdn_bwd_reference` setting requires B200 and uses the bundled
manifest. The optional `modal.patch_fla_fp32_state_dot` setting makes FLA's
GDN/KDA backward reference accumulate its chunk-end state dot in FP32 on any
GPU; it cannot be combined with the lock. Both launchers optionally accept a
top-level `dataset` object in Hugging Face mode. The sandbox then downloads and
serves only the selected workloads instead of the bundled data; `modal.dataset`
is rejected. Multiturn tasks select theirs with `workload_uuid`, which must
match the `DEFINITION_NAME` and `WORKLOAD_UUID` in their test script. Every
attempt of a multiturn run must resolve the same dataset revision as its first
attempt; pin `dataset.revision` to resume across dataset updates. Multiturn
configs may also set `language` to `cuda` (the default) or `triton`.

`examples/` holds B200 CUDA and Triton configs for GDN and KDA prefill/backward
with the `*-blackwell-*-unified` prompt tags: GPT-6 coding agents and Gemini 3.1
Pro multiturn runs. The multiturn configs use the tests under
`multiturn/tests/{cuda,triton}/` and the prompt configs in
`examples/prompt_configs/`, and write under `~/ptxbench-runs`, which must exist:

```bash
python multiturn/modal/run_modal_coding_agent.py multiturn/modal/examples/codex_gpt6_cuda_unified_b200.json
python multiturn/modal/modal_profile_runtime.py multiturn/modal/examples/multiturn_gemini31_triton_unified_b200.json
```

Both launchers own and release their Modal sandbox. The multiturn watcher can
resume a completed root after interruption; the coding-agent launcher follows
the evaluator's normal run and persistent-agent rules. Outputs and credentials
belong in local experiment directories, not this source directory.
