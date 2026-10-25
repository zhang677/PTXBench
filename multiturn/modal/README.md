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
manifest. The coding-agent launcher optionally accepts `modal.dataset` in
Hugging Face mode to download selected workloads.

Both launchers own and release their Modal sandbox. The multiturn watcher can
resume a completed root after interruption; the coding-agent launcher follows
the evaluator's normal run and persistent-agent rules. Outputs and credentials
belong in local experiment directories, not this source directory.
