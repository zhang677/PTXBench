# Qwen3.6 MHA Prompt Patch Trace

Date written: 2026-07-14

## Artifact

Patch fragment:

`/home/ubuntu/AccRL/fib_runtime/structural_doc/patterns/qwen36-27b-mha-patch.md`

This file is a hand-written prompt patch fragment, not a generated prompt doc.
It was then referenced by `hub.json` aliases and included in generated
`*-mha-patched.md` prompt documents.

## Generation Trace

1. The source analysis was the `mha_with_lse_d128` failure study for:

   `/home/ubuntu/AccRL-exps/eval_runs/2026-0610-0910`

   The analysis extracted 156 per-turn kernels and found 66 invalid-argument
   failures, including 15 explicit `CUDA_ERROR_INVALID_VALUE` failures on
   `cuTensorMapEncodeTiled`.

2. The failure analysis identified prompt-level constraints Qwen needed:

   - state a kernel contract before writing code
   - use valid BF16 TMA 128B-swizzle dimensions for flattened `[B*H*S, D]`
   - check all CUDA and driver calls
   - opt into large dynamic shared memory with `cudaFuncSetAttribute`
   - implement online softmax with `m_i`, `l_i`, and old-`O` rescaling
   - avoid unsynchronized shared accumulator updates
   - finish with a descriptor/barrier/WGMMA/output/launch self-audit

3. Those lessons were distilled into the seven-line fragment now stored at
   `structural_doc/patterns/qwen36-27b-mha-patch.md`.

4. Git confirms the fragment was added in commit:

   `317eaa11070dbb13b93d71a302046c4321af8b58` (`Add patch`),
   authored on `2026-06-11 15:53:18 +0000`.

   The same commit updated:

   `/home/ubuntu/AccRL/fib_runtime/multiturn/prompt_configs/hub.json`

   and generated prompt docs such as:

   - `hopper-07-mha-patched.md`
   - `hopper-010-mha-patched.md`
   - `hopper-no-hint-mha-patched.md`

5. The `hub.json` aliases use the original prompt tag plus this patch fragment.
   Example shape:

   ```json
   "hopper-07-mha-patched": [
     "hopper-07",
     "structural_doc/patterns/qwen36-27b-mha-patch.md"
   ]
   ```

6. The patched config files in `/home/ubuntu/AccRL-exps/prompt_configs/`
   kept the original schema and changed only `prompt_tag` values to the
   `*-mha-patched` aliases. `build_doc_v2.py --force` then generated the
   concrete prompt markdowns.

## Correct-Kernel Use In The Error Analysis

The `ERROR_ANALYSIS.md` file has two relevant phases.

Early analysis was failure-driven. It used Qwen-generated failed kernels from
`2026-0610-0910` and an isolated reference kernel only for descriptor and launch
contract evidence. That early isolated reference was explicitly not treated as
end-to-end correct because it still reported a later misaligned-address runtime
fault.

Later, the analysis did bring in a verified correct kernel:

```text
/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340/success/exp_004/record.json
/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340/success/exp_004/kernel_v0.cu
```

That record matched the exact task:

```text
definition=mha_with_lse_d128
uuid=bc38b351-d595-451b-9153-8e225702e53b
B=4, H=48, S=4096, D=128
status=PASSED
```

The kernel was copied into:

`/home/ubuntu/AccRL/fib_runtime/mini_swe_agent_docker/isolated/mha_with_lse_d128/kernel.cu`

and verified with:

```bash
PROFILE_BASE_URL=http://localhost:10000 python test.py
```

Recorded result:

```text
status=PASSED
max_absolute_error=0.0009765625
latency_ms=4.307388
reference_latency_ms=2.639804
speedup_factor=0.612855
```

The prompt patch was therefore derived from both failed-kernel clustering and a
later comparison against one verified correct `mha_with_lse_d128` anchor.

