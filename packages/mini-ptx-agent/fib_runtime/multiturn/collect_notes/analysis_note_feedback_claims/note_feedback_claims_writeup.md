# Note Feedback Retrieval Claims

## Scope

- Note-only roots: `qwen36-27b-retrieved-notes-d128-4defs-*`.
- Fixed-kernel roots: `qwen36-27b-note-feedback2-mha-d128-d96-8defs-*`.
- Unit of analysis is the trajectory and the per-turn feedback messages in `trajectories/exp_*.json`; `summary.json` is not used as the primary outcome because several roots were resumed subsets.

## Claim 1: Notes alone did not help

The four note-only d128 roots contain `32` trajectories. They produced `0` trajectories with any passing turn and `0` saved passing kernels.

This is the strongest available artifact-backed conclusion: the notes were present in feedback, but no trajectory reached a saved correct kernel. The failure mode is not that retrieval was absent; it is that prose repair notes were insufficient for Qwen3.6-27B to synthesize a legal, correct Hopper implementation.

## Claim 2: BM25 is adequate for selecting concrete success kernels, but the model mostly imitates

The fixed-kernel run has `32` trajectories and `39` saved passing kernels. By dimension:

- `d128`: `13/16` trajectories passed, `36` saved kernels, pass rate `81.2%`.
- `d96`: `3/16` trajectories passed, `3` saved kernels, pass rate `18.8%`.

The retrieval corpus is d128-only, so the d96 rows are a transfer/generalization test. The much stronger d128 result and weak d96 result are consistent with in-context imitation or local adaptation of concrete examples, not robust extrapolation of the algorithm across head dimensions.

BM25 itself looks adequate as a sparse selector: retrieved definitions are concentrated on structurally related MHA forward/backward and causal/noncausal variants rather than arbitrary kernels.

Retrieved definition counts:
- `mha_with_lse_d128`: `60`
- `mha_bwd_d128`: `57`
- `mha_with_lse_d128_causal`: `49`
- `mha_bwd_d128_causal`: `40`

Next-turn outcomes after a retrieved fixed kernel was injected:
- `FAILED`: `115`
- `COMPILE_ERROR`: `36`
- `PASSED`: `31`
- `NO_NEXT_EVAL`: `24`

Most-used retrieved definition/speedup pairs:
- `mha_with_lse_d128` @ `0.31027x`: `26` uses, `8` next-turn passes.
- `mha_bwd_d128_causal` @ `0.00996955x`: `22` uses, `0` next-turn passes.
- `mha_bwd_d128` @ `0.609697x`: `21` uses, `0` next-turn passes.
- `mha_bwd_d128` @ `0.0946552x`: `15` uses, `0` next-turn passes.
- `mha_with_lse_d128_causal` @ `0.0597501x`: `14` uses, `1` next-turn passes.
- `mha_with_lse_d128` @ `0.0255242x`: `13` uses, `3` next-turn passes.
- `mha_with_lse_d128` @ `0.542423x`: `12` uses, `1` next-turn passes.
- `mha_with_lse_d128_causal` @ `0.158081x`: `8` uses, `0` next-turn passes.
- `mha_with_lse_d128_causal` @ `0.16063x`: `8` uses, `0` next-turn passes.
- `mha_with_lse_d128_causal` @ `0.482831x`: `7` uses, `1` next-turn passes.

## Figures

- `pass_rate_by_run.png`: note-only versus fixed-kernel pass rates by task.
- `fixed_kernel_d128_vs_d96.png`: d128 versus d96 pass rate under fixed-kernel retrieval.
- `retrieval_definition_heatmap.png`: BM25 target-definition to retrieved-definition counts.
- `retrieved_speedup_vs_next_turn.png`: retrieved fixed-kernel speedup versus next-turn behavior.

## Interpretation

A good next experiment is to make the success-kernel injection less copy-heavy and more transform-heavy: show the correct kernel plus a compact structured diff against the retrieved wrong kernel, and explicitly ask for which constants/layout choices must change for the target definition. That would test whether the model can adapt the example instead of copying its shape-specific implementation.

## Caveats

This is observational evidence from the current artifacts, not a randomized causal estimate. The fixed-kernel run also has `8` trajectories ending in context-length `BadRequestError`, leaving `11` planned turns unattempted. The pass-rate comparison is therefore trajectory-level and uses any observed passing kernel, not final-turn-only success.
