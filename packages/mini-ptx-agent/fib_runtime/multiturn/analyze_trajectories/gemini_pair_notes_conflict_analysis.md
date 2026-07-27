# Gemini Pair-Notes Conflict Analysis

Date: 2026-07-09

## Scope

This memo records the artifact-backed checks around whether Qwen3.6-27B's uncertainty in
`qwen36-27b-pair-notes-fixit` was caused by contradictory Gemini fixed kernels.

Primary artifacts:

- Qwen pair-notes run:
  `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-pair-notes-fixit`
- Confusion supplement:
  `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-pair-notes-fixit/analysis/qwen_confusion_marker_supplement.md`
- Row-level comparison table:
  `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-pair-notes-fixit/analysis/qwen_vs_gemini_kernel_comparison.csv`
- Row-level uncertainty marker counts:
  `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-pair-notes-fixit/analysis/qwen_reasoning_confusion_markers.csv`
- Pair-notes config:
  `/home/ubuntu/AccRL-exps/tasks/collect_notes/outputs/2026-0624-0939-kernel-fix-notes-full/pair_notes_fixit/qwen36_27b_pair_notes_fixit_config.json`
- Collected per-pair notes:
  `/home/ubuntu/AccRL-exps/tasks/collect_notes/outputs/2026-0624-0939-kernel-fix-notes-full/notes.jsonl`

## Result

The hypothesis is not supported as the primary explanation.

The paired Gemini correct kernels do contain legal-but-divergent Hopper conventions:

- `gemini_has_cluster_tma`: 106/159
- `gemini_has_cta_tma`: 149/159
- both CTA and cluster TMA patterns in the same kernel: 96/159
- WGMMA descriptor helpers with base-offset handling: 80/159
- WGMMA descriptor helpers without base-offset handling: 69/159
- dummy or validator-only TMA/WGMMA instruction stubs: 8/159

But those Gemini-side variants did not predict higher Qwen uncertainty at row level. Using the
Hopper-specific uncertainty sum:

`descriptor_lbo_sbo_swizzle_uncertainty + tma_dimension_coordinate_uncertainty + mbarrier_parity_tx_uncertainty + wgmma_fragment_mapping_uncertainty`

and stratifying by definition:

- `cluster_tma`: residual diff `-27.6`, permutation p approx `0.532`
- `both_cta_cluster`: residual diff `+10.8`, p approx `0.803`
- `desc_base_offset`: residual diff `-1.7`, p approx `0.971`
- `desc_no_base`: residual diff `+50.5`, p approx `0.229`
- `dummy_instr`: residual diff `-252.1`, p approx `0.006`, opposite direction
- `tma_no_expect`: residual diff `-247.3`, p approx `0.008`, opposite direction

The stronger positive signals were note/prompt content about TMA dimensions:

- notes mentioning `boxDim={1,...}`-style content: residual diff `+97.8`, p approx `0.033`
- notes mentioning `globalDim` / dimension order: residual diff `+134.7`, p approx `0.017`

Interpretation: Qwen's uncertainty is better explained by unresolved TMA coordinate/layout ambiguity and lossy note summaries than by direct contradiction among the fixed Gemini kernels.

## `__shared__ __align__` Follow-Up

The exact `__shared__ __align__(...)` syntax was not a major independent confusion source.

Important distinction:

- Raw grep is misleading because `__shared__ __align__(...)` appears in every trajectory through the system/reference material and often through the initial broken kernel.
- Qwen itself used the exact syntax in generated code in 47/159 runs and reasoned about it in 48/159 runs.
- Those runs were not worse:
  - assistant exact syntax usage: Hopper uncertainty mean `660` vs `707` without it
  - `extern __shared__ __align__` in code: `675` vs `697`
  - alignment-related feedback errors: `603` vs `705`

What did show up was broader shared-memory layout trouble:

- `Misaligned shared or local address`: 15 feedback hits
- `Invalid __shared__ write`: 23 feedback hits
- one bad `alignas(16)` pointer-cast compile error
- two `extern __attribute__((shared))` invalid-storage-class errors

Interpretation: the spelling `__shared__ __align__` was not the problem by itself. The real issue was matching shared-memory alignment, swizzle layout, descriptor base/offset, and WGMMA's logical tile interpretation.

## Practical Takeaway

Do not frame the pair-notes failure as "Gemini examples contradicted each other and confused Qwen." A better framing is:

Qwen saw enough evidence to adopt the right primitive vocabulary (`CUtensorMap`, `cp.async.bulk.tensor`, mbarrier, WGMMA), but the per-pair summaries did not preserve enough exact working structure to resolve Hopper-specific layout choices. Future prompts should provide either the actual correct kernel text or a structured wrong-vs-correct diff around:

- TMA descriptor `globalDim`, `globalStrides`, `boxDim`, and coordinate order
- `shared::cta` vs `shared::cluster` preconditions
- mbarrier expected transaction byte accounting and phase parity
- WGMMA descriptor fields, including when base-offset bits matter
- shared-memory tile layout, swizzle alignment, and WGMMA fragment mapping
- softmax/LSE integration around the WGMMA/TMA skeleton
