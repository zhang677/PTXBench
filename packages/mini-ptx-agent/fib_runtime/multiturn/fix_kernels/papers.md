# Papers Relevant to the Fixit v0/v1/v2 Regression

Last updated: 2026-06-19.

These papers are relevant to the observed pattern where `Qwen3.6-27B-fixit-v1` and `Qwen3.6-27B-fixit-v2` were trained on more fixit data than `v0` but scored worse on downstream evals.

## Data Quality Beats Raw Quantity

- LIMA: Less Is More for Alignment
  - URL: https://arxiv.org/abs/2305.11206
  - Why it matters: Shows that a small number of carefully curated SFT examples can teach response behavior effectively. This supports treating fixit row quality and task match as more important than raw parquet row count.

- AlpaGasus: Training A Better Alpaca with Fewer Data
  - URL: https://arxiv.org/abs/2307.08701
  - Why it matters: Filters Alpaca from 52k examples to 9k higher-quality examples and obtains better results. This is a close analogue for filtering fixit examples by correctness margin, speedup, and repair value.

- From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
  - URL: https://arxiv.org/abs/2308.12032
  - Why it matters: Uses a model-guided difficulty/usefulness criterion to select a small high-value subset for instruction tuning. This maps directly to selecting fixit examples by whether they teach a useful repair, not merely whether the final kernel passes.

- DataComp-LM: In search of the next generation of training sets for language models
  - URL: https://arxiv.org/abs/2406.11794
  - Why it matters: Although pretraining-focused, it is strong evidence that dataset curation and filtering can dominate raw token count. It is useful background for arguing that "more tokens" is not the right objective.

## Difficulty, Mixtures, and Generalization

- Data Difficulty and the Generalization-Extrapolation Tradeoff in LLM Fine-Tuning
  - URL: https://arxiv.org/abs/2605.12906
  - Why it matters: Argues that the best data difficulty depends on data budget and that difficulty selection changes generalization. For fixit, adding low-margin or off-distribution repair examples can change what the model extrapolates to at eval time.

- Scaling Instruction-Finetuned Language Models
  - URL: https://arxiv.org/abs/2210.11416
  - Why it matters: Shows task mixture can help, but only under carefully scaled multi-task instruction tuning. The fixit runs are the opposite regime: very small data, high LR, no validation, and mixed CUDA problem families.

## Imitation and Reasoning Trace Quality

- The False Promise of Imitating Proprietary LLMs
  - URL: https://arxiv.org/abs/2305.15717
  - Why it matters: Teacher-output imitation can teach style without transferring robust capability. The fixit parquets are Kimi-generated reasoning plus code; added rows may teach the look of repair reasoning without improving kernel synthesis.

- Can Language Models Perform Robust Reasoning in Chain-of-thought Prompting with Noisy Rationales?
  - URL: https://arxiv.org/abs/2410.23856
  - Why it matters: Shows noisy or irrelevant rationales can sharply hurt reasoning. This is relevant when synthesized fixit rationales are long, inconsistent, or attached to low-quality fixes.

- Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning
  - URL: https://arxiv.org/abs/2605.21127
  - Why it matters: Fine-tuning can alter explicit reasoning behavior in ways not captured by final-answer metrics. The fixit evals should inspect generated repair traces and code behavior, not only final correctness.

- Structural Rationale Distillation via Reasoning Space Compression
  - URL: https://arxiv.org/abs/2605.07139
  - Why it matters: Argues that inconsistent teacher rationales burden the student and that compressing reasoning paths can improve distillation. This is relevant because the fixit data contains diverse freeform Kimi repair reasoning.

## Forgetting and Robustness Loss

- An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning
  - URL: https://arxiv.org/abs/2308.08747
  - Why it matters: Fine-tuning on new data can degrade prior capabilities. The fixit models get worse on GEMM even though the added data is MHA-oriented, which is consistent with forgetting or capability drift.

- Revisiting Catastrophic Forgetting in Large Language Model Tuning
  - URL: https://arxiv.org/abs/2406.04836
  - Why it matters: Connects catastrophic forgetting with loss landscape flatness and mitigation strategies. This is useful if we want to test lower LR, fewer steps, SAM-style mitigation, or checkpoint selection.

- Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution
  - URL: https://arxiv.org/abs/2202.10054
  - Why it matters: Shows fine-tuning can hurt out-of-distribution robustness even when in-distribution performance improves. Here, narrow fixit SFT can distort the base model's general kernel-generation behavior.

- Robust fine-tuning of zero-shot models
  - URL: https://arxiv.org/abs/2109.01903
  - Why it matters: Weight interpolation between base and fine-tuned models can preserve robustness. It suggests a possible mitigation for LoRA/checkpoint interpolation or earlier checkpoint selection.

## Narrow Fine-Tuning Can Cause Broad Behavioral Shifts

- Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs
  - URL: https://arxiv.org/abs/2502.17424
  - Why it matters: The topic is safety rather than CUDA quality, but the mechanism is relevant: narrow fine-tuning can produce broad unintended behavior changes outside the training slice.

## Standard SFT Scale References

These are useful for calibrating whether the fixit runs have enough SFT data to support a strong conclusion. The important comparison unit is usually independent examples/tasks plus held-out eval coverage, not just total tokens.

- LIMA: Less Is More for Alignment
  - URL: https://arxiv.org/abs/2305.11206
  - Scale reference: around 1,000 carefully curated SFT examples.
  - Why it matters: This is a "small data can work" reference, but it is still much larger in example count than the current fixit sets. `fixit-v2` has 140 kept rows, so it is below even the small-data alignment regime in terms of independent examples.

- AlpaGasus: Training A Better Alpaca with Fewer Data
  - URL: https://arxiv.org/abs/2307.08701
  - Scale reference: filters Alpaca from about 52k instruction examples to about 9k higher-quality examples.
  - Why it matters: This supports filtering by quality, but also shows that "filtered small" in instruction-tuning papers often still means thousands of examples, not tens or low hundreds.

- Tulu 2: Advancing Open Language Models with Instruction Tuning and RLHF
  - URL: https://arxiv.org/abs/2311.10702
  - Scale reference: broad instruction-tuning mixtures are in the hundreds-of-thousands-of-examples regime.
  - Why it matters: This is not a direct CUDA-fixit analogue, but it is a useful upper reference for general-purpose SFT claims. It argues against making broad Qwen3.6-27B SFT conclusions from 38-140 examples.

- Code Alpaca: An Instruction-following LLaMA Model for Code Generation
  - URL: https://github.com/sahil280114/codealpaca
  - Scale reference: about 20k code instruction examples.
  - Why it matters: Even lightweight code-instruction SFT commonly uses tens of thousands of examples, far above the current fixit row count.

- WizardCoder: Empowering Code Large Language Models with Evol-Instruct
  - URL: https://arxiv.org/abs/2306.08568
  - Scale reference: about 78k evolved code instruction examples.
  - Why it matters: This is a closer code-generation reference than general chat alignment. It suggests that robust code-SFT claims usually need far more independent examples than the current fixit parquets.

- Magicoder: Source Code Is All You Need
  - URL: https://arxiv.org/abs/2312.02120
  - Scale reference: about 75k synthetic code instruction examples.
  - Why it matters: Another code-SFT reference in the tens-of-thousands regime. It is useful when deciding whether a CUDA fixit result is a real data-scaling conclusion or just small-sample variance plus task-mixture shift.

- DeepSeek-Coder: When the Large Language Model Meets Programming
  - URL: https://arxiv.org/abs/2401.14196
  - Scale reference: instruction/code tuning is reported at very large code-data scale, far beyond the current fixit setting.
  - Why it matters: This is a broad code-model scaling reference. It should not be used as the minimum target for narrow fixit, but it frames what "standard" code SFT looks like at production scale.

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
  - URL: https://arxiv.org/abs/2501.12948
  - Scale reference: distilled reasoning models use hundreds of thousands of examples.
  - Why it matters: This is relevant because fixit data is reasoning-plus-code distillation. It suggests that stable reasoning distillation claims generally require far more examples than the current setup.

## Token-Scale Interpretation for Current Fixit Runs

The current fixit rows are unusually long, so token count looks larger than example diversity:

| run | kept rows | unique total tokens | unique target/loss tokens | 5-epoch total-token exposure |
| --- | ---: | ---: | ---: | ---: |
| fixit-v0 | 38 | 1.9M | 0.70M | 9.6M |
| fixit-v1 | 74 | 4.0M | 1.61M | 19.9M |
| fixit-v2 | 140 | 7.5M | 2.84M | 37.3M |

Practical calibration:

- Minimum credible internal conclusion: about 500-1,000 independent fixit examples, roughly 25M-50M unique SFT tokens at the current row length.
- Stronger narrow CUDA-repair claim: about 2k-10k examples, roughly 100M-500M unique SFT tokens, with task-balanced splits and held-out evals.
- Broad code-SFT claim: tens of thousands of examples and hundreds of millions or more tokens, closer to WizardCoder/Magicoder/DeepSeek-Coder scale.

The current runs have nontrivial token exposure because each example is long, but they are still very small by example count. For conclusions about Qwen3.6-27B, the first priority should be more independent fixit tasks and balanced held-out evals, not simply repeating the same long rows for more epochs.

## Practical Takeaways for Fixit

- Treat `raw_rows` as a weak signal. Track post-filter rows, task mixture, speedup distribution, repair-stage distribution, and duplicate wrong inputs.
- Treat token count as necessary but insufficient. At the current row length, 140 rows already create 7.5M unique tokens, but that is still only 140 independent repair examples.
- Avoid training on all passing fixes if many are low-speedup or low-margin. For v2, every row added beyond v1 had speedup below 0.15.
- Keep eval-turn limits comparable. Some v1 MHA eval artifacts only contain 4 turns, so comparing them to 8-turn v0/v2 runs is misleading.
- Add validation or checkpoint selection. The current runs use 5 epochs with the same LR and no eval during training; larger datasets get many more optimizer steps.
