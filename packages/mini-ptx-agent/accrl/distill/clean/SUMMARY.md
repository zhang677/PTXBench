# Reasoning-pairs cleaning

Tooling for diagnosing and filtering `reasoning_pairs.jsonl` files produced by
`accrl.distill.run_experiment`. Originally written to investigate why SFT of
Qwen3.5 35B on the Kimi K2.6 reasoning traces in
`AccRL-exps/distill/experiments/trajectory_reasoning_kimi2.6——1228/reasoning_pairs.jsonl`
diverged (loss blow-up).

## Bugs found in `reasoning_pairs.jsonl`

Audit was run on the 215-record file; numbers below are from that audit.

### 1. Hidden-thinking leak — 4 records (HIGH confidence cause of loss blow-up)

Records: indices `95, 146, 148, 162`.

The Kimi K2.6 post-processor sometimes failed to strip the hidden CoT from the
visible answer. For these records the `reasoning` field actually contains:

```
<hidden thinking trace, 30k–63k chars> </think> <my_reasoning>
<visible reasoning trace>
```

`accrl/distill/sft/build_sft_dataset.py` wraps `reasoning` as
`<think>\n{reasoning.strip()}\n</think>`, so the assistant target ends up with
**a premature `</think>` token in the middle**:

```
<think>
…hidden trace… </think> <my_reasoning>
…visible trace…
</think>
```

`</think>` is a single special token in Qwen's tokenizer; training the model to
emit it mid-thought and then continue generating produces large CE spikes and
malformed targets. Highest-confidence single cause for loss instability.

### 2. Training on failing kernels — 122/215 records (57%)

`passed=False` and `speedup=None` for the majority. The reasoning still ends
with confident statements like "I'm now ready to write the kernel," but the
kernel that followed failed compilation or correctness. The student is taught
to confidently justify broken designs.

### 3. Chinese language leakage — 13/215 reasonings

Kimi (Chinese-trained) drops CJK characters mid-English in records:
`50, 144, 71, 17, 118, 2, 81, 5, 96, 97, 32, …`

Examples:

- "…`wait_group 0` 没有其他待处理的 TMA stores…"
- "**从Turn 2 的正确设计恢复**"
- "…98KB \* 2 = 196KB 接近极限…"

A handful of out-of-distribution token sequences in a small (215-row) dataset
can produce per-step loss spikes.

### 4. Sequence-length pressure

Estimated tokens (chars/3.5):

| field         | mean | max  |
|---------------|------|------|
| input         | 42 k | 65 k |
| reasoning     | 7 k  | 22 k |
| total prompt  | 50 k | 79 k |

97.7% of records exceed 32k tokens; 8.8% exceed 64k. With
`MAX_TOKENS_PER_GPU=32768` × `CP_SIZE=4` ≈ 128k effective window, all rows
nominally fit, but the truncation/padding boundary is close.

Note: `sft_data/glm_kimi_intersection/length_stats.json` reports
`total_tokens.max = 2` for every variant — the tokenizer never actually ran in
that stats step, so the team likely doesn't realize the true sequence lengths.

### 5. Dataset narrowness

All 215 records are the same definition (`gemm_n7168_k5120`), 8 turns × ~27
exp_ids. After GLM intersection the SFT set is 189 rows of one task with very
long, highly variable targets (9k–77k chars). High per-batch gradient variance.

### Other observations (no obvious problem)

- No empty/whitespace-only fields, no control/private-use unicode,
  no truncated code fences.
- 1 record (idx 89) doesn't end with standard punctuation — possible
  truncation, low impact.
- No chat-template special tokens (`<|im_end|>` etc.) leaked into reasoning.

## Scripts

Both scripts are runnable as modules from the `AccRL` repo root.

### `inspect_reasoning_pairs.py`

Read-only audit. Prints shape, length stats, metadata distributions, every leak
class above, repetition / truncation hints, and estimated token-bucket
occupancy.

```bash
cd /home/ubuntu/AccRL
python3 -m accrl.distill.clean.inspect_reasoning_pairs \
    '/home/ubuntu/AccRL-exps/distill/experiments/trajectory_reasoning_kimi2.6——1228/reasoning_pairs.jsonl'
```

### `clean_reasoning_pairs.py`

Filter the file. Default filters drop the high-confidence problems; other
filters are opt-in via flags.

Default-on (turn off with `--keep-*`):

| filter                  | drops records where…                                              |
|-------------------------|-------------------------------------------------------------------|
| hidden-thinking leak    | `reasoning` contains `</think>`, `<think>`, `<my_reasoning>`, `</my_reasoning>` |
| chat/EOS special tokens | `reasoning` contains `<|im_end|>`, `<|endoftext|>`, `<unk>`, etc. |
| failed kernel           | `metadata.passed is False` or `metadata.speedup is None`          |
| min length              | `len(reasoning) < --min-reasoning-chars` (default 200)            |

Opt-in:

| flag                    | effect                                                    |
|-------------------------|-----------------------------------------------------------|
| `--drop-chinese`        | drop reasonings containing CJK characters                 |
| `--drop-no-punct-ending`| drop reasonings not ending in `.!?"'`)] ` ` (trunc indicator) |

```bash
cd /home/ubuntu/AccRL

# Dry-run first to see what would happen.
python3 -m accrl.distill.clean.clean_reasoning_pairs \
    '/home/ubuntu/AccRL-exps/distill/experiments/trajectory_reasoning_kimi2.6——1228/reasoning_pairs.jsonl' \
    -o /tmp/cleaned.jsonl --dry-run

# Conservative clean (recommended first attempt against the blow-up):
# only drop the 4 leak rows + 13 Chinese-leak rows, keep failed kernels.
# 215 → 198 rows.
python3 -m accrl.distill.clean.clean_reasoning_pairs \
    '/home/ubuntu/AccRL-exps/distill/experiments/trajectory_reasoning_kimi2.6——1228/reasoning_pairs.jsonl' \
    -o '/home/ubuntu/AccRL-exps/distill/experiments/trajectory_reasoning_kimi2.6——1228/reasoning_pairs.cleaned.jsonl' \
    --keep-failed --drop-chinese

# Strict clean: also drop failed-kernel rows (215 → 90 rows).
python3 -m accrl.distill.clean.clean_reasoning_pairs \
    '/home/ubuntu/AccRL-exps/distill/experiments/trajectory_reasoning_kimi2.6——1228/reasoning_pairs.jsonl' \
    -o '/home/ubuntu/AccRL-exps/distill/experiments/trajectory_reasoning_kimi2.6——1228/reasoning_pairs.strict.jsonl' \
    --drop-chinese
```

The cleaning script also writes a `<output>.report.json` next to its output
listing every dropped record with `(line, reason, exp_id, turn, passed,
speedup, reasoning_chars)` for auditing.

## Recommended workflow

1. Run `inspect_reasoning_pairs.py` to confirm the leak counts on the file
   you're about to train on.
2. Run `clean_reasoning_pairs.py --dry-run --keep-failed --drop-chinese` to
   preview the conservative filter.
3. Produce a cleaned JSONL and re-run `accrl.distill.sft.build_sft_dataset` on
   it. Re-train. If loss is still unstable, escalate to the strict filter
   (drop failed kernels) and consider re-running the data generation with a
   stricter system prompt that forbids language switching, since the small
   dataset size makes a few outliers disproportionately influential.
