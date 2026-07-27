# Reverse-CoT Distillation

Generate reasoning tokens for expert kernel trajectories. A strong model (GLM-5.1, Kimi-K2.6, MiniMax-M2.7, or DeepSeek-V4-Pro) explains the thought process behind Gemini's kernel optimization decisions, producing `<my_reasoning>...</my_reasoning>` traces (extracted) plus the model's hidden CoT (raw `reasoning_content` / `reasoning` field) for SFT training.

## Directory Structure

```
accrl/distill/
├── run_experiment.py       # Entry point
├── inspector.py            # Textual TUI for reasoning_pairs.jsonl
├── utils.py                # Extraction, selectors, context builders, prompts, ExperimentConfig
└── configs/                # Experiment YAML configs
    ├── trajectory_reasoning.yaml          # GLM-5.1 thinking-off, all-with-history (canonical)
    ├── trajectory_reasoning_kimi.yaml     # Kimi-K2.6 thinking-on
    ├── trajectory_reasoning_deepseek.yaml # DeepSeek-V4-Pro thinking-on (endpoint unstable)
    ├── full_context_best_kernel.yaml      # Best kernel only, v1 prompt
    └── full_context_best_kernel_v2.yaml   # Best kernel only, deep prompt

../AccRL-exps/distill/                    # Lives in sister repo
├── gemini_turns.jsonl                    # Extracted turns from Gemini trajectories (55 turns)
└── experiments/<exp_name>/
    ├── config.yaml                       # Resolved ExperimentConfig
    ├── provenance.json                   # Git SHA, command, timestamp, dirty/untracked files
    └── reasoning_pairs.jsonl             # {input, reasoning, thinking, metadata} per pair
```

## Quick Start

### 1. Extract turns from eval trajectories (one-time)

```bash
python -c "
from accrl.distill.utils import extract_eval_dir
from pathlib import Path
import json

turns = extract_eval_dir(Path('../AccRL-exps/eval_runs/2026-0414-1214'))
with open('../AccRL-exps/distill/gemini_turns.jsonl', 'w') as f:
    for t in turns:
        f.write(json.dumps(t) + '\n')
print(f'{len(turns)} turns extracted')
"
```

### 2. Run reasoning generation

Tree must be clean (`run_experiment.py` refuses dirty trees). Add the proxy for non-Kimi endpoints — Kimi via `pod-internal.fbinfra.net` does NOT need it.

```bash
HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 \
  python -m accrl.distill.run_experiment \
    --config accrl/distill/configs/trajectory_reasoning.yaml \
    --turns ../AccRL-exps/distill/gemini_turns.jsonl \
    --output-dir ../AccRL-exps/distill/experiments/trajectory_reasoning/
```

Default `--max-concurrent` is 16. Pass `--allow-dirty` for one-off smoke tests (records the bypass in `provenance.json`).

Monitor progress (results stream incrementally):
```bash
wc -l ../AccRL-exps/distill/experiments/trajectory_reasoning/reasoning_pairs.jsonl
```

### 3. Inspect results

**Textual TUI** for distillation pairs:
```bash
python -m accrl.distill.inspector ../AccRL-exps/distill/experiments/trajectory_reasoning_glm_v3/reasoning_pairs.jsonl
```
Keys: `]`/`[` next/prev record, `Tab`/`Shift-Tab` next/prev section, `Ctrl-D`/`Ctrl-U` page within, `q` quit.

**Raw Gemini trajectories** (mini-swe-agent inspector — installed):
```bash
python -m minisweagent.run.utilities.inspector ../AccRL-exps/eval_runs/2026-0414-1214/trajectories/exp_000.json
```

**Quick stats**:
```bash
python -c "
import json, statistics
records = [json.loads(l) for l in open('<path>/reasoning_pairs.jsonl')]
lens = [len(r['reasoning']) for r in records]
print(f'pairs: {len(records)}')
print(f'reasoning: mean={statistics.mean(lens):.0f}, median={statistics.median(lens):.0f}, range={min(lens)}-{max(lens)}')
if 'thinking' in records[0]:
    th = [len(r.get('thinking','')) for r in records]
    print(f'thinking:  mean={statistics.mean(th):.0f}, median={statistics.median(th):.0f}')
"
```

## How It Works

### Pipeline

```
Gemini eval trajectories (eval_runs/.../trajectories/*.json)
    ↓ extract_eval_dir()
Turn records (gemini_turns.jsonl)
    ↓ selector (best_per_trajectory / passing_only / all_with_history)
Selected turns
    ↓ context builder (build_context / build_trajectory_context)
Prompts with context
    ↓ reasoning prompt (v1 / v2_deep / v3_trajectory)
Full prompts
    ↓ reasoning model via litellm (+ custom AsyncOpenAI for pod-internal endpoints)
Response: content (visible) + reasoning_content/reasoning (hidden CoT)
    ↓ _extract_reasoning(content)  → reasoning (extracted from <my_reasoning>...</my_reasoning>)
    ↓ raw reasoning_content/reasoning field → thinking
    ↓ quality filter (length + numeric grounding)
reasoning_pairs.jsonl: {input, reasoning, thinking, metadata}
```

### Output schema (`reasoning_pairs.jsonl`, one record per line)

```json
{
  "system_prompt": "...",
  "input": "<full prompt sent to model>",
  "reasoning": "<extracted from <my_reasoning>...</my_reasoning> in content>",
  "thinking": "<raw reasoning_content (GLM/Kimi) or reasoning (DeepSeek) field; \"\" if not exposed>",
  "metadata": {"exp_id": "...", "turn": 0, "speedup": 0.497, "passed": true, "definition_name": "..."}
}
```

### Selectors

- `best_per_trajectory`: Highest speedup per trajectory (8 examples from 8 trajectories)
- `passing_only`: All turns with speedup > 0
- `all_with_history`: Every turn, annotated with full preceding history (for trajectory reasoning)

### Reasoning Prompts

- `v1`: Standard — analyze problem, design tiling, plan memory/compute
- `v2_deep`: Maximum depth — 7 sections, hardware constraints, calculations
- `v3_trajectory`: Multi-turn aware — first-person, explains what changed from previous turn and why

### Context Builders

- `build_context`: Reference manual + task + expert kernel
- `build_trajectory_context`: Reference manual + task + all previous turns (kernel + verdict) + current kernel (no verdict)

## Models (`MODEL_CONFIGS` in `run_experiment.py`)

| Name | Endpoint | Needs proxy | Custom AsyncOpenAI client | CoT field |
|---|---|---|---|---|
| `GLM-5.1` | `llm.gateway.msl-cw...` /playground/glm5.1/v1 | yes | no | `reasoning_content` |
| `Kimi-K2.6` | `pod-internal.fbinfra.net/v1` | **no** | **yes** (X-Predictor-Tier header + smc_tier_name query + verify_ssl=False) | `reasoning_content` |
| `DeepSeek-V4-Pro` | `llm.gateway.msl-cw...` /rift/deepseek-v4/v1 | yes | no | `reasoning` |
| `MiniMax-M2.7` | `llm.gateway.msl-cw...` /playground/minimax/v1 | yes | no | n/a |

Both `reasoning_content` and `reasoning` are captured by `_generate_one` via a `getattr` chain.

### `enable_thinking` semantics (per model)

Plumbed through `ExperimentConfig.enable_thinking`, persisted in `config.yaml`, injected into `extra_body={"chat_template_kwargs": {"enable_thinking": ...}}` for GLM/Kimi/DeepSeek.

- **GLM-5.1**: `false` puts the entire output budget into the visible `<my_reasoning>` block. v3 dataset uses this (38.5k char mean, vs 4.2k with thinking on).
- **Kimi-K2.6**: `true` writes meta-planning to `reasoning_content` (saved as `thinking` field) and the polished `<my_reasoning>` block to `content`. Kimi v1 uses this.
- **DeepSeek-V4-Pro**: same toggle, CoT lands in `reasoning` field.

## Adding a new reasoning model

1. Add entry to `MODEL_CONFIGS` in `run_experiment.py` with `model`, `api_base`, `api_key`. For pod-internal-style gateways add `default_headers`, `default_query`, `verify_ssl: False` — `run()` builds a custom `AsyncOpenAI` client when `default_query` is set or `verify_ssl=False`, then passes it via `client=` to `litellm.acompletion`.
2. If the model exposes a `chat_template_kwargs.enable_thinking` knob, add the model name to the tuple in `run()` that injects `extra_body`.
3. If the model puts CoT in a non-standard field, extend the `getattr` chain in `_generate_one` (currently checks `reasoning_content` then `reasoning`).
4. Probe new endpoints with `httpx.post(...)` first to confirm URL form (`/v1` or not), model name, and CoT field name before wiring litellm — saves debugging time when 10-min timeouts hit.

## Reproducibility (enforced)

`run_experiment.py` refuses to launch if the AccRL working tree has uncommitted changes to tracked files. Pass `--allow-dirty` for throwaway smoke tests (records the bypass in `provenance.json`). Untracked files don't block but are recorded.

`provenance.json` schema:
```json
{
  "command": "<full shell command>",
  "argv": [...],
  "git_sha": "<HEAD SHA>",
  "git_dirty_files": ["<porcelain status lines>"],
  "git_untracked_files": [...],
  "allow_dirty_bypass": false,
  "timestamp_utc": "...",
  "turns_path": "<abs path>",
  "output_dir": "<abs path>"
}
```

## Throughput diagnostics

After a run, extract signals from the log:

```bash
LOG=/tmp/run.log
echo "Retries: $(grep -c 'Retrying request' $LOG)"
echo "FAILED:  $(grep -c FAILED $LOG)"
grep ' OK ' $LOG | awk '{print $1, $2}' | python3 -c "
import sys
from datetime import datetime
ts = [datetime.strptime(l.strip().split(',')[0], '%Y-%m-%d %H:%M:%S') for l in sys.stdin if l.strip()]
gaps = sorted((ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1))
total = (ts[-1]-ts[0]).total_seconds()/60
print(f'pairs={len(ts)} | gap median={gaps[len(gaps)//2]:.0f}s | wall={total:.0f}min | {len(ts)/(total/60):.1f} pairs/hr')
"
```

Saturation logic:
- Wall ≈ (per-call latency × pairs) / concurrency → slots fully utilized.
- Retry rate >50% or FAILED >10% → endpoint pushed past capacity, lower concurrency.
- Retries ~0 and idle slots → safe to raise concurrency.

## Current Data

Source: `eval_runs/2026-0414-1214` — 8 Gemini 3.1 Pro trajectories on `gemm_n7168_k5120` (55 turns total with kernel_code)

| Dataset | Pairs | Reasoning Model | Mean reasoning | Notes |
|---|---|---|---|---|
| `trajectory_reasoning_glm_v3` | 50/55 | GLM-5.1 (thinking off) | 38.5k chars | Committed; output-only (no `thinking` field) |
| `trajectory_reasoning_kimi_v1` | 52/55 | Kimi-K2.6 (thinking on) | ~13k reasoning + ~57k thinking | Both fields populated |

## Tests

```bash
python -m pytest tests/test_distill.py -v --noconftest
```
