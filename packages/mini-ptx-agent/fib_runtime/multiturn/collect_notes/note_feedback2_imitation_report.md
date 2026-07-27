# Qwen3.6-27B Note-Feedback2 Imitation Report

Date: 2026-07-10

## Scope

Run roots inspected:

- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96`
- `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal`

Artifacts inspected:

- `trajectories/exp_*.json`
- the assistant kernel in the message immediately after a user feedback message containing `Best fixed kernel:`
- the retrieved `Best fixed kernel` code blocks embedded in that preceding feedback message

## Method

For each feedback-to-next-response pair:

1. Extract the next assistant kernel from the first fenced C/C++ code block.
2. Extract all retrieved `Best fixed kernel` code blocks from the immediately preceding user feedback message.
3. Strip C/C++ line and block comments from both sides.
4. Normalize whitespace per line and drop empty lines.
5. Compare the next assistant kernel against each retrieved fixed kernel and keep the closest match.

Primary imitation definition:

- strict imitation: normalized line-sequence similarity `>= 0.95`

Additional reference thresholds:

- exact copy: comment-stripped normalized code is identical
- broader near-copy: line similarity `>= 0.95` or token 9-shingle Jaccard similarity `>= 0.95`
- loose similarity: normalized line-sequence similarity `>= 0.90`

## Result

Strict imitation count: **22**

Other counts:

- exact copies: **1**
- broader near-copies: **23**
- loose `>= 0.90` line-similarity matches: **39**

Coverage:

- trajectories inspected: **32**
- feedback-to-next-kernel comparisons: **174**

## Per-Folder Counts

| Folder suffix | Comparisons | Exact | Strict line >= 0.95 | Loose line >= 0.90 | Token9 >= 0.95 | Broader near-copy |
|---|---:|---:|---:|---:|---:|---:|
| `mha-bwd-d128` | 20 | 0 | 6 | 6 | 3 | 6 |
| `mha-bwd-d128-causal` | 24 | 1 | 3 | 6 | 3 | 3 |
| `mha-bwd-d96` | 24 | 0 | 1 | 1 | 0 | 1 |
| `mha-bwd-d96-causal` | 20 | 0 | 0 | 3 | 0 | 0 |
| `mha-d128` | 16 | 0 | 6 | 8 | 4 | 7 |
| `mha-d128-causal` | 19 | 0 | 4 | 7 | 0 | 4 |
| `mha-d96` | 26 | 0 | 1 | 4 | 0 | 1 |
| `mha-d96-causal` | 25 | 0 | 1 | 4 | 1 | 1 |
| **Total** | **174** | **1** | **22** | **39** | **11** | **23** |

## Strict Imitation Cases

| Run folder suffix | Experiment | Next turn | Line similarity | Token9 similarity | Exact | Fixed variant |
|---|---|---:|---:|---:|---|---:|
| `mha-bwd-d128` | `exp_000` | 2 | 0.9764 | 0.9584 | no | 1 |
| `mha-bwd-d128` | `exp_001` | 2 | 0.9967 | 0.9928 | no | 1 |
| `mha-bwd-d128` | `exp_001` | 6 | 0.9934 | 0.9841 | no | 1 |
| `mha-bwd-d128` | `exp_003` | 2 | 0.9693 | 0.9329 | no | 1 |
| `mha-bwd-d128` | `exp_003` | 4 | 0.9779 | 0.9443 | no | 1 |
| `mha-bwd-d128` | `exp_003` | 8 | 0.9656 | 0.9116 | no | 1 |
| `mha-bwd-d128-causal` | `exp_001` | 3 | 0.9801 | 0.9683 | no | 1 |
| `mha-bwd-d128-causal` | `exp_002` | 8 | 1.0000 | 1.0000 | yes | 1 |
| `mha-bwd-d128-causal` | `exp_003` | 2 | 0.9741 | 0.9737 | no | 1 |
| `mha-bwd-d96` | `exp_002` | 2 | 0.9576 | 0.8800 | no | 1 |
| `mha-d128` | `exp_000` | 4 | 0.9930 | 0.9829 | no | 1 |
| `mha-d128` | `exp_000` | 6 | 0.9536 | 0.9011 | no | 1 |
| `mha-d128` | `exp_001` | 2 | 0.9518 | 0.9049 | no | 1 |
| `mha-d128` | `exp_002` | 3 | 0.9652 | 0.9797 | no | 1 |
| `mha-d128` | `exp_002` | 5 | 0.9684 | 0.9392 | no | 1 |
| `mha-d128` | `exp_002` | 6 | 0.9835 | 0.9687 | no | 1 |
| `mha-d128-causal` | `exp_000` | 2 | 0.9508 | 0.9216 | no | 1 |
| `mha-d128-causal` | `exp_003` | 6 | 0.9605 | 0.9373 | no | 1 |
| `mha-d128-causal` | `exp_003` | 7 | 0.9593 | 0.9021 | no | 1 |
| `mha-d128-causal` | `exp_003` | 8 | 0.9607 | 0.9107 | no | 1 |
| `mha-d96` | `exp_000` | 3 | 0.9542 | 0.9300 | no | 1 |
| `mha-d96-causal` | `exp_002` | 6 | 0.9909 | 0.9788 | no | 1 |

## Interpretation

Using the strict comment-stripped line-similarity threshold, there are **22 clear imitations** across the eight folders. Only one is an exact copy, but many strict matches are close enough that the next-round kernel is effectively a lightly modified version of a retrieved correct kernel.
