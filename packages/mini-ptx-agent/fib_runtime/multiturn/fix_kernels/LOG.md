# fix_kernels Run Log

This log inventories the `/home/ubuntu/AccRL-exps` artifacts tied to
`/home/ubuntu/AccRL/fib_runtime/multiturn/fix_kernels`.

## Fixit Kernel Sets

These are selected failure/error-kernel manifests and prompt configs for fix-it
style follow-up runs.

| Artifact | Rows | What it does |
| --- | ---: | --- |
| `/home/ubuntu/AccRL-exps/tasks/test-kernels-fixit-qwen36-27b.csv` | 75 | Test-sized set of failed/error kernels for Qwen 27B fix-it evaluation. |
| `/home/ubuntu/AccRL-exps/prompt_configs/test-fixit-qwen36-27b.json` | 75 | Prompt config derived from the test fix-it kernel set. |
| `/home/ubuntu/AccRL-exps/tasks/scale-kernels-fixit-qwen36-27b.csv` | 526 | Larger scale fix-it candidate set. |
| `/home/ubuntu/AccRL-exps/tasks/scale-d128-kernels-fixit-qwen36-27b.csv` | 222 | D128 subset of the scale fix-it candidate set. |
| `/home/ubuntu/AccRL-exps/prompt_configs/scale-d128-kernels-fixit-qwen36-27b.json` | 222 | Prompt config derived from the D128 scale fix-it set. |

Related eval outputs include:

- `/home/ubuntu/AccRL-exps/eval_runs/test-fixit-qwen36-27b-gemini`
- `/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-gemini`
- `/home/ubuntu/AccRL-exps/sft_experiments/test-fixit-qwen36-27b-gemini-glm/`
