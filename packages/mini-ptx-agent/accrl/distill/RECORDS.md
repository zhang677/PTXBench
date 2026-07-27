This is a record of distill experiments.

provenance: The command takes in trajectories and generates reasoning pairs. The provenance.json resides in the output dirs that are usually named as AccRL-exps/distill/experiments/trajectory_reasoning_xxx

manifest: Reasoning pairs => Parquet script. The manifest.json resides in AccRL-exps/sft_data/xxx/data

DataLink: The link to parquet
Missing now:
1. How was /home/chengze/work/AccRL-exps/distill/gemini_turns_0422.jsonl generated?
    - python3 -m accrl.distill.extract_turns /home/chengze/work/AccRL-exps/eval_runs/2026-0422-1002 /home/chengze/work/AccRL-exps/distill/gemini_turns_0422.jsonl
2. What are the "free-generation eval prompts" stored in `AccRL-exps/sft_experiments/glm_kimi_intersection/eval`?
    - The “free-generation eval prompts” are prompts used to ask the SFT model to generate freely, it's basically replay what we asked Gemini to do. `accrl/distill/sft/prepare_sft_eval_prompts.py --turns gemini_turns_0422.json`
3. The files listed in README.md are not actually in `AccRL-exps/sft_experiments/glm_kimi_intersection/eval`
4. What is `reasoning_raw` in inspector.py?


