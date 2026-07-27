# Wrong kernel sources
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d64
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d64-causal
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d128
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-d128-causal
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d64
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d64-causal
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d128
/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-linfo-mha-bwd-d128-causal
All 5 prompt-tags and all 8 turns

# Fixit config
Only hopper-07 and hopper-08 for fwd and only hopper-012 and hopper-013 for bwd
fix-it turns: 5
fix-it target speedup: 0.15x
min-speedup: 0.0x

TODO: Add a reblance after glm synthesizes reasoning

# Eval defintions (12 problems)
Use config: /home/ubuntu/AccRL-exps/prompt_configs/2026-0428-1632.json
mha_with_lse_d96
mha_with_lse_d96_causal
mha_bwd_d96
mha_bwd_d96_causal
gqa_paged_decode_h32_kv8_d128_ps64
gqa_paged_prefill_causal_h32_kv4_d128_ps64
gqa_ragged_prefill_causal_h32_kv8_d128
mla_paged_decode_h16_ckv512_kpe64_ps1
mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
mla_ragged_prefill_causal_h16_qk192_vo128
Use config: /home/ubuntu/AccRL-exps/prompt_configs/2026-0507-1340.json
fp8_mha_with_lse_d64
fp8_mha_with_lse_d128