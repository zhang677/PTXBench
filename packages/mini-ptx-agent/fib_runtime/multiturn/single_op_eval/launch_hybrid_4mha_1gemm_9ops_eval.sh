#!/usr/bin/env bash
set -euo pipefail

RUN_DATE="${RUN_DATE}"
MODEL_NAME="${MODEL_NAME}"
SERVICE_URL="${SERVICE_URL}"
ACCRL_MODEL_HOST="${ACCRL_MODEL_HOST}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MAX_PROFILES="${MAX_PROFILES:-8}"
RUNNER="${RUNNER:-/home/ubuntu/AccRL/fib_runtime/multiturn/run_parallel_v2.py}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/home/ubuntu/AccRL-exps/eval_runs}"
CONFIG="${CONFIG:-/home/ubuntu/AccRL-exps/prompt_configs/2026-0428-1632.json}"
LLM_API_TIMEOUT=3600

RUN_ARGS=(
    --timeout 86400
    --without-local-gpu
    --model "$MODEL_NAME"
    --gpu-arch hopper
    --max-parallel "$MAX_PARALLEL"
    --max-profiles "$MAX_PROFILES"
    --service-url "$SERVICE_URL"
    --turn-timeout 980
)

run_one() {
  local output_root="$1"
  shift
  local mode_args=()
  if [[ -d "$output_root" ]]; then
    mode_args=(--resume)
  fi

  LLM_API_TIMEOUT="$LLM_API_TIMEOUT" ACCRL_MODEL_HOST="$ACCRL_MODEL_HOST" MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 python "$RUNNER" \
    --output-root "$output_root" \
    "$@" \
    "${mode_args[@]}" \
    "${RUN_ARGS[@]}"
}

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d128" \
  --definition mha_with_lse_d128 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d128_bc38b351-d595-451b-9153-8e225702e53b.py \
  --config /home/ubuntu/AccRL-exps/prompt_configs/2026-0605-mha-p4-mha-patched.json

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-d128-causal" \
  --definition mha_with_lse_d128_causal \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0426-1410/mha_with_lse_d128_causal_6d2f67a7-225a-4af5-87d3-cbb99b496325.py \
  --config /home/ubuntu/AccRL-exps/prompt_configs/2026-0605-mha-p4-mha-patched.json

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d128" \
  --definition mha_bwd_d128 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_38c3b07c-f006-5f5e-9860-ba214c805a6b.py \
  --config /home/ubuntu/AccRL-exps/prompt_configs/2026-0605-mha-bwd-p4-mha-patched.json

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-mha-bwd-d128-causal" \
  --definition mha_bwd_d128_causal \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0427-1308/mha_bwd_d128_causal_c119b3f0-c051-5e96-9c2a-2268d992fe1a.py \
  --config /home/ubuntu/AccRL-exps/prompt_configs/2026-0605-mha-bwd-p4-mha-patched.json

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-gemm" \
  --definition gemm_n7168_k5120 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0413-1611/gemm_n7168_k5120_94920358-01a8-4c5b-9209-3103fd490e94.py \
  --config /home/ubuntu/AccRL-exps/prompt_configs/gemm-5-r8-p4.json

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-gdn_decode_qk8_v16_d128_k_last" \
  --definition gdn_decode_qk8_v16_d128_k_last \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gdn_decode_qk8_v16_d128_k_last_35b0abd8-5892-44f2-a152-52cba6b59493.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-gdn_prefill_qk8_v16_d128_k_last" \
  --definition gdn_prefill_qk8_v16_d128_k_last \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gdn_prefill_qk8_v16_d128_k_last_ba7d3e67-2d8e-4173-87e1-533c3a323b3f.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-gqa_paged_decode_h24_kv8_d128_ps1" \
  --definition gqa_paged_decode_h24_kv8_d128_ps1 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gqa_paged_decode_h24_kv8_d128_ps1_12e644fb-9783-46be-9763-f69561dfc700.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-gqa_paged_prefill_causal_h24_kv8_d128_ps1" \
  --definition gqa_paged_prefill_causal_h24_kv8_d128_ps1 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gqa_paged_prefill_causal_h24_kv8_d128_ps1_2bcc7cdd-53db-4da4-844f-07ff4f3439b6.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-gqa_ragged_prefill_causal_h32_kv4_d128" \
  --definition gqa_ragged_prefill_causal_h32_kv4_d128 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/gqa_ragged_prefill_causal_h32_kv4_d128_007ddabb-3c8c-48a1-a693-c0618d32243c.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-mla_paged_decode_h16_ckv512_kpe64_ps1" \
  --definition mla_paged_decode_h16_ckv512_kpe64_ps1 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/mla_paged_decode_h16_ckv512_kpe64_ps1_939f995a-1ab2-4d19-8d94-50f07e73542d.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-mla_paged_prefill_causal_h16_ckv512_kpe64_ps1" \
  --definition mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_54187805-1b18-4d39-83ca-46332f85da9e.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-mla_ragged_prefill_causal_h16_qk192_vo128" \
  --definition mla_ragged_prefill_causal_h16_qk192_vo128 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/mla_ragged_prefill_causal_h16_qk192_vo128_1fe95283-ade9-4efa-8df0-8cd15dc8b09e.py \
  --config "$CONFIG"

run_one "$OUTPUT_ROOT_BASE/$RUN_DATE-fused_add_rmsnorm_h7168" \
  --definition fused_add_rmsnorm_h7168 \
  --test-path /home/ubuntu/AccRL/fib_runtime/multiturn/single_op_eval/fused_add_rmsnorm_h7168_d9c27791-1f27-4feb-adb6-aa8a8c20556f.py \
  --config "$CONFIG"
