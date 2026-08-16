#!/usr/bin/env bash
# Reproduce the full prepare + verify workflow for mha_with_lse_fp8 and
# mha_with_lse_fp8_causal.
#
# Assumes the `acc` conda env is active (i.e. `/home/ubuntu/miniconda3/envs/acc/bin/python`
# is on PATH). `conda activate acc` first if it is not.
#
# Note: /home/ubuntu/flashinfer-bench/flashinfer_bench/bench/evaluators/{default,lowbit}.py
# must contain the try/except NotImplementedError wrappers around torch.isinf /
# torch.isnan; without them the FP8 output tensor trips the eval harness.

set -euo pipefail

# --- 1. generate definitions, workloads, safetensors blobs into accrl-training
# Wipe prior blobs so stale uuids do not accumulate.
rm -rf /home/ubuntu/accrl-training/blob/workloads/attention/mha_with_lse_fp8 \
       /home/ubuntu/accrl-training/blob/workloads/attention/mha_with_lse_fp8_causal

cd /home/ubuntu/AccRL/fib_runtime/multiturn/fp8-mha-with-lse-problems
python scripts/prepare_fp8_mha.py --overwrite

# --- 2. stage a tmp folder for `flashinfer-bench serve --local`
TMP=/tmp/fp8_mha_serve_$(date +%s)
mkdir -p "$TMP/definitions/attention" "$TMP/workloads/attention" "$TMP/blob/workloads/attention"
cp /home/ubuntu/accrl-training/definitions/attention/mha_with_lse_fp8.json        "$TMP/definitions/attention/"
cp /home/ubuntu/accrl-training/definitions/attention/mha_with_lse_fp8_causal.json "$TMP/definitions/attention/"
cp /home/ubuntu/accrl-training/workloads/attention/mha_with_lse_fp8.jsonl         "$TMP/workloads/attention/"
cp /home/ubuntu/accrl-training/workloads/attention/mha_with_lse_fp8_causal.jsonl  "$TMP/workloads/attention/"
cp -r /home/ubuntu/accrl-training/blob/workloads/attention/mha_with_lse_fp8        "$TMP/blob/workloads/attention/"
cp -r /home/ubuntu/accrl-training/blob/workloads/attention/mha_with_lse_fp8_causal "$TMP/blob/workloads/attention/"
cp /home/ubuntu/AccRL/fib_runtime/multiturn/fp8-mha-with-lse-problems/scripts/acc_config.yaml "$TMP/acc_config.yaml"
echo "TMP=$TMP"

# --- 3. kill any flashinfer-bench server already on :10000 (only PIDs you own),
#        then launch a fresh one.
OLD_PID=$(ss -tlnp 2>/dev/null | grep ':10000' | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
if [ -n "${OLD_PID:-}" ]; then
  kill "$OLD_PID" 2>/dev/null || true
  sleep 2
  kill -9 "$OLD_PID" 2>/dev/null || true
fi

LOG=/tmp/fb_serve.log
: > "$LOG"
cd /home/ubuntu/flashinfer-bench
TVM_FFI_CUDA_ARCH_LIST="9.0a" \
PYTHONPATH=/home/ubuntu/AccRL:${PYTHONPATH:-} \
  nohup flashinfer-bench serve --local "$TMP" --port 10000 --timeout 20 \
        --config "$TMP/acc_config.yaml" > "$LOG" 2>&1 &
sleep 6
tail -15 "$LOG"
ss -tlnp 2>/dev/null | grep ':10000' || { echo "no listener on :10000"; exit 1; }

# --- 4. verify via the service (cuDNN-vs-cuDNN smoke test on all 12 workloads)
cd /home/ubuntu/AccRL/fib_runtime/multiturn/fp8-mha-with-lse-problems
PYTHONPATH=/home/ubuntu/AccRL:${PYTHONPATH:-} \
  python scripts/verify_via_service.py --deadline-s 1200
