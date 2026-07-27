# GDN Workload Selection Criteria

This document records the criteria used to choose 6 representative workloads for
each of the 6 GDN problems imported from `flashinfer-ai/flashinfer-trace`.

## Scope

Only GDN problems with workload rows are included:

- `gdn_decode_qk4_v8_d128_k_last`
- `gdn_decode_qk8_v16_d128_k_last`
- `gdn_mtp_qk4_v8_d128_k_last`
- `gdn_mtp_qk8_v16_d128_k_last`
- `gdn_prefill_qk4_v8_d128_k_last`
- `gdn_prefill_qk8_v16_d128_k_last`

The three qk16/v32 tasks are excluded because they do not have workload rows:

- `gdn_decode_qk16_v32_d128_k_last`
- `gdn_prefill_qk16_v32_d128_k_last`
- `gdn_mtp_qk16_v32_d128_k_last`

Solutions are not included for `accrl-training`.

## General Selection Rules

- Select exactly 6 workloads per included problem.
- Keep only rows whose referenced safetensor blobs exist locally.
- Prefer axis coverage over random sampling or first-N rows.
- Use comparable axis points across qk4/v8 and qk8/v16 variants where possible,
  so paired problems differ mainly by head configuration rather than workload
  shape.

## Decode Problems

Decode workloads vary by `batch_size` only. The selected rows cover the available
batch-size range from single-request decode to large concurrent decode:

- `batch_size = 1`
- `batch_size = 4`
- `batch_size = 8`
- `batch_size = 16`
- `batch_size = 32`
- `batch_size = 64`

This applies to:

- `gdn_decode_qk4_v8_d128_k_last`
- `gdn_decode_qk8_v16_d128_k_last`

## MTP Problems

MTP workloads have fixed `seq_len = 4` and `pool_size = 49`; the meaningful
varying axis is `batch_size`. The selected rows cover small through largest
available batch sizes:

- `batch_size = 1`
- `batch_size = 4`
- `batch_size = 8`
- `batch_size = 16`
- `batch_size = 32`
- `batch_size = 48`

This applies to:

- `gdn_mtp_qk4_v8_d128_k_last`
- `gdn_mtp_qk8_v16_d128_k_last`

## Prefill Problems

Prefill workloads vary by total token count and packing structure. The selected
rows cover both sequence length scale and packed sequence count:

- Tiny single sequence: `total_seq_len = 6`, `num_seqs = 1`
- Small packed batch: `total_seq_len = 32`, `num_seqs = 2`
- Medium packed batch: `total_seq_len = 401`, `num_seqs = 4`
- Near-1k packed batch: `total_seq_len = 959`, `num_seqs = 4`
- Long sparse packed batch: `total_seq_len = 3271`, `num_seqs = 2`
- Max-token packed batch: `total_seq_len = 8192`, using the largest available
  `num_seqs` for that problem variant

For the max-token packed batch:

- qk4/v8 uses `total_seq_len = 8192`, `num_seqs = 57`
- qk8/v16 uses `total_seq_len = 8192`, `num_seqs = 51`

This applies to:

- `gdn_prefill_qk4_v8_d128_k_last`
- `gdn_prefill_qk8_v16_d128_k_last`

## Verification

The final selection contains 36 workloads total: 6 workloads for each of 6
problems. Each selected row was checked against its JSONL source file and the
referenced safetensor file path.
