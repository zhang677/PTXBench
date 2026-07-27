# Definition Taxonomy Tree

Generated from:

- `/home/ubuntu/AccRL/fib_runtime/multiturn/profile_mha/results/perf.csv`
- `/home/ubuntu/AccRL-exps/tasks/clean_data/artifacts/hopper_profile_one_large_per_definition.csv`
- `/home/ubuntu/AccRL-exps/tasks/clean_data/artifacts/hopper_profile_one_per_definition.csv`

Current source shape:

- `perf.csv`: 132 rows, 22 unique `definition_name` values, 3 suites. Each definition is swept over `seq_len` values `512, 1024, 2048, 4096, 8192, 16384`.
- `hopper_profile_one_large_per_definition.csv`: 32 rows, 32 unique `definition` values, one selected workload per definition.
- `hopper_profile_one_per_definition.csv`: 60 rows, 60 unique `definition` values, one selected workload per definition; only its 6 `gdn` definitions are added here.

Legend:

- `dN`: attention head dimension `N`.
- `hN`: query heads `N`.
- `kvN`: key/value heads `N`.
- `psN`: paged-KV page size `N`.
- `ckvN`: compressed KV dimension `N`.
- `kpeN`: key positional-encoding dimension `N`.
- `qkN`: query/key dimension `N`.
- `vN`: value dimension `N`.
- `voN`: value/output dimension `N`.
- `k_last`: K dimension is stored as the last layout axis.
- `causal`: causal mask enabled.
- `with_lse`: forward attention returns or uses log-sum-exp metadata.
- `ragged`: variable-length contiguous layout.
- `paged`: paged KV-cache layout.

## Tree

```text
attention
|-- dense MHA profile benchmarks
|   |-- forward
|   |   |-- with_lse
|   |   |   |-- non-FP8
|   |   |   |   |-- non-causal
|   |   |   |   |   |-- mha_with_lse_d64
|   |   |   |   |   |-- mha_with_lse_d96
|   |   |   |   |   |-- mha_with_lse_d128
|   |   |   |   |   `-- mha_with_lse_d256
|   |   |   |   `-- causal
|   |   |   |       |-- mha_with_lse_d64_causal
|   |   |   |       |-- mha_with_lse_d96_causal
|   |   |   |       |-- mha_with_lse_d128_causal
|   |   |   |       `-- mha_with_lse_d256_causal
|   |   |   `-- FP8
|   |   |       |-- non-causal
|   |   |       |   |-- fp8_mha_with_lse_d64
|   |   |       |   |-- fp8_mha_with_lse_d96
|   |   |       |   |-- fp8_mha_with_lse_d128
|   |   |       |   `-- fp8_mha_with_lse_d256
|   |   |       `-- causal
|   |   |           |-- fp8_mha_with_lse_d64_causal
|   |   |           |-- fp8_mha_with_lse_d96_causal
|   |   |           |-- fp8_mha_with_lse_d128_causal
|   |   |           `-- fp8_mha_with_lse_d256_causal
|   `-- backward
|       `-- MHA backward
|           |-- non-causal
|           |   |-- mha_bwd_d64
|           |   |-- mha_bwd_d96
|           |   `-- mha_bwd_d128
|           `-- causal
|               |-- mha_bwd_d64_causal
|               |-- mha_bwd_d96_causal
|               `-- mha_bwd_d128_causal
`-- FlashInfer Hopper selected workloads
    |-- attention_decode
    |   |-- gqa_paged
    |   |   |-- ps1
    |   |   |   |-- gqa_paged_decode_h5_kv1_d128_ps1
    |   |   |   |-- gqa_paged_decode_h20_kv4_d128_ps1
    |   |   |   |-- gqa_paged_decode_h24_kv8_d128_ps1
    |   |   |   |-- gqa_paged_decode_h32_kv4_d128_ps1
    |   |   |   |-- gqa_paged_decode_h32_kv8_d128_ps1
    |   |   |   `-- gqa_paged_decode_h48_kv8_d128_ps1
    |   |   `-- ps64
    |   |       |-- gqa_paged_decode_h20_kv4_d128_ps64
    |   |       |-- gqa_paged_decode_h24_kv4_d128_ps64
    |   |       |-- gqa_paged_decode_h32_kv4_d128_ps64
    |   |       `-- gqa_paged_decode_h32_kv8_d128_ps64
    |   `-- mla_paged
    |       `-- ps1
    |           `-- mla_paged_decode_h16_ckv512_kpe64_ps1
    |-- attention_prefill
    |   |-- gqa_paged
    |   |   `-- causal
    |   |       |-- ps1
    |   |       |   |-- gqa_paged_prefill_causal_h5_kv1_d128_ps1
    |   |       |   |-- gqa_paged_prefill_causal_h20_kv4_d128_ps1
    |   |       |   |-- gqa_paged_prefill_causal_h24_kv4_d128_ps1
    |   |       |   |-- gqa_paged_prefill_causal_h24_kv8_d128_ps1
    |   |       |   |-- gqa_paged_prefill_causal_h32_kv4_d128_ps1
    |   |       |   |-- gqa_paged_prefill_causal_h32_kv8_d128_ps1
    |   |       |   `-- gqa_paged_prefill_causal_h40_kv10_d128_ps1
    |   |       `-- ps64
    |   |           |-- gqa_paged_prefill_causal_h16_kv1_d128_ps64
    |   |           |-- gqa_paged_prefill_causal_h16_kv2_d128_ps64
    |   |           |-- gqa_paged_prefill_causal_h20_kv4_d128_ps64
    |   |           |-- gqa_paged_prefill_causal_h24_kv4_d128_ps64
    |   |           |-- gqa_paged_prefill_causal_h24_kv8_d128_ps64
    |   |           `-- gqa_paged_prefill_causal_h32_kv4_d128_ps64
    |   |-- gqa_ragged
    |   |   `-- causal
    |   |       |-- d128
    |   |       |   |-- gqa_ragged_prefill_causal_h20_kv4_d128
    |   |       |   |-- gqa_ragged_prefill_causal_h24_kv8_d128
    |   |       |   |-- gqa_ragged_prefill_causal_h32_kv4_d128
    |   |       |   |-- gqa_ragged_prefill_causal_h32_kv8_d128
    |   |       |   `-- gqa_ragged_prefill_causal_h32_kv16_d128
    |   |       `-- d256
    |   |           `-- gqa_ragged_prefill_causal_h8_kv1_d256
    |   |-- mla_paged
    |   |   `-- causal
    |   |       `-- ps1
    |   |           `-- mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
    |   `-- mla_ragged
    |       `-- causal
    |           `-- mla_ragged_prefill_causal_h16_qk192_vo128
    `-- gdn
        |-- gdn_decode
        |   |-- qk4_v8_d128_k_last
        |   |   `-- gdn_decode_qk4_v8_d128_k_last
        |   `-- qk8_v16_d128_k_last
        |       `-- gdn_decode_qk8_v16_d128_k_last
        |-- gdn_mtp
        |   |-- qk4_v8_d128_k_last
        |   |   `-- gdn_mtp_qk4_v8_d128_k_last
        |   `-- qk8_v16_d128_k_last
        |       `-- gdn_mtp_qk8_v16_d128_k_last
        `-- gdn_prefill
            |-- qk4_v8_d128_k_last
            |   `-- gdn_prefill_qk4_v8_d128_k_last
            `-- qk8_v16_d128_k_last
                `-- gdn_prefill_qk8_v16_d128_k_last
```

## Classification Axes

The tree uses these primary axes, in order:

1. Workload family: dense synthetic MHA profile benchmark vs. FlashInfer Hopper selected workload.
2. Phase: forward, backward, decode, or prefill.
3. Attention/kernel family: MHA, GQA, MLA, or GDN.
4. Memory layout: dense, paged KV-cache, or ragged variable-length.
5. Feature flags: `with_lse`, FP8, and causal masking.
6. Shape parameters encoded in definition names: query heads, KV heads, head dimension, page size, and MLA/GDN-specific latent dimensions.

## Source Counts

```text
perf.csv
|-- mha_with_lse: 8 definitions, 48 rows
|-- fp8_mha_with_lse: 8 definitions, 48 rows
`-- mha_bwd: 6 definitions, 36 rows

hopper_profile_one_large_per_definition.csv
|-- attention_decode: 11 definitions
|   |-- gqa_paged: 10
|   `-- mla_paged: 1
`-- attention_prefill: 21 definitions
    |-- gqa_paged: 13
    |-- gqa_ragged: 6
    |-- mla_paged: 1
    `-- mla_ragged: 1

hopper_profile_one_per_definition.csv
`-- gdn: 6 definitions
    |-- gdn_decode: 2
    |-- gdn_mtp: 2
    `-- gdn_prefill: 2
```
