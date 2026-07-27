# note-feedback2 retrieval analysis for 8 MHA folders

Run glob: `/home/ubuntu/AccRL-exps/eval_runs/qwen36-27b-note-feedback2-mha-d128-d96-8defs-*`
Rows: 206 retrieval entries from 8 run dirs; each row is one retrieved fixed kernel in one feedback message. 39 turns PASSED. 8 * 8 * 4 = 256 turns in total. The other 11 turns exceed token limit. 
Notes corpus: `/home/ubuntu/AccRL-exps/tasks/collect_notes/outputs/mha-d128-4def-kernel-fix-notes-full/notes.jsonl`

## Overall

- Retrieved definitions: mha_with_lse_d128=60, mha_bwd_d128=57, mha_with_lse_d128_causal=49, mha_bwd_d128_causal=40
- Retrieved fixed-kernel speedups: min=0.00181162x, max=0.617647x, unique definition/speedup pairs=26
- Current-turn behavior when retrieval was attached: COMPILE_ERROR=67, FAILED:INCORRECT_NUMERICALx1=67, FAILED:RUNTIME_ERRORx1=50, FAILED:TIMEOUTx1=22
- Next-turn behavior counts: FAILED:INCORRECT_NUMERICALx1=59, FAILED:RUNTIME_ERRORx1=39, COMPILE_ERROR=36, NO_NEXT_EVAL_OBSERVED=24, FAILED:TIMEOUTx1=17, PASSED speedup 0.549x=2, PASSED speedup 0.173x=2, PASSED speedup 0.027x=2, PASSED speedup 0.139x=1, PASSED speedup 0.136x=1, PASSED speedup 0.546x=1, PASSED speedup 0.170x=1, PASSED speedup 0.089x=1, PASSED speedup 0.098x=1, PASSED speedup 0.323x=1, PASSED speedup 0.328x=1, PASSED speedup 0.336x=1, PASSED speedup 0.417x=1, PASSED speedup 0.503x=1, PASSED speedup 0.499x=1, PASSED speedup 0.026x=1, PASSED speedup 0.197x=1, PASSED speedup 0.213x=1, PASSED speedup 0.100x=1, PASSED speedup 0.080x=1, PASSED speedup 0.102x=1, PASSED speedup 0.278x=1, PASSED speedup 0.167x=1, PASSED speedup 0.164x=1, PASSED speedup 0.163x=1, PASSED speedup 0.058x=1, PASSED speedup 0.477x=1, PASSED speedup 0.012x=1

## Unique retrieved definition/speedup pairs

- `mha_with_lse_d128` @ `0.310270202x`: 26
- `mha_bwd_d128_causal` @ `0.009969548x`: 22
- `mha_bwd_d128` @ `0.609697494x`: 21
- `mha_bwd_d128` @ `0.094655208x`: 15
- `mha_with_lse_d128_causal` @ `0.059750075x`: 14
- `mha_with_lse_d128` @ `0.025524163x`: 13
- `mha_with_lse_d128` @ `0.542423145x`: 12
- `mha_with_lse_d128_causal` @ `0.158081207x`: 8
- `mha_with_lse_d128_causal` @ `0.160630227x`: 8
- `mha_with_lse_d128_causal` @ `0.482830848x`: 7
- `mha_bwd_d128` @ `0.054817905x`: 6
- `mha_bwd_d128_causal` @ `0.159359689x`: 6
- `mha_bwd_d128` @ `0.022062692x`: 6
- `mha_with_lse_d128` @ `0.20273466x`: 5
- `mha_bwd_d128_causal` @ `0.1699274x`: 4
- `mha_bwd_d128` @ `0.138801194x`: 4
- `mha_bwd_d128_causal` @ `0.00181162x`: 4
- `mha_bwd_d128_causal` @ `0.009690682x`: 4
- `mha_with_lse_d128_causal` @ `0.231110159x`: 4
- `mha_with_lse_d128` @ `0.219989369x`: 4
- `mha_with_lse_d128_causal` @ `0.204503484x`: 4
- `mha_bwd_d128` @ `0.546974942x`: 3
- `mha_with_lse_d128_causal` @ `0.160685548x`: 3
- `mha_bwd_d128` @ `0.214417523x`: 1
- `mha_bwd_d128` @ `0.193459544x`: 1
- `mha_with_lse_d128_causal` @ `0.617647294x`: 1

## Per run summary

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128
- task_definition: `mha_bwd_d128`
- retrieval rows: 23; retrieved definitions: mha_bwd_d128_causal=9, mha_with_lse_d128=7, mha_bwd_d128=7
- retrieved definition/speedup pairs: mha_with_lse_d128@0.310270202x=7, mha_bwd_d128_causal@0.009969548x=6, mha_bwd_d128@0.138801194x=4, mha_bwd_d128@0.546974942x=3, mha_bwd_d128_causal@0.1699274x=1, mha_bwd_d128_causal@0.00181162x=1, mha_bwd_d128_causal@0.009690682x=1
- next-turn behavior: FAILED:INCORRECT_NUMERICALx1=12, NO_NEXT_EVAL_OBSERVED=3, FAILED:RUNTIME_ERRORx1=2, PASSED speedup 0.549x=2, COMPILE_ERROR=1, PASSED speedup 0.139x=1, PASSED speedup 0.136x=1, PASSED speedup 0.546x=1

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal
- task_definition: `mha_bwd_d128_causal`
- retrieval rows: 27; retrieved definitions: mha_bwd_d128=14, mha_bwd_d128_causal=7, mha_with_lse_d128=6
- retrieved definition/speedup pairs: mha_bwd_d128@0.094655208x=7, mha_with_lse_d128@0.310270202x=6, mha_bwd_d128@0.054817905x=6, mha_bwd_d128_causal@0.1699274x=3, mha_bwd_d128_causal@0.009690682x=3, mha_bwd_d128_causal@0.009969548x=1, mha_bwd_d128@0.214417523x=1
- next-turn behavior: FAILED:INCORRECT_NUMERICALx1=8, FAILED:TIMEOUTx1=5, COMPILE_ERROR=4, NO_NEXT_EVAL_OBSERVED=3, PASSED speedup 0.173x=2, FAILED:RUNTIME_ERRORx1=2, PASSED speedup 0.170x=1, PASSED speedup 0.089x=1, PASSED speedup 0.098x=1

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96
- task_definition: `mha_bwd_d96`
- retrieval rows: 28; retrieved definitions: mha_bwd_d128_causal=14, mha_bwd_d128=14
- retrieved definition/speedup pairs: mha_bwd_d128@0.609697494x=14, mha_bwd_d128_causal@0.009969548x=7, mha_bwd_d128_causal@0.159359689x=6, mha_bwd_d128_causal@0.00181162x=1
- next-turn behavior: FAILED:RUNTIME_ERRORx1=12, COMPILE_ERROR=7, FAILED:INCORRECT_NUMERICALx1=5, NO_NEXT_EVAL_OBSERVED=4

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal
- task_definition: `mha_bwd_d96_causal`
- retrieval rows: 31; retrieved definitions: mha_bwd_d128=21, mha_bwd_d128_causal=10
- retrieved definition/speedup pairs: mha_bwd_d128@0.094655208x=8, mha_bwd_d128_causal@0.009969548x=8, mha_bwd_d128@0.609697494x=7, mha_bwd_d128@0.022062692x=6, mha_bwd_d128_causal@0.00181162x=2
- next-turn behavior: FAILED:INCORRECT_NUMERICALx1=10, COMPILE_ERROR=7, FAILED:RUNTIME_ERRORx1=7, NO_NEXT_EVAL_OBSERVED=4, FAILED:TIMEOUTx1=3

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128
- task_definition: `mha_with_lse_d128`
- retrieval rows: 18; retrieved definitions: mha_with_lse_d128=13, mha_with_lse_d128_causal=4, mha_bwd_d128=1
- retrieved definition/speedup pairs: mha_with_lse_d128@0.025524163x=5, mha_with_lse_d128@0.310270202x=4, mha_with_lse_d128_causal@0.231110159x=4, mha_with_lse_d128@0.20273466x=3, mha_bwd_d128@0.193459544x=1, mha_with_lse_d128@0.219989369x=1
- next-turn behavior: FAILED:TIMEOUTx1=3, COMPILE_ERROR=2, PASSED speedup 0.027x=2, PASSED speedup 0.323x=1, PASSED speedup 0.328x=1, PASSED speedup 0.336x=1, PASSED speedup 0.417x=1, PASSED speedup 0.503x=1, PASSED speedup 0.499x=1, NO_NEXT_EVAL_OBSERVED=1, PASSED speedup 0.026x=1, PASSED speedup 0.197x=1, FAILED:RUNTIME_ERRORx1=1, PASSED speedup 0.213x=1

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal
- task_definition: `mha_with_lse_d128_causal`
- retrieval rows: 20; retrieved definitions: mha_with_lse_d128=20
- retrieved definition/speedup pairs: mha_with_lse_d128@0.310270202x=9, mha_with_lse_d128@0.542423145x=6, mha_with_lse_d128@0.219989369x=3, mha_with_lse_d128@0.20273466x=2
- next-turn behavior: COMPILE_ERROR=5, FAILED:RUNTIME_ERRORx1=3, FAILED:INCORRECT_NUMERICALx1=2, FAILED:TIMEOUTx1=2, PASSED speedup 0.100x=1, PASSED speedup 0.080x=1, PASSED speedup 0.102x=1, PASSED speedup 0.278x=1, NO_NEXT_EVAL_OBSERVED=1, PASSED speedup 0.167x=1, PASSED speedup 0.164x=1, PASSED speedup 0.163x=1

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96
- task_definition: `mha_with_lse_d96`
- retrieval rows: 30; retrieved definitions: mha_with_lse_d128_causal=24, mha_with_lse_d128=6
- retrieved definition/speedup pairs: mha_with_lse_d128_causal@0.059750075x=8, mha_with_lse_d128_causal@0.160630227x=8, mha_with_lse_d128_causal@0.158081207x=6, mha_with_lse_d128@0.542423145x=6, mha_with_lse_d128_causal@0.482830848x=2
- next-turn behavior: FAILED:INCORRECT_NUMERICALx1=12, FAILED:RUNTIME_ERRORx1=7, COMPILE_ERROR=5, NO_NEXT_EVAL_OBSERVED=4, FAILED:TIMEOUTx1=2

### qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal
- task_definition: `mha_with_lse_d96_causal`
- retrieval rows: 29; retrieved definitions: mha_with_lse_d128_causal=21, mha_with_lse_d128=8
- retrieved definition/speedup pairs: mha_with_lse_d128@0.025524163x=8, mha_with_lse_d128_causal@0.059750075x=6, mha_with_lse_d128_causal@0.482830848x=5, mha_with_lse_d128_causal@0.204503484x=4, mha_with_lse_d128_causal@0.160685548x=3, mha_with_lse_d128_causal@0.158081207x=2, mha_with_lse_d128_causal@0.617647294x=1
- next-turn behavior: FAILED:INCORRECT_NUMERICALx1=10, FAILED:RUNTIME_ERRORx1=5, COMPILE_ERROR=5, NO_NEXT_EVAL_OBSERVED=4, FAILED:TIMEOUTx1=2, PASSED speedup 0.058x=1, PASSED speedup 0.477x=1, PASSED speedup 0.012x=1

## Detailed rows

| run | exp | feedback_after_turn | retrieved definition | speedup | next turn behavior |
|---|---:|---:|---|---:|---|
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 1 | mha_bwd_d128_causal | 0.169927x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 2 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 3 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 4 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 5 | mha_with_lse_d128 | 0.31027x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 6 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 7 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_000 | 8 | mha_with_lse_d128 | 0.31027x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_001 | 1 | mha_bwd_d128 | 0.138801x | PASSED speedup 0.139x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_001 | 4 | mha_bwd_d128 | 0.138801x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_001 | 5 | mha_bwd_d128 | 0.138801x | PASSED speedup 0.136x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_001 | 7 | mha_bwd_d128 | 0.138801x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 1 | mha_bwd_d128_causal | 0.00181162x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 2 | mha_bwd_d128_causal | 0.00996955x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 3 | mha_bwd_d128_causal | 0.00996955x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 4 | mha_bwd_d128_causal | 0.00996955x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 5 | mha_bwd_d128_causal | 0.00996955x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 6 | mha_bwd_d128_causal | 0.00996955x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 7 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_002 | 8 | mha_bwd_d128_causal | 0.00969068x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_003 | 1 | mha_bwd_d128 | 0.546975x | PASSED speedup 0.549x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_003 | 3 | mha_bwd_d128 | 0.546975x | PASSED speedup 0.546x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128 | exp_003 | 7 | mha_bwd_d128 | 0.546975x | PASSED speedup 0.549x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 1 | mha_bwd_d128_causal | 0.00996955x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 2 | mha_bwd_d128 | 0.0946552x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 3 | mha_bwd_d128 | 0.0946552x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 4 | mha_bwd_d128 | 0.0946552x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 5 | mha_bwd_d128 | 0.0946552x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 6 | mha_bwd_d128 | 0.0946552x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 7 | mha_bwd_d128 | 0.0946552x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_000 | 8 | mha_bwd_d128 | 0.0946552x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_001 | 1 | mha_bwd_d128_causal | 0.169927x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_001 | 2 | mha_bwd_d128_causal | 0.169927x | PASSED speedup 0.173x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_001 | 4 | mha_with_lse_d128 | 0.31027x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_001 | 5 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_001 | 6 | mha_with_lse_d128 | 0.31027x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_001 | 7 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.173x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_002 | 1 | mha_bwd_d128_causal | 0.169927x | PASSED speedup 0.170x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_002 | 3 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.089x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_002 | 5 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_002 | 6 | mha_bwd_d128_causal | 0.00969068x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_002 | 7 | mha_bwd_d128_causal | 0.00969068x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_002 | 8 | mha_bwd_d128_causal | 0.00969068x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_003 | 1 | mha_bwd_d128 | 0.214418x | PASSED speedup 0.098x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_003 | 3 | mha_bwd_d128 | 0.0548179x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_003 | 4 | mha_bwd_d128 | 0.0548179x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_003 | 5 | mha_bwd_d128 | 0.0548179x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_003 | 6 | mha_bwd_d128 | 0.0548179x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_003 | 7 | mha_bwd_d128 | 0.0548179x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d128-causal | exp_003 | 8 | mha_bwd_d128 | 0.0548179x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_000 | 1 | mha_bwd_d128_causal | 0.15936x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_000 | 2 | mha_bwd_d128_causal | 0.15936x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_000 | 3 | mha_bwd_d128_causal | 0.15936x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_000 | 4 | mha_bwd_d128_causal | 0.15936x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_000 | 5 | mha_bwd_d128_causal | 0.15936x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_000 | 6 | mha_bwd_d128_causal | 0.15936x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_001 | 1 | mha_bwd_d128 | 0.609697x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_001 | 2 | mha_bwd_d128 | 0.609697x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_001 | 3 | mha_bwd_d128 | 0.609697x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_001 | 4 | mha_bwd_d128 | 0.609697x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_001 | 5 | mha_bwd_d128 | 0.609697x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_001 | 6 | mha_bwd_d128 | 0.609697x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_001 | 7 | mha_bwd_d128 | 0.609697x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_002 | 1 | mha_bwd_d128 | 0.609697x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_002 | 2 | mha_bwd_d128 | 0.609697x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_002 | 3 | mha_bwd_d128 | 0.609697x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_002 | 4 | mha_bwd_d128 | 0.609697x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_002 | 5 | mha_bwd_d128 | 0.609697x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_002 | 6 | mha_bwd_d128 | 0.609697x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_002 | 7 | mha_bwd_d128 | 0.609697x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 1 | mha_bwd_d128_causal | 0.00181162x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 2 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 3 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 4 | mha_bwd_d128_causal | 0.00996955x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 5 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 6 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 7 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96 | exp_003 | 8 | mha_bwd_d128_causal | 0.00996955x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 1 | mha_bwd_d128 | 0.0946552x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 2 | mha_bwd_d128 | 0.0946552x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 3 | mha_bwd_d128 | 0.0946552x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 4 | mha_bwd_d128 | 0.0946552x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 5 | mha_bwd_d128 | 0.0946552x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 6 | mha_bwd_d128 | 0.0946552x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 7 | mha_bwd_d128 | 0.0946552x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_000 | 8 | mha_bwd_d128 | 0.0946552x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 1 | mha_bwd_d128_causal | 0.00181162x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 2 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 3 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 4 | mha_bwd_d128_causal | 0.00996955x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 5 | mha_bwd_d128_causal | 0.00996955x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 6 | mha_bwd_d128_causal | 0.00996955x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 7 | mha_bwd_d128_causal | 0.00996955x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_001 | 8 | mha_bwd_d128_causal | 0.00996955x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 1 | mha_bwd_d128_causal | 0.00181162x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 2 | mha_bwd_d128_causal | 0.00996955x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 3 | mha_bwd_d128 | 0.0220627x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 4 | mha_bwd_d128 | 0.0220627x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 5 | mha_bwd_d128 | 0.0220627x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 6 | mha_bwd_d128 | 0.0220627x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 7 | mha_bwd_d128 | 0.0220627x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_002 | 8 | mha_bwd_d128 | 0.0220627x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_003 | 1 | mha_bwd_d128 | 0.609697x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_003 | 2 | mha_bwd_d128 | 0.609697x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_003 | 3 | mha_bwd_d128 | 0.609697x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_003 | 4 | mha_bwd_d128 | 0.609697x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_003 | 5 | mha_bwd_d128 | 0.609697x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_003 | 6 | mha_bwd_d128 | 0.609697x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-bwd-d96-causal | exp_003 | 7 | mha_bwd_d128 | 0.609697x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_000 | 1 | mha_bwd_d128 | 0.19346x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_000 | 2 | mha_with_lse_d128 | 0.31027x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_000 | 3 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.323x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_000 | 5 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.328x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_000 | 7 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.336x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_001 | 1 | mha_with_lse_d128_causal | 0.23111x | PASSED speedup 0.417x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_001 | 3 | mha_with_lse_d128_causal | 0.23111x | PASSED speedup 0.503x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_001 | 5 | mha_with_lse_d128_causal | 0.23111x | PASSED speedup 0.499x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_001 | 8 | mha_with_lse_d128_causal | 0.23111x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_002 | 1 | mha_with_lse_d128 | 0.0255242x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_002 | 2 | mha_with_lse_d128 | 0.0255242x | PASSED speedup 0.027x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_002 | 4 | mha_with_lse_d128 | 0.0255242x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_002 | 5 | mha_with_lse_d128 | 0.0255242x | PASSED speedup 0.027x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_002 | 7 | mha_with_lse_d128 | 0.0255242x | PASSED speedup 0.026x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_003 | 1 | mha_with_lse_d128 | 0.219989x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_003 | 2 | mha_with_lse_d128 | 0.202735x | PASSED speedup 0.197x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_003 | 4 | mha_with_lse_d128 | 0.202735x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128 | exp_003 | 5 | mha_with_lse_d128 | 0.202735x | PASSED speedup 0.213x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_000 | 1 | mha_with_lse_d128 | 0.219989x | PASSED speedup 0.100x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_000 | 3 | mha_with_lse_d128 | 0.202735x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_000 | 4 | mha_with_lse_d128 | 0.202735x | PASSED speedup 0.080x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_000 | 6 | mha_with_lse_d128 | 0.219989x | PASSED speedup 0.102x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_001 | 1 | mha_with_lse_d128 | 0.542423x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_001 | 2 | mha_with_lse_d128 | 0.542423x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_001 | 3 | mha_with_lse_d128 | 0.542423x | PASSED speedup 0.278x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_001 | 5 | mha_with_lse_d128 | 0.542423x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_001 | 6 | mha_with_lse_d128 | 0.219989x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_001 | 7 | mha_with_lse_d128 | 0.542423x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_001 | 8 | mha_with_lse_d128 | 0.542423x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_002 | 1 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.167x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_002 | 7 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.164x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_003 | 1 | mha_with_lse_d128 | 0.31027x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_003 | 2 | mha_with_lse_d128 | 0.31027x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_003 | 3 | mha_with_lse_d128 | 0.31027x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_003 | 4 | mha_with_lse_d128 | 0.31027x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_003 | 5 | mha_with_lse_d128 | 0.31027x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_003 | 6 | mha_with_lse_d128 | 0.31027x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d128-causal | exp_003 | 7 | mha_with_lse_d128 | 0.31027x | PASSED speedup 0.163x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 1 | mha_with_lse_d128_causal | 0.482831x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 2 | mha_with_lse_d128_causal | 0.482831x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 3 | mha_with_lse_d128_causal | 0.158081x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 4 | mha_with_lse_d128_causal | 0.158081x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 5 | mha_with_lse_d128_causal | 0.158081x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 6 | mha_with_lse_d128_causal | 0.158081x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 7 | mha_with_lse_d128_causal | 0.158081x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_000 | 8 | mha_with_lse_d128_causal | 0.158081x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_001 | 1 | mha_with_lse_d128 | 0.542423x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_001 | 2 | mha_with_lse_d128 | 0.542423x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_001 | 3 | mha_with_lse_d128 | 0.542423x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_001 | 4 | mha_with_lse_d128 | 0.542423x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_001 | 5 | mha_with_lse_d128 | 0.542423x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_001 | 6 | mha_with_lse_d128 | 0.542423x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 1 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 2 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 3 | mha_with_lse_d128_causal | 0.0597501x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 4 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 5 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 6 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 7 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_002 | 8 | mha_with_lse_d128_causal | 0.0597501x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 1 | mha_with_lse_d128_causal | 0.16063x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 2 | mha_with_lse_d128_causal | 0.16063x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 3 | mha_with_lse_d128_causal | 0.16063x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 4 | mha_with_lse_d128_causal | 0.16063x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 5 | mha_with_lse_d128_causal | 0.16063x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 6 | mha_with_lse_d128_causal | 0.16063x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 7 | mha_with_lse_d128_causal | 0.16063x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96 | exp_003 | 8 | mha_with_lse_d128_causal | 0.16063x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_000 | 1 | mha_with_lse_d128_causal | 0.617647x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_000 | 2 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_000 | 3 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_000 | 4 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_000 | 5 | mha_with_lse_d128_causal | 0.0597501x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_000 | 6 | mha_with_lse_d128_causal | 0.0597501x | PASSED speedup 0.058x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_000 | 8 | mha_with_lse_d128_causal | 0.0597501x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 1 | mha_with_lse_d128 | 0.0255242x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 2 | mha_with_lse_d128 | 0.0255242x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 3 | mha_with_lse_d128 | 0.0255242x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 4 | mha_with_lse_d128 | 0.0255242x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 5 | mha_with_lse_d128 | 0.0255242x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 6 | mha_with_lse_d128 | 0.0255242x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 7 | mha_with_lse_d128 | 0.0255242x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_001 | 8 | mha_with_lse_d128 | 0.0255242x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_002 | 1 | mha_with_lse_d128_causal | 0.482831x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_002 | 2 | mha_with_lse_d128_causal | 0.482831x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_002 | 3 | mha_with_lse_d128_causal | 0.482831x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_002 | 4 | mha_with_lse_d128_causal | 0.482831x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_002 | 5 | mha_with_lse_d128_causal | 0.482831x | PASSED speedup 0.477x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_002 | 7 | mha_with_lse_d128_causal | 0.158081x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_002 | 8 | mha_with_lse_d128_causal | 0.158081x | NO_NEXT_EVAL_OBSERVED |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_003 | 1 | mha_with_lse_d128_causal | 0.204503x | COMPILE_ERROR |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_003 | 2 | mha_with_lse_d128_causal | 0.204503x | PASSED speedup 0.012x |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_003 | 4 | mha_with_lse_d128_causal | 0.160686x | FAILED:INCORRECT_NUMERICALx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_003 | 5 | mha_with_lse_d128_causal | 0.160686x | FAILED:TIMEOUTx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_003 | 6 | mha_with_lse_d128_causal | 0.160686x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_003 | 7 | mha_with_lse_d128_causal | 0.204503x | FAILED:RUNTIME_ERRORx1 |
| qwen36-27b-note-feedback2-mha-d128-d96-8defs-mha-d96-causal | exp_003 | 8 | mha_with_lse_d128_causal | 0.204503x | NO_NEXT_EVAL_OBSERVED |

Full CSV rows: `/home/ubuntu/AccRL/fib_runtime/multiturn/collect_notes/note_feedback2_8runs_retrieval_rows.csv`
