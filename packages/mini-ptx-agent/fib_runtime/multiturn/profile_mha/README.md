# MHA profile results

`profile_all_mha.sh` reruns all MHA profiling through the heavy-mounted
`fib-profile-heavy` service and writes:

- `/home/ubuntu/accrl-training-heavy/perf.csv`
- `/home/ubuntu/accrl-training-heavy/perf_parts/*.csv`

The copied results from the latest run are under `results/`:

- `results/perf.csv`: combined MHA perf CSV
- `results/perf_parts/mha_with_lse_perf.csv`
- `results/perf_parts/mha_bwd_perf.csv`
- `results/perf_parts/fp8_mha_with_lse_perf.csv`

Coverage for the copied combined CSV is 132 rows: 22 MHA definitions times 6
sequence lengths.
