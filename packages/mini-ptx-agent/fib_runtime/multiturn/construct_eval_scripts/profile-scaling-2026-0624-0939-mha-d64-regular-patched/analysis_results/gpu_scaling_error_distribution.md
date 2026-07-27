# GPU-scaling turn error distribution

Source: exporter-produced `figures/turn_correctness_arch.csv` files routed by `analysis_results/gpu_scaling_export_experiments.csv`.

| correctness | g=2 | g=4 | g=6 | g=8 |
| --- | ---: | ---: | ---: | ---: |
| Correct | 59 (9.22%) | 44 (6.89%) | 48 (7.51%) | 75 (11.72%) |
| Compilation error | 174 (27.19%) | 183 (28.64%) | 163 (25.51%) | 160 (25.00%) |
| Extraction error | 35 (5.47%) | 32 (5.01%) | 26 (4.07%) | 29 (4.53%) |
| Kernel Execution Timeout | 67 (10.47%) | 61 (9.55%) | 79 (12.36%) | 73 (11.41%) |
| Numerical error | 199 (31.09%) | 180 (28.17%) | 183 (28.64%) | 187 (29.22%) |
| Other error | 0 (0.00%) | 4 (0.63%) | 4 (0.63%) | 2 (0.31%) |
| Runtime error | 106 (16.56%) | 135 (21.13%) | 136 (21.28%) | 114 (17.81%) |
| **Total exported turns** | 640 | 639 | 639 | 640 |

Coverage: `g=2` has 4 runs and 640 rows, `g=4` has 4 runs and 639 rows, `g=6` has 4 runs and 639 rows, `g=8` has 4 runs and 640 rows.
