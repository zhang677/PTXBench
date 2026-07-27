# Failure Recovery Analysis

- Selected runs CSV: `/home/ubuntu/AccRL-exps/tasks/selected-runs-2026-0610-1600.csv`
- Failure labels: `Runtime error`, `Numerical error`, `Kernel Execution Timeout`, `Compilation error`
- Selected rows: 8
- Unique exp dirs: 8
- Total turn rows: 856
- Total trajectories: 107
- Adjacent failure-to-`Correct` pairs: 128
- Adjacent pairs with no earlier `Correct` in the trajectory: 77
- Detail CSV: `/home/ubuntu/AccRL/fib_runtime/multiturn/collect_kernels/analysis/failure-recovery-runtime-numerical-timeout-compilation.csv`

## Selected Metadata

- model: gemini-3.1-pro-preview
- arch: hopper
- definition: mha_bwd_d128, mha_bwd_d128_causal, mha_bwd_h48_d128, mha_with_lse_d128, mha_with_lse_d128_causal, mha_with_lse_h48_d128

## Counts By Run

| exp_dir | selected rows | turn rows | trajectories | adjacent pairs | no-prior-Correct pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | 1 | 128 | 16 | 20 | 13 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | 1 | 128 | 16 | 17 | 10 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | 1 | 40 | 5 | 6 | 2 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | 1 | 40 | 5 | 6 | 4 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0340` | 1 | 40 | 5 | 4 | 1 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | 1 | 160 | 20 | 36 | 18 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | 1 | 160 | 20 | 21 | 15 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | 1 | 160 | 20 | 18 | 14 |

## Counts By Failure Label

| failure label | adjacent pairs | no-prior-Correct pairs |
| --- | ---: | ---: |
| `Runtime error` | 42 | 25 |
| `Numerical error` | 50 | 34 |
| `Kernel Execution Timeout` | 22 | 9 |
| `Compilation error` | 14 | 9 |

## First Failure Turn Distribution

| failure turn | adjacent pairs | no-prior-Correct pairs |
| ---: | ---: | ---: |
| 0 | 29 | 29 |
| 1 | 20 | 18 |
| 2 | 19 | 11 |
| 3 | 18 | 10 |
| 4 | 12 | 3 |
| 5 | 13 | 2 |
| 6 | 17 | 4 |

## No-Prior-Correct Pairs

| exp_dir | trajectory | failure label | failure turn | correct turn | correct speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_000` | `Runtime error` | 1 | 2 | 0.5198350214752019 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_001` | `Numerical error` | 1 | 2 | 0.4818100776048782 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_002` | `Runtime error` | 4 | 5 | 0.36969853196149177 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_003` | `Numerical error` | 1 | 2 | 0.3161954453687572 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_004` | `Runtime error` | 0 | 1 | 0.2752440993123349 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_006` | `Runtime error` | 1 | 2 | 0.7128510609847244 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_007` | `Runtime error` | 0 | 1 | 0.4911737612212462 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_008` | `Runtime error` | 1 | 2 | 0.4459376650566165 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_009` | `Numerical error` | 1 | 2 | 0.472612779961506 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_010` | `Numerical error` | 2 | 3 | 0.3522406297065836 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_011` | `Numerical error` | 3 | 4 | 0.5623787665053528 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_013` | `Runtime error` | 6 | 7 | 0.46919387040436233 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_014` | `Compilation error` | 0 | 1 | 0.179857665936391 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_000` | `Runtime error` | 2 | 3 | 0.24383642400215663 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_001` | `Runtime error` | 0 | 1 | 0.22802195422084973 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_002` | `Compilation error` | 0 | 1 | 0.12783512345590503 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_005` | `Numerical error` | 6 | 7 | 0.1954106004689403 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_007` | `Numerical error` | 2 | 3 | 0.224768099012604 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_009` | `Numerical error` | 1 | 2 | 0.24285018073218664 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_010` | `Compilation error` | 0 | 1 | 0.40417260304615554 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_013` | `Numerical error` | 3 | 4 | 0.40117757031488244 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_014` | `Numerical error` | 6 | 7 | 0.14778263256056115 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_015` | `Numerical error` | 1 | 2 | 0.04110037584835741 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_001` | `Runtime error` | 2 | 3 | 0.4423237178884754 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_003` | `Compilation error` | 0 | 1 | 0.5991844270509635 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_000` | `Runtime error` | 0 | 1 | 0.2924491670585398 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_002` | `Numerical error` | 3 | 4 | 0.11182363444431237 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_003` | `Compilation error` | 1 | 2 | 0.2646580263076392 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_004` | `Numerical error` | 6 | 7 | 0.23490529973439989 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0340` | `exp_002` | `Kernel Execution Timeout` | 1 | 2 | 0.17918539993107435 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_000` | `Runtime error` | 0 | 1 | 0.3238746313409191 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_001` | `Numerical error` | 0 | 1 | 0.47744263067035164 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_002` | `Numerical error` | 3 | 4 | 0.3049046080662063 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_004` | `Runtime error` | 1 | 2 | 0.0539774525327902 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_005` | `Numerical error` | 3 | 4 | 0.48859032468390995 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_006` | `Compilation error` | 0 | 1 | 0.29910217533146033 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_007` | `Runtime error` | 0 | 1 | 0.4720338248142955 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_008` | `Runtime error` | 0 | 1 | 0.23850464501769605 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_010` | `Runtime error` | 0 | 1 | 0.42257800745528257 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_011` | `Runtime error` | 0 | 1 | 0.20601453155903418 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_012` | `Runtime error` | 0 | 1 | 0.2732024702199943 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_013` | `Kernel Execution Timeout` | 0 | 1 | 0.1975372385750982 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_014` | `Runtime error` | 0 | 1 | 0.34712527034942836 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_015` | `Numerical error` | 3 | 4 | 0.22261827224269526 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_016` | `Runtime error` | 0 | 1 | 0.5475752806793048 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_017` | `Kernel Execution Timeout` | 1 | 2 | 0.17992535106843074 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_018` | `Kernel Execution Timeout` | 1 | 2 | 0.3580527025555938 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_019` | `Compilation error` | 0 | 1 | 0.33091298288270216 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_000` | `Numerical error` | 4 | 5 | 0.17516111494376502 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_001` | `Kernel Execution Timeout` | 2 | 3 | 0.21204019050789313 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_006` | `Kernel Execution Timeout` | 5 | 6 | 0.1706351977310609 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_007` | `Numerical error` | 1 | 2 | 0.46530365976686777 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_008` | `Numerical error` | 4 | 5 | 0.15223268562389589 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_009` | `Numerical error` | 3 | 4 | 0.1673738724273677 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_010` | `Numerical error` | 2 | 3 | 0.326325142026823 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_011` | `Numerical error` | 1 | 2 | 0.1701598065218904 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_012` | `Numerical error` | 0 | 1 | 0.22551033705124734 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_013` | `Numerical error` | 3 | 4 | 0.45073425963681274 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_014` | `Numerical error` | 3 | 4 | 0.21181668889246735 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_015` | `Runtime error` | 0 | 1 | 0.5090559184104752 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_016` | `Compilation error` | 0 | 1 | 0.1729399702293596 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_018` | `Numerical error` | 1 | 2 | 0.3439911568555166 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_019` | `Runtime error` | 2 | 3 | 0.23567391274094746 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_001` | `Numerical error` | 2 | 3 | 0.24959079392897057 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_002` | `Numerical error` | 5 | 6 | 0.24079781688252516 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_005` | `Runtime error` | 2 | 3 | 0.2818966781365094 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_006` | `Runtime error` | 0 | 1 | 0.4545621917972395 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_008` | `Compilation error` | 0 | 1 | 0.03517023377249276 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_009` | `Numerical error` | 0 | 1 | 0.059835914330318554 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_011` | `Numerical error` | 0 | 1 | 0.10888972703554908 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_012` | `Numerical error` | 1 | 2 | 0.3280515662048358 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_013` | `Kernel Execution Timeout` | 1 | 2 | 0.3186400427943165 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_014` | `Numerical error` | 0 | 1 | 0.25876472448179877 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_015` | `Kernel Execution Timeout` | 3 | 4 | 0.29902404783666464 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_017` | `Numerical error` | 2 | 3 | 0.39444270070705995 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_018` | `Runtime error` | 2 | 3 | 0.33280167895554796 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_019` | `Kernel Execution Timeout` | 0 | 1 | 0.38464315782227915 |

## All Adjacent Pairs

| exp_dir | trajectory | failure label | failure turn | correct turn | no prior Correct | correct speedup |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_000` | `Runtime error` | 1 | 2 | true | 0.5198350214752019 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_000` | `Numerical error` | 5 | 6 | false | 0.6379676264358038 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_001` | `Numerical error` | 1 | 2 | true | 0.4818100776048782 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_001` | `Numerical error` | 3 | 4 | false | 0.5836784937212944 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_002` | `Runtime error` | 4 | 5 | true | 0.36969853196149177 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_003` | `Numerical error` | 1 | 2 | true | 0.3161954453687572 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_004` | `Runtime error` | 0 | 1 | true | 0.2752440993123349 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_004` | `Runtime error` | 2 | 3 | false | 0.4291134093657865 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_004` | `Runtime error` | 4 | 5 | false | 0.41349299208034723 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_006` | `Runtime error` | 1 | 2 | true | 0.7128510609847244 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_007` | `Runtime error` | 0 | 1 | true | 0.4911737612212462 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_007` | `Numerical error` | 4 | 5 | false | 0.5705094707079865 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_008` | `Runtime error` | 1 | 2 | true | 0.4459376650566165 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_009` | `Numerical error` | 1 | 2 | true | 0.472612779961506 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_010` | `Numerical error` | 2 | 3 | true | 0.3522406297065836 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_010` | `Numerical error` | 6 | 7 | false | 0.35701569780930803 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_011` | `Numerical error` | 3 | 4 | true | 0.5623787665053528 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_013` | `Runtime error` | 6 | 7 | true | 0.46919387040436233 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_014` | `Compilation error` | 0 | 1 | true | 0.179857665936391 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0503-2313` | `exp_014` | `Runtime error` | 2 | 3 | false | 0.19996530283576758 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_000` | `Runtime error` | 2 | 3 | true | 0.24383642400215663 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_000` | `Numerical error` | 5 | 6 | false | 0.11254292685030505 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_001` | `Runtime error` | 0 | 1 | true | 0.22802195422084973 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_002` | `Compilation error` | 0 | 1 | true | 0.12783512345590503 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_002` | `Numerical error` | 3 | 4 | false | 0.2566353976248708 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_002` | `Numerical error` | 6 | 7 | false | 0.24828161675468294 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_005` | `Numerical error` | 6 | 7 | true | 0.1954106004689403 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_006` | `Runtime error` | 6 | 7 | false | 0.14362497844706604 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_007` | `Numerical error` | 2 | 3 | true | 0.224768099012604 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_007` | `Runtime error` | 6 | 7 | false | 0.2898686184346728 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_009` | `Numerical error` | 1 | 2 | true | 0.24285018073218664 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_010` | `Compilation error` | 0 | 1 | true | 0.40417260304615554 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_013` | `Numerical error` | 3 | 4 | true | 0.40117757031488244 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_013` | `Numerical error` | 6 | 7 | false | 0.24502797570885892 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_014` | `Numerical error` | 6 | 7 | true | 0.14778263256056115 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_015` | `Numerical error` | 1 | 2 | true | 0.04110037584835741 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0506-0930` | `exp_015` | `Runtime error` | 4 | 5 | false | 0.5194512081644476 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_001` | `Runtime error` | 2 | 3 | true | 0.4423237178884754 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_001` | `Runtime error` | 5 | 6 | false | 0.5488001557038642 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_003` | `Compilation error` | 0 | 1 | true | 0.5991844270509635 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_003` | `Kernel Execution Timeout` | 3 | 4 | false | 0.5104567666601753 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_003` | `Compilation error` | 5 | 6 | false | 0.4548798265719838 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0529-2340` | `exp_004` | `Runtime error` | 3 | 4 | false | 0.5189864009188245 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_000` | `Runtime error` | 0 | 1 | true | 0.2924491670585398 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_002` | `Numerical error` | 3 | 4 | true | 0.11182363444431237 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_002` | `Numerical error` | 5 | 6 | false | 0.047670764193837076 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_003` | `Compilation error` | 1 | 2 | true | 0.2646580263076392 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_003` | `Numerical error` | 6 | 7 | false | 0.2963224913016865 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0140` | `exp_004` | `Numerical error` | 6 | 7 | true | 0.23490529973439989 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0340` | `exp_000` | `Numerical error` | 1 | 2 | false | 0.3514277773203047 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0340` | `exp_002` | `Kernel Execution Timeout` | 1 | 2 | true | 0.17918539993107435 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0340` | `exp_002` | `Runtime error` | 5 | 6 | false | 0.1723062781268753 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0530-0340` | `exp_003` | `Runtime error` | 3 | 4 | false | 0.3290139406884958 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_000` | `Runtime error` | 0 | 1 | true | 0.3238746313409191 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_001` | `Numerical error` | 0 | 1 | true | 0.47744263067035164 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_002` | `Numerical error` | 3 | 4 | true | 0.3049046080662063 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_003` | `Compilation error` | 1 | 2 | false | 0.10216124437597392 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_003` | `Compilation error` | 3 | 4 | false | 0.5630823770447169 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_003` | `Kernel Execution Timeout` | 6 | 7 | false | 0.5755552392750148 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_004` | `Runtime error` | 1 | 2 | true | 0.0539774525327902 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_005` | `Numerical error` | 3 | 4 | true | 0.48859032468390995 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_005` | `Kernel Execution Timeout` | 5 | 6 | false | 0.38741016927607774 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_006` | `Compilation error` | 0 | 1 | true | 0.29910217533146033 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_007` | `Runtime error` | 0 | 1 | true | 0.4720338248142955 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_007` | `Runtime error` | 2 | 3 | false | 0.5257891824381384 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_008` | `Runtime error` | 0 | 1 | true | 0.23850464501769605 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_008` | `Runtime error` | 2 | 3 | false | 0.2062116418669793 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_008` | `Runtime error` | 4 | 5 | false | 0.3267678241494033 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_009` | `Kernel Execution Timeout` | 5 | 6 | false | 0.27756019320073827 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_010` | `Runtime error` | 0 | 1 | true | 0.42257800745528257 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_011` | `Runtime error` | 0 | 1 | true | 0.20601453155903418 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_011` | `Kernel Execution Timeout` | 4 | 5 | false | 0.27184524119694436 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_011` | `Kernel Execution Timeout` | 6 | 7 | false | 0.3606121181289925 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_012` | `Runtime error` | 0 | 1 | true | 0.2732024702199943 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_012` | `Runtime error` | 2 | 3 | false | 0.31587132055248507 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_013` | `Kernel Execution Timeout` | 0 | 1 | true | 0.1975372385750982 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_013` | `Numerical error` | 2 | 3 | false | 0.24546011418742825 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_013` | `Kernel Execution Timeout` | 4 | 5 | false | 0.2066774219564048 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_013` | `Kernel Execution Timeout` | 6 | 7 | false | 0.2895064160033712 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_014` | `Runtime error` | 0 | 1 | true | 0.34712527034942836 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_014` | `Numerical error` | 3 | 4 | false | 0.4442849659324333 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_015` | `Numerical error` | 3 | 4 | true | 0.22261827224269526 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_016` | `Runtime error` | 0 | 1 | true | 0.5475752806793048 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_016` | `Numerical error` | 3 | 4 | false | 0.5532895686811813 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_017` | `Kernel Execution Timeout` | 1 | 2 | true | 0.17992535106843074 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_017` | `Runtime error` | 6 | 7 | false | 0.456186507215153 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_018` | `Kernel Execution Timeout` | 1 | 2 | true | 0.3580527025555938 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_019` | `Compilation error` | 0 | 1 | true | 0.33091298288270216 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140` | `exp_019` | `Kernel Execution Timeout` | 2 | 3 | false | 0.4995106306179069 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_000` | `Numerical error` | 4 | 5 | true | 0.17516111494376502 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_001` | `Kernel Execution Timeout` | 2 | 3 | true | 0.21204019050789313 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_006` | `Kernel Execution Timeout` | 5 | 6 | true | 0.1706351977310609 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_007` | `Numerical error` | 1 | 2 | true | 0.46530365976686777 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_007` | `Numerical error` | 4 | 5 | false | 0.45443792848490216 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_008` | `Numerical error` | 4 | 5 | true | 0.15223268562389589 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_009` | `Numerical error` | 3 | 4 | true | 0.1673738724273677 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_009` | `Compilation error` | 5 | 6 | false | 0.16844457251937925 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_010` | `Numerical error` | 2 | 3 | true | 0.326325142026823 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_010` | `Kernel Execution Timeout` | 6 | 7 | false | 0.18231099707761464 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_011` | `Numerical error` | 1 | 2 | true | 0.1701598065218904 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_011` | `Compilation error` | 5 | 6 | false | 0.16134569926997508 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_012` | `Numerical error` | 0 | 1 | true | 0.22551033705124734 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_013` | `Numerical error` | 3 | 4 | true | 0.45073425963681274 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_014` | `Numerical error` | 3 | 4 | true | 0.21181668889246735 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_015` | `Runtime error` | 0 | 1 | true | 0.5090559184104752 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_016` | `Compilation error` | 0 | 1 | true | 0.1729399702293596 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_018` | `Numerical error` | 1 | 2 | true | 0.3439911568555166 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_018` | `Numerical error` | 6 | 7 | false | 0.27740143300221876 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_019` | `Runtime error` | 2 | 3 | true | 0.23567391274094746 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240` | `exp_019` | `Runtime error` | 5 | 6 | false | 0.022751926097657874 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_001` | `Numerical error` | 2 | 3 | true | 0.24959079392897057 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_002` | `Numerical error` | 5 | 6 | true | 0.24079781688252516 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_005` | `Runtime error` | 2 | 3 | true | 0.2818966781365094 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_005` | `Runtime error` | 4 | 5 | false | 0.22212765984937977 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_006` | `Runtime error` | 0 | 1 | true | 0.4545621917972395 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_006` | `Kernel Execution Timeout` | 4 | 5 | false | 0.29179946811912255 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_008` | `Compilation error` | 0 | 1 | true | 0.03517023377249276 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_009` | `Numerical error` | 0 | 1 | true | 0.059835914330318554 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_011` | `Numerical error` | 0 | 1 | true | 0.10888972703554908 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_011` | `Kernel Execution Timeout` | 6 | 7 | false | 0.10066063029425508 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_012` | `Numerical error` | 1 | 2 | true | 0.3280515662048358 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_013` | `Kernel Execution Timeout` | 1 | 2 | true | 0.3186400427943165 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_014` | `Numerical error` | 0 | 1 | true | 0.25876472448179877 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_015` | `Kernel Execution Timeout` | 3 | 4 | true | 0.29902404783666464 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_017` | `Numerical error` | 2 | 3 | true | 0.39444270070705995 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_018` | `Runtime error` | 2 | 3 | true | 0.33280167895554796 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_019` | `Kernel Execution Timeout` | 0 | 1 | true | 0.38464315782227915 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340` | `exp_019` | `Kernel Execution Timeout` | 2 | 3 | false | 0.5686115214273347 |
