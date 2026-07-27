# Success Diff Investigation

- 2026-0610-1100-complete: `/home/ubuntu/AccRL-exps/eval_runs/2026-0610-1100-complete/success-diffs.md`
- 2026-0610-0900: `/home/ubuntu/AccRL-exps/eval_runs/2026-0610-0900/success-diffs.md`
- Parsed success sections: 2026-0610-1100-complete=109, 2026-0610-0900=50
- Declared success counts: 2026-0610-1100-complete=109, 2026-0610-0900=50
- Success kernels differing from original files: 2026-0610-1100-complete=99 yes / 10 no / 0 unknown; 2026-0610-0900=47 yes / 3 no / 0 unknown

## Successful Exp Ids

- Common: 37 (exp_019, exp_053, exp_066, exp_068, exp_070, exp_071, exp_072, exp_073, exp_074, exp_075, exp_076, exp_084, exp_085, exp_086, exp_089, exp_120, exp_121, exp_124, exp_125, exp_126, exp_127, exp_154, exp_155, exp_161, exp_183, exp_192, exp_193, exp_194, exp_195, exp_196, exp_226, exp_228, exp_229, exp_230, exp_231, exp_232, exp_233)
- Only in 2026-0610-1100-complete: 28 (exp_007, exp_018, exp_021, exp_022, exp_033, exp_034, exp_040, exp_051, exp_054, exp_056, exp_067, exp_069, exp_088, exp_090, exp_092, exp_100, exp_108, exp_114, exp_115, exp_140, exp_148, exp_149, exp_150, exp_151, exp_160, exp_182, exp_186, exp_188)
- Only in 2026-0610-0900: 13 (exp_011, exp_012, exp_023, exp_052, exp_077, exp_079, exp_102, exp_137, exp_167, exp_189, exp_197, exp_227, exp_235)

## Original Kernels

- Common original kernel paths: 26
- Only in 2026-0610-1100-complete: 17
- Only in 2026-0610-0900: 7

### Original kernels only in 2026-0610-1100-complete
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_001/kernel_t3.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t4.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_004/kernel_t7.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_005/kernel_t4.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_006/kernel_t1.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_008/kernel_t3.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_009/kernel_t0.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_009/kernel_t1.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t5.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t6.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_013/kernel_t7.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_014/kernel_t5.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_015/kernel_t4.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_019/kernel_t4.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_001/kernel_t3.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_007/kernel_t2.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_005/kernel_t3.cu`

### Original kernels only in 2026-0610-0900
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_002/kernel_t4.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_002/kernel_t5.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_011/kernel_t3.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_014/kernel_t1.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_019/kernel_t1.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_014/kernel_t4.cu`
- `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t6.cu`

## Successes Only In 2026-0610-1100-complete

| exp | original kernel | differs from original | speedup | success hash |
| --- | --- | --- | ---: | --- |
| exp_007 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_001/kernel_t3.cu` | yes (+20/-14, 13 hunks) | 0.365515 | `50313eeaf47c` |
| exp_018 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t0.cu` | yes (+83/-26, 33 hunks) | 0.429755 | `eddf5bc3c700` |
| exp_021 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t4.cu` | yes (+9/-9, 4 hunks) | 0.561755 | `a026acb74173` |
| exp_022 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t7.cu` | yes (+22/-7, 4 hunks) | 0.589853 | `3bab4881b3c3` |
| exp_033 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_004/kernel_t7.cu` | yes (+44/-29, 15 hunks) | 0.285807 | `cf63632fba5d` |
| exp_034 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_005/kernel_t4.cu` | yes (+80/-79, 16 hunks) | 0.460367 | `97e3c060cee3` |
| exp_040 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_006/kernel_t1.cu` | yes (+36/-41, 11 hunks) | 0.299785 | `6f156a03f377` |
| exp_051 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_008/kernel_t3.cu` | yes (+49/-49, 3 hunks) | 0.203144 | `ee9656bbbfca` |
| exp_054 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_009/kernel_t0.cu` | yes (+26/-18, 13 hunks) | 0.466658 | `25e180180c1d` |
| exp_056 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_009/kernel_t1.cu` | yes (+154/-98, 18 hunks) | 0.416323 | `b3c75320298a` |
| exp_067 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_010/kernel_t1.cu` | yes (+74/-83, 22 hunks) | 0.423243 | `1ca427f0221e` |
| exp_069 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_010/kernel_t2.cu` | yes (+15/-43, 11 hunks) | 0.535013 | `dee541952f6a` |
| exp_088 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t4.cu` | yes (+41/-45, 24 hunks) | 0.275317 | `80edb7c8fba5` |
| exp_090 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t5.cu` | yes (+4/-4, 1 hunks) | 0.374692 | `731031f705bd` |
| exp_092 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t6.cu` | yes (+10/-10, 3 hunks) | 0.415087 | `6b443b0c8f36` |
| exp_100 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_013/kernel_t7.cu` | yes (+32/-16, 10 hunks) | 0.288119 | `0504c63e2fdb` |
| exp_108 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_014/kernel_t5.cu` | yes (+59/-56, 23 hunks) | 0.397291 | `ec56d1604063` |
| exp_114 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_015/kernel_t4.cu` | yes (+13/-27, 14 hunks) | 0.237818 | `fb42e95dee63` |
| exp_115 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_015/kernel_t4.cu` | yes (+35/-46, 19 hunks) | 0.342721 | `57964eb7ba64` |
| exp_140 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_019/kernel_t4.cu` | yes (+130/-134, 35 hunks) | 0.395569 | `f67ed21fbfe0` |
| exp_148 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_001/kernel_t3.cu` | no | 0.216239 | `e363cf9a23e1` |
| exp_149 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_001/kernel_t3.cu` | yes (+4/-4, 2 hunks) | 0.214106 | `16db135496d6` |
| exp_150 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_007/kernel_t2.cu` | yes (+49/-48, 31 hunks) | 0.431203 | `96803f8d780a` |
| exp_151 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_007/kernel_t2.cu` | yes (+142/-63, 20 hunks) | 0.0562328 | `75d7fd39c9ff` |
| exp_160 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_010/kernel_t3.cu` | yes (+19/-19, 10 hunks) | 0.322609 | `cddaf2f6a952` |
| exp_182 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_001/kernel_t3.cu` | yes (+28/-31, 12 hunks) | 0.252008 | `cecbb109403b` |
| exp_186 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_005/kernel_t3.cu` | yes (+20/-19, 7 hunks) | 0.0514361 | `4e1a80c560d5` |
| exp_188 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_005/kernel_t5.cu` | no | 0.223206 | `444c024e657c` |

## Successes Only In 2026-0610-0900

| exp | original kernel | differs from original | speedup | success hash |
| --- | --- | --- | ---: | --- |
| exp_011 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_002/kernel_t4.cu` | yes (+9/-9, 5 hunks) | 0.313259 | `9e682d93d67f` |
| exp_012 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_002/kernel_t5.cu` | yes (+8/-8, 4 hunks) | 0.427035 | `87ebc83da6ca` |
| exp_023 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t7.cu` | yes (+2/-2, 1 hunks) | 0.592198 | `2b9770734681` |
| exp_052 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_008/kernel_t5.cu` | yes (+1/-1, 1 hunks) | 0.333085 | `778bfbd26f3d` |
| exp_077 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_011/kernel_t2.cu` | yes (+2/-2, 1 hunks) | 0.232817 | `e2a1abc03899` |
| exp_079 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_011/kernel_t3.cu` | yes (+1/-1, 1 hunks) | 0.294423 | `a824996973ce` |
| exp_102 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_014/kernel_t1.cu` | yes (+6/-6, 4 hunks) | 0.344502 | `fe58d4fb99c6` |
| exp_137 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_019/kernel_t1.cu` | yes (+6/-6, 4 hunks) | 0.338886 | `3fa7d671e9af` |
| exp_167 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_014/kernel_t4.cu` | yes (+29/-35, 9 hunks) | 0.215558 | `5874d6690855` |
| exp_189 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_005/kernel_t5.cu` | no | 0.230899 | `444c024e657c` |
| exp_197 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_006/kernel_t5.cu` | yes (+2/-2, 2 hunks) | 0.288975 | `1cd342c2d185` |
| exp_227 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t1.cu` | yes (+5/-5, 3 hunks) | 0.378458 | `29651d56c468` |
| exp_235 | `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t6.cu` | yes (+2/-2, 1 hunks) | 0.276681 | `fa2862cff7df` |

## Common Exp Comparison

| exp | same original path | same success file content | 2026-0610-1100-complete differs from original | 2026-0610-0900 differs from original | 2026-0610-1100-complete speedup | 2026-0610-0900 speedup |
| --- | --- | --- | --- | --- | ---: | ---: |
| exp_019 | yes | no | no | yes (+2/-2, 1 hunks) | 0.423519 | 0.425022 |
| exp_053 | yes | no | yes (+4/-6, 5 hunks) | yes (+2/-2, 2 hunks) | 0.326135 | 0.340628 |
| exp_066 | yes | no | yes (+5/-5, 4 hunks) | yes (+4/-4, 3 hunks) | 0.427947 | 0.426163 |
| exp_068 | yes | no | yes (+91/-99, 27 hunks) | yes (+12/-12, 8 hunks) | 0.505004 | 0.52797 |
| exp_070 | yes | no | yes (+259/-304, 8 hunks) | yes (+5/-5, 5 hunks) | 0.551394 | 0.543635 |
| exp_071 | yes | no | yes (+41/-48, 24 hunks) | yes (+13/-12, 9 hunks) | 0.541317 | 0.536901 |
| exp_072 | yes | no | yes (+25/-24, 17 hunks) | yes (+24/-24, 16 hunks) | 0.362496 | 0.356575 |
| exp_073 | yes | no | yes (+8/-8, 8 hunks) | yes (+16/-16, 8 hunks) | 0.368346 | 0.359694 |
| exp_074 | yes | no | yes (+44/-26, 25 hunks) | yes (+5/-10, 9 hunks) | 0.205449 | 0.201652 |
| exp_075 | yes | no | yes (+27/-20, 26 hunks) | yes (+5/-7, 6 hunks) | 0.197493 | 0.206204 |
| exp_076 | yes | no | yes (+145/-202, 28 hunks) | yes (+4/-4, 2 hunks) | 0.295357 | 0.231712 |
| exp_084 | yes | no | yes (+3/-2, 2 hunks) | yes (+6/-6, 2 hunks) | 0.269834 | 0.266769 |
| exp_085 | yes | yes | yes (+6/-6, 2 hunks) | yes (+6/-6, 2 hunks) | 0.267755 | 0.268035 |
| exp_086 | yes | no | yes (+4/-4, 1 hunks) | yes (+10/-10, 3 hunks) | 0.311902 | 0.320088 |
| exp_089 | yes | no | yes (+20/-16, 12 hunks) | yes (+8/-8, 4 hunks) | 0.331288 | 0.339265 |
| exp_120 | yes | no | yes (+71/-95, 15 hunks) | yes (+8/-8, 8 hunks) | 0.311862 | 0.265433 |
| exp_121 | yes | no | yes (+2/-2, 2 hunks) | yes (+9/-8, 9 hunks) | 0.257574 | 0.260658 |
| exp_124 | yes | no | yes (+6/-6, 4 hunks) | yes (+10/-10, 6 hunks) | 0.565926 | 0.57753 |
| exp_125 | yes | no | no | yes (+10/-10, 6 hunks) | 0.556345 | 0.570357 |
| exp_126 | yes | no | yes (+92/-185, 27 hunks) | yes (+36/-36, 22 hunks) | 0.440295 | 0.412708 |
| exp_127 | yes | no | yes (+12/-12, 6 hunks) | yes (+36/-36, 22 hunks) | 0.403776 | 0.412614 |
| exp_154 | yes | no | yes (+16/-15, 17 hunks) | yes (+1/-1, 1 hunks) | 0.46185 | 0.461744 |
| exp_155 | yes | no | yes (+1/-1, 1 hunks) | no | 0.4654 | 0.457242 |
| exp_161 | yes | no | yes (+18/-25, 7 hunks) | yes (+1/-1, 1 hunks) | 0.322347 | 0.319851 |
| exp_183 | yes | no | yes (+245/-207, 41 hunks) | yes (+2/-5, 5 hunks) | 0.12978 | 0.240528 |
| exp_192 | yes | no | yes (+25/-23, 7 hunks) | yes (+4/-2, 2 hunks) | 0.335071 | 0.454142 |
| exp_193 | yes | no | yes (+2/-2, 2 hunks) | yes (+1/-1, 1 hunks) | 0.448458 | 0.44554 |
| exp_194 | yes | no | no | yes (+4/-1, 2 hunks) | 0.348013 | 0.347689 |
| exp_195 | yes | no | yes (+66/-93, 19 hunks) | yes (+1/-1, 1 hunks) | 0.339026 | 0.348595 |
| exp_196 | yes | no | yes (+15/-24, 6 hunks) | yes (+1/-1, 1 hunks) | 0.283056 | 0.286309 |
| exp_226 | yes | no | yes (+62/-99, 31 hunks) | yes (+8/-3, 4 hunks) | 0.383698 | 0.378862 |
| exp_228 | yes | no | yes (+63/-54, 17 hunks) | yes (+2/-2, 1 hunks) | 0.463967 | 0.568986 |
| exp_229 | yes | yes | yes (+2/-2, 1 hunks) | yes (+2/-2, 1 hunks) | 0.576587 | 0.57534 |
| exp_230 | yes | no | yes (+14/-27, 13 hunks) | no | 0.418029 | 0.413501 |
| exp_231 | yes | no | yes (+2/-2, 1 hunks) | yes (+2/-2, 1 hunks) | 0.405396 | 0.411328 |
| exp_232 | yes | no | yes (+21/-17, 8 hunks) | yes (+2/-2, 1 hunks) | 0.454002 | 0.442194 |
| exp_233 | yes | no | yes (+50/-27, 22 hunks) | yes (+2/-2, 1 hunks) | 0.486405 | 0.440455 |

## Original Kernel Groups

| original kernel | reports | 2026-0610-1100-complete exps | 2026-0610-0900 exps | 2026-0610-1100-complete diff count | 2026-0610-0900 diff count | 2026-0610-1100-complete unique success hashes | 2026-0610-0900 unique success hashes | shared success hashes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_001/kernel_t3.cu` | 2026-0610-1100-complete | exp_007 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_002/kernel_t4.cu` | 2026-0610-0900 | - | exp_011 | - | 1/1 changed | 0 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_002/kernel_t5.cu` | 2026-0610-0900 | - | exp_012 | - | 1/1 changed | 0 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t0.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_018, exp_018, exp_019 | exp_019 | 2/3 changed | 1/1 changed | 3 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t4.cu` | 2026-0610-1100-complete | exp_021 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_003/kernel_t7.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_022 | exp_023 | 1/1 changed | 1/1 changed | 1 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_004/kernel_t7.cu` | 2026-0610-1100-complete | exp_033 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_005/kernel_t4.cu` | 2026-0610-1100-complete | exp_034, exp_034, exp_034 | - | 3/3 changed | - | 3 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_006/kernel_t1.cu` | 2026-0610-1100-complete | exp_040, exp_040 | - | 2/2 changed | - | 2 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_008/kernel_t3.cu` | 2026-0610-1100-complete | exp_051 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_008/kernel_t5.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_053 | exp_052, exp_053 | 1/1 changed | 2/2 changed | 1 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_009/kernel_t0.cu` | 2026-0610-1100-complete | exp_054 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_009/kernel_t1.cu` | 2026-0610-1100-complete | exp_056 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_010/kernel_t1.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_066, exp_067, exp_067, exp_067 | exp_066 | 4/4 changed | 1/1 changed | 4 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_010/kernel_t2.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_068, exp_068, exp_069 | exp_068 | 3/3 changed | 1/1 changed | 3 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_010/kernel_t3.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_070, exp_070, exp_071, exp_071 | exp_070, exp_071 | 4/4 changed | 2/2 changed | 4 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_010/kernel_t4.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_072, exp_073 | exp_072, exp_073 | 2/2 changed | 2/2 changed | 2 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_011/kernel_t1.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_074, exp_074, exp_075, exp_075, exp_075 | exp_074, exp_075 | 5/5 changed | 2/2 changed | 5 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_011/kernel_t2.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_076, exp_076 | exp_076, exp_077 | 2/2 changed | 2/2 changed | 2 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_011/kernel_t3.cu` | 2026-0610-0900 | - | exp_079 | - | 1/1 changed | 0 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t1.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_084, exp_085 | exp_084, exp_085 | 2/2 changed | 2/2 changed | 2 | 2 | 1 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t3.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_086 | exp_086 | 1/1 changed | 1/1 changed | 1 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t4.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_088, exp_088, exp_089, exp_089 | exp_089 | 4/4 changed | 1/1 changed | 4 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t5.cu` | 2026-0610-1100-complete | exp_090 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_012/kernel_t6.cu` | 2026-0610-1100-complete | exp_092 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_013/kernel_t7.cu` | 2026-0610-1100-complete | exp_100, exp_100 | - | 2/2 changed | - | 2 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_014/kernel_t1.cu` | 2026-0610-0900 | - | exp_102 | - | 1/1 changed | 0 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_014/kernel_t5.cu` | 2026-0610-1100-complete | exp_108, exp_108 | - | 2/2 changed | - | 2 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_015/kernel_t4.cu` | 2026-0610-1100-complete | exp_114, exp_114, exp_114, exp_115, exp_115 | - | 4/5 changed | - | 5 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_015/kernel_t7.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_120, exp_120, exp_120, exp_121 | exp_120, exp_121 | 4/4 changed | 2/2 changed | 4 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_016/kernel_t4.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_124, exp_125 | exp_124, exp_125 | 1/2 changed | 2/2 changed | 2 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_016/kernel_t5.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_126, exp_126, exp_127, exp_127 | exp_126, exp_127 | 4/4 changed | 2/2 changed | 3 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_019/kernel_t1.cu` | 2026-0610-0900 | - | exp_137 | - | 1/1 changed | 0 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2140/kernels/exp_019/kernel_t4.cu` | 2026-0610-1100-complete | exp_140, exp_140 | - | 2/2 changed | - | 2 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_001/kernel_t3.cu` | 2026-0610-1100-complete | exp_148, exp_148, exp_149 | - | 1/3 changed | - | 2 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_007/kernel_t2.cu` | 2026-0610-1100-complete | exp_150, exp_151, exp_151 | - | 3/3 changed | - | 3 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_007/kernel_t5.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_154, exp_154, exp_155 | exp_154, exp_155 | 2/3 changed | 1/2 changed | 3 | 2 | 2 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_010/kernel_t3.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_160, exp_160, exp_161, exp_161 | exp_161 | 3/4 changed | 1/1 changed | 4 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2240/kernels/exp_014/kernel_t4.cu` | 2026-0610-0900 | - | exp_167 | - | 1/1 changed | 0 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_001/kernel_t3.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_182, exp_182, exp_182, exp_182, exp_183, exp_183, exp_183 | exp_183 | 7/7 changed | 1/1 changed | 7 | 1 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_005/kernel_t3.cu` | 2026-0610-1100-complete | exp_186 | - | 1/1 changed | - | 1 | 0 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_005/kernel_t5.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_188 | exp_189 | 0/1 changed | 0/1 changed | 1 | 1 | 1 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_006/kernel_t1.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_192, exp_192, exp_192, exp_193 | exp_192, exp_193 | 4/4 changed | 2/2 changed | 4 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_006/kernel_t2.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_194, exp_195, exp_195 | exp_194, exp_195 | 2/3 changed | 2/2 changed | 3 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_006/kernel_t5.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_196, exp_196 | exp_196, exp_197 | 2/2 changed | 2/2 changed | 2 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t1.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_226, exp_226, exp_226 | exp_226, exp_227 | 3/3 changed | 2/2 changed | 3 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t3.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_228, exp_228, exp_229 | exp_228, exp_229 | 3/3 changed | 2/2 changed | 2 | 2 | 1 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t4.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_230, exp_230, exp_231 | exp_230, exp_231 | 3/3 changed | 1/2 changed | 3 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t5.cu` | 2026-0610-1100-complete, 2026-0610-0900 | exp_232, exp_232, exp_233 | exp_232, exp_233 | 2/3 changed | 2/2 changed | 3 | 2 | 0 |
| `/home/ubuntu/AccRL-exps/eval_runs/2026-0609-2340/kernels/exp_019/kernel_t6.cu` | 2026-0610-0900 | - | exp_235 | - | 1/1 changed | 0 | 1 | 0 |
