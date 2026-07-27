# GLM 5.2 vs Gemini d128 Fix-It Kernel Quality

Generated: 2026-07-01

## Scope

Compared these two eval roots:

- GLM 5.2: `/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-glm52`
- Gemini: `/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-gemini`

The two roots use identical input plans:

- `222` planned fix-it prompts in each root
- same `exp_index`
- same source failed kernel paths
- same source error log paths
- same definitions
- same test paths
- same `target_speedup = 0.15`

That makes a direct paired comparison by `exp_NNN` valid.

## Important Artifact Note

The GLM root's existing `figures/turn_correctness_arch.csv` is stale or incomplete for this comparison:

- GLM CSV rows: `618`
- GLM CSV trajectories covered: `124 / 222`
- GLM current trajectory files: `222`

Gemini's CSV covers all `222` trajectories, but for symmetry and current-state correctness I computed the main comparison from current `trajectories/*.json` for both roots. I used `success/exp_*/kernel_v*.cu` as a cross-check for real fixed-kernel materialization.

## Method

For each trajectory, I reused the same classification and speedup extraction logic from:

`/home/ubuntu/AccRL/benchmark/export_turn_correctness_arch.py`

Relevant functions:

- `extract_turn_sequence(traj)`: pairs each assistant kernel response with its following user evaluation feedback and classifies the turn.
- `classify_turn(content)`: maps evaluation feedback to labels such as `Correct`, `Numerical error`, `Compilation error`, `Runtime error`, and `Kernel Execution Timeout`.
- `extract_turn_speedups(traj)`: extracts per-turn speedups from `extra.traces[*].evaluation.performance.speedup_factor`, falling back to `extra.min_speedup`.

For each planned `exp_NNN`, I computed:

- `any_correct`: whether any turn was classified `Correct`
- `first_correct_turn`: first turn with `Correct`
- `best_correct_speedup`: maximum speedup among `Correct` turns
- `best_correct_turn`: turn where that maximum occurred
- target hit: `best_correct_speedup >= 0.15`

I also counted materialized fixed kernels from non-backup success directories containing `kernel_v*.cu`.

## Reproduction Script

Run from anywhere on this host:

```bash
python - <<'PY'
import json, importlib.util, statistics
from pathlib import Path
from collections import Counter, defaultdict

spec = importlib.util.spec_from_file_location(
    "exporter",
    "/home/ubuntu/AccRL/benchmark/export_turn_correctness_arch.py",
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

roots = {
    "glm": Path("/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-glm52"),
    "gemini": Path("/home/ubuntu/AccRL-exps/eval_runs/scale-d128-fixit-qwen36-27b-gemini"),
}

allres = {}

for name, root in roots.items():
    plan = json.load(open(root / "plan.json"))["plan"]
    rows = []

    for tp in sorted((root / "trajectories").glob("*.json")):
        traj = json.load(open(tp))
        seq = exporter.extract_turn_sequence(traj)
        speedups = exporter.extract_turn_speedups(traj)
        for turn, correctness in enumerate(seq):
            rows.append(
                {
                    "trajectory_id": tp.stem,
                    "turn": turn,
                    "correctness": correctness,
                    "speedup": speedups.get(turn),
                }
            )

    bytraj = defaultdict(list)
    for row in rows:
        bytraj[row["trajectory_id"]].append(row)

    summary = []
    for i, item in enumerate(plan):
        tid = f"exp_{i:03d}"
        turns = sorted(bytraj.get(tid, []), key=lambda row: row["turn"])
        correct = [row for row in turns if row["correctness"] == "Correct"]
        speedup_rows = [
            row for row in correct
            if row["speedup"] is not None
        ]
        best = None
        best_turn = None
        if speedup_rows:
            best_row = max(speedup_rows, key=lambda row: float(row["speedup"]))
            best = float(best_row["speedup"])
            best_turn = best_row["turn"]
        summary.append(
            {
                **item,
                "trajectory_id": tid,
                "n_rows": len(turns),
                "any_correct": bool(correct),
                "first_correct_turn": min((row["turn"] for row in correct), default=None),
                "best_correct_speedup": best,
                "best_correct_turn": best_turn,
            }
        )

    allres[name] = {"rows": rows, "summary": summary, "plan": plan}

    best = [
        row["best_correct_speedup"]
        for row in summary
        if row["best_correct_speedup"] is not None
    ]

    print(f"\n== {name} ==")
    print("plan", len(plan))
    print("trajectory_files", len(list((root / "trajectories").glob("*.json"))))
    print("rows", len(rows))
    print("unique_trajectories", len(bytraj))
    print("turn_counts", Counter(len(v) for v in bytraj.values()))
    print("row_correctness", Counter(row["correctness"] for row in rows))
    print("any_correct", len(best), f"{len(best) / len(summary):.1%}")
    print("best_speedup_mean", statistics.mean(best))
    print("best_speedup_median", statistics.median(best))
    print("target_hits_ge_0.15", sum(value >= 0.15 for value in best))

    bydef = defaultdict(list)
    for row in summary:
        bydef[row["definition"]].append(row)
    for definition, rows_for_definition in sorted(bydef.items()):
        vals = [
            row["best_correct_speedup"]
            for row in rows_for_definition
            if row["best_correct_speedup"] is not None
        ]
        print(
            definition,
            "n", len(rows_for_definition),
            "fixed", len(vals),
            "target_hits", sum(value >= 0.15 for value in vals),
            "median", statistics.median(vals) if vals else None,
            "mean", statistics.mean(vals) if vals else None,
        )

print("\n== paired ==")
glm = allres["glm"]["summary"]
gemini = allres["gemini"]["summary"]
counts = Counter()
deltas = []
for g, m in zip(glm, gemini):
    gs = g["best_correct_speedup"]
    ms = m["best_correct_speedup"]
    if gs is not None and ms is not None:
        counts["both_fixed"] += 1
        deltas.append(gs - ms)
    elif gs is not None:
        counts["glm_only"] += 1
    elif ms is not None:
        counts["gemini_only"] += 1
    else:
        counts["neither"] += 1

print(counts)
print("both_delta_mean", statistics.mean(deltas))
print("both_delta_median", statistics.median(deltas))
print("glm_better_when_both_fixed", sum(delta > 0 for delta in deltas))
print("gemini_better_when_both_fixed", sum(delta < 0 for delta in deltas))

for name, root in roots.items():
    success_dirs = []
    kernel_files = 0
    for d in (root / "success").glob("exp_*"):
        if d.is_dir() and ".bak" not in d.name:
            kernels = list(d.glob("kernel_v*.cu"))
            if kernels:
                success_dirs.append(d.name)
                kernel_files += len(kernels)
    print(name, "success_dirs_with_kernel_v", len(success_dirs), "kernel_files", kernel_files)
PY
```

## Results

### Overall

| Metric | GLM 5.2 | Gemini |
|---|---:|---:|
| Planned prompts | 222 | 222 |
| Trajectory files | 222 | 222 |
| Current trajectory-derived turn rows | 1093 | 992 |
| Fixed trajectories with real `kernel_v*.cu` | 46 | 117 |
| Correct kernel attempts | 64 | 145 |
| Any Correct within run | 46/222 = 20.7% | 117/222 = 52.7% |
| Best-correct mean speedup | 0.1113 | 0.1911 |
| Best-correct median speedup | 0.0558 | 0.1697 |
| Best-correct max speedup | 0.4758 | 0.6176 |
| Target hits, `best_correct_speedup >= 0.15` | 15/222 = 6.8% | 74/222 = 33.3% |
| Target hits among fixed trajectories | 15/46 = 32.6% | 74/117 = 63.2% |

### By Definition

| Definition | GLM fixed | Gemini fixed | GLM target hits | Gemini target hits | GLM median speedup | Gemini median speedup |
|---|---:|---:|---:|---:|---:|---:|
| `mha_bwd_d128` | 8/72 = 11.1% | 32/72 = 44.4% | 2 | 18 | 0.0526 | 0.1734 |
| `mha_bwd_d128_causal` | 15/74 = 20.3% | 40/74 = 54.1% | 1 | 20 | 0.0328 | 0.1330 |
| `mha_with_lse_d128_causal` | 23/76 = 30.3% | 45/76 = 59.2% | 12 | 36 | 0.1553 | 0.2227 |

### By Turn

| Turn | GLM attempts | GLM Correct | GLM target hits | GLM median Correct speedup | Gemini attempts | Gemini Correct | Gemini target hits | Gemini median Correct speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 222 | 3 | 0 | 0.0670 | 222 | 5 | 2 | 0.0141 |
| 1 | 222 | 11 | 1 | 0.0322 | 220 | 24 | 13 | 0.2032 |
| 2 | 221 | 20 | 4 | 0.0360 | 206 | 40 | 23 | 0.1573 |
| 3 | 217 | 15 | 6 | 0.0673 | 183 | 40 | 22 | 0.1610 |
| 4 | 211 | 15 | 4 | 0.0587 | 161 | 36 | 14 | 0.0834 |

### Paired Outcome by Prompt

| Pair category | Count |
|---|---:|
| Both GLM and Gemini fixed | 29 |
| Gemini only fixed | 88 |
| GLM only fixed | 17 |
| Neither fixed | 88 |

Among the 29 prompts where both produced at least one Correct kernel:

- Gemini had the higher best speedup in 21 cases.
- GLM had the higher best speedup in 8 cases.
- Mean GLM-minus-Gemini best-speedup delta: `-0.1033`
- Median GLM-minus-Gemini best-speedup delta: `-0.1273`

## Interpretation

Gemini is clearly higher quality on this d128 fix-it set.

It fixed more than 2.5x as many planned prompts:

- GLM: `46 / 222`
- Gemini: `117 / 222`

It also produced many more kernels that met the target speedup:

- GLM: `15 / 222`
- Gemini: `74 / 222`

The difference is not only pass rate. Gemini's fixed kernels are also faster on average and by median:

- Mean best-correct speedup: Gemini `0.1911` vs GLM `0.1113`
- Median best-correct speedup: Gemini `0.1697` vs GLM `0.0558`

GLM does have isolated strong wins, especially in `mha_with_lse_d128_causal`, but those are sparse. As a source of high-quality fixed kernels for downstream SFT or eval reuse, Gemini is the better run.

## Notable GLM Wins

These are GLM-only or GLM-better cases with strong speedups:

| Trajectory | Definition | GLM best speedup | Gemini best speedup | Note |
|---|---|---:|---:|---|
| `exp_199` | `mha_with_lse_d128_causal` | 0.4758 | no fix | GLM-only |
| `exp_152` | `mha_with_lse_d128_causal` | 0.3310 | no fix | GLM-only |
| `exp_178` | `mha_with_lse_d128_causal` | 0.3054 | 0.1581 | GLM better |
| `exp_163` | `mha_with_lse_d128_causal` | 0.2941 | 0.0014 | GLM better |
| `exp_022` | `mha_bwd_d128` | 0.2739 | no fix | GLM-only |

These are useful if we want to mine GLM for extra kernels, but they do not change the aggregate conclusion.

## Notable Gemini-Only Wins

Top Gemini-only speedups:

| Trajectory | Definition | Gemini best speedup |
|---|---|---:|
| `exp_160` | `mha_with_lse_d128_causal` | 0.6176 |
| `exp_042` | `mha_bwd_d128` | 0.6097 |
| `exp_041` | `mha_bwd_d128` | 0.5470 |
| `exp_207` | `mha_with_lse_d128_causal` | 0.5231 |
| `exp_150` | `mha_with_lse_d128_causal` | 0.4828 |
| `exp_157` | `mha_with_lse_d128_causal` | 0.4785 |
| `exp_203` | `mha_with_lse_d128_causal` | 0.4553 |
| `exp_201` | `mha_with_lse_d128_causal` | 0.4001 |

## Conclusion

For this exact d128 fix-it input set, Gemini is the stronger fixed-kernel source:

- higher fix rate
- higher target-speedup hit rate
- higher median and mean speedup
- better paired performance when both runs fix the same prompt

Recommended use:

- Use Gemini as the primary source for fixed kernels.
- Optionally union in GLM-only successes, especially the higher-speed `mha_with_lse_d128_causal` cases, after deduping and validating against the same benchmark contract.
