# Benchmark exports and plot viewer

The repository-root `benchmark/export_turn_correctness_arch.py` creates the
canonical per-turn CSV for each evaluation run. Correct CUDA turns receive
`sass_arch_tag` only after two checks:

1. `nvcc` builds the candidate for the requested architecture and `cuobjdump`
   finds a selected instruction family in the embedded cubin.
2. FIBServe profiles that static-positive candidate with Nsight Compute and
   reports positive predicate-true execution for that family.

Hopper (`H`) recognizes GMMA and TMA transfer instructions. Blackwell (`B`)
recognizes TCGEN/TMEM instruction families. A static match by itself never sets
`sass_arch_tag`.

## Export per-turn results

Prepare a manifest with one row per evaluation run:

```csv
model,arch,definition,workload,exp_dir
example-model,hopper,gemm_n7168_k5120,94920358-01a8-4c5b-9209-3103fd490e94,/path/to/eval-run
```

Then run:

```bash
python benchmark/export_turn_correctness_arch.py \
  --experiments-csv experiments.csv \
  --base-url http://localhost:10000 \
  --num-compile-parallel 4 \
  --num-parallel 2
```

The output is written directly to
`<exp_dir>/figures/turn_correctness_arch.csv`; there is no intermediate CSV or
merge step. Static and dynamic results are cached below `figures/`. Use
`--force` to rewrite the output, `--force-static` to rebuild cubin evidence, and
`--force-profile` to rerun dynamic profiling. An existing output that does not
have the native SASS schema must be replaced with `--force`.

`sass_verification_status` distinguishes missing source, static absence,
inspection/profile failures, dynamic non-execution, and verified dynamic
presence. `--continue-on-static-error` and `--continue-on-profile-error` retain
rows with explicit failure statuses. For workflows that need only correctness
and speedup, `--skip-sass-verification` writes blank tags with status
`not_requested`.

## Build architecture plot data

After native per-turn exports exist, create the viewer's verified-SASS series:

```bash
python benchmark/aggregate_sass_metrics.py --experiments-csv experiments.csv
```

The aggregator rejects per-turn CSVs that lack the native SASS columns. It
marks every output row with `tag_evidence=dynamic_sass`, and the viewer filters
the architecture metric on that marker so only dynamically verified rows are
presented as SASS evidence. Model release dates come from
`benchmark/plot_viewer_v1/data/model_release_dates.csv`; a manifest-level
`date` column overrides them.

## Open the viewer

From the repository root, run:

```bash
python -m http.server 8765 --bind 0.0.0.0 --directory benchmark
```

Then open <http://localhost:8765/plot_viewer_v1/>.

The viewer and its checked-in data are self-contained under
`benchmark/plot_viewer_v1/`.
