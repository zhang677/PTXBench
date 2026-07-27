# /debug metadata sources

This note documents where each `logs[0].metadata` section from
`flashinfer-bench` `/debug` comes from. The current implementation is in
`/home/ubuntu/flashinfer-bench/flashinfer_bench/agents/debug.py`.

## Raw inputs

`flashinfer_bench_debug_solution()` builds one `raw_output` string from:

- compute-sanitizer stdout/stderr and return code for each requested sanitizer pass
- timeout messages from the sanitizer subprocess
- optional direct coredump replay output after a sanitizer timeout
- optional `cuda-gdb` batch backtrace output when a CUDA coredump file exists

It writes the complete raw text to `<debug_dir>/debug_raw.log`, then calls
`format_debug_metadata(..., raw_log=raw_output, coredumps=coredumps, ...)`.

The `coredumps` input is collected from `debug_dir.glob("cuda_coredump_*")`,
excluding files whose names contain `pipe`.

## Metadata sections

### FlashInfer CUDA debug report

Source: request/runtime context, not parser output.

Fields are formatted from:

- `solution.name`
- `solution.definition`
- `workload.uuid`
- `workload.axes`
- `device`
- `debug_dir`

This section is always emitted with `exist: true` in the normal formatter.
For early validation failures, `_metadata_response()` returns only this section
with `exist: false` and the error message.

### Most precise CUDA fault

Primary source: `extract_sanitizer_faults(raw_log)`.

It parses high-signal compute-sanitizer fault blocks:

- fault kind from sanitizer headers such as invalid, misaligned, race,
  uninitialized, barrier, warp, out-of-range, or stack overflow faults
- thread and block from `by thread (...) in block (...)`
- address lines containing `Address `
- source frames from `in <file>.cu:<line>`
- whether a frame is a `Device Frame`

Fallback source: `infer_cuda_api_faults(raw_log, entry_lines)`.

The fallback looks for `CUDA error ... at ...kernel.cu:<line>`. If the reported
line is a `cudaGetLastError` or `cudaStreamSynchronize` check site, it walks
backward through the submitted entry source to find a likely preceding kernel
launch line containing `<<<` or `cudaLaunch`.

The section has `exist: true` if either sanitizer faults or inferred CUDA API
faults were found.

### CUDA core dump analysis

Source: `extract_cuda_gdb_insights(raw_log)`.

This parser consumes the `cuda-gdb` text appended to `raw_log` after a coredump
is found and inspected with:

```bash
cuda-gdb -batch -ex "target cudacore <dump>" -ex bt
```

It extracts:

- diagnostics matching `CUDA Exception:...`,
  `Program received/terminated with signal...`, or `ERROR: cuda-gdb...`
- focus lines matching `Current focus set to ...`
- backtrace frames matching `#<index> ... in <function> at <file>.cu:<line>`
- inline call sites matching `inlined from ...kernel.cu:<line>`

The formatter then maps frame line numbers onto the submitted entry source,
adds source text when available, and adds timeout/wait/barrier interpretation
when frame text or source contains tokens such as `wait`, `barrier`,
`mbarrier`, `syncthreads`, `spin`, or `poll`.

The section has `exist: true` when any coredump diagnostics, focuses, or frames
produce rendered lines.

### Primary diagnostics

Source: direct regex scan over the full `raw_log`.

The formatter collects unique lines matching:

- `CUDA error...`
- `CUDA Exception:...`
- `Invalid ...`
- `Race reported...`
- `ERROR:...`
- `WARNING:...detected issues...`
- timeout messages like `... timed out after <N> seconds.`

It skips known noisy snippets:

- `cuGetProcAddress_v2`
- `Variable environment CUDA_`

The section has `exist: true` when at least one diagnostic line remains after
filtering.

### Source locations

Source: `extract_cuda_source_lines(raw_log)`.

This scans the entire raw log for generic CUDA source references using these
patterns:

- `kernel.cu:<line>` or `kernel.cu(<line>)`
- `at <file>.cu:<line>`
- `File "<file>.cu", line <line>`

It deduplicates exact `(path, line)` pairs while preserving first-seen order.
The formatter renders at most the first 8 locations.

If the submitted entry source is available, the raw path is used only for
deduplication. Display uses the submitted entry source path plus the matched
line number, followed by a source excerpt from the submitted kernel with
`source_context_lines` lines of context.

If the entry source is unavailable, the raw path and line are shown with an
`entry source unavailable` note.

The section has `exist: true` when at least one source-looking location was
found anywhere in `raw_log`.

### CUDA core dumps

Source: the `coredumps` filesystem list passed into `format_debug_metadata()`.

Each path is a retained CUDA coredump file under `debug_dir`.

If coredump files exist and `cuda-gdb` produced source frames, the section notes
that source-level coredump analysis is reported above and sets `exist: false`.
That makes renderers prefer `CUDA core dump analysis` as the actionable section.

If coredump files exist but no source-level `cuda-gdb` frames were parsed, the
section keeps the dump paths and sets `exist: true` for manual follow-up.

If no coredump files exist, it reports that no CUDA core dump was produced and
sets `exist: false`.

## Important interpretation

`Source locations` is broad evidence: any source-looking line mentioned by
compute-sanitizer, CUDA runtime errors, compiler warnings, or `cuda-gdb` can
appear there.

`Most precise CUDA fault` is narrower and usually more actionable for memory
errors because it is based on structured sanitizer fault blocks first, then CUDA
API error-site inference.

`CUDA core dump analysis` is the more actionable section for timeout/hang cases
when a coredump and source-level `cuda-gdb` backtrace exist.
