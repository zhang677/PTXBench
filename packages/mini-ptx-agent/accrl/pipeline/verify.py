"""Verify Definition references are buildable and runnable via flashinfer-bench.

Uses BuilderRegistry to compile the reference, gen_inputs to create test data,
and runs the reference on each workload trace.
"""

import logging
import traceback

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.timing import time_runnable
from flashinfer_bench.bench.utils import compute_error_stats, gen_inputs
from flashinfer_bench.compile.registry import BuilderRegistry
from flashinfer_bench.data import Definition, Solution, Trace, TraceSet

logger = logging.getLogger(__name__)


def verify_definition(
    defn: Definition,
    traces: list[Trace],
    device: str = "cuda",
    profile: bool = False,
    warmup: int = 10,
    iterations: int = 50,
    min_latency_ms: float = 1.0,
    max_latency_ms: float = 15.0,
) -> tuple[bool, str]:
    """Build reference for a definition and run it on each workload trace.

    Returns (passed, message) where message describes the first failure
    or a success summary.
    """
    # Build reference
    try:
        registry = BuilderRegistry.get_instance()
        ref = registry.build_reference(defn)
    except Exception as e:
        return False, f"Build failed: {e}"

    # Run on each workload
    n_passed = 0
    latencies: list[float] = []
    for i, trace in enumerate(traces):
        try:
            inputs = gen_inputs(defn, trace.workload, device=device)
            result = ref(*inputs)
            n_passed += 1
        except Exception as e:
            return False, (
                f"Runtime error on workload {i} "
                f"(axes={trace.workload.axes}): {e}"
            )

        if profile:
            try:
                latency_ms = time_runnable(ref, inputs, warmup, iterations, device)
                latencies.append(latency_ms)
                if latency_ms < min_latency_ms or latency_ms > max_latency_ms:
                    return False, (
                        f"Latency out of range on workload {i} "
                        f"(axes={trace.workload.axes}): "
                        f"{latency_ms:.2f} ms not in [{min_latency_ms}, {max_latency_ms}] ms"
                    )
            except Exception as e:
                return False, (
                    f"Profiling error on workload {i} "
                    f"(axes={trace.workload.axes}): {e}"
                )

    if profile and latencies:
        lo, hi = min(latencies), max(latencies)
        return True, f"Passed: {n_passed}/{len(traces)} workloads (latency: {lo:.1f}\u2013{hi:.1f} ms)"

    return True, f"Passed: {n_passed}/{len(traces)} workloads"


def verify_solution(
    defn: Definition,
    solution: Solution,
    traces: list[Trace],
    device: str = "cuda",
) -> tuple[bool, str]:
    """Compile a solution via BuilderRegistry, run on each workload, compare against reference.

    Returns (passed, message).
    """
    registry = BuilderRegistry.get_instance()

    # Build reference
    try:
        ref = registry.build_reference(defn)
    except Exception as e:
        return False, f"Reference build failed: {e}"

    # Build solution
    try:
        sol = registry.build(defn, solution)
    except Exception as e:
        return False, f"Solution build failed: {e}"

    cfg = BenchmarkConfig(atol=1e-2, rtol=1e-2, required_matched_ratio=0.999)

    n_passed = 0
    for i, trace in enumerate(traces):
        try:
            inputs = gen_inputs(defn, trace.workload, device=device)

            # Run reference
            ref_outputs = ref.call_value_returning(*inputs)
            if not isinstance(ref_outputs, tuple):
                ref_outputs = (ref_outputs,)

            # Run solution
            sol_outputs = sol.call_value_returning(*inputs)
            if not isinstance(sol_outputs, tuple):
                sol_outputs = (sol_outputs,)

            # Compare each output
            for j, (sol_out, ref_out) in enumerate(zip(sol_outputs, ref_outputs)):
                _, _, exceeds, matched = compute_error_stats(sol_out, ref_out, cfg)
                if exceeds:
                    out_name = list(defn.outputs.keys())[j]
                    return False, (
                        f"Workload {i} output '{out_name}': "
                        f"matched_ratio={matched:.4f} (axes={trace.workload.axes})"
                    )

            n_passed += 1
        except Exception as e:
            return False, (
                f"Runtime error on workload {i} "
                f"(axes={trace.workload.axes}): {e}"
            )

    return True, f"Passed: {n_passed}/{len(traces)} workloads"


def verify_traceset(
    root: str,
    definitions: list[str] | None = None,
    device: str = "cuda",
    profile: bool = False,
    warmup: int = 10,
    iterations: int = 50,
    min_latency_ms: float = 1.0,
    max_latency_ms: float = 15.0,
) -> dict[str, tuple[bool, str]]:
    """Verify all (or selected) definitions in a traceset directory.

    Returns {name: (passed, message)}.
    """
    ts = TraceSet.from_path(root)
    results = {}

    names = definitions if definitions else list(ts.definitions.keys())
    for name in names:
        if name not in ts.definitions:
            results[name] = (False, f"Definition '{name}' not found in traceset")
            continue

        defn = ts.definitions[name]
        traces = ts.workloads.get(name, [])
        if not traces:
            results[name] = (False, "No workload traces found")
            continue

        logger.info("Verifying %s (%d workloads)...", name, len(traces))
        try:
            passed, msg = verify_definition(
                defn, traces, device=device, profile=profile,
                warmup=warmup, iterations=iterations,
                min_latency_ms=min_latency_ms, max_latency_ms=max_latency_ms,
            )
        except Exception:
            passed, msg = False, f"Unexpected error:\n{traceback.format_exc()}"

        results[name] = (passed, msg)
        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s — %s", name, status, msg)

    return results


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Verify traceset definitions")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Traceset root directory",
    )
    parser.add_argument(
        "--definitions",
        nargs="*",
        default=None,
        help="Specific definitions to verify (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile reference latency on each workload",
    )
    parser.add_argument(
        "--min-latency-ms",
        type=float,
        default=1.0,
        help="Minimum acceptable reference latency in ms (default: 1.0)",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=15.0,
        help="Maximum acceptable reference latency in ms (default: 15.0)",
    )
    args = parser.parse_args()

    results = verify_traceset(
        args.input_dir, args.definitions, args.device,
        profile=args.profile,
        min_latency_ms=args.min_latency_ms,
        max_latency_ms=args.max_latency_ms,
    )

    n_pass = sum(1 for p, _ in results.values() if p)
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} passed, {n_fail} failed out of {len(results)}")
    for name, (passed, msg) in sorted(results.items()):
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}: {msg}")

    sys.exit(0 if n_fail == 0 else 1)
