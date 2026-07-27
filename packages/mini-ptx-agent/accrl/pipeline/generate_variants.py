"""Generate workload Trace variants by sampling variable axis values.

This is a lightweight utility — the selection of axis ranges and values
that matter for real workloads requires LLM intelligence and is handled
by the collect/specialize skills.
"""

import itertools
import random
import uuid
from flashinfer_bench.data import Definition, Trace
from flashinfer_bench.data.workload import Workload


def generate_workloads(
    defn: Definition,
    var_axis_ranges: dict[str, list[int]],
    n: int = 5,
    seed: int | None = None,
) -> list[Trace]:
    """Generate workload Traces with sampled variable axis values.

    Parameters
    ----------
    defn : Definition
        The kernel definition (axes determine which are variable vs const).
    var_axis_ranges : dict[str, list[int]]
        For each variable axis name, a list of candidate values to sample from.
    n : int
        Maximum number of workload traces to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[Trace]
        Workload traces (solution=None, evaluation=None).
    """
    if seed is not None:
        random.seed(seed)

    # Determine const axes from the definition
    const_axes = {}
    var_axes_names = []
    for axis_name, axis_spec in defn.axes.items():
        if axis_spec.type == "const":
            const_axes[axis_name] = axis_spec.value
        else:
            var_axes_names.append(axis_name)

    # Validate that var_axis_ranges covers all variable axes
    missing = set(var_axes_names) - set(var_axis_ranges.keys())
    if missing:
        raise ValueError(
            f"Missing ranges for variable axes: {missing}. "
            f"Provide ranges for all variable axes: {var_axes_names}"
        )

    # Generate all combinations, then sample
    axis_lists = [var_axis_ranges[name] for name in var_axes_names]
    all_combos = list(itertools.product(*axis_lists))

    if len(all_combos) <= n:
        selected = all_combos
    else:
        selected = random.sample(all_combos, n)

    traces = []
    for combo in selected:
        axes = dict(const_axes)
        for name, value in zip(var_axes_names, combo):
            axes[name] = value

        wl = Workload(
            axes=axes,
            inputs={},
            uuid=str(uuid.uuid4()),
        )
        traces.append(Trace(definition=defn.name, workload=wl))

    return traces


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate workload traces for a definition"
    )
    parser.add_argument(
        "--definition",
        required=True,
        help="Path to definition JSON file",
    )
    parser.add_argument(
        "--ranges",
        required=True,
        help='JSON string of axis ranges, e.g. \'{"M": [128, 256, 512], "N": [64, 128]}\'',
    )
    parser.add_argument("--n", type=int, default=5, help="Max workloads to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    from flashinfer_bench.data import Definition

    defn = Definition.model_validate_json(open(args.definition).read())
    ranges = json.loads(args.ranges)
    traces = generate_workloads(defn, ranges, n=args.n, seed=args.seed)

    for t in traces:
        print(t.model_dump_json())
