"""Write Definition and Workload data to flashinfer-trace folder structure.

Target layout:
    root/
    ├── definitions/{op_type}/{name}.json
    ├── workloads/{op_type}/{name}.jsonl
    ├── solutions/{op_type}/{name}/
    └── traces/{op_type}/{name}.jsonl
"""

import logging
from pathlib import Path

from flashinfer_bench.data import Definition, Solution, Trace, TraceSet

logger = logging.getLogger(__name__)


def write_definition(defn: Definition, root: Path) -> Path:
    """Write a Definition JSON to definitions/{op_type}/{name}.json.

    Returns the path of the written file.
    """
    root = Path(root)
    out_dir = root / "definitions" / defn.op_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{defn.name}.json"
    out_path.write_text(defn.model_dump_json(indent=2))
    logger.info("Wrote definition %s -> %s", defn.name, out_path)
    return out_path


def write_workloads(defn: Definition, traces: list[Trace], root: Path) -> Path:
    """Write workload Traces to workloads/{op_type}/{name}.jsonl.

    Each trace must be a workload trace (solution=None, evaluation=None)
    and reference the given definition name.

    Returns the path of the written file.
    """
    root = Path(root)
    out_dir = root / "workloads" / defn.op_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{defn.name}.jsonl"

    lines = []
    for t in traces:
        assert t.is_workload_trace(), f"Trace has solution/evaluation set: {t}"
        assert t.definition == defn.name, (
            f"Trace definition mismatch: {t.definition} != {defn.name}"
        )
        lines.append(t.model_dump_json())

    out_path.write_text("\n".join(lines) + "\n" if lines else "")
    logger.info("Wrote %d workloads for %s -> %s", len(traces), defn.name, out_path)
    return out_path


def write_solution(defn: Definition, solution: Solution, root: Path) -> Path:
    """Write a Solution JSON to solutions/{op_type}/{name}/{solution.name}.json.

    Returns the path of the written file.
    """
    root = Path(root)
    out_dir = root / "solutions" / defn.op_type / defn.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{solution.name}.json"
    out_path.write_text(solution.model_dump_json(indent=2))
    logger.info("Wrote solution %s -> %s", solution.name, out_path)
    return out_path


def write_traceset(defn: Definition, traces: list[Trace], root: Path) -> None:
    """Write both definition and workload traces to the traceset directory."""
    write_definition(defn, root)
    write_workloads(defn, traces, root)


def validate_traceset(root: Path) -> TraceSet:
    """Load a traceset directory with TraceSet.from_path() to validate structure.

    Raises if the directory structure is invalid.
    """
    root = Path(root)
    ts = TraceSet.from_path(str(root))
    logger.info(
        "Validated traceset at %s: %d definitions, %d workload groups",
        root,
        len(ts.definitions),
        len(ts.workloads),
    )
    return ts


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Validate a traceset directory")
    parser.add_argument("--input-dir", required=True, help="Traceset root directory")
    args = parser.parse_args()

    ts = validate_traceset(Path(args.input_dir))
    for name, defn in ts.definitions.items():
        n_wl = len(ts.workloads.get(name, []))
        print(f"  {name} ({defn.op_type}): {n_wl} workloads")
