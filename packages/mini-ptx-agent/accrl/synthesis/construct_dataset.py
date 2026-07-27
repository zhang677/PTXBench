"""Construct a HuggingFace dataset from the kernel database.

Each dataset item represents one RL task: a kernel definition paired with
its workloads.  The resulting dataset can be saved locally or pushed to the
HuggingFace Hub, and later converted to miles JSONL via
``download_and_process.py``.

Only identifiers are stored — full Definition/Workload objects are resolved
from the database at training time by ``generate_with_agent_env.py``.

Fields per row:
  - definition_name: str
  - workload_uuids:  str  (JSON list of UUIDs)
  - language:        str
  - num_workloads:   int

Usage:
    python accrl/synthesis/construct_dataset.py \\
        --output data/kernel_tasks \\
        --language triton \\
        --max-workloads-per-def 10
"""

import argparse
import json
import logging
import random

from datasets import Dataset

from fib_runtime.db_client import KernelDB

logger = logging.getLogger(__name__)


def construct_dataset(
    db: KernelDB,
    language: str = "triton",
    definitions: list[str] | None = None,
    max_workloads_per_def: int = 10,
    seed: int = 42,
) -> Dataset:
    """Build a HuggingFace Dataset from the kernel database.

    Args:
        db: KernelDB instance.
        language: Target language for kernel generation.
        definitions: Optional list of definition names to include.
        max_workloads_per_def: Max workloads per definition.
        seed: Random seed for reproducibility.

    Returns:
        A ``datasets.Dataset`` with one row per (definition, workloads) pair.
    """
    random.seed(seed)

    all_defs = db.list_definitions()
    if definitions:
        name_set = set(definitions)
        all_defs = [d for d in all_defs if d.name in name_set]

    logger.info("Found %d definitions", len(all_defs))

    records: list[dict] = []
    for defn in all_defs:
        workloads = db.list_workloads(
            definition_name=defn.name, limit=max_workloads_per_def
        )
        if not workloads:
            logger.info("  %s: 0 workloads — skipped", defn.name)
            continue

        logger.info("  %s: %d workloads", defn.name, len(workloads))

        records.append(
            {
                "definition_name": defn.name,
                "workload_uuids": json.dumps([w.uuid for w in workloads]),
                "language": language,
                "num_workloads": len(workloads),
            }
        )

    random.shuffle(records)
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct HuggingFace dataset from kernel database"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for HF dataset"
    )
    parser.add_argument(
        "--language",
        default="triton",
        choices=["triton", "python", "cuda"],
    )
    parser.add_argument("--definitions", nargs="*", default=None)
    parser.add_argument("--max-workloads-per-def", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional HuggingFace Hub repo name",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    db = KernelDB()
    ds = construct_dataset(
        db,
        language=args.language,
        definitions=args.definitions,
        max_workloads_per_def=args.max_workloads_per_def,
        seed=args.seed,
    )

    ds.save_to_disk(args.output)
    logger.info("Saved dataset with %d rows to %s", len(ds), args.output)

    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub)
        logger.info("Pushed to Hub: %s", args.push_to_hub)


if __name__ == "__main__":
    main()
