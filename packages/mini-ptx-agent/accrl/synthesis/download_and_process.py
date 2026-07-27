"""Convert a kernel-task dataset to miles JSONL format.

Input: HuggingFace dataset (local directory, local JSONL, or Hub path)
with slim fields produced by ``construct_dataset.py``:
    definition_name, workload_uuids, language, num_workloads

Output: Miles JSONL where each line is:
    {"metadata": {<slim task identifiers>}}

Full Definition/Workload objects are resolved from the database at training
time by ``generate_with_agent_env.py``.

Usage:
    # From saved HF dataset directory:
    python accrl/synthesis/download_and_process.py \\
        --input data/kernel_tasks \\
        --output data/kernel_tasks_miles.jsonl

    # From HuggingFace Hub:
    python accrl/synthesis/download_and_process.py \\
        --input my-org/kernel-tasks \\
        --output data/kernel_tasks_miles.jsonl
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_miles_format(
    input_path: str,
    output_path: str,
    limit: int | None = None,
    split: str = "train",
) -> None:
    """Convert kernel-task JSONL to miles format.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        limit: Optional cap on number of samples.
        split: Dataset split name (included in metadata).
    """
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if limit and count >= limit:
                break

            instance = json.loads(line)

            metadata = {
                "definition_name": instance["definition_name"],
                "workload_uuids": json.loads(instance["workload_uuids"]),
                "language": instance["language"],
                "num_workloads": instance["num_workloads"],
                "split": split,
            }

            miles_sample = {
                "metadata": metadata,
            }

            fout.write(json.dumps(miles_sample) + "\n")
            count += 1

    logger.info("Converted %d samples: %s -> %s", count, input_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert kernel-task dataset to miles JSONL"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="HF dataset dir, local JSONL, or HuggingFace Hub path",
    )
    parser.add_argument("--output", required=True, help="Output miles JSONL path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_path = Path(args.input)

    if input_path.exists() and input_path.suffix == ".jsonl":
        # Local JSONL file — convert directly.
        logger.info("Processing local JSONL file: %s", args.input)
        convert_to_miles_format(args.input, args.output, args.limit, args.split)

    elif input_path.exists() and input_path.is_dir():
        # Saved HF dataset directory — load, export to JSONL, convert.
        from datasets import load_from_disk

        logger.info("Loading HF dataset from disk: %s", args.input)
        ds = load_from_disk(str(input_path))

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as tmp:
                tmp_path = tmp.name
            ds.to_json(tmp_path)
            convert_to_miles_format(tmp_path, args.output, split=args.split)
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    else:
        # Assume HuggingFace Hub path.
        from datasets import load_dataset

        logger.info("Loading from HuggingFace Hub: %s (split=%s)", args.input, args.split)
        ds = load_dataset(args.input, split=args.split)

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as tmp:
                tmp_path = tmp.name
            ds.to_json(tmp_path)
            convert_to_miles_format(tmp_path, args.output, split=args.split)
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    logger.info("Done.")


if __name__ == "__main__":
    main()
