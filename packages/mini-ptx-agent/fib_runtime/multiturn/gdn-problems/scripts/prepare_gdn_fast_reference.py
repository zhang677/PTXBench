#!/usr/bin/env python3
"""Use copied GDN solution code as the definition reference.

The original GDN Python references are correct but slow enough that service
verification spends most of its time running the baseline/reference path. This
script preserves each original reference in the definition description and then
replaces the definition's `reference` field with the selected copied solution
source.

Run from anywhere:

    python scripts/prepare_gdn_fast_reference.py --dataset-root /home/ubuntu/accrl-training

By default the script updates `/home/ubuntu/accrl-training`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOLUTION_FILES = {
    "gdn_decode_qk4_v8_d128_k_last": "flashinfer_wrapper_9b7f1e.json",
    "gdn_decode_qk8_v16_d128_k_last": "flashinfer_wrapper_a5e9d2.json",
    "gdn_mtp_qk4_v8_d128_k_last": "flashinfer_wrapper_a3d7c2.json",
    "gdn_mtp_qk8_v16_d128_k_last": "flashinfer_wrapper_b5e9f1.json",
    "gdn_prefill_qk4_v8_d128_k_last": "flashinfer_wrapper_c3f8a1.json",
    "gdn_prefill_qk8_v16_d128_k_last": "flashinfer_wrapper_b7d4e2.json",
}

ARCHIVE_MARKER = "## Original Python Reference"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def solution_source(dataset_root: Path, definition_name: str) -> str:
    solution_path = (
        dataset_root
        / "solutions"
        / "gdn"
        / definition_name
        / SOLUTION_FILES[definition_name]
    )
    solution = load_json(solution_path)
    if solution.get("definition") != definition_name:
        raise ValueError(
            f"{solution_path} has definition={solution.get('definition')!r}, "
            f"expected {definition_name!r}"
        )

    sources = solution.get("sources") or []
    main_sources = [s for s in sources if s.get("path") == "main.py"]
    if len(main_sources) != 1:
        raise ValueError(f"{solution_path} expected exactly one main.py source")
    content = main_sources[0].get("content")
    if not isinstance(content, str) or "def run(" not in content:
        raise ValueError(f"{solution_path} main.py source does not define run()")
    return make_reference_safe(add_cuda_device_guard(content, solution_path), definition_name)


def add_cuda_device_guard(content: str, solution_path: Path) -> str:
    """Run the GDN reference under the input tensor's CUDA device.

    The profiling service can execute workloads on a non-zero GPU while the
    process default CUDA device remains 0. This mirrors the MHA BWD reference
    pattern and avoids device-index mismatches inside CUDA libraries.
    """
    if "with torch.cuda.device(q.device):" in content:
        return content

    lines = content.rstrip().splitlines()
    run_idx = next((i for i, line in enumerate(lines) if line.startswith("def run(")), None)
    if run_idx is None:
        raise ValueError(f"{solution_path} main.py source does not define top-level run()")

    guarded = lines[: run_idx + 1]
    guarded.append("    with torch.cuda.device(q.device):")
    for line in lines[run_idx + 1 :]:
        guarded.append(("    " + line) if line else "")
    return "\n".join(guarded) + "\n"


def make_reference_safe(content: str, definition_name: str) -> str:
    """Add the same minimal wrapper used by copied GDN solutions.

    The wrapper enters the active CUDA device and clones only mutable state
    buffers before calling the original FlashInfer kernel.
    """
    wrapper_marker = "# Clone mutable state inputs before running the GDN kernel."
    old_wrapper_marker = "# Clone mutable state inputs for evaluator baseline profiling."
    if wrapper_marker in content:
        return content
    if old_wrapper_marker in content:
        return content.replace(old_wrapper_marker, wrapper_marker)

    if "_decode_" in definition_name or "_prefill_" in definition_name:
        needle = "    with torch.cuda.device(q.device):\n"
        replacement = (
            needle
            + f"        {wrapper_marker}\n"
            + "        state = state.clone() if isinstance(state, torch.Tensor) else state\n"
        )
    elif "_mtp_" in definition_name:
        needle = "    with torch.cuda.device(q.device):\n"
        replacement = (
            needle
            + f"        {wrapper_marker}\n"
            + "        initial_state = initial_state.clone() if isinstance(initial_state, torch.Tensor) else initial_state\n"
            + "        if isinstance(intermediate_states_buffer, torch.Tensor):\n"
            + "            intermediate_states_buffer = intermediate_states_buffer.clone()\n"
        )
    else:
        raise ValueError(f"unexpected GDN definition name: {definition_name}")

    if needle not in content:
        raise ValueError(f"reference for {definition_name} is missing CUDA device guard")
    return content.replace(needle, replacement, 1)


def archive_reference(description: str, reference: str) -> str:
    if ARCHIVE_MARKER in description:
        return description
    return (
        description.rstrip()
        + "\n\n"
        + ARCHIVE_MARKER
        + "\n\n"
        + "The operator is expressed as a Python reference below, which is only for correctness not performance.\n\n"
        + "```python\n"
        + reference.rstrip()
        + "\n```\n"
    )


def prepare_definition(dataset_root: Path, definition_name: str, dry_run: bool) -> bool:
    definition_path = dataset_root / "definitions" / "gdn" / f"{definition_name}.json"
    definition = load_json(definition_path)
    if definition.get("name") != definition_name:
        raise ValueError(
            f"{definition_path} has name={definition.get('name')!r}, expected {definition_name!r}"
        )

    old_reference = definition.get("reference")
    if not isinstance(old_reference, str) or not old_reference.strip():
        raise ValueError(f"{definition_path} has no string reference field")

    new_reference = solution_source(dataset_root, definition_name)
    new_description = archive_reference(definition.get("description", ""), old_reference)
    changed = (
        definition.get("reference") != new_reference
        or definition.get("description", "") != new_description
    )
    if not changed:
        print(f"unchanged {definition_path}")
        return False

    definition["description"] = new_description
    definition["reference"] = new_reference
    if dry_run:
        print(f"would update {definition_path}")
    else:
        dump_json(definition_path, definition)
        print(f"updated {definition_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ubuntu/accrl-training"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    updated = 0
    for definition_name in SOLUTION_FILES:
        if prepare_definition(dataset_root, definition_name, args.dry_run):
            updated += 1
    action = "would update" if args.dry_run else "updated"
    print(f"{action} {updated} GDN definitions")


if __name__ == "__main__":
    main()
