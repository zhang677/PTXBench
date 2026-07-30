"""Download a Tinker LoRA checkpoint and convert to a servable artifact.

Reads `checkpoints.jsonl` from a training run, selects the requested
checkpoint (default ``final``), and downloads its ``sampler_path`` via
``tinker_cookbook.weights.download``. From the raw Tinker adapter, this
script can produce either (or both) of:

* a standalone PEFT adapter (``--peft-output``) via
  ``tinker_cookbook.weights.build_lora_adapter`` — small, paired with the
  base model at serve time (see ``serve_peft.sh``).
* a full merged HuggingFace model (``--hf-output``) via
  ``tinker_cookbook.weights.build_hf_model`` — base + LoRA merged into
  one model directory, served standalone without ``--enable-lora`` (see
  ``serve_full_weight.sh``).

At least one of ``--peft-output`` / ``--hf-output`` must be provided.

See https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/

Usage:
    python download_weights.py \
        --checkpoints-jsonl <run_dir>/checkpoints.jsonl \
        (--peft-output <run_dir>/peft_adapter_final |
         --hf-output  <run_dir>/hf_merged_final) \
        [--checkpoint-name final] \
        [--base-model Qwen/Qwen3.5-35B-A3B] \
        [--adapter-dir <run_dir>/tinker_adapter_final]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


def find_checkpoint(checkpoints_jsonl: Path, name: str) -> dict:
    with checkpoints_jsonl.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]
    for entry in entries:
        if entry.get("name") == name:
            return entry
    available = ", ".join(e.get("name", "?") for e in entries)
    raise SystemExit(
        f"checkpoint {name!r} not found in {checkpoints_jsonl}. Available: {available}"
    )


def download_adapter(
    *,
    tinker_path: str,
    output_dir: Path,
    timeout_seconds: float,
) -> str:
    """Download with a timeout long enough for Tinker to build large archives."""
    import tinker
    from tinker_cookbook.weights._download import _safe_extract_tar

    service_client = tinker.ServiceClient(timeout=timeout_seconds)
    rest_client = service_client.create_rest_client()
    response = rest_client.get_checkpoint_archive_url_from_tinker_path(
        tinker_path
    ).result()

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        archive_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(response.url, str(archive_path))
        _safe_extract_tar(archive_path, output_dir)
    finally:
        archive_path.unlink(missing_ok=True)
    return str(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-jsonl", type=Path, required=True)
    parser.add_argument("--peft-output", type=Path, default=None,
                        help="Destination directory for the PEFT-format adapter.")
    parser.add_argument("--hf-output", type=Path, default=None,
                        help="Destination directory for the merged full-weight "
                             "HuggingFace model (base + LoRA merged).")
    parser.add_argument("--checkpoint-name", default="final")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--adapter-dir", type=Path, default=None,
                        help="Where to extract the raw Tinker adapter. "
                             "Defaults to <output>/../tinker_adapter_<name>.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download/re-convert even if outputs already exist.")
    parser.add_argument(
        "--tinker-timeout-seconds",
        type=float,
        default=1800,
        help="HTTP timeout while Tinker builds the checkpoint archive (default: 1800).",
    )
    args = parser.parse_args()

    if args.peft_output is None and args.hf_output is None:
        raise SystemExit("must provide at least one of --peft-output or --hf-output")

    if "TINKER_API_KEY" not in os.environ:
        raise SystemExit("TINKER_API_KEY must be set")

    entry = find_checkpoint(args.checkpoints_jsonl, args.checkpoint_name)
    sampler_path = entry["sampler_path"]
    print(f"checkpoint  : {entry['name']} (epoch={entry['epoch']}, batch={entry['batch']})")
    print(f"sampler_path: {sampler_path}")

    primary_output: Path = (
        args.peft_output.resolve() if args.peft_output is not None
        else args.hf_output.resolve()
    )
    adapter_dir: Path = (
        args.adapter_dir.resolve()
        if args.adapter_dir is not None
        else primary_output.parent / f"tinker_adapter_{args.checkpoint_name}"
    )

    from tinker_cookbook import weights

    raw_marker = adapter_dir / "adapter_model.safetensors"
    if raw_marker.exists() and not args.force:
        print(f"raw Tinker adapter already present at {adapter_dir} -- skipping download")
    else:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_adapter(
            tinker_path=sampler_path,
            output_dir=adapter_dir,
            timeout_seconds=args.tinker_timeout_seconds,
        )
        print(f"raw Tinker adapter downloaded to: {downloaded}")

    if args.peft_output is not None:
        peft_output = args.peft_output.resolve()
        peft_marker = peft_output / "adapter_config.json"
        if peft_marker.exists() and not args.force:
            print(f"PEFT adapter already present at {peft_output} -- skipping conversion")
        else:
            if peft_output.exists() and args.force:
                import shutil
                shutil.rmtree(peft_output)
            peft_output.parent.mkdir(parents=True, exist_ok=True)
            weights.build_lora_adapter(
                base_model=args.base_model,
                adapter_path=str(adapter_dir),
                output_path=str(peft_output),
            )
            print(f"PEFT adapter saved to: {peft_output}")

    if args.hf_output is not None:
        hf_output = args.hf_output.resolve()
        # build_hf_model errors out if the output dir already exists, so check
        # for a completed merge ourselves (config.json is the standard marker).
        hf_marker = hf_output / "config.json"
        if hf_marker.exists() and not args.force:
            print(f"merged HF model already present at {hf_output} -- skipping merge")
        else:
            if hf_output.exists() and args.force:
                import shutil
                shutil.rmtree(hf_output)
            hf_output.parent.mkdir(parents=True, exist_ok=True)
            weights.build_hf_model(
                base_model=args.base_model,
                adapter_path=str(adapter_dir),
                output_path=str(hf_output),
            )
            print(f"merged HF model saved to: {hf_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
