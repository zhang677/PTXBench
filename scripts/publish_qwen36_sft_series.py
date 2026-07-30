#!/usr/bin/env python3
"""Stage and privately publish the PTXBench Qwen3.6-27B s0-s6 artifacts.

The staging command copies the byte-exact historical training parquets and
standard PEFT adapter directories into one dataset repository tree and seven
model repository trees. The upload command always creates private repositories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from safetensors import safe_open

NAMESPACE = "Genghan"
DATASET_REPO = f"{NAMESPACE}/PTXBench-Qwen3.6-27B-SFT"
BASE_MODEL = "Qwen/Qwen3.6-27B"
BASE_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
COLLECTION_TITLE = "PTXBench Qwen3.6-27B SFT Series"


@dataclass(frozen=True)
class Variant:
    label: str
    internal_tag: str
    config: str
    template: str
    synthesizer: str
    rows: int
    parquet_sha256: str

    @property
    def model_repo(self) -> str:
        return f"{NAMESPACE}/PTXBench-Qwen3.6-27B-{self.label}"


VARIANTS = (
    Variant(
        "s0",
        "sft-v4",
        "8ops-Extended",
        "KernelGen",
        "GLM-5.2",
        494,
        "5416899bf9f8312e1e5361dc12f20613246ef597f073852224c81eb314c10ff8",
    ),
    Variant(
        "s1",
        "fixit-v2-glm",
        "4ops",
        "Fixit",
        "GLM-5.2",
        158,
        "6a42a93125cc8c6dfc0bb52ec1c807e0e6b39fd8f04c43300d82f82ad866d4de",
    ),
    Variant(
        "s2",
        "fixit-v2-glm-8turns",
        "4ops-Extended",
        "Fixit",
        "GLM-5.2",
        259,
        "06d78bfa4c6f68ec73d7f8d57df4eb31c2a946d1915edfa1c19202e7b74448b1",
    ),
    Variant(
        "s3",
        "fixit-v4",
        "8ops-Extended",
        "Fixit",
        "GLM-5.2",
        406,
        "8aa2bf4f4c34e542bcc13c66f16e18f51d0da20426823108c8d9778c0002b0b8",
    ),
    Variant(
        "s4",
        "fixit-v5",
        "8ops-Post-balanced",
        "Fixit",
        "GLM-5.2",
        170,
        "eeda2fa32ded94675fa8b800b3a7be110d79eec195945e836d9141c11f4b3664",
    ),
    Variant(
        "s5",
        "fixit-v5-full",
        "8ops-Pre-balanced",
        "Fixit",
        "GLM-5.2",
        258,
        "e59150ed81e28edf92710f06760a77077d54d588a3b85ec8d641d52dca5f33d5",
    ),
    Variant(
        "s6",
        "fixit-v6",
        "8ops-Pre-balanced",
        "Fixit",
        "Qwen3.6-27B",
        258,
        "55122991454816ab51ffbc10d5be2547bb7000b44e8f4df93aa7804a67a60689",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        return {row["tag"]: row for row in csv.DictReader(stream)}


def load_final_checkpoint(path: Path) -> dict:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    finals = [record for record in records if record.get("name") == "final"]
    if len(finals) != 1:
        raise ValueError(f"{path}: expected one final checkpoint, found {len(finals)}")
    return finals[0]


def validate_parquet(path: Path, variant: Variant) -> dict:
    metadata = pq.ParquetFile(path).metadata
    actual_hash = sha256(path)
    if metadata.num_rows != variant.rows:
        raise ValueError(
            f"{variant.label}: expected {variant.rows} rows, got {metadata.num_rows}"
        )
    if actual_hash != variant.parquet_sha256:
        raise ValueError(
            f"{variant.label}: expected {variant.parquet_sha256}, got {actual_hash}"
        )
    table = pq.read_table(path, columns=["messages"])
    expected_roles = (
        ["system", "user", "assistant"]
        if variant.label == "s0"
        else ["system", "user", "assistant", "user", "assistant"]
    )
    expected_masks = None if variant.label == "s0" else [0, 0, 0, 0, 1]
    for index, messages in enumerate(table["messages"].to_pylist()):
        roles = [message.get("role") for message in messages]
        if roles != expected_roles:
            raise ValueError(
                f"{variant.label} row {index}: roles {roles!r}, "
                f"expected {expected_roles!r}"
            )
        if expected_masks is not None:
            masks = [message.get("step_loss_mask") for message in messages]
            if masks != expected_masks:
                raise ValueError(
                    f"{variant.label} row {index}: masks {masks!r}, "
                    f"expected {expected_masks!r}"
                )
    return {
        "rows": metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": actual_hash,
        "message_roles": expected_roles,
        "step_loss_masks": expected_masks,
    }


def validate_adapter(path: Path) -> dict:
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(
            f"{path}: expected adapter_config.json and adapter_model.safetensors"
        )
    config = json.loads(config_path.read_text())
    if config.get("base_model_name_or_path") != BASE_MODEL:
        raise ValueError(
            f"{path}: base model is {config.get('base_model_name_or_path')!r}, "
            f"expected {BASE_MODEL!r}"
        )
    target_modules = set(config.get("target_modules", []))
    if "in_proj_qkv" not in target_modules:
        raise ValueError(f"{path}: fused in_proj_qkv is missing from target_modules")
    if target_modules.intersection({"in_proj_q", "in_proj_k", "in_proj_v"}):
        raise ValueError(f"{path}: unfused linear-attention targets remain")
    if config.get("rank_pattern", {}).get("in_proj_qkv") != 96:
        raise ValueError(f"{path}: expected rank_pattern in_proj_qkv=96")
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        qkv_keys = [key for key in keys if ".in_proj_qkv." in key]
        split_keys = [
            key
            for key in keys
            if any(f".in_proj_{projection}." in key for projection in ("q", "k", "v"))
        ]
        if len(qkv_keys) != 96 or split_keys:
            raise ValueError(
                f"{path}: expected 96 fused QKV tensors and no split tensors; "
                f"found {len(qkv_keys)} fused and {len(split_keys)} split"
            )
        tensor_count = len(keys)
    return {
        "files": {
            child.name: {"bytes": child.stat().st_size, "sha256": sha256(child)}
            for child in sorted(path.iterdir())
            if child.is_file()
        },
        "peft_type": config.get("peft_type"),
        "rank": config.get("r"),
        "target_modules": config.get("target_modules"),
        "rank_pattern": config.get("rank_pattern"),
        "tensor_count": tensor_count,
    }


def dataset_card() -> str:
    configs = "\n".join(
        f"  - config_name: {variant.label}\n"
        f"    data_files:\n"
        f"      - split: train\n"
        f"        path: data/{variant.label}/train.parquet"
        for variant in VARIANTS
    )
    rows = "\n".join(
        f"| {v.label} | `{v.internal_tag}` | {v.config} | {v.template} | "
        f"{v.synthesizer} | {v.rows} | `{v.parquet_sha256}` |"
        for v in VARIANTS
    )
    return f"""---
configs:
{configs}
language:
  - code
pretty_name: PTXBench Qwen3.6-27B SFT Series
tags:
  - cuda
  - ptx
  - supervised-fine-tuning
  - code-generation
---

# PTXBench Qwen3.6-27B SFT datasets

This private repository contains the byte-exact parquet files used to train
PTXBench Qwen3.6-27B-s0 through Qwen3.6-27B-s6. Load one release as:

```python
from datasets import load_dataset

dataset = load_dataset("{DATASET_REPO}", "s0", split="train", token=True)
```

| Config | Internal recipe | Scope | Template | Reasoning synthesizer | Rows | SHA-256 |
| --- | --- | --- | --- | --- | ---: | --- |
{rows}

The parquet `metadata` column retains historical machine-local paths because
these are the exact training artifacts. Those paths are provenance strings and
are not required to load or train on the `messages` column.

The s0 rows contain `system`, `user`, and `assistant` messages. The s1-s6
Fixit rows contain `system`, `user`, `assistant`, `user`, and `assistant`
messages, with only the final assistant message carrying loss mask 1.

All seven runs used `{BASE_MODEL}`, five epochs, learning rate `4.65e-4`,
maximum length 65,536, and LoRA rank 32. See `manifest.json` for exact
artifact and checkpoint provenance.

These data contain generated reasoning and CUDA kernels. Repository access is
private; no public redistribution license is asserted by this card.
"""


def merge_script(variant: Variant) -> str:
    return f'''"""Merge the {variant.label} PEFT adapter into its pinned base model."""

from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

BASE_MODEL = "{BASE_MODEL}"
BASE_REVISION = "{BASE_MODEL_REVISION}"
ADAPTER = "{variant.model_repo}"
OUTPUT = "PTXBench-Qwen3.6-27B-{variant.label}-merged"

base = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    token=True,
)
model = PeftModel.from_pretrained(base, ADAPTER, token=True)
merged = model.merge_and_unload(safe_merge=True)
merged.save_pretrained(OUTPUT, safe_serialization=True, max_shard_size="5GB")
AutoProcessor.from_pretrained(
    BASE_MODEL, revision=BASE_REVISION, token=True
).save_pretrained(OUTPUT)
'''


def model_card(variant: Variant) -> str:
    return f"""---
base_model: {BASE_MODEL}
base_model_relation: finetune
datasets:
  - {DATASET_REPO}
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - lora
  - cuda
  - ptx
  - code-generation
---

# PTXBench Qwen3.6-27B-{variant.label}

This private repository contains the final PEFT LoRA adapter for
PTXBench Qwen3.6-27B-{variant.label}. It was trained from `{BASE_MODEL}` with
rank 32 on the `{variant.label}` configuration of `{DATASET_REPO}`.

Qwen3.6 represents each linear-attention Q/K/V input projection as one fused
`in_proj_qkv` module in Transformers. The release adapter therefore rewrites
each independently trained rank-32 Q/K/V triplet as one mathematically
equivalent block-diagonal rank-96 adapter. All other modules remain rank 32.
See `FUSION_PROVENANCE.json` for the conversion details.

| Field | Value |
| --- | --- |
| Internal recipe | `{variant.internal_tag}` |
| Training rows | {variant.rows} |
| Epochs | 5 |
| Learning rate | `4.65e-4` |
| Maximum length | 65,536 |
| Release-tested reconstruction base revision | `{BASE_MODEL_REVISION}` |

## Load without merging

```python
from peft import PeftModel
from transformers import AutoModelForMultimodalLM

base = AutoModelForMultimodalLM.from_pretrained(
    "{BASE_MODEL}",
    revision="{BASE_MODEL_REVISION}",
    torch_dtype="auto",
    device_map="auto",
    token=True,
)
model = PeftModel.from_pretrained(base, "{variant.model_repo}", token=True)
```

Run `python merge_adapter.py` to create a standalone merged model directory.
The merge requires the full base-model download and substantial CPU memory and
disk space; it does not require a GPU.

This checkpoint is intended for research on CUDA kernel generation and repair.
It inherits the limitations of the base model and may emit incorrect or unsafe
CUDA. Generated kernels must be compiled and evaluated in an isolated
environment before use.
"""


def stage(args: argparse.Namespace) -> None:
    mapping = load_mapping(args.mapping_csv)
    dataset_root = args.stage_root / "dataset"
    models_root = args.stage_root / "models"
    dataset_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "series": "PTXBench Qwen3.6-27B s0-s6",
        "visibility": "private",
        "dataset_repo": DATASET_REPO,
        "base_model": BASE_MODEL,
        "release_tested_base_model_revision": BASE_MODEL_REVISION,
        "variants": [],
    }

    for variant in VARIANTS:
        row = mapping[variant.internal_tag]
        parquet_path = Path(row["data"])
        checkpoint_path = Path(row["checkpoint"])
        parquet = validate_parquet(parquet_path, variant)
        final_checkpoint = load_final_checkpoint(checkpoint_path)
        run_config = json.loads((checkpoint_path.parent / "config.json").read_text())

        dataset_path = dataset_root / "data" / variant.label / "train.parquet"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet_path, dataset_path)

        model_root = models_root / variant.label
        model_root.mkdir(parents=True, exist_ok=True)
        adapter_info = None
        adapter_source = args.adapter_root / variant.label / "peft_adapter_final"
        if adapter_source.is_dir():
            validate_adapter(adapter_source)
            for child in adapter_source.iterdir():
                if child.is_file():
                    shutil.copy2(child, model_root / child.name)
            staged_config_path = model_root / "adapter_config.json"
            staged_config = json.loads(staged_config_path.read_text())
            staged_config["revision"] = BASE_MODEL_REVISION
            staged_config_path.write_text(
                json.dumps(staged_config, indent=2, sort_keys=True) + "\n"
            )
            adapter_info = validate_adapter(model_root)

        training = {
            "model_name": run_config.get("model_name"),
            "renderer_name": run_config.get("renderer_name"),
            "learning_rate": run_config.get("learning_rate"),
            "num_epochs": run_config.get("num_epochs"),
            "lora_rank": run_config.get("lora_rank"),
            "load_checkpoint_path": run_config.get("load_checkpoint_path"),
            "final_checkpoint": final_checkpoint,
            "parquet": parquet,
        }
        (model_root / "training_manifest.json").write_text(
            json.dumps(training, indent=2, sort_keys=True) + "\n"
        )
        (model_root / "README.md").write_text(model_card(variant))
        (model_root / "merge_adapter.py").write_text(merge_script(variant))
        (model_root / "requirements.txt").write_text(
            "transformers>=5.5,<6\npeft>=0.18.1,<1\nsafetensors>=0.7,<1\n"
        )

        manifest["variants"].append(
            {
                "label": variant.label,
                "internal_tag": variant.internal_tag,
                "config": variant.config,
                "template": variant.template,
                "reasoning_synthesizer": variant.synthesizer,
                "dataset_repo": DATASET_REPO,
                "model_repo": variant.model_repo,
                "parquet": parquet,
                "final_checkpoint": final_checkpoint,
                "adapter": adapter_info,
            }
        )

    (dataset_root / "README.md").write_text(dataset_card())
    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def upload(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    dataset_root = args.stage_root / "dataset"
    if not (dataset_root / "manifest.json").is_file():
        raise ValueError(f"{dataset_root}: staged dataset manifest is missing")
    for variant in VARIANTS:
        validate_adapter(args.stage_root / "models" / variant.label)

    api.create_repo(
        DATASET_REPO, repo_type="dataset", private=True, exist_ok=True
    )
    api.update_repo_settings(DATASET_REPO, repo_type="dataset", private=True)
    dataset_commit = api.upload_folder(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        folder_path=dataset_root,
        commit_message="Upload exact PTXBench Qwen3.6-27B s0-s6 training parquets",
    )
    if not api.repo_info(DATASET_REPO, repo_type="dataset").private:
        raise RuntimeError(f"{DATASET_REPO} is not private")
    print(f"dataset_commit={dataset_commit.oid}")

    for variant in VARIANTS:
        model_root = args.stage_root / "models" / variant.label
        api.create_repo(variant.model_repo, private=True, exist_ok=True)
        api.update_repo_settings(variant.model_repo, private=True)
        model_commit = api.upload_folder(
            repo_id=variant.model_repo,
            folder_path=model_root,
            commit_message=f"Upload final PTXBench Qwen3.6-27B-{variant.label} adapter",
        )
        if not api.repo_info(variant.model_repo).private:
            raise RuntimeError(f"{variant.model_repo} is not private")
        print(f"{variant.label}_commit={model_commit.oid}")
    collection = api.create_collection(
        COLLECTION_TITLE,
        namespace=NAMESPACE,
        description=(
            "Private collection of the exact s0-s6 training parquets and final "
            "PEFT adapters used by the PTXBench Qwen3.6-27B SFT series."
        ),
        private=True,
        exists_ok=True,
    )
    api.update_collection_metadata(collection.slug, private=True)
    api.add_collection_item(
        collection.slug, DATASET_REPO, "dataset", exists_ok=True
    )
    for variant in VARIANTS:
        api.add_collection_item(
            collection.slug, variant.model_repo, "model", exists_ok=True
        )
    print(f"private_collection={collection.slug}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("stage", "upload"))
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("/home/ubuntu/AccRL/benchmark/sft_mapping.csv"),
    )
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "stage":
        stage(args)
    else:
        upload(args)


if __name__ == "__main__":
    main()
