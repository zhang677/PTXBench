import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "fuse_qwen36_peft_adapter.py"
SPEC = importlib.util.spec_from_file_location("fuse_qwen36_peft_adapter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
fuse_adapter = MODULE.fuse_adapter


def test_fuse_split_qkv_is_exact(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    config = {
        "alpha_pattern": {},
        "base_model_name_or_path": "Qwen/Qwen3.6-27B",
        "inference_mode": False,
        "lora_alpha": 2,
        "peft_type": "LORA",
        "r": 2,
        "rank_pattern": {},
        "target_modules": ["in_proj_q", "in_proj_k", "in_proj_v", "out_proj"],
        "use_rslora": False,
    }
    (source / "adapter_config.json").write_text(json.dumps(config))

    prefix = "base_model.model.layers.0.linear_attn"
    weights = {}
    generator = torch.Generator().manual_seed(17)
    for projection, output_features in (("q", 3), ("k", 2), ("v", 4)):
        weights[f"{prefix}.in_proj_{projection}.lora_A.weight"] = torch.randn(
            2, 5, generator=generator
        )
        weights[f"{prefix}.in_proj_{projection}.lora_B.weight"] = torch.randn(
            output_features, 2, generator=generator
        )
    save_file(weights, source / "adapter_model.safetensors")

    assert fuse_adapter(source, output) == 1

    fused = load_file(output / "adapter_model.safetensors")
    fused_delta = (
        fused[f"{prefix}.in_proj_qkv.lora_B.weight"]
        @ fused[f"{prefix}.in_proj_qkv.lora_A.weight"]
    )
    expected_delta = torch.cat(
        [
            weights[f"{prefix}.in_proj_{projection}.lora_B.weight"]
            @ weights[f"{prefix}.in_proj_{projection}.lora_A.weight"]
            for projection in ("q", "k", "v")
        ],
        dim=0,
    )
    assert torch.equal(fused_delta, expected_delta)

    fused_config = json.loads((output / "adapter_config.json").read_text())
    assert fused_config["rank_pattern"]["in_proj_qkv"] == 6
    assert fused_config["alpha_pattern"]["in_proj_qkv"] == 6
    assert "in_proj_qkv" in fused_config["target_modules"]
    assert not {"in_proj_q", "in_proj_k", "in_proj_v"}.intersection(
        fused_config["target_modules"]
    )
