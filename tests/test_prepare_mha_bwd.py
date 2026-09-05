import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = (
    REPO_ROOT
    / "packages"
    / "mini-ptx-agent"
    / "fib_runtime"
    / "multiturn"
    / "mha-bwd-problems"
    / "scripts"
    / "prepare_mha_bwd.py"
)


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_mha_bwd", PREPARE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mha_bwd_preparer_keeps_do_out_of_safetensors():
    module = load_prepare_module()
    item = module.build_workload_entry("mha_bwd_d128", 128, 4096)
    inputs = item["workload"]["inputs"]

    assert list(inputs) == ["Q", "K", "V", "O", "dO", "L"]
    assert inputs["dO"] == {"type": "random"}
    for name in ["Q", "K", "V", "O", "L"]:
        assert inputs[name]["type"] == "safetensors"
        assert inputs[name]["tensor_key"] == name

    bf16 = torch.ones(2, dtype=torch.bfloat16)
    fp32 = torch.ones(2, dtype=torch.float32)
    blob = module.build_blob_tensors(bf16, bf16, bf16, bf16, fp32)
    assert list(blob) == ["Q", "K", "V", "O", "L"]
    assert "dO" not in blob
