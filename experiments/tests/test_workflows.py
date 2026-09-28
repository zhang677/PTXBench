"""Integration checks for the public workflow boundaries, without GPU services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.fixit.build_repair_config import build_config
from experiments.shared.run_manifest import ROOT, command, path, read_manifest


def test_repair_config_passes_failed_kernel_to_shared_runner(tmp_path: Path) -> None:
    kernel = tmp_path / "failed.cu"
    log = tmp_path / "error.txt"
    kernel.write_text("extern \"C\" __global__ void broken() {}\n")
    log.write_text("Numerical error: output mismatch\n")
    test_script = ROOT / "multiturn/tests/cuda/mha_with_lse_d128.py"
    config = build_config(
        [{
            "error_kernel_path": str(kernel),
            "error_log_path": str(log),
            "prompt_tag": "hopper-07",
            "definition": "mha_with_lse_d128",
            "test_path": str(test_script),
        }],
        num_turns=5,
        target_speedup=0.15,
        prompt_dir=tmp_path / "repair_prompts",
    )
    assert len(config) == 1
    entry = config[0]
    prompt = Path(entry["user_template"])
    assert prompt.is_file()
    assert "{task_content}" in prompt.read_text()
    assert kernel.read_text().strip() in prompt.read_text()
    assert log.read_text().strip() in prompt.read_text()
    assert entry["definition"] == "mha_with_lse_d128"
    assert entry["num_turns"] == 5
    assert entry["error_kernel_path"] == str(kernel)


def test_manifests_and_coding_agent_examples_share_prompt_registry() -> None:
    for workflow in ("fixit", "kernelgen"):
        for manifest in ("source-runs.csv", "eval-runs.csv"):
            rows = read_manifest(ROOT / "experiments" / workflow / manifest)
            assert rows
            assert all(path(row["test_path"]).is_file() for row in rows)
    hub = ROOT / "multiturn/prompts/hub.json"
    fragments = ROOT / "multiturn/prompts/fragments"
    for example in ("gemini_gemm", "gpt56_gemm"):
        config_path = ROOT / "ptxbench-eval/examples" / example / "experiment.json"
        config = json.loads(config_path.read_text())
        for key, target in (("prompt_hub", hub), ("prompt_files_root", fragments)):
            assert (config_path.parent / config[key]).resolve() == target
        assert set(run["prompt_tag"] for run in config["runs"]) <= set(json.loads(hub.read_text()))
