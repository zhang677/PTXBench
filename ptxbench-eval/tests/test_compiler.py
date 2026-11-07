import subprocess

import pytest

from ptxbench_eval import compiler

H100 = "arch=compute_90a,code=sm_90a"


@pytest.fixture
def fake_nvcc(monkeypatch):
    calls = []
    monkeypatch.setattr(
        compiler.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=None),
    )

    def warning(kernel_path, nvcc_gencode, tvm_ffi_dir):
        calls.append(nvcc_gencode)
        return "INSTRUCTION WARNING: try wgmma."

    monkeypatch.setattr(compiler, "_instruction_warning", warning)
    return calls


@pytest.mark.parametrize("prompt_tag", [None, "h100-fp8", "hopper-conv"])
def test_other_prompt_tags_keep_arch_instruction_warning(fake_nvcc, prompt_tag):
    result = compiler.compile_source("// kernel", nvcc_gencode=H100, prompt_tag=prompt_tag)
    assert result["ok"]
    assert result["isa_warning"].startswith("INSTRUCTION WARNING")
    assert fake_nvcc == [H100]


@pytest.mark.parametrize("prompt_tag", ["hopper-reduce", "blackwell-reduce"])
def test_reduce_prompt_tags_skip_instruction_warning(fake_nvcc, prompt_tag):
    result = compiler.compile_source("// kernel", nvcc_gencode=H100, prompt_tag=prompt_tag)
    assert result["ok"]
    assert result["isa_warning"] == ""
    assert fake_nvcc == []
