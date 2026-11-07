import subprocess
from pathlib import Path

import pytest

from ptxbench_eval import compiler

H100 = "arch=compute_90a,code=sm_90a"
B200 = "arch=compute_100a,code=sm_100a"


@pytest.fixture
def fake_nvcc(monkeypatch):
    """nvcc stub: the --ptx pass writes kernel.ptx containing the chosen PTX text."""
    state = {"ptx": ""}

    def run(command, *args, **kwargs):
        if "--ptx" in command:
            out = command[command.index("-o") + 1]
            (Path(kwargs["cwd"]) / out).write_text(state["ptx"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=None)

    monkeypatch.setattr(compiler.subprocess, "run", run)
    return state


def warning(nvcc_gencode, prompt_tag, ptx):
    return compiler.compile_source("// kernel", nvcc_gencode=nvcc_gencode,
                                   prompt_tag=prompt_tag)["isa_warning"]


@pytest.mark.parametrize("prompt_tag", [None, "h100-fp8", "hopper-conv"])
def test_other_prompt_tags_use_arch_instructions(fake_nvcc, prompt_tag):
    fake_nvcc["ptx"] = "st.async.shared::cluster"
    text = warning(H100, prompt_tag, fake_nvcc["ptx"])
    assert "wgmma." in text and "cp.async.bulk.tensor" in text


@pytest.mark.parametrize("gencode,prompt_tag", [(H100, "hopper-reduce"), (B200, "blackwell-reduce")])
def test_reduce_prompt_tags_use_st_async(fake_nvcc, gencode, prompt_tag):
    fake_nvcc["ptx"] = "ld.global.v4.u32"
    text = warning(gencode, prompt_tag, fake_nvcc["ptx"])
    assert text.startswith("INSTRUCTION WARNING")
    assert "st.async" in text and "wgmma." not in text and "tcgen05" not in text


def test_reduce_prompt_tag_satisfied_by_st_async(fake_nvcc):
    fake_nvcc["ptx"] = "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32"
    assert warning(H100, "hopper-reduce", fake_nvcc["ptx"]) == ""


def test_required_instructions_lookup():
    assert compiler.required_instructions(H100, "hopper-reduce") == {"st.async"}
    assert compiler.required_instructions(H100, "h100-fp8") == compiler.REQUIRED_INSTS[H100]
    assert compiler.required_instructions("arch=compute_80,code=sm_80") == set()
