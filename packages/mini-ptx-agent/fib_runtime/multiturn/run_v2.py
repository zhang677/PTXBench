#!/usr/bin/env python3
"""Multiturn kernel generation using mini-swe-agent infrastructure.

The model outputs CUDA kernel code directly each turn (no bash commands).
Docker container handles compilation, the profiling service handles evaluation.
Uses DefaultAgent + DockerEnvironment for trajectory logging/inspection compatibility.

Unlike run.py / run_v1.py, the system prompt is composed from a `prompt_tag`
that indexes into `prompt_configs/hub.json`. The assembled base prompt is
expected to already exist at `prompt_configs/<prompt_tag>.md` — build it first
with `build_doc_v2.py <config.json>`. For parallel sweeps driven by a config
JSON, use `run_parallel_v2.py`.

Usage:
    # One-time: assemble the base prompt for the tag(s) you'll use
    python build_doc_v2.py prompt_configs/2026-0421-2352.json --force  # --force to rebuild even if the .md files already exist

    # Single run:
    python run_v2.py --definition gemm_n6144_k4096 --model gemini-3.1-pro-preview \
        --test-path ../mini_swe_agent_docker/envs/test_profile_cuda_gemm_n6144_k4096.py \
        --log-path trajectory.json --prompt-tag hopper-00 \
        --max-turns 5 --target-speedup 1.5
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SCRIPT_DIR, SYSTEM_INSTRUCTIONS, run_main_v2  # noqa: F401

logger = logging.getLogger(__name__)



def build_system_prompt(prompt_tag: str, gpu_arch: str) -> str:
    """Build system prompt from base prompt file and any additional instructions.

    Wraps in {% raw %}...{% endraw %} because base_prompt contains PTX inline
    assembly with {%...} syntax that conflicts with Jinja2 template rendering.
    """
    base_prompt_path = SCRIPT_DIR / "prompt_configs" / f"{prompt_tag}.md"
    assert base_prompt_path.exists(), f"Base prompt file {base_prompt_path} does not exist. Did you run build_doc_v2.py to generate it?"
    base_prompt = base_prompt_path.read_text()

    TVM_FFI_EXAMPLE_PATH = SCRIPT_DIR.parent / "mini_swe_agent_docker/envs/example.cu"
    
    base_prompt += f"""
Here is an example of how to use TVM-FFI. You should use TVM-FFI to wrap you kernel.
```cpp
{TVM_FFI_EXAMPLE_PATH.read_text()}
```

"""
    if gpu_arch == "hopper":
        extra_prompt = "\n\n You are targeting NVIDIA Hopper architecture GPUs. Use the provided structural docs to understand the hardware features and how to optimize for them. \n\n"
        base_prompt += extra_prompt
    elif gpu_arch == "blackwell":
        extra_prompt = "\n\n You are targeting NVIDIA Blackwell architecture GPUs. Use the provided structural docs to understand the hardware features and how to optimize for them. \n\n"
        base_prompt += extra_prompt
    else:
        raise ValueError(f"Unsupported GPU architecture: {gpu_arch}")
    return "{% raw %}" + SYSTEM_INSTRUCTIONS + base_prompt + "{% endraw %}"


if __name__ == "__main__":
    run_main_v2(build_system_prompt)
