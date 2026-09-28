#!/usr/bin/env python3
"""Multiturn kernel generation using mini-swe-agent infrastructure.

The model outputs CUDA kernel code directly each turn (no bash commands).
Docker container handles compilation, the profiling service handles evaluation.
Uses DefaultAgent + DockerEnvironment for trajectory logging/inspection compatibility.

The system prompt uses `prompts/assembled/<prompt_tag>.md`.
For parallel sweeps driven by a config JSON, use `run_parallel_v2.py`.

Usage:
    python run_v2.py --definition gemm_n7168_k5120 --model gemini-3.1-pro-preview \
        --test-path tests/cuda/gemm_n7168_k5120.py \
        --log-path trajectory.json --prompt-tag hopper-00 \
        --max-turns 5 --target-speedup 1.5
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: F401
    SCRIPT_DIR,
    SYSTEM_INSTRUCTIONS,
    TRITON_SYSTEM_INSTRUCTIONS,
    run_main_v2,
)

logger = logging.getLogger(__name__)



def build_system_prompt(prompt_tag: str, gpu_arch: str) -> str:
    """Build system prompt from base prompt file and any additional instructions.

    Wraps in {% raw %}...{% endraw %} because base_prompt contains PTX inline
    assembly with {%...} syntax that conflicts with Jinja2 template rendering.
    """
    base_prompt_path = SCRIPT_DIR / "prompts/assembled" / f"{prompt_tag}.md"
    assert base_prompt_path.exists(), f"Bundled prompt file is missing: {base_prompt_path}"
    base_prompt = base_prompt_path.read_text()

    TVM_FFI_EXAMPLE_PATH = SCRIPT_DIR / "prompts/examples/cuda.cu"
    
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


def build_triton_system_prompt(prompt_tag: str, gpu_arch: str) -> str:
    """Build the Triton prompt without CUDA C++/TVM-FFI material."""
    base_prompt_path = SCRIPT_DIR / "prompts/assembled" / f"{prompt_tag}.md"
    assert base_prompt_path.exists(), (
        f"Base prompt file {base_prompt_path} does not exist. "
        "The prompt document is missing."
    )
    base_prompt = base_prompt_path.read_text()

    triton_example_path = (
        SCRIPT_DIR / "prompts/examples/triton.py"
    )
    base_prompt += f"""
Here is an example of the required Triton integration pattern. Follow its
destination-passing wrapper, metadata-only host code, current-device selection,
and explicit launch structure.
```python
{triton_example_path.read_text()}
```

"""

    if gpu_arch == "hopper":
        target_prompt = (
            "\n\nYou are targeting NVIDIA Hopper SM90/SM90a GPUs with Triton. "
            "Use the provided Triton structural documentation for this architecture.\n\n"
        )
    elif gpu_arch == "blackwell":
        target_prompt = (
            "\n\nYou are targeting NVIDIA Blackwell SM100/SM100a GPUs with Triton. "
            "Use the provided Triton structural documentation for this architecture.\n\n"
        )
    else:
        raise ValueError(f"Unsupported GPU architecture: {gpu_arch}")

    return (
        "{% raw %}"
        + TRITON_SYSTEM_INSTRUCTIONS
        + base_prompt
        + target_prompt
        + "{% endraw %}"
    )


if __name__ == "__main__":
    run_main_v2(
        build_system_prompt,
        build_triton_system_prompt,
    )
