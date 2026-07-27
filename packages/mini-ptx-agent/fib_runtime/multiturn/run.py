#!/usr/bin/env python3
"""Multiturn kernel generation using mini-swe-agent infrastructure.

The model outputs CUDA kernel code directly each turn (no bash commands).
Docker container handles compilation, the profiling service handles evaluation.
Uses DefaultAgent + DockerEnvironment for trajectory logging/inspection compatibility.

Usage:
    python run.py --definition gemm_n6144_k4096 --model claude-opus-4.8-xhigh \
        --test-path ../mini_swe_agent_docker/envs/test_profile_cuda_gemm_n6144_k4096.py \
        --log-path trajectory.json --max-turns 5 --target-speedup 1.5
"""

import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SCRIPT_DIR, SYSTEM_INSTRUCTIONS, run_main

logger = logging.getLogger(__name__)


def build_system_prompt(gpu_arch: str = "hopper") -> str:
    """Build system prompt from base prompt file and any additional instructions.

    Wraps in {% raw %}...{% endraw %} because base_prompt contains PTX inline
    assembly with {%...} syntax that conflicts with Jinja2 template rendering.
    """
    base_prompt_path = SCRIPT_DIR / f"base_prompt_{gpu_arch}.txt"
    if not base_prompt_path.exists():
        subprocess.run(
            ["python", "stitch_base_prompt.py", "--gpu-arch", gpu_arch],
            check=True,
        )
    base_prompt = base_prompt_path.read_text()
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
    run_main(build_system_prompt)
