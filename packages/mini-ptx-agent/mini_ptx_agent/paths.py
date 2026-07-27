"""Portable filesystem layout for PTXBench and mini-ptx-agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PTXBenchPaths:
    repo_root: Path
    mini_ptx_agent_root: Path
    data_root: Path
    multiturn_root: Path
    construct_eval_root: Path
    config_root: Path
    fixit_v6_project: Path
    eval_runs_root: Path


def resolve_paths() -> PTXBenchPaths:
    package_root = Path(__file__).resolve().parents[1]
    inferred_repo_root = package_root.parents[1]
    repo_root = Path(os.environ.get("PTXBENCH_ROOT", inferred_repo_root)).expanduser().resolve()
    mini_root = Path(os.environ.get("MINI_PTX_AGENT_ROOT", package_root)).expanduser().resolve()
    data_root = Path(os.environ.get("PTXBENCH_DATA_ROOT", repo_root / "data")).expanduser().resolve()
    multiturn_root = mini_root / "fib_runtime" / "multiturn"
    config_root = Path(os.environ.get("PTXBENCH_CONFIG_ROOT", repo_root / "configs")).expanduser().resolve()
    project = Path(
        os.environ.get(
            "PTXBENCH_FIXIT_PROJECT",
            data_root / "sft_experiments" / "test-fixit-qwen36-27b-gemini-glm",
        )
    ).expanduser().resolve()
    return PTXBenchPaths(
        repo_root=repo_root,
        mini_ptx_agent_root=mini_root,
        data_root=data_root,
        multiturn_root=multiturn_root,
        construct_eval_root=multiturn_root / "construct_eval_scripts",
        config_root=config_root,
        fixit_v6_project=project,
        eval_runs_root=data_root / "eval_runs",
    )

