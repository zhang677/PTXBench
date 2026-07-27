from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = ROOT / "packages" / "mini-ptx-agent"
CONSTRUCT_ROOT = MINI_ROOT / "fib_runtime" / "multiturn" / "construct_eval_scripts"
FIXIT_ROOT = CONSTRUCT_ROOT / "fixit-v6-scripts"


def test_active_fixit_v6_sources_have_no_legacy_absolute_roots() -> None:
    paths = [
        *FIXIT_ROOT.glob("*.sh"),
        CONSTRUCT_ROOT / "fixit_downstream_process.py",
        CONSTRUCT_ROOT / "watch_eval_common.sh",
        MINI_ROOT / "fib_runtime" / "multiturn" / "run_parallel_v2.py",
        MINI_ROOT / "fib_runtime" / "multiturn" / "common.py",
        MINI_ROOT / "accrl" / "distill" / "inspector.py",
    ]
    legacy_roots = ("/home/ubuntu/AccRL", "/home/ubuntu/AccRL-exps")
    for path in paths:
        text = path.read_text()
        for root in legacy_roots:
            assert root not in text, f"{path.relative_to(ROOT)} still contains {root}"


def test_fixit_v6_static_preflight() -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "smoke_fixit_v6.sh"), "--check"],
        cwd=ROOT,
        env={
            **os.environ,
            "PTXBENCH_ROOT": str(ROOT),
            "MINI_PTX_AGENT_ROOT": str(MINI_ROOT),
            "PTXBENCH_DATA_ROOT": str(ROOT / "data"),
        },
        check=True,
    )


def test_path_resolver_uses_repo_layout() -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from mini_ptx_agent.paths import resolve_paths

    paths = resolve_paths()
    assert paths.repo_root == ROOT
    assert paths.mini_ptx_agent_root == MINI_ROOT
    assert paths.multiturn_root == MINI_ROOT / "fib_runtime" / "multiturn"
    assert paths.config_root == ROOT / "configs"
