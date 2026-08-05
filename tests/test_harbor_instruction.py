from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = ROOT / "integrations" / "harbor" / "render_instruction.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("harbor_render_instruction", RENDERER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_renderer_matches_accrl_definition_encoding(monkeypatch) -> None:
    renderer = load_renderer()
    definition = {
        "name": "gemm_n7168_k5120",
        "op_type": "gemm",
        "reference": "def run(A, B):\n    return A @ B.T",
        "tags": ["status:reference"],
    }
    requested: list[tuple[str, float]] = []

    def fake_urlopen(url: str, timeout: float):
        requested.append((url, timeout))
        return io.BytesIO(json.dumps(definition).encode())

    monkeypatch.setattr(renderer, "urlopen", fake_urlopen)
    loaded = renderer.load_definition(
        "http://profile.example:11000/",
        "gemm_n7168_k5120",
        3.0,
    )
    rendered = renderer.render_instruction("Task:\n{task_content}\n", loaded)

    assert requested == [
        ("http://profile.example:11000/definitions/gemm_n7168_k5120", 3.0)
    ]
    assert rendered == "Task:\n" + json.dumps(
        {
            "name": "gemm_n7168_k5120",
            "op_type": "gemm",
            "reference": "def run(A, B):\n    return A @ B.T",
        },
        indent=2,
    ) + "\n"
    assert loaded["tags"] == ["status:reference"]


def test_checked_in_harbor_template_has_one_task_marker() -> None:
    renderer = load_renderer()
    template = (
        ROOT
        / "integrations"
        / "harbor"
        / "tasks"
        / "gemm_n7168_k5120"
        / "instruction.template.md"
    ).read_text()

    assert template.count(renderer.TASK_CONTENT_MARKER) == 1
