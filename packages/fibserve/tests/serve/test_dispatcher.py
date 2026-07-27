"""Tests for the serve dispatcher API."""

from typing import Any, Dict, List

import pytest

from flashinfer_bench.serve import dispatcher as dispatcher_mod

try:
    from httpx import ASGITransport, AsyncClient
except ImportError:
    pytest.skip("httpx not installed", allow_module_level=True)

pytestmark = pytest.mark.asyncio


class FakeDispatcher:
    def __init__(self) -> None:
        self.urls = ["http://backend-a"]
        self.task_to_url: Dict[str, str] = {}
        self.forwarded: List[Dict[str, Any]] = []

    async def pick_submit_backend(self) -> str:
        return self.urls[0]

    async def forward(self, method: str, url: str, path: str, **kwargs: Any) -> tuple[int, Any]:
        self.forwarded.append({"method": method, "url": url, "path": path, "kwargs": kwargs})
        return 200, {"task_id": "debug-task", "normalized_solution_name": "solution_1"}

    async def close(self) -> None:
        pass


async def test_debug_submit_is_forwarded_and_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(dispatcher_mod, "_dispatcher", dispatcher)

    async with AsyncClient(
        transport=ASGITransport(app=dispatcher_mod.app), base_url="http://test"
    ) as client:
        resp = await client.post("/debug", json={"solution": {"name": "solution_1"}})

    assert resp.status_code == 200
    assert resp.json() == {"task_id": "debug-task", "normalized_solution_name": "solution_1"}
    assert dispatcher.forwarded == [
        {
            "method": "POST",
            "url": "http://backend-a",
            "path": "/debug",
            "kwargs": {"json": {"solution": {"name": "solution_1"}}},
        }
    ]
    assert dispatcher.task_to_url == {"debug-task": "http://backend-a"}
