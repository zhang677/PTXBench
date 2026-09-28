"""HTTP dispatcher for FlashInfer-Bench serve backends.

Fronts N backend servers (each running ``flashinfer_bench serve``) behind a single
endpoint. Submit endpoints (/evaluate, /profile, /sanitize, /debug) are routed
to the least-loaded backend by queue size, and the returned task_id is recorded
so that subsequent /tasks/{task_id} lookups go to the backend that owns the task.

Usage:
    python dispatcher.py --urls localhost:10000 localhost:10001
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request

logger = logging.getLogger("fib-dispatcher")


def _normalize_url(u: str) -> str:
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u.rstrip("/")


class Dispatcher:
    """Routes requests across backend servers and tracks task ownership."""

    def __init__(self, urls: List[str], request_timeout: float = 3600.0):
        self.urls = [_normalize_url(u) for u in urls]
        self.client = httpx.AsyncClient(
            timeout=request_timeout,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
        )
        self.health_client = httpx.AsyncClient(
            timeout=10.0, limits=httpx.Limits(max_connections=128, max_keepalive_connections=32)
        )
        self.task_to_url: Dict[str, str] = {}
        self._rr_idx = 0
        self._rr_lock = asyncio.Lock()

    async def close(self) -> None:
        await asyncio.gather(self.client.aclose(), self.health_client.aclose())

    async def queue_sizes(self) -> List[Tuple[Optional[int], str]]:
        """Return (queue_size, url) per backend. queue_size is None if unhealthy."""

        async def probe(url: str) -> Tuple[Optional[int], str]:
            try:
                r = await self.health_client.get(f"{url}/health", timeout=5.0)
                if r.status_code == 200:
                    body = r.json()
                    workers = body.get("workers")
                    if isinstance(workers, list) and workers:
                        if not any(bool(w.get("healthy")) for w in workers if isinstance(w, dict)):
                            return None, url
                    return body.get("queue_size", 0), url
            except Exception as e:
                logger.debug("health probe failed for %s: %s", url, e)
            return None, url

        return list(await asyncio.gather(*(probe(u) for u in self.urls)))

    async def pick_submit_backend(self) -> str:
        """Pick the least-loaded healthy backend; round-robin among ties."""
        sizes = await self.queue_sizes()
        healthy = [(s, u) for s, u in sizes if s is not None]
        if healthy:
            min_size = min(s for s, _ in healthy)
            candidates = [u for s, u in healthy if s == min_size]
            async with self._rr_lock:
                url = candidates[self._rr_idx % len(candidates)]
                self._rr_idx += 1
            return url
        async with self._rr_lock:
            url = self.urls[self._rr_idx % len(self.urls)]
            self._rr_idx += 1
        return url

    async def forward(self, method: str, url: str, path: str, **kwargs: Any) -> Tuple[int, Any]:
        r = await self.client.request(method, f"{url}{path}", **kwargs)
        try:
            body: Any = r.json()
        except ValueError:
            body = r.text
        return r.status_code, body

    async def try_each(self, method: str, path: str, **kwargs: Any) -> Tuple[int, Any]:
        """Try each backend in order until one returns a non-5xx response."""
        last: Tuple[int, Any] = (503, "no backends available")
        for url in self.urls:
            try:
                code, body = await self.forward(method, url, path, **kwargs)
                if code < 500:
                    return code, body
                last = (code, body)
            except Exception as e:
                last = (502, str(e))
        return last


_dispatcher: Optional[Dispatcher] = None


def _get_dispatcher() -> Dispatcher:
    if _dispatcher is None:
        raise RuntimeError("Dispatcher not initialized")
    return _dispatcher


def _raise(code: int, body: Any) -> None:
    detail = body["detail"] if isinstance(body, dict) and "detail" in body else body
    raise HTTPException(code, detail=detail)


def _raise_backend_error(url: str, path: str, exc: httpx.RequestError) -> None:
    if isinstance(exc, httpx.TimeoutException):
        status_code = 504
        detail = f"Backend timed out while handling {path}: {url}"
    else:
        status_code = 502
        detail = f"Backend request failed while handling {path}: {url}: {exc}"
    logger.warning("%s", detail)
    raise HTTPException(status_code, detail=detail) from exc


@asynccontextmanager
async def _lifespan(app):
    del app
    yield
    if _dispatcher is not None:
        await _dispatcher.close()


app = FastAPI(title="FlashInfer-Bench Dispatcher", lifespan=_lifespan)


@app.get("/")
async def root() -> Dict[str, Any]:
    d = _get_dispatcher()
    return {
        "name": "FlashInfer-Bench Dispatcher",
        "backends": d.urls,
        "tracked_tasks": len(d.task_to_url),
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    d = _get_dispatcher()
    sizes = await d.queue_sizes()
    backends = [{"url": u, "queue_size": s, "healthy": s is not None} for s, u in sizes]
    total = sum(s for s, _ in sizes if s is not None)
    any_healthy = any(s is not None for s, _ in sizes)
    return {
        "status": "ok" if any_healthy else "unhealthy",
        "backends": backends,
        "queue_size": total,
    }


async def _forward_any_get(path: str) -> Any:
    code, body = await _get_dispatcher().try_each("GET", path)
    if code >= 400:
        _raise(code, body)
    return body


@app.get("/definitions")
async def list_definitions() -> Any:
    return await _forward_any_get("/definitions")


@app.get("/definitions/{name}")
async def get_definition(name: str) -> Any:
    return await _forward_any_get(f"/definitions/{name}")


@app.get("/definitions/{name}/workloads")
async def list_workloads(name: str) -> Any:
    return await _forward_any_get(f"/definitions/{name}/workloads")


@app.get("/workloads/{uuid}")
async def get_workload(uuid: str) -> Any:
    return await _forward_any_get(f"/workloads/{uuid}")


async def _submit(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    d = _get_dispatcher()
    url = await d.pick_submit_backend()
    try:
        code, data = await d.forward("POST", url, path, json=body)
    except httpx.RequestError as e:
        _raise_backend_error(url, path, e)
    if code >= 400:
        _raise(code, data)
    if isinstance(data, dict) and "task_id" in data:
        d.task_to_url[data["task_id"]] = url
        logger.info("%s task_id=%s -> %s", path, data["task_id"], url)
    return data


@app.post("/evaluate")
async def evaluate(request: Request) -> Any:
    return await _submit("/evaluate", await request.json())


@app.post("/profile")
async def profile(request: Request) -> Any:
    return await _submit("/profile", await request.json())


@app.post("/sanitize")
async def sanitize(request: Request) -> Any:
    return await _submit("/sanitize", await request.json())


@app.post("/debug")
async def debug(request: Request) -> Any:
    return await _submit("/debug", await request.json())


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, timeout: float = Query(default=0, ge=0, le=3600)) -> Any:
    d = _get_dispatcher()
    url = d.task_to_url.get(task_id)
    if url is None:
        raise HTTPException(404, detail=f"Task not found: {task_id}")
    path = f"/tasks/{task_id}"
    try:
        code, data = await d.forward("GET", url, path, params={"timeout": timeout})
    except httpx.RequestError as e:
        _raise_backend_error(url, path, e)
    if code >= 400:
        _raise(code, data)
    return data


@app.post("/tasks/batch")
async def tasks_batch(request: Request) -> List[Any]:
    """Split the batch by owning backend, query in parallel, return original order."""
    d = _get_dispatcher()
    body = await request.json()
    task_ids: List[str] = body.get("task_ids", [])
    timeout: float = float(body.get("timeout", 0))

    by_url: Dict[str, List[str]] = {}
    for tid in task_ids:
        url = d.task_to_url.get(tid)
        if url is None:
            raise HTTPException(404, detail=f"Task not found: {tid}")
        by_url.setdefault(url, []).append(tid)

    async def fetch(url: str, ids: List[str]) -> List[Any]:
        try:
            code, data = await d.forward(
                "POST", url, "/tasks/batch", json={"task_ids": ids, "timeout": timeout}
            )
        except httpx.RequestError as e:
            _raise_backend_error(url, "/tasks/batch", e)
        if code >= 400:
            _raise(code, data)
        return data

    grouped = await asyncio.gather(*(fetch(u, ids) for u, ids in by_url.items()))
    flat: Dict[str, Any] = {}
    for ids, results in zip(by_url.values(), grouped):
        for tid, res in zip(ids, results):
            flat[tid] = res
    return [flat[t] for t in task_ids]


@app.post("/shutdown")
async def shutdown() -> Dict[str, Any]:
    """Shut down every backend."""
    d = _get_dispatcher()

    async def kill(url: str) -> None:
        try:
            await d.client.post(f"{url}/shutdown", timeout=5.0)
        except Exception as e:
            logger.warning("shutdown %s failed: %s", url, e)

    await asyncio.gather(*(kill(u) for u in d.urls))
    return {"status": "shutting_down"}


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashInfer-Bench serve dispatcher")
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="Backend server URLs (host:port or http://host:port)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    global _dispatcher
    _dispatcher = Dispatcher(args.urls)

    logger.info(
        "dispatcher listening on %s:%d, backends=%s", args.host, args.port, _dispatcher.urls
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
