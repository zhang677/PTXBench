from __future__ import annotations

import argparse
import concurrent.futures
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException

from .evaluator import EvaluationInfrastructureError, Evaluator
from .feedback import FEEDBACK_STYLES, validate_feedback_style
from .models import ContractError
from .recovery import FileProfileRecoveryController, ProfileRecoveryGate
from .registry import Registry


class EvaluationJobs:
    """Keep evaluations owned by the gateway after an HTTP client disconnects."""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="ptxbench-evaluation"
        )
        self.lock = threading.Lock()
        self.inflight: set[concurrent.futures.Future[dict[str, Any]]] = set()

    def submit(
        self, payload: dict[str, Any], token: str
    ) -> concurrent.futures.Future[dict[str, Any]]:
        future = self.executor.submit(self.evaluator.evaluate, payload, token)
        with self.lock:
            self.inflight.add(future)
        future.add_done_callback(self._forget)
        return future

    def _forget(
        self,
        completed: concurrent.futures.Future[dict[str, Any]],
    ) -> None:
        with self.lock:
            self.inflight.discard(completed)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


def create_app(
    *,
    root: str,
    profile_base_url: str,
    evaluator: Evaluator | None = None,
    feedback_style: str | None = None,
    profile_recovery_control_dir: str | None = None,
    profile_recovery_timeout: float = 720.0,
    profile_recovery_poll_interval: float = 1.0,
    max_profile_recoveries: int = 1,
) -> FastAPI:
    # None means the app setting was omitted; injected evaluators own their mode.
    if feedback_style is not None:
        feedback_style = validate_feedback_style(feedback_style)
    if evaluator is not None and feedback_style is not None:
        if feedback_style != evaluator.feedback_style:
            raise ValueError("app feedback_style differs from the injected evaluator")
    registry = Registry(root)
    if evaluator is None:
        recovery_gate = None
        if profile_recovery_control_dir:
            control_dir = Path(profile_recovery_control_dir).resolve()
            if not control_dir.is_relative_to(registry.root):
                raise ValueError("profile recovery control directory must be under root")
            controller = FileProfileRecoveryController(
                control_dir=control_dir,
                profile_base_url=profile_base_url,
                timeout=profile_recovery_timeout,
                poll_interval=profile_recovery_poll_interval,
            )
            recovery_gate = ProfileRecoveryGate(controller)
        evaluator = Evaluator(
            registry=registry,
            profile_base_url=profile_base_url,
            recovery_gate=recovery_gate,
            max_profile_recoveries=max_profile_recoveries,
            feedback_style=feedback_style if feedback_style is not None else "default",
        )
    recovery_gate = evaluator.recovery_gate
    evaluation_jobs = EvaluationJobs(evaluator)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            evaluation_jobs.shutdown()

    app = FastAPI(title="PTXBench evaluation gateway", version="1", lifespan=lifespan)
    app.state.evaluation_jobs = evaluation_jobs

    @app.get("/health")
    def health() -> dict[str, Any]:
        recovery = (
            recovery_gate.snapshot()
            if recovery_gate is not None
            else {"enabled": False}
        )
        return {
            "status": "ok",
            "profile_base_url": profile_base_url,
            "root": str(registry.root),
            "feedback_style": evaluator.feedback_style,
            "profile_recovery": recovery,
        }

    @app.post("/v1/evaluate")
    def evaluate(
        payload: dict[str, Any], authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer run token")
        try:
            future = evaluation_jobs.submit(
                payload, authorization.removeprefix("Bearer ").strip()
            )
            return future.result()
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ContractError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=f"usage checkpoint unavailable: {exc}") from exc
        except EvaluationInfrastructureError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ptxbench-eval-server")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--profile-base-url",
        default=os.environ.get("PROFILE_BASE_URL", "http://127.0.0.1:10000"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--feedback-style", choices=FEEDBACK_STYLES, default="default")
    parser.add_argument("--profile-recovery-control-dir")
    parser.add_argument("--profile-recovery-timeout", type=float, default=720.0)
    parser.add_argument("--profile-recovery-poll-interval", type=float, default=1.0)
    parser.add_argument("--max-profile-recoveries", type=int, default=1)
    args = parser.parse_args(argv)
    if args.profile_recovery_timeout <= 0 or args.profile_recovery_poll_interval <= 0:
        parser.error("profile recovery timeout and poll interval must be positive")
    if args.max_profile_recoveries < 0:
        parser.error("max profile recoveries must be non-negative")
    uvicorn.run(
        create_app(
            root=args.root,
            profile_base_url=args.profile_base_url,
            feedback_style=args.feedback_style,
            profile_recovery_control_dir=args.profile_recovery_control_dir,
            profile_recovery_timeout=args.profile_recovery_timeout,
            profile_recovery_poll_interval=args.profile_recovery_poll_interval,
            max_profile_recoveries=args.max_profile_recoveries,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
