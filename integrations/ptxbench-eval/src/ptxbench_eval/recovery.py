from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, local
from urllib.parse import quote

import requests

from .util import atomic_write_json, read_json, utc_now


class ProfileRecoveryError(RuntimeError):
    pass


class ProfileRequestInterrupted(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class ProfileRequirement:
    definition: str
    workload_uuid: str

    def as_dict(self) -> dict[str, str]:
        return {
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
        }


RecoveryCallback = Callable[[int, str, tuple[ProfileRequirement, ...]], None]


class ProfileRecoveryGate:
    """Coordinate one profiling-service recovery across concurrent evaluations."""

    def __init__(self, recover: RecoveryCallback):
        self._recover = recover
        self._condition = Condition()
        self._local = local()
        self._generation = 0
        self._recovering = False
        self._active_attempts = 0
        self._requirements: set[ProfileRequirement] = set()
        self._failures: dict[int, str] = {}
        self._last_error: str | None = None

    @contextlib.contextmanager
    def attempt(self, requirement: ProfileRequirement) -> Iterator[int]:
        with self._condition:
            while self._recovering:
                self._condition.wait()
            generation = self._generation
            failure = self._failures.get(generation)
            if failure is not None:
                raise ProfileRecoveryError(failure)
            self._requirements.add(requirement)
            self._active_attempts += 1
            previous_generation = getattr(self._local, "generation", None)
            self._local.generation = generation
        try:
            yield generation
        finally:
            if previous_generation is None:
                try:
                    del self._local.generation
                except AttributeError:
                    pass
            else:
                self._local.generation = previous_generation
            with self._condition:
                self._active_attempts -= 1
                self._condition.notify_all()

    def raise_if_failed(self) -> None:
        """Fail new evaluations while a failed recovery incident is latched."""

        with self._condition:
            failure = self._failures.get(self._generation)
            if failure is not None:
                raise ProfileRecoveryError(failure)

    def permit_profile_request(self) -> None:
        generation = getattr(self._local, "generation", None)
        if generation is None:
            return
        with self._condition:
            if self._recovering or generation != self._generation:
                raise ProfileRequestInterrupted(
                    "profiling request paused for service recovery"
                )

    def recover(self, observed_generation: int, reason: str) -> int:
        with self._condition:
            failure = self._failures.get(observed_generation)
            if failure is not None:
                raise ProfileRecoveryError(failure)
            if observed_generation != self._generation:
                return self._generation
            if self._recovering:
                while (
                    self._recovering
                    and observed_generation == self._generation
                ):
                    self._condition.wait()
                failure = self._failures.get(observed_generation)
                if failure is not None:
                    raise ProfileRecoveryError(failure)
                return self._generation

            self._recovering = True
            while self._active_attempts:
                self._condition.wait()
            requirements = tuple(sorted(self._requirements))

        try:
            self._recover(observed_generation, reason, requirements)
        except Exception as exc:
            message = f"profiling-service recovery failed: {exc}"
            with self._condition:
                self._failures[observed_generation] = message
                self._last_error = message
                self._recovering = False
                self._condition.notify_all()
            raise ProfileRecoveryError(message) from exc

        with self._condition:
            self._generation += 1
            self._requirements = set()
            self._last_error = None
            self._recovering = False
            self._condition.notify_all()
            return self._generation

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "enabled": True,
                "generation": self._generation,
                "recovering": self._recovering,
                "active_attempts": self._active_attempts,
                "last_error": self._last_error,
            }


class ProfileServiceReadiness:
    def __init__(self, *, profile_base_url: str, session: requests.Session | None = None):
        self.profile_base_url = profile_base_url.rstrip("/")
        self.session = session or requests.Session()

    def check(
        self,
        requirements: tuple[ProfileRequirement, ...],
        *,
        definitions: tuple[str, ...] = (),
        timeout: float = 10.0,
    ) -> None:
        try:
            health_response = self.session.get(
                f"{self.profile_base_url}/health", timeout=(5, timeout)
            )
            health_response.raise_for_status()
            health = health_response.json()
            if not isinstance(health, dict):
                raise TypeError("health response is not an object")
            backends = health.get("backends") or health.get("workers") or []
            healthy = (
                health.get("status") == "ok"
                and isinstance(backends, list)
                and bool(backends)
                and all(
                    isinstance(backend, dict) and backend.get("healthy") is True
                    for backend in backends
                )
                and int(health.get("queue_size") or 0) == 0
            )
            if not healthy:
                raise ValueError(f"profiling service is not ready: {health}")
            required_definitions = set(definitions)
            required_definitions.update(item.definition for item in requirements)
            for definition_name in sorted(required_definitions):
                definition = quote(definition_name, safe="")
                response = self.session.get(
                    f"{self.profile_base_url}/definitions/{definition}",
                    timeout=(5, timeout),
                )
                response.raise_for_status()
                definition_payload = response.json()
                if not isinstance(definition_payload, dict):
                    raise TypeError(
                        f"definition response for {definition_name} is not an object"
                    )
                if definition_payload.get("name") != definition_name:
                    raise ValueError(
                        "profiling service returned the wrong definition: "
                        f"expected {definition_name}, got "
                        f"{definition_payload.get('name')}"
                    )
            for requirement in requirements:
                definition = quote(requirement.definition, safe="")
                response = self.session.get(
                    f"{self.profile_base_url}/definitions/{definition}/workloads",
                    timeout=(5, timeout),
                )
                response.raise_for_status()
                workloads = response.json()
                if not isinstance(workloads, list):
                    raise TypeError(
                        f"workloads response for {requirement.definition} is not a list"
                    )
                available = {
                    str(item.get("uuid") or item.get("workload_uuid"))
                    for item in workloads
                    if isinstance(item, dict)
                    and (item.get("uuid") or item.get("workload_uuid")) is not None
                }
                if requirement.workload_uuid not in available:
                    raise ValueError(
                        f"profiling service lacks workload "
                        f"{requirement.workload_uuid} for {requirement.definition}"
                    )
        except (requests.RequestException, ValueError, TypeError) as exc:
            raise ProfileRecoveryError(str(exc)) from exc

    def wait(
        self,
        requirements: tuple[ProfileRequirement, ...],
        *,
        definitions: tuple[str, ...] = (),
        timeout: float,
        poll_interval: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "no readiness response"
        while time.monotonic() < deadline:
            try:
                self.check(
                    requirements,
                    definitions=definitions,
                    timeout=min(10, max(deadline - time.monotonic(), 0.1)),
                )
                return
            except ProfileRecoveryError as exc:
                last_error = str(exc)
            sleep(min(poll_interval, max(deadline - time.monotonic(), 0)))
        raise ProfileRecoveryError(
            f"profiling service did not become ready within {timeout:g} seconds: "
            f"{last_error}"
        )


class FileProfileRecoveryController:
    """Request a host-side restart and verify FIBServe before releasing the gate."""

    def __init__(
        self,
        *,
        control_dir: Path,
        profile_base_url: str,
        timeout: float = 720.0,
        poll_interval: float = 1.0,
        session: requests.Session | None = None,
    ):
        self.control_dir = control_dir.resolve()
        self.request_dir = self.control_dir / "requests"
        self.result_dir = self.control_dir / "results"
        self.profile_base_url = profile_base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session = session or requests.Session()
        self.readiness = ProfileServiceReadiness(
            profile_base_url=self.profile_base_url,
            session=self.session,
        )

    def __call__(
        self,
        generation: int,
        reason: str,
        requirements: tuple[ProfileRequirement, ...],
    ) -> None:
        request_id = str(uuid.uuid4())
        request_path = self.request_dir / f"{request_id}.json"
        result_path = self.result_dir / f"{request_id}.json"
        deadline = time.monotonic() + self.timeout
        atomic_write_json(
            request_path,
            {
                "schema_version": 1,
                "request_id": request_id,
                "generation": generation,
                "requested_at": utc_now(),
                "reason": reason,
                "requirements": [item.as_dict() for item in requirements],
            },
        )

        while time.monotonic() < deadline:
            result = read_json(result_path)
            if isinstance(result, dict) and result.get("request_id") == request_id:
                if result.get("ok") is not True:
                    detail = result.get("error") or result.get("output") or "unknown error"
                    raise ProfileRecoveryError(f"restart helper failed: {detail}")
                self._wait_until_ready(requirements, deadline)
                return
            time.sleep(min(self.poll_interval, max(deadline - time.monotonic(), 0)))
        raise ProfileRecoveryError(
            f"restart helper did not answer within {self.timeout:g} seconds"
        )

    def _wait_until_ready(
        self,
        requirements: tuple[ProfileRequirement, ...],
        deadline: float,
    ) -> None:
        remaining = max(deadline - time.monotonic(), 0.1)
        self.readiness.wait(
            requirements,
            timeout=remaining,
            poll_interval=self.poll_interval,
        )
