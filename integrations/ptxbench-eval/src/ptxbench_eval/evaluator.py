from __future__ import annotations

import copy
import hashlib
from typing import Any

from .compiler import compile_source
from .events import RawEventLog
from .feedback import (
    finish_feedback,
    has_runtime_failure,
    prepend_log,
    render_debug_metadata,
    replace_runtime_logs,
    sanitizer_failure_status,
    sanitizer_traces,
    validate_feedback_style,
    validation_error,
)
from .fibserve import FibServeClient, FibServeError
from .models import ContractError, WorkloadManifest
from .recovery import ProfileRecoveryError, ProfileRecoveryGate, ProfileRequirement
from .registry import Registry
from .util import FileLockUnavailable, atomic_write_json, file_lock, read_json


class EvaluationInfrastructureError(RuntimeError):
    pass


class Evaluator:
    def __init__(
        self,
        *,
        registry: Registry,
        profile_base_url: str,
        client: FibServeClient | None = None,
        queue_timeout: int = 240,
        sanitizer_timeout: int = 120,
        max_check_runs: int = 2,
        max_replays: int = 2,
        max_profile_recoveries: int = 1,
        recovery_gate: ProfileRecoveryGate | None = None,
        usage_wait_timeout: float = 20.0,
        feedback_style: str = "default",
    ):
        self.feedback_style = validate_feedback_style(feedback_style)
        self.registry = registry
        self.client = client or FibServeClient(
            profile_base_url,
            request_permit=(
                recovery_gate.permit_profile_request if recovery_gate is not None else None
            ),
        )
        self.queue_timeout = queue_timeout
        self.sanitizer_timeout = sanitizer_timeout
        self.max_check_runs = max_check_runs
        self.max_replays = max_replays
        self.max_profile_recoveries = max_profile_recoveries
        self.recovery_gate = recovery_gate
        self.usage_wait_timeout = usage_wait_timeout

    def evaluate(self, payload: dict[str, Any], token: str) -> dict[str, Any]:
        run_id = str(payload.get("run_id") or "")
        action_id = str(payload.get("action_id") or "")
        if not run_id or not action_id:
            raise ContractError("run_id and action_id are required")
        source = payload.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ContractError("source must be a non-empty string")
        record = self.registry.authenticate(run_id, token)
        if "task_view" in payload:
            raise ContractError("task_view is controlled by the prepared run")
        if record.get("feedback_style", "default") != self.feedback_style:
            raise ContractError("gateway feedback_style differs from the prepared run")
        manifest = self.registry.validate_manifest(record, payload.get("workload"))
        if "feedback_style" in payload:
            raise ContractError("feedback_style is controlled by the gateway")
        forbidden = sorted({"compile", "validation_error"}.intersection(payload))
        if forbidden:
            raise ContractError(
                "client-supplied evaluation evidence is forbidden: " + ", ".join(forbidden)
            )
        log = RawEventLog(self.registry, run_id, action_id)
        result_path = self.registry.action_path(run_id, action_id)
        existing = read_json(result_path)
        if isinstance(existing, dict) and existing.get("state") == "completed":
            return self._replay_response(existing, log)
        if self.recovery_gate is not None:
            try:
                self.recovery_gate.raise_if_failed()
            except ProfileRecoveryError as exc:
                log.emit("profile_recovery_latched", error=str(exc))
                self._request_profile_stop_all(str(exc))
                raise EvaluationInfrastructureError(str(exc)) from exc
        try:
            with (
                file_lock(self.registry.run_action_lock_path(run_id), blocking=False),
                file_lock(self.registry.action_lock_path(run_id, action_id)),
            ):
                existing = read_json(result_path)
                if isinstance(existing, dict) and existing.get("state") == "completed":
                    return self._replay_response(existing, log)
                previous_checkpoint = self.registry.last_canonical_usage_checkpoint(
                    run_id
                )
                self.registry.begin_action(run_id, action_id)
                try:
                    log.emit("evaluation_admitted", feedback_style=self.feedback_style)
                    log.emit(
                        "evaluation_started",
                        manifest_digest=manifest.digest,
                        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        feedback_style=self.feedback_style,
                    )
                    try:
                        usage_snapshot = self.registry.wait_for_new_usage(
                            run_id,
                            after_source_event=previous_checkpoint.get("source_event"),
                            after_updated_at=previous_checkpoint.get(
                                "checkpoint_updated_at"
                            ),
                            timeout=self.usage_wait_timeout,
                        )
                    except TimeoutError as exc:
                        current_snapshot = self.registry.usage_snapshot(run_id)
                        details = {
                            "scope": "evaluation_checkpoint",
                            "timeout_seconds": self.usage_wait_timeout,
                            "last_committed_source_event": previous_checkpoint.get(
                                "source_event"
                            ),
                            "last_committed_updated_at": previous_checkpoint.get(
                                "checkpoint_updated_at"
                            ),
                            "current_source_event": current_snapshot.get("source_event"),
                            "current_updated_at": current_snapshot.get("updated_at"),
                        }
                        log.emit("usage_checkpoint_stalled", **details)
                        self.registry.request_stop(
                            run_id,
                            reason="usage_stream_stalled",
                            details=details,
                        )
                        raise EvaluationInfrastructureError(str(exc)) from exc
                    log.emit("usage_checkpoint_acquired", usage_snapshot=usage_snapshot)
                    response = self._evaluate_bound(
                        record=record,
                        manifest=manifest,
                        payload=payload,
                        source=source,
                        log=log,
                        usage_snapshot=usage_snapshot,
                    )
                    atomic_write_json(
                        result_path,
                        {"state": "completed", "response": response},
                    )
                    log.emit("evaluation_committed", response=response)
                    return response
                except Exception as exc:
                    log.emit("evaluation_not_committed", error=repr(exc))
                    raise
                finally:
                    self.registry.end_action(run_id, action_id)
        except FileLockUnavailable as exc:
            log.emit("evaluation_rejected_busy")
            raise ContractError(
                "another evaluation is already in progress for this run; "
                "wait for its feedback before submitting again"
            ) from exc

    def _replay_response(self, existing: dict[str, Any], log: RawEventLog) -> dict[str, Any]:
        response = dict(existing["response"])
        if response.get("feedback_style", "default") != self.feedback_style:
            raise ContractError("saved action feedback_style differs from the gateway")
        response["idempotent_replay"] = True
        log.emit("idempotent_result_replayed", feedback_style=self.feedback_style)
        return response

    def _evaluate_bound(
        self,
        *,
        record: dict[str, Any],
        manifest: WorkloadManifest,
        payload: dict[str, Any],
        source: str,
        log: RawEventLog,
        usage_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        validation = validation_error(source)
        log.emit("source_validation_completed", rejected=bool(validation), feedback=validation)
        compile_result: dict[str, Any] = {
            "ok": False,
            "output": "",
            "isa_warning": "",
            "infrastructure_error": False,
        }
        if not validation:
            log.emit("compilation_started", nvcc_gencode=manifest.data["nvcc_gencode"])
            compile_result = compile_source(
                source,
                nvcc_gencode=manifest.data["nvcc_gencode"],
            )
            log.emit("compilation_completed", result=compile_result)
            if compile_result.get("infrastructure_error"):
                raise EvaluationInfrastructureError(
                    "controlled endpoint compiler unavailable: "
                    + str(compile_result.get("output") or "unknown compiler error")
                )
        compile_output = str(compile_result.get("output") or "")
        replay_count = 0
        recovery_count = 0
        if validation:
            feedback = str(validation)
            traces = None
            all_passed = False
            min_speedup = 0.0
        elif not compile_result.get("ok"):
            feedback = f"test.py failed (returncode 1):\n{compile_output}\nFailed to compile kernel"
            traces = None
            all_passed = False
            min_speedup = 0.0
        else:
            solution = manifest.solution(source)
            requirement = ProfileRequirement(
                definition=str(manifest.data["definition"]),
                workload_uuid=str(manifest.data["workload_uuid"]),
            )
            while True:
                generation: int | None = None
                try:
                    if self.recovery_gate is None:
                        traces = self._profile_once(solution, manifest, log)
                        self._validate_evaluate_traces(traces, manifest)
                    else:
                        with self.recovery_gate.attempt(requirement) as generation:
                            traces = self._profile_once(solution, manifest, log)
                            self._validate_evaluate_traces(traces, manifest)
                    break
                except FibServeError as exc:
                    log.emit(
                        "profile_anomaly_detected",
                        replay_count=replay_count,
                        recovery_count=recovery_count,
                        retryable=exc.retryable,
                        requires_restart=exc.requires_restart,
                        error=str(exc),
                    )
                    if exc.requires_restart and self.recovery_gate is not None:
                        if recovery_count >= self.max_profile_recoveries:
                            raise EvaluationInfrastructureError(str(exc)) from exc
                        assert generation is not None
                        log.emit(
                            "profile_recovery_requested",
                            generation=generation,
                            error=str(exc),
                        )
                        try:
                            restored_generation = self.recovery_gate.recover(
                                generation, str(exc)
                            )
                        except ProfileRecoveryError as recovery_exc:
                            log.emit(
                                "profile_recovery_failed",
                                generation=generation,
                                error=str(recovery_exc),
                            )
                            self._request_profile_stop_all(str(recovery_exc))
                            raise EvaluationInfrastructureError(
                                str(recovery_exc)
                            ) from recovery_exc
                        recovery_count += 1
                        log.emit(
                            "profile_recovery_completed",
                            previous_generation=generation,
                            generation=restored_generation,
                            recovery_count=recovery_count,
                        )
                        continue
                    if not exc.retryable or replay_count >= self.max_replays:
                        raise EvaluationInfrastructureError(str(exc)) from exc
                    replay_count += 1
                    log.emit("anomaly_replay_started", replay_count=replay_count)
                except ProfileRecoveryError as recovery_exc:
                    log.emit(
                        "profile_recovery_latched",
                        error=str(recovery_exc),
                    )
                    self._request_profile_stop_all(str(recovery_exc))
                    raise EvaluationInfrastructureError(
                        str(recovery_exc)
                    ) from recovery_exc
            prepend_log(traces, compile_output)
            prepend_log(traces, str(compile_result.get("isa_warning") or ""))
            feedback, all_passed, min_speedup = finish_feedback(
                traces, target_speedup=float(manifest.data.get("target_speedup", 0) or 0),
                feedback_style=self.feedback_style,
            )
        self.registry.append_turn(
            run_id=record["run_id"],
            action_id=str(payload["action_id"]),
            source=source,
            feedback=feedback,
            traces=traces,
            all_passed=all_passed,
            min_speedup=min_speedup,
            usage_snapshot=usage_snapshot,
            replay_count=replay_count,
            recovery_count=recovery_count,
        )
        return {
            "schema_version": 1,
            "run_id": record["run_id"],
            "action_id": payload["action_id"],
            "task_id": record["task_id"],
            "feedback": feedback,
            "feedback_style": self.feedback_style,
            # Full diagnostic evidence was saved above in canonical metadata and
            # raw profile events. JSON/file-exchange clients see the same detail
            # level as the formatted feedback, including on idempotent replay.
            "traces": traces if self.feedback_style == "default" else None,
            "all_passed": all_passed,
            "min_speedup": min_speedup,
            "returncode": 0 if all_passed else 1,
            "replay_count": replay_count,
            "recovery_count": recovery_count,
            "cumulative_usage_at_kernel_request": usage_snapshot["usage"],
            "idempotent_replay": False,
        }

    def _request_profile_stop_all(self, error: str) -> None:
        details = {"error": error}
        if self.recovery_gate is not None:
            details["recovery"] = self.recovery_gate.snapshot()
        for path in sorted((self.registry.root / "runs").glob("*.json")):
            self.registry.request_stop(
                path.stem,
                reason="profile_recovery_failed",
                details=details,
            )

    def _profile_once(
        self,
        solution: dict[str, Any],
        manifest: WorkloadManifest,
        log: RawEventLog,
    ) -> list[dict[str, Any]]:
        workload_uuid = manifest.data["workload_uuid"]
        sanitizer_status = None
        failing_logs: Any = None
        for index in range(self.max_check_runs):
            log.emit("sanitize_pass_started", pass_index=index)
            result = self.client.submit_and_poll(
                "sanitize",
                {
                    "solution": solution,
                    "workload_uuids": [workload_uuid],
                    "sanitizer_types": ["memcheck"],
                    "print_limit": 1,
                    "max_lines": None,
                    "sanitizer_timeout": self.sanitizer_timeout,
                },
                self.sanitizer_timeout + self.queue_timeout,
                log.emit,
            )
            if result.get("status") == "failed":
                raise FibServeError(
                    f"sanitize task failed: {result.get('error', 'unknown')}",
                    requires_restart=True,
                )
            sanitizer_status = sanitizer_failure_status(result.get("logs"))
            if sanitizer_status:
                failing_logs = copy.deepcopy(result.get("logs"))
                break
        if failing_logs is not None:
            traces = sanitizer_traces(failing_logs, sanitizer_status or "RUNTIME_ERROR")
            report = self._debug(solution, workload_uuid, sanitizer_status or "RUNTIME_ERROR", log)
            replace_runtime_logs(traces, report)
            return traces
        result = self.client.submit_and_poll(
            "evaluate",
            {"solution": solution, "workload_uuids": [workload_uuid]},
            self.queue_timeout,
            log.emit,
        )
        if result.get("status") == "failed":
            raise FibServeError(
                f"evaluate task failed: {result.get('error', 'unknown')}",
                requires_restart=True,
            )
        traces = copy.deepcopy(result.get("traces"))
        if not isinstance(traces, list):
            raise FibServeError("evaluate response has no traces")
        if has_runtime_failure(traces):
            statuses = {
                (trace.get("evaluation") or {}).get("status")
                for trace in traces
                if isinstance(trace, dict)
            }
            report = self._debug(
                solution,
                workload_uuid,
                "TIMEOUT" if "TIMEOUT" in statuses else "RUNTIME_ERROR",
                log,
            )
            replace_runtime_logs(traces, report)
        return traces

    def _debug(
        self,
        solution: dict[str, Any],
        workload_uuid: str,
        status: str,
        log: RawEventLog,
    ) -> str:
        try:
            result = self.client.submit_and_poll(
                "debug",
                {
                    "solution": solution,
                    "workload_uuids": [workload_uuid],
                    "sanitizer_types": ["memcheck"],
                    "timeout": self.sanitizer_timeout,
                    "max_lines": None,
                    "print_limit": 100,
                    "source_context_lines": 4,
                    "enable_coredump": status == "TIMEOUT",
                    "coredump_grace_seconds": 30,
                },
                self.sanitizer_timeout + self.queue_timeout,
                log.emit,
            )
        except FibServeError as exc:
            if exc.requires_restart:
                raise
            log.emit("debug_failed_original_log_retained", error=str(exc))
            return ""
        logs = result.get("logs") or []
        metadata = logs[0].get("metadata") if logs and isinstance(logs[0], dict) else None
        return render_debug_metadata(metadata)

    @staticmethod
    def _validate_evaluate_traces(
        traces: list[dict[str, Any]] | None, manifest: WorkloadManifest
    ) -> None:
        if traces is None:
            return
        expected_definition = manifest.data["definition"]
        expected_uuid = manifest.data["workload_uuid"]
        for trace in traces:
            if not isinstance(trace, dict):
                raise FibServeError("trace is not an object")
            definition = trace.get("definition")
            if definition is not None and definition != expected_definition:
                raise FibServeError(
                    f"trace definition mismatch: expected {expected_definition}, got {definition}"
                )
            workload = trace.get("workload") or {}
            uuid_value = workload.get("uuid") or workload.get("workload_uuid")
            if uuid_value is not None and uuid_value != expected_uuid:
                raise FibServeError(
                    f"trace workload mismatch: expected {expected_uuid}, got {uuid_value}"
                )
