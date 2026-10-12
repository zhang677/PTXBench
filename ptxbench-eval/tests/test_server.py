import concurrent.futures
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ptxbench_eval import server


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("fail", [False, True])
def test_complete_evaluations_hold_slots_until_success_or_failure(limit, fail):
    started = [threading.Event() for _ in range(limit + 1)]
    second_stage = [threading.Event() for _ in range(limit + 1)]
    finish_first_stage = threading.Event()
    finish_evaluation = threading.Event()

    def evaluate(payload, token):
        index = payload["index"]
        started[index].set()
        assert finish_first_stage.wait(5)
        second_stage[index].set()
        assert finish_evaluation.wait(5)
        if fail:
            raise server.EvaluationInfrastructureError("profiling failed")
        return {"index": index}

    jobs = server.EvaluationJobs(SimpleNamespace(evaluate=evaluate), max_profiles=limit)
    try:
        futures = [jobs.submit({"index": i}, "token") for i in range(limit + 1)]
        for event in started[:limit]:
            assert event.wait(5)
        assert not started[limit].wait(0.1)
        finish_first_stage.set()
        for event in second_stage[:limit]:
            assert event.wait(5)
        # Finishing an inner stage must not admit another complete evaluation.
        assert not started[limit].wait(0.1)
        assert all(not future.done() for future in futures)
        finish_evaluation.set()
        for i, future in enumerate(futures):
            if fail:
                with pytest.raises(server.EvaluationInfrastructureError, match="profiling failed"):
                    future.result(timeout=5)
            else:
                assert future.result(timeout=5) == {"index": i}
        assert started[limit].is_set()
    finally:
        finish_first_stage.set()
        finish_evaluation.set()
        jobs.shutdown()


@pytest.mark.parametrize("limit", [None, 1, 2])
def test_http_evaluation_stays_synchronous_and_health_reports_limit(tmp_path, limit):
    started = threading.Event()
    release = threading.Event()

    def evaluate(payload, token):
        assert token == "test-token"
        started.set()
        assert release.wait(5)
        return {"result": "completed"}

    evaluator = SimpleNamespace(evaluate=evaluate, recovery_gate=None, feedback_style="default")
    app = server.create_app(
        root=str(tmp_path), profile_base_url="http://profile.test",
        evaluator=evaluator, max_profiles=limit,
    )
    with TestClient(app) as client, concurrent.futures.ThreadPoolExecutor() as pool:
        try:
            assert client.get("/health").json()["max_profiles"] == limit
            response = pool.submit(
                client.post, "/v1/evaluate", json={},
                headers={"Authorization": "Bearer test-token"},
            )
            assert started.wait(5)
            assert not response.done()
            release.set()
            result = response.result(timeout=5)
            assert result.status_code == 200
            assert result.json() == {"result": "completed"}
        finally:
            release.set()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_limit_rejected(limit):
    with pytest.raises(ValueError, match="max_profiles must be a positive integer"):
        server.EvaluationJobs(SimpleNamespace(), max_profiles=limit)


def test_cli_passes_limit_to_gateway(tmp_path, monkeypatch):
    def run(app, **kwargs):
        jobs = app.state.evaluation_jobs
        try:
            assert jobs.max_profiles == 2
        finally:
            jobs.shutdown()

    monkeypatch.setattr(server.uvicorn, "run", run)
    server.main(["--root", str(tmp_path), "--max-profiles", "2"])


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_cli_rejects_nonpositive_limit(tmp_path, limit):
    with pytest.raises(SystemExit) as exc:
        server.main(["--root", str(tmp_path), "--max-profiles", limit])
    assert exc.value.code == 2
