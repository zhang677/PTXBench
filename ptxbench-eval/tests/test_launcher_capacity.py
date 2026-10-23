"""Exercise batch cancellation with real workers/registry and simulated Docker."""

import io
import json
import threading
from types import SimpleNamespace

import pytest

from ptxbench_eval import launcher
from ptxbench_eval.registry import Registry


def capacity_event(*, retry=False, code="serverOverloaded"):
    return {
        "event": "codex_app_server",
        "payload": {
            "method": "error",
            "params": {
                "willRetry": retry,
                "error": {
                    "codexErrorInfo": code,
                    "message": "Selected model is at capacity. Please try a different model.",
                },
            },
        },
    }


@pytest.mark.parametrize("event", [
    {},
    {"event": "codex_app_server", "payload": None},
    {"event": "codex_app_server", "payload": {"method": "error", "params": None}},
    {"event": "codex_app_server", "payload": {
        "method": "error", "params": {"willRetry": False, "error": "at capacity"},
    }},
    {"event": "non_json_output", "text": "Selected model is at capacity."},
    capacity_event(retry=True),
    capacity_event(retry=None),
    capacity_event(retry=0),
    capacity_event(code="other"),
])
def test_only_structured_terminal_capacity_errors_abort(event):
    assert launcher._codex_capacity_error(event) is None


class FakeAgent:
    def __init__(self, *, events=(), before=None, hang=False, returncode=0, late=False):
        self.pid = 123
        self.stdin = io.StringIO()
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.reaped = False
        self.hang = hang
        self.returncode = returncode
        self.late = late
        self.stdout = self.stream(events, before)

    def stream(self, events, before):
        self.started.set()
        if before is not None:
            before()
        for event in events:
            yield json.dumps(event) + "\n"
        if self.hang:
            assert self.stopped.wait(5), "active agent was never stopped"

    def wait(self):
        if self.hang:
            assert self.stopped.wait(5), "agent process was never stopped"
        self.reaped = True
        return self.returncode


class FakeWatcher:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.on_wait = lambda: None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.on_wait()
        self.returncode = 0
        return 0


@pytest.fixture
def batch(tmp_path, monkeypatch):
    registry = Registry(tmp_path / "experiment")
    manifest = {
        "schema_version": 1,
        "task_id": "task",
        "definition": "gemm",
        "workload_uuid": "workload",
        "language": "cuda",
        "binding": "tvm-ffi",
        "nvcc_gencode": "arch=compute_90a,code=sm_90a",
        "target_hardware": ["H100"],
        "definition_payload": {"name": "gemm", "reference": "def run(a): return a"},
    }
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode":"chatgpt"}')
    config = {
        "agent": "codex", "model": "test-model", "agent_image": "test-image",
        "codex_auth_file": str(auth), "language": "cuda", "gateway_url": "http://test",
        "max_parallel": 2, "runs": [],
    }
    agents = {}
    started = []
    stops = []
    watcher = FakeWatcher()
    state = SimpleNamespace(
        registry=registry, config=config, agents=agents, started=started, stops=stops,
        watcher=watcher, on_stop=lambda names: None,
    )

    def add(run_id, **kwargs):
        registry.create_run(
            run_id=run_id, manifest=manifest, agent_prompt="task", token_budget=1000,
            harness="codex", model="test-model", prompt_tag="test",
        )
        registry.prompt_path(run_id).write_text("task")
        workspace = registry.root / "workspaces" / run_id
        workspace.mkdir()
        (workspace / "kernel.cu").write_text("// saved kernel\n")
        config["runs"].append({"run_id": run_id})
        agents[run_id] = FakeAgent(**kwargs)
        return agents[run_id]

    def popen(command, **kwargs):
        if command[0] != "docker":
            assert "ptxbench_eval.token_watcher" in command
            return watcher
        name = command[command.index("--name") + 1]
        run_id = next(key for key in agents if launcher._container_name(key) == name)
        started.append(run_id)
        return agents[run_id]

    def stop(command, **kwargs):
        assert command[:4] == ["docker", "stop", "-t", "10"]
        names = command[4:]
        stops.append(names)
        state.on_stop(names)
        for run_id in list(started):
            agent = agents[run_id]
            if launcher._container_name(run_id) in names:
                if agent.late:
                    # Docker run was spawned, but its container did not exist
                    # when the first stop reached the daemon.
                    agent.late = False
                else:
                    agent.stopped.set()
        return SimpleNamespace(returncode=0)

    state.add = add
    monkeypatch.setattr(launcher, "_registry_for_launch", lambda config: registry)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.subprocess, "run", stop)
    return state


@pytest.mark.parametrize("late", [False, True])
def test_capacity_stops_active_and_queued_runs_preserving_results(batch, monkeypatch, late):
    active = batch.add("active", hang=True, returncode=143, late=late)

    def wait_for_active():
        assert active.started.wait(5)

    source = batch.add("source", events=[capacity_event()], before=wait_for_active, hang=True)
    batch.add("queued")
    batch.add("completed")
    batch.registry.update_run("completed", status="completed", best_speedup=2.1)
    completed = batch.registry.run_path("completed").read_bytes()
    saved = {
        path: path.read_bytes()
        for path in [
            batch.registry.trajectory_path(run_id)
            for run_id in batch.agents
        ] + list((batch.registry.root / "workspaces").glob("*/kernel.cu"))
    }
    exchange_stops = []

    def start_exchange(*args):
        stop = threading.Event()
        exchange_stops.append(stop)
        return stop, SimpleNamespace(join=lambda timeout: None)

    monkeypatch.setattr(launcher, "_uses_eval_exchange", lambda config: True)
    monkeypatch.setattr(launcher, "_start_eval_exchange", start_exchange)

    def watcher_reconcile():
        # Simulate the watcher reconciling a coincident turn-limit crossing.
        for run_id in ("source", "active"):
            batch.registry.update_run(run_id, status="turn_limit_exhausted")

    batch.watcher.on_wait = watcher_reconcile
    with pytest.raises(RuntimeError, match="Model at capacity for source"):
        launcher.launch(batch.config)

    assert set(batch.started) == {"source", "active"}
    assert source.reaped and active.reaped
    assert source.stopped.is_set() and active.stopped.is_set()
    assert batch.watcher.terminated
    assert len(exchange_stops) == 2 and all(stop.is_set() for stop in exchange_stops)
    record = batch.registry.get_run("source")
    assert record["status"] == "infra_failed"
    assert record["failure"]["kind"] == "model_capacity"
    assert record["failure"]["error"] == capacity_event()["payload"]["params"]["error"]
    assert record["process"]["ended_at"]
    for run_id in ("active", "queued"):
        record = batch.registry.get_run(run_id)
        assert record["status"] == "cancelled"
        assert record["failure"]["source_run_id"] == "source"
    assert batch.registry.run_path("completed").read_bytes() == completed
    assert all(path.read_bytes() == data for path, data in saved.items())
    events = [json.loads(line) for line in batch.registry.harness_events_path("source").read_text().splitlines()]
    assert events[0]["payload"] == capacity_event()
    if late:
        assert sum("ptxbench-active" in names for names in batch.stops) >= 2


@pytest.mark.parametrize("event,returncode", [
    (capacity_event(retry=True), 0),
    (capacity_event(retry=None), 1),
    (capacity_event(code="other"), 1),
])
def test_non_capacity_failure_and_retryable_error_do_not_cancel_batch(batch, event, returncode):
    batch.config["max_parallel"] = 1
    batch.add("first", events=[event], returncode=returncode)
    batch.add("queued")
    assert launcher.launch(batch.config) == returncode
    assert batch.started == ["first", "queued"]
    assert batch.registry.get_run("queued")["status"] == "completed"
    assert batch.registry.get_run("first")["status"] == ("failed" if returncode else "completed")
    assert not batch.stops
    assert not batch.watcher.terminated


def test_capacity_during_preparation_prevents_late_worker_start(batch, monkeypatch):
    preparing = threading.Event()
    release = threading.Event()
    validate = launcher._validate_agent_task

    def delayed_validate(registry, run_id, manifest, task_view):
        if run_id == "preparing":
            preparing.set()
            assert release.wait(5)
        validate(registry, run_id, manifest, task_view)

    def wait_for_preparation():
        assert preparing.wait(5)

    batch.add("source", events=[capacity_event()], before=wait_for_preparation, hang=True)
    batch.add("preparing")
    batch.add("queued")
    batch.on_stop = lambda names: release.set()
    monkeypatch.setattr(launcher, "_validate_agent_task", delayed_validate)
    with pytest.raises(RuntimeError, match="Model at capacity"):
        launcher.launch(batch.config)
    assert batch.started == ["source"]
    assert batch.registry.get_run("preparing")["status"] == "cancelled"
    assert batch.registry.get_run("queued")["status"] == "cancelled"


def test_capacity_does_not_start_next_run_with_one_worker(batch):
    batch.config["max_parallel"] = 1
    batch.add("source", events=[capacity_event()], hang=True)
    batch.add("queued")
    with pytest.raises(RuntimeError, match="Model at capacity"):
        launcher.launch(batch.config)
    assert batch.started == ["source"]
    assert batch.registry.get_run("queued")["status"] == "cancelled"


def test_capacity_cancels_siblings_before_originating_worker_finishes(batch):
    active = batch.add("active", hang=True)
    stopping_source = threading.Event()
    batch_stop = threading.Event()

    def wait_for_active():
        assert active.started.wait(5)

    batch.add("source", events=[capacity_event()], before=wait_for_active, hang=True)

    def slow_source_stop(names):
        if names == ["ptxbench-source"]:
            stopping_source.set()
            assert batch_stop.wait(5), "batch waited for the failed worker to finish"
        else:
            assert stopping_source.wait(5)
            batch_stop.set()

    batch.on_stop = slow_source_stop
    with pytest.raises(RuntimeError, match="Model at capacity"):
        launcher.launch(batch.config)
    assert active.stopped.is_set()
    assert batch.registry.get_run("active")["status"] == "cancelled"


def test_persistent_capacity_checkpoints_then_resumes_only_unfinished_runs(batch):
    batch.config['agent'] = 'codex-persistent'
    batch.config['max_parallel'] = 1
    error = capacity_event()['payload']['params']['error']
    batch.add('source', events=[capacity_event(), {'event': 'codex_capacity_checkpoint', 'error': error}], hang=True)
    batch.add('queued')
    batch.add('completed')
    batch.registry.update_run('completed', status='completed', best_speedup=2.1)
    completed = batch.registry.run_path('completed').read_bytes()
    assert launcher.launch(batch.config) == 75
    assert batch.started == ['source']
    assert batch.registry.get_run('source')['status'] == 'waiting_for_capacity'
    assert batch.registry.get_run('queued')['status'] == 'waiting_for_capacity'
    assert batch.registry.run_path('completed').read_bytes() == completed
    assert batch.registry.canonical_turn_count('source') == 0
    batch.agents['source'] = FakeAgent(events=[{'event': 'result', 'source': 'codex_app_server',
        'result': {'status': 'completed', 'usage': {'total_tokens': 25, 'input_tokens': 25}}}])
    batch.watcher.returncode = None
    assert launcher.launch(batch.config) == 0
    assert batch.started == ['source', 'source', 'queued']
    assert batch.registry.run_path('completed').read_bytes() == completed
    assert batch.registry.usage_snapshot('source')['usage']['total_tokens'] == 25
    events = [json.loads(line) for line in batch.registry.harness_events_path('source').read_text().splitlines()]
    assert [e['line_number'] for e in events] == [1, 2, 3]


def test_persistent_raw_capacity_does_not_stop_before_checkpoint(batch):
    batch.config['agent'] = 'codex-persistent'
    batch.add('source', events=[capacity_event()], returncode=1)
    assert launcher.launch(batch.config) == 1
    assert not batch.stops
    assert batch.registry.get_run('source')['status'] == 'failed'


def test_recovery_starts_one_run_before_restoring_parallelism(batch):
    batch.config['agent'] = 'codex-persistent'
    (batch.registry.root / 'capacity.json').write_text('{"status":"available"}')
    def before_output():
        assert batch.started == ['first']
    first = batch.add('first', events=[{'event': 'codex_model_available'}], before=before_output, hang=True)
    batch.add('second', before=lambda: first.stopped.set())
    assert launcher.launch(batch.config) == 0
    assert batch.started == ['first', 'second']


def test_first_resumed_non_capacity_failure_does_not_start_siblings(batch):
    batch.config['agent'] = 'codex-persistent'
    (batch.registry.root / 'capacity.json').write_text('{"status":"available"}')
    batch.add('first', returncode=1)
    batch.add('second')
    with pytest.raises(RuntimeError, match='first resumed Codex run failed'):
        launcher.launch(batch.config)
    assert batch.started == ['first']
