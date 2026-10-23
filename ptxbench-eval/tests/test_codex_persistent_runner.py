import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

import pytest

from ptxbench_eval import codex_persistent_runner as runner
from ptxbench_eval import launcher


@pytest.mark.parametrize('settings', [
    {'gpu_policy': 'other'}, {'max_wait_seconds': True}, {'initial_delay_seconds': 0},
    {'probe_timeout_seconds': float('nan')}, {'max_delay_seconds': 5}, {'unknown': 1},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        runner.CapacitySettings.from_config({'agent': 'codex-persistent', 'codex_persistent': settings})


@pytest.mark.parametrize('policy,expected', [
    ('retain', ['acquire', 'run', 'wait', 'heartbeat', 'run', 'release']),
    ('release', ['acquire', 'run', 'release', 'wait', 'acquire', 'run', 'release']),
])
def test_capacity_lifecycle(policy, expected):
    events = []
    statuses = iter([75, 0])
    def attempt():
        events.append('run')
        return next(statuses)
    def wait(config, heartbeat):
        events.append('wait')
        heartbeat()
    config = {'agent': 'codex-persistent', 'codex_persistent': {'gpu_policy': policy}}
    assert runner.supervise(config, acquire=lambda: events.append('acquire'),
        release=lambda: events.append('release'), run_attempt=attempt,
        heartbeat=lambda: events.append('heartbeat'), wait=wait) == 0
    assert events == expected


def test_failed_cleanup_prevents_reacquisition():
    calls = []
    def release():
        calls.append('release')
        raise RuntimeError('release not verified')
    with pytest.raises(RuntimeError, match='release not verified'):
        runner.supervise({'agent': 'codex-persistent', 'codex_persistent': {'gpu_policy': 'release'}},
            acquire=lambda: calls.append('acquire'), release=release,
            run_attempt=lambda: 75, wait=lambda **kw: pytest.fail('must not probe'))
    assert calls.count('acquire') == 1


def test_non_capacity_errors_do_not_retry():
    calls = []
    assert runner.supervise({'agent': 'codex-persistent'},
        acquire=lambda: calls.append('acquire'), release=lambda: calls.append('release'),
        run_attempt=lambda: 1, wait=lambda **kw: pytest.fail('must not retry')) == 1
    assert calls == ['acquire', 'release']


def test_wait_backoff_and_retry_after(tmp_path, monkeypatch):
    clock = [0.0]
    times = []
    monkeypatch.setattr(runner.random, 'uniform', lambda *args: 1)
    runner.save_json(tmp_path / 'capacity.json', {'retry_after_seconds': 45})
    def probe(config):
        times.append(clock[0])
        return len(times) == 4
    runner.wait_for_capacity({'agent': 'codex-persistent', 'experiment_root': str(tmp_path)},
        probe=probe, sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        monotonic=lambda: clock[0])
    assert times == [45, 105, 225, 465]
    assert json.loads((tmp_path / 'capacity.json').read_text())['status'] == 'available'


def test_wait_deadline_keeps_checkpoint(tmp_path):
    with pytest.raises(TimeoutError):
        runner.wait_for_capacity({'agent': 'codex-persistent', 'experiment_root': str(tmp_path),
                                 'codex_persistent': {'max_wait_seconds': 1}},
                                probe=lambda c: pytest.fail('must not probe before requested delay'))
    assert json.loads((tmp_path / 'capacity.json').read_text())['status'] == 'waiting_deadline_exceeded'


FAKE_CODEX = r'''#!/usr/bin/env python3
import json, os, pathlib, sys
root = pathlib.Path(os.environ['CODEX_HOME'])
def send(value):
    print(json.dumps(value), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    method = request['method']
    with (root / 'requests.jsonl').open('a') as log:
        log.write(json.dumps(request) + '\n')
    if 'id' not in request:
        continue
    result = {}
    if method in ('thread/start', 'thread/resume'):
        assert request['params'].get('ephemeral') is not True
        if method == 'thread/resume':
            assert request['params']['threadId'] == 'saved-thread'
            assert 'serviceName' not in request['params']
        result = {'thread': {'id': 'saved-thread'}}
    if method == 'turn/start':
        result = {'turn': {'id': 'turn-1'}}
    send({'id': request['id'], 'result': result})
    if method == 'turn/start':
        failed = not (root / 'capacity-seen').exists()
        usage = 10 if failed else 20
        send({'method': 'thread/tokenUsage/updated', 'params': {
            'tokenUsage': {'total': {'totalTokens': usage, 'inputTokens': usage}}}})
        if failed:
            (root / 'capacity-seen').touch()
            send({'method': 'error', 'params': {'willRetry': False, 'error': {
                'codexErrorInfo': 'serverOverloaded', 'message': 'at capacity'}}})
        send({'method': 'turn/completed', 'params': {'turn': {
            'id': 'turn-1', 'status': 'failed' if failed else 'completed'}}})
'''


def test_real_runner_process_resumes_saved_thread_after_capacity(tmp_path):
    binary = tmp_path / 'codex'
    binary.write_text(FAKE_CODEX.replace('#!/usr/bin/env python3', '#!' + sys.executable))
    binary.chmod(0o755)
    state = tmp_path / 'state'
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'CODEX_HOME': str(state), 'PTXBENCH_CODEX_STATE_DIR': str(tmp_path / 'durable')}
    auth = json.dumps({'auth_mode': 'chatgpt', 'tokens': {'access_token': 'test-only'}})
    command = [sys.executable, '-m', 'ptxbench_eval.codex_persistent_runner',
               '--model', 'unchanged-model', '--prompt', 'original task', '--print-timeout', '10s']
    first = subprocess.run(command, input=auth, capture_output=True, text=True, env=env, timeout=15)
    assert first.returncode == 75, first.stdout + first.stderr
    checkpoint = json.loads((tmp_path / 'durable/session.json').read_text())
    assert checkpoint['thread_id'] == 'saved-thread'
    assert checkpoint['usage']['total_tokens'] == 10
    assert checkpoint['status'] == 'waiting_for_capacity'
    assert not (state / 'auth.json').exists()
    assert not list((tmp_path / 'durable').rglob('auth.json'))
    shutil.rmtree(state)  # Simulate the container tmpfs disappearing completely.
    second = subprocess.run(command, input=auth, capture_output=True, text=True, env=env, timeout=15)
    assert second.returncode == 0, second.stdout + second.stderr
    requests = [json.loads(line) for line in (state / 'requests.jsonl').read_text().splitlines()]
    assert [r['method'] for r in requests].count('thread/start') == 1
    assert [r['method'] for r in requests].count('thread/resume') == 1
    turns = [r for r in requests if r['method'] == 'turn/start']
    assert turns[0]['params']['input'][0]['text'] == 'original task'
    assert turns[1]['params']['input'][0]['text'] == runner.CONTINUE_PROMPT
    assert json.loads((tmp_path / 'durable/session.json').read_text())['usage']['total_tokens'] == 20
    changed = command.copy()
    changed[changed.index('--model') + 1] = 'another-model'
    third = subprocess.run(changed, input=auth, capture_output=True, text=True, env=env, timeout=15)
    assert third.returncode == 1
    assert 'does not match' in third.stdout


def test_agent_routing_and_runtime_mounts():
    config = {'agent': 'codex-persistent', 'model': 'model', 'serviceTier': 'priority',
              'codex_use_mcp': True, 'effort': 'xhigh'}
    argv = launcher._agent_argv(config, 'task')
    assert argv[0] == 'ptxbench-codex-persistent'
    assert '--evaluation-via-mcp' in argv
    assert argv[argv.index('--service-tier') + 1] == 'priority'
    assert launcher._agent_container_tmpfs_args('codex-persistent')
    assert launcher._agent_container_tmpfs_args('codex')


def test_config_identity_allows_only_lease_endpoints():
    a = {'agent': 'codex-persistent', 'model': 'm', 'profile_base_url': 'old',
         'profile_service': {'host': 'old', 'fib_devices': 'cuda:0'}}
    b = {**a, 'profile_base_url': 'new', 'profile_service': {'host': 'new', 'fib_devices': 'cuda:0'}}
    assert runner.config_identity(a) == runner.config_identity(b)
    b['model'] = 'other'
    assert runner.config_identity(a) != runner.config_identity(b)


def test_manual_stop_during_wait_releases_without_reacquisition():
    calls = []
    def stop(config, heartbeat):
        raise SystemExit(143)
    with pytest.raises(SystemExit):
        runner.supervise({'agent': 'codex-persistent'}, acquire=lambda: calls.append('acquire'),
                         release=lambda: calls.append('release'), run_attempt=lambda: 75, wait=stop)
    assert calls == ['acquire', 'release']


def test_goal_resume_preserves_original_objective_and_usage(tmp_path):
    fake = FAKE_CODEX.replace("if method == 'turn/start':", "if method in ('turn/start', 'thread/goal/set'):")
    fake = fake.replace("result = {'turn': {'id': 'turn-1'}}", "result = {'goal': {'status': 'active'}}")
    fake += "\n"
    fake = fake.replace("        send({'method': 'turn/completed'", "        if not failed:\n            send({'method': 'thread/goal/updated', 'params': {'goal': {'threadId': 'saved-thread', 'status': 'complete'}}})\n        send({'method': 'turn/completed'")
    binary = tmp_path / 'codex'
    binary.write_text(fake.replace('#!/usr/bin/env python3', '#!' + sys.executable))
    binary.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'CODEX_HOME': str(tmp_path / 'temporary'), 'PTXBENCH_CODEX_STATE_DIR': str(tmp_path / 'durable')}
    command = [sys.executable, '-m', 'ptxbench_eval.codex_persistent_runner', '--model', 'model',
               '--prompt', 'original objective', '--goal', '--print-timeout', '10s']
    auth = json.dumps({'auth_mode': 'chatgpt', 'tokens': {'access_token': 'test-only'}})
    for expected in (75, 0):
        result = subprocess.run(command, input=auth, capture_output=True, text=True, env=env, timeout=15)
        assert result.returncode == expected, result.stdout + result.stderr
    requests = [json.loads(line) for line in (tmp_path / 'temporary/requests.jsonl').read_text().splitlines()]
    goals = [r['params'] for r in requests if r['method'] == 'thread/goal/set']
    assert goals == [{'threadId': 'saved-thread', 'status': 'active', 'objective': 'original objective'},
                     {'threadId': 'saved-thread', 'status': 'active'}]
