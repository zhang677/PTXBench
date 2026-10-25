"""Shared local lifecycle for GPU-owning Modal and Radix launchers."""
from __future__ import annotations

import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
import uuid


def execute(config, backend):
    from ptxbench_eval import launcher
    from ptxbench_eval.codex_persistent_runner import (
        CAPACITY_EXIT, CapacitySettings, save_json, supervise,
    )
    from ptxbench_eval.feedback import feedback_style_from_config
    from ptxbench_eval.registry import Registry, TERMINAL_STATES
    from run_modal_coding_agent import (
        build_images, check_gateway_port, docker_wrapper, output, run, wait_ready,
    )

    original = copy.deepcopy(config)
    root = Path(config['experiment_root'])
    persistent = config['agent'] == 'codex-persistent'
    settings = CapacitySettings.from_config(config)
    if root.exists():
        if not persistent or not (root / 'source_config.json').is_file():
            raise ValueError('existing experiment is not a resumable persistent run')
        if json.loads((root / 'source_config.json').read_text()) != original:
            raise ValueError('resume config differs from the saved source config')
    else:
        root.mkdir(parents=True)
        save_json(root / 'source_config.json', original)
    # Held across waits and reacquisition, including separate invocations.
    with (root / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved_runs = [root / 'runs' / (r['run_id'] + '.json') for r in config['runs']]
        if persistent and all(p.exists() for p in saved_runs):
            records = [json.loads(p.read_text()) for p in saved_runs]
            if all(r['status'] in TERMINAL_STATES for r in records):
                return int(any(r['status'] in {'failed', 'infra_failed', 'cancelled'} for r in records))
        gateway = 'ptxbench-gateway-' + re.sub(r'[^A-Za-z0-9_.-]', '-', root.name)
        names = {gateway, *(launcher._container_name(r['run_id']) for r in config['runs'])}
        conflicts = names.intersection(output('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines())
        if conflicts:
            raise RuntimeError(f'conflicting Docker containers: {sorted(conflicts)}')
        address = urlsplit(config['gateway_url'])
        check_gateway_port(address.hostname, address.port)
        build_images(config)
        if persistent:
            try:
                run('docker', 'run', '--rm', '--network', 'none', '--entrypoint',
                    'ptxbench-codex-persistent', config['agent_image'], '--help',
                    stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError('agent image lacks the persistent runner; rebuild with build_images=true') from exc
        process = None
        effective = None
        registry = None
        attempt = len(list((root / 'attempts').glob('*'))) if (root / 'attempts').exists() else 0
        label = 'ptxbench.lease-launch=' + uuid.uuid4().hex
        old_path = os.environ['PATH']
        with tempfile.TemporaryDirectory(prefix='ptxbench-lease-') as temporary:
            rootless = 'name=rootless' in output('docker', 'info', '--format', '{{json .SecurityOptions}}')
            docker_wrapper(Path(temporary), label, rootless)
            os.environ['PATH'] = temporary + os.pathsep + old_path

            def stop_local():
                nonlocal process
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    # Launcher cancellation stops the owned agents before exiting.
                    try:
                        process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                process = None
                ids = output('docker', 'ps', '-aq', '--filter', 'label=' + label).split()
                if ids:
                    try:
                        run('docker', 'stop', '-t', '10', *ids, timeout=45)
                        with (root / 'gateway.log').open('a') as log:
                            subprocess.run(['docker', 'logs', gateway], stdout=log, stderr=log, timeout=15)
                    finally:
                        run('docker', 'rm', '-f', *ids, timeout=45)
                if output('docker', 'ps', '-aq', '--filter', 'label=' + label).strip():
                    raise RuntimeError('owned containers remain after cleanup')

            def release():
                try:
                    stop_local()
                finally:
                    backend.release()

            def acquire():
                nonlocal effective, registry, attempt
                attempt += 1
                folder = root / 'attempts' / f'{attempt:04d}'
                folder.mkdir(parents=True)
                effective = backend.acquire(folder)
                save_json(folder / 'config.json', effective)
                # Validate all saved workload/prompt hashes on each new GPU.
                registry = launcher.prepare(effective)
                args = ['--max-profiles', str(config['max_profiles'])] if 'max_profiles' in config else []
                run('docker', 'run', '--detach', '--name', gateway, '--network', 'host',
                    '-v', f'{root}:{root}', config['gateway_image'], '--root', root,
                    '--profile-base-url', effective['profile_base_url'], '--host', address.hostname,
                    '--port', address.port, '--feedback-style', feedback_style_from_config(config), *args)
                wait_ready(config['gateway_url'], lambda h: (
                    h.get('status') == 'ok' and h.get('root') == str(root)
                    and h.get('profile_base_url') == effective['profile_base_url']
                    and h.get('feedback_style') == feedback_style_from_config(config)
                    and h.get('max_profiles') == config.get('max_profiles')),
                    lambda: output('docker', 'inspect', '-f', '{{.State.Running}}', gateway) == 'true', 60)

            def run_attempt():
                nonlocal process
                saved = root / 'attempts' / f'{attempt:04d}' / 'config.json'
                process = subprocess.Popen([sys.executable, '-m', 'ptxbench_eval.launcher',
                                            'launch', str(saved)], start_new_session=True)
                next_status = time.monotonic()
                while process.poll() is None:
                    backend.check()
                    if time.monotonic() >= next_status:
                        launcher.print_status(root)
                        next_status = time.monotonic() + config['status_interval_seconds']
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                status = process.returncode
                process = None
                if status == CAPACITY_EXIT and persistent:
                    # Agent processes are stopped; the gateway commits evaluations
                    # already submitted before any GPU is released.
                    deadline = time.monotonic() + settings.drain_timeout_seconds
                    while any(registry.get_run(r['run_id']).get('active_actions')
                              or registry.get_run(r['run_id']).get('active_action') for r in config['runs']):
                        backend.check()
                        if time.monotonic() >= deadline:
                            raise TimeoutError('evaluations did not drain; refusing automatic replay')
                        time.sleep(1)
                    for item in config['runs']:
                        run_id = item['run_id']
                        record = registry.get_run(run_id)
                        if record['status'] in TERMINAL_STATES:
                            continue
                        started_at = (record.get('process') or {}).get('started_at')
                        if started_at:
                            snapshot = root / 'codex_home' / run_id / 'persistent' / 'snapshot.json'
                            if (not snapshot.is_file() or json.loads(snapshot.read_text())['saved_at']
                                    < datetime.fromisoformat(started_at).timestamp()):
                                raise RuntimeError(f'{run_id}: interrupted session was not checkpointed; refusing replay')
                        trajectory = json.loads(registry.trajectory_path(run_id).read_text())
                        feedback = [m for m in trajectory['messages'] if m['role'] == 'user'
                                    and (m.get('extra') or {}).get('raw_output') is not None]
                        save_json(root / 'workspaces' / run_id / '.ptxbench-resume.json', {
                            'completed_evaluations': registry.canonical_turn_count(run_id),
                            'token_usage': registry.usage_snapshot(run_id)['usage'],
                            'latest_feedback': feedback[-1]['content'] if feedback else None,
                        })
                return status

            try:
                return supervise(config, acquire=acquire, release=release,
                                 run_attempt=run_attempt, heartbeat=backend.check)
            finally:
                os.environ['PATH'] = old_path


def install_signal_handlers():
    def interrupted(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
