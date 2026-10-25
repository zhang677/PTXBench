#!/usr/bin/env python3
"""Own one Modal profiling sandbox and one resumable local runner attempt.

Use the acc environment. Arguments after -- are executed without a shell;
their one __MODAL_PROFILE_URL__ token is replaced after service readiness.
Exit 75 requests infrastructure recovery, 2 requires configuration/operator
attention, and other exit codes are the child's status (130/143 on signals).
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen
import uuid

from modal_profile_runtime import collect_reference_records, GPU_ARCH, MODAL_DOCKERFILE, SANDBOX_START, modal_base_image, modal_environment

HERE = Path(__file__).resolve().parent
MULTITURN = HERE.parent
URL_TOKEN = '__MODAL_PROFILE_URL__'
RETRY = 75


class ConfigurationError(RuntimeError):
    pass


class AlreadyRunning(ConfigurationError):
    pass


class InfrastructureError(RuntimeError):
    pass


def log(message):
    print(f'{datetime.now(timezone.utc).isoformat()} [modal-profile] {message}', flush=True)


def write_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def healthy(data, *, idle=False):
    if not isinstance(data, dict):
        return False
    workers = data.get('workers') or data.get('backends') or []
    return (data.get('status') == 'ok' and len(workers) == 1
            and isinstance(workers[0], dict) and workers[0].get('healthy') is True
            and (not idle or data.get('queue_size') == 0))


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-name', required=True)
    parser.add_argument('--gpu', choices=GPU_ARCH, required=True)
    parser.add_argument('--lock-gdn-bwd-reference', action='store_true',
                        help='pin the validated GDN backward reference configuration on B200')
    parser.add_argument('--cpu', type=positive, required=True)
    parser.add_argument('--memory-mib', type=positive, required=True)
    parser.add_argument('--timeout-seconds', type=positive, required=True)
    parser.add_argument('--ready-timeout-seconds', type=positive, required=True)
    parser.add_argument('--poll-interval-seconds', type=positive, required=True)
    parser.add_argument('--health-failure-threshold', type=positive, required=True)
    parser.add_argument('--drain-timeout-seconds', type=positive, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--working-dir', type=Path, required=True)
    parser.add_argument('--definitions', nargs='+', required=True)
    parser.add_argument('--output-roots', nargs='+', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ['--']:
        args.command.pop(0)
    return args


def validate(args):
    try:
        modal_environment(args.gpu, getattr(args, 'lock_gdn_bwd_reference', False))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if args.command.count(URL_TOKEN) != 1:
        raise ConfigurationError(f'child argv must contain exactly one {URL_TOKEN}')
    if any(arg.startswith('PROFILE_BASE_URL=') for arg in args.command):
        raise ConfigurationError('child argv must not override PROFILE_BASE_URL')
    if not 0 < args.ready_timeout_seconds <= args.timeout_seconds <= 86400:
        raise ConfigurationError('require readiness timeout <= sandbox timeout <= 86400')
    for name in ('state_dir', 'working_dir'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_roots = [p.expanduser().resolve() for p in args.output_roots]
    if not args.working_dir.is_dir():
        raise ConfigurationError(f'missing working directory: {args.working_dir}')
    for path in [MULTITURN / 'run_parallel_v2.py']:
        if not path.is_file():
            raise ConfigurationError(f'missing required input: {path}')
    if len(set(args.output_roots)) != len(args.output_roots):
        raise ConfigurationError('output roots must be distinct')
    for root in args.output_roots:
        if root == Path('/') or root == args.state_dir or root in args.state_dir.parents:
            raise ConfigurationError('state directory must be outside evaluation roots')
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]+', d) for d in args.definitions):
        raise ConfigurationError('invalid definition name')
    if not shutil.which('docker') or not shutil.which(args.command[0]):
        raise ConfigurationError('Docker and the child executable must be available')


@contextmanager
def run_lock(state_dir, name='run.lock'):
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / name).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AlreadyRunning(f'another supervisor owns {state_dir}') from None
        yield


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True, stderr=subprocess.PIPE, timeout=30).strip()


def remove_containers(label):
    ids = docker('ps', '-aq', '--filter', 'label=' + label).split()
    if ids:
        docker('rm', '-f', *ids)


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def owned_runner_group(state):
    pid = state.get('runner_pid')
    if not pid:
        return None
    if state.get('runner_start') is not None and process_start(pid) == state['runner_start']:
        return pid
    # The group leader may have died while a run_v2 child was still querying.
    # Verify the attempt token on a surviving member before signalling its group.
    token = ('PTXBENCH_MODAL_ATTEMPT=' + state['attempt']).encode()
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if os.getpgid(int(entry.name)) == pid and token in (entry / 'environ').read_bytes().split(b'\0'):
                return pid
        except (OSError, ValueError):
            continue
    return None


def recover_orphan(args, modal):
    """The lock is held; only resources recorded by the prior owner are touched."""
    path = args.state_dir / 'active.json'
    if not path.exists():
        return
    state = json.loads(path.read_text())
    log(f'recovering resources from attempt {state["attempt"]}')
    errors = []
    try:
        pid = owned_runner_group(state)
        if pid:
            signal_group(pid, signal.SIGTERM)
            time.sleep(1)
            signal_group(pid, signal.SIGKILL)
        remove_containers(state['container_label'])
    except Exception as exc:
        errors.append(str(exc))
    try:
        if state.get('sandbox_id'):
            sandbox = modal.Sandbox.from_id(state['sandbox_id'])
        else:
            sandbox = modal.Sandbox.from_name(state['app_name'], state['sandbox_name'])
        sandbox.terminate(wait=True)
    except modal.exception.NotFoundError:
        pass
    except Exception as exc:
        errors.append(str(exc))
    if errors:
        raise ConfigurationError('prior attempt cleanup failed: ' + '; '.join(errors))
    path.unlink()


def write_docker_wrapper(directory, label, rootless):
    binary = shutil.which('docker')
    command = [binary, 'run', '--label', label]
    if rootless:
        command += ['--user', '0:0']
    wrapper = directory / 'docker'
    wrapper.write_text('#!/bin/bash\nif [[ "${1:-}" == run ]]; then\n shift\n exec '
                       + shlex.join(command) + ' "$@"\nfi\nexec ' + shlex.quote(binary) + ' "$@"\n')
    wrapper.chmod(0o700)


def preflight_runner(args):
    if not any(Path(arg).name == 'run_parallel_v2.py' for arg in args.command):
        return
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([*args.command, '--help'], cwd=args.working_dir,
                            env=environment, timeout=120, stdout=subprocess.DEVNULL)
    if result.returncode:
        raise ConfigurationError('runner import preflight failed; no GPU was allocated')
    image = (args.command[args.command.index('--image') + 1]
             if '--image' in args.command else 'mini-swe-eval:latest')
    docker('image', 'inspect', image)


class Attempt:
    def __init__(self, args, modal, interrupted):
        self.args, self.modal, self.interrupted = args, modal, interrupted
        self.name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12]
        self.directory = args.state_dir / self.name
        self.directory.mkdir(mode=0o700)
        self.active_path = args.state_dir / 'active.json'
        self.stop_file = self.directory / 'stop'
        self.failure_file = self.directory / 'profile_failure.json'
        self.sandbox = self.process = None
        self.readers = []
        self.url = None
        self.metadata = {
            'attempt': self.name, 'app_name': args.app_name,
            'sandbox_name': 'accrl-' + self.name,
            'container_label': 'accrl.modal-attempt=' + self.name,
            'gpu': args.gpu, 'cpu': args.cpu, 'memory_mib': args.memory_mib,
            'timeout_seconds': args.timeout_seconds,
            'ready_timeout_seconds': args.ready_timeout_seconds,
            'poll_interval_seconds': args.poll_interval_seconds,
            'health_failure_threshold': args.health_failure_threshold,
            'drain_timeout_seconds': args.drain_timeout_seconds,
            'definitions': args.definitions, 'output_roots': list(map(str, args.output_roots)),
            'working_dir': str(args.working_dir),
            'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'startup_sha256': hashlib.sha256(SANDBOX_START.encode()).hexdigest(),
        }
        if getattr(args, 'experiment_config', None) is not None:
            config_path = self.directory / 'experiment_config.json'
            write_json(config_path, args.experiment_config)
            self.metadata['experiment_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()

    def save(self):
        write_json(self.directory / 'modal.json', self.metadata)
        write_json(self.active_path, self.metadata)

    def check_interrupted(self):
        if self.interrupted:
            raise SystemExit(128 + self.interrupted[0])

    def stream(self, stream, path):
        try:
            with path.open('w') as target:
                for chunk in stream:
                    target.write(chunk)
                    target.flush()
        except Exception as exc:
            log(f'log stream interrupted: {exc}')

    def startup_failure(self, status):
        for reader in self.readers:
            reader.join(timeout=5)
        logs = '\n'.join(p.read_text(errors='replace') for p in self.directory.glob('modal.*.log'))
        # These are deterministic PTXBench input/setup errors, not candidate errors.
        fatal = ('has SHA-256', 'entrypoint changed:', 'unexpected workload',
                 'unexpected tensor keys', 'unexpected input layout', 'unexpected blob path',
                 '[entrypoint] ERROR:', 'MODAL_PROFILE_FATAL:')
        error = ConfigurationError if any(marker in logs for marker in fatal) else InfrastructureError
        raise error(f'profiler exited during preparation (rc={status}); see {self.directory}')

    def start(self):
        args, modal = self.args, self.modal
        deadline = time.monotonic() + args.ready_timeout_seconds
        self.metadata['dockerfile_sha256'] = hashlib.sha256(MODAL_DOCKERFILE.encode()).hexdigest()
        self.save()  # Name is durable even if creation is interrupted before returning an ID.
        self.check_interrupted()
        image = modal_base_image(modal)
        app = modal.App.lookup(args.app_name, create_if_missing=True)
        log(f'requesting {args.gpu} sandbox {self.metadata["sandbox_name"]}; '
            f'allocation/startup budget={args.ready_timeout_seconds}s')
        async def create():
            task = asyncio.ensure_future(modal.Sandbox.create.aio(
                'bash', '-c', SANDBOX_START, app=app, name=self.metadata['sandbox_name'],
                image=image, gpu=args.gpu, cpu=args.cpu, memory=args.memory_mib,
                timeout=args.timeout_seconds, encrypted_ports=[10000],
                env=modal_environment(args.gpu, getattr(args, 'lock_gdn_bwd_reference', False))))
            try:
                while not task.done():
                    self.check_interrupted()
                    if time.monotonic() >= deadline:
                        raise InfrastructureError('sandbox allocation/readiness deadline exceeded')
                    await asyncio.wait({task}, timeout=0.2)
                return task.result()
            except BaseException:
                # Modal's create coroutine terminates a sandbox cancelled while
                # scheduling. Named cleanup below also covers ambiguous RPCs.
                task.cancel()
                try:
                    self.sandbox = await task
                except BaseException:
                    pass
                raise
        with modal.enable_output():
            self.sandbox = asyncio.run(create())
        self.metadata['sandbox_id'] = self.sandbox.object_id
        self.save()
        log(f'sandbox={self.sandbox.object_id}; preparation logs={self.directory}')
        for stream, name in [(self.sandbox.stdout, 'modal.stdout.log'),
                             (self.sandbox.stderr, 'modal.stderr.log')]:
            reader = threading.Thread(target=self.stream, args=(stream, self.directory / name), daemon=True)
            reader.start()
            self.readers.append(reader)
        self.check_interrupted()
        self.url = self.sandbox.tunnels(timeout=min(50, args.ready_timeout_seconds))[10000].url.rstrip('/')
        self.metadata['profile_base_url'] = self.url
        self.save()
        last_error = 'no healthy worker'
        while time.monotonic() < deadline:
            self.check_interrupted()
            status = self.sandbox.poll()
            if status is not None:
                self.startup_failure(status)
            try:
                if healthy(read_json(self.url + '/health'), idle=True):
                    for definition in args.definitions:
                        try:
                            read_json(self.url + '/definitions/' + quote(definition, safe=''))
                        except HTTPError as exc:
                            if exc.code == 404:
                                raise ConfigurationError(f'definition missing from sandbox: {definition}') from exc
                            raise
                    break
            except (OSError, ValueError) as exc:
                last_error = str(exc)
            self.interruptible_wait(min(2, args.poll_interval_seconds))
        else:
            raise InfrastructureError(f'profiler readiness timed out: {last_error}')
        probe = self.sandbox.exec('cat', '/workspace/modal-runtime.json', timeout=30)
        runtime = json.loads(probe.stdout.read())
        probe.wait()
        if probe.returncode:
            raise InfrastructureError('could not collect sandbox runtime provenance')
        self.metadata['runtime'] = runtime
        self.save()
        log(f'profiling service ready: {self.url}')

    def interruptible_wait(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check_interrupted()
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))

    def launch(self):
        rootless = 'name=rootless' in docker('info', '--format', '{{json .SecurityOptions}}')
        directory = self.directory / 'bin'
        directory.mkdir()
        write_docker_wrapper(directory, self.metadata['container_label'], rootless)
        command = [self.url if arg == URL_TOKEN else arg for arg in self.args.command]
        environment = dict(os.environ)
        environment.update(
            PROFILE_BASE_URL=self.url, PTXBENCH_ANSWER_DRAIN_GUARD='1',
            PTXBENCH_MODAL_ATTEMPT=self.name,
            PTXBENCH_PROFILE_CONFLICT_STOP_FILE=str(self.stop_file),
            PATH=str(directory) + os.pathsep + environment['PATH'],
            PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        )
        # The argv contains experiment options only; never serialize the environment.
        write_json(self.directory / 'runner_argv.json', command)
        self.check_interrupted()
        self.process = subprocess.Popen(command, cwd=self.args.working_dir, env=environment,
                                        start_new_session=True)
        self.metadata.update(runner_pid=self.process.pid, runner_start=process_start(self.process.pid))
        self.save()

    def monitor(self):
        failures, next_probe = 0, 0
        while True:
            self.check_interrupted()
            # Check before child exit: a worker can publish its failure and exit
            # between polls even when the profiling service's /health is green.
            if self.failure_file.exists():
                raise InfrastructureError('runner reported profiling infrastructure failure')
            status = self.process.poll()
            if status is not None:
                return status if status >= 0 else 128 - status
            if time.monotonic() >= next_probe:
                try:
                    status = self.sandbox.poll()
                    if status is not None:
                        raise InfrastructureError(f'sandbox exited rc={status}')
                    if healthy(read_json(self.url + '/health')):
                        failures = 0
                    else:
                        failures += 1
                except InfrastructureError:
                    raise
                except Exception as exc:
                    failures += 1
                    log(f'health probe failed ({failures}/{self.args.health_failure_threshold}): {type(exc).__name__}')
                if failures >= self.args.health_failure_threshold:
                    raise InfrastructureError(f'profiling service unhealthy for {failures} consecutive probes')
                next_probe = time.monotonic() + self.args.poll_interval_seconds
            self.interruptible_wait(0.2)

    def stop_profile_workers(self):
        # Model calls run on the host. Only workers already holding/waiting for
        # a profiling ticket are terminated during the answer-drain period.
        for root in self.args.output_roots:
            path = root / '.profile_slots/ticket_state.json'
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for group in ('holders', 'waiters'):
                for entry in (data.get(group) or {}).values():
                    pid = entry.get('pid')
                    if type(pid) is not int or pid <= 0:
                        continue
                    try:
                        if os.getpgid(pid) == self.process.pid:
                            os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def drain(self):
        self.stop_file.touch()
        # First prevent any further results from the failed service. Releasing
        # the GPU here also avoids paying for a sandbox while model calls drain.
        actions = [lambda: remove_containers(self.metadata['container_label'])]
        if self.sandbox is not None:
            if getattr(self.args, 'lock_gdn_bwd_reference', False) and self.sandbox.poll() is None:
                actions.append(lambda: collect_reference_records(self.sandbox, self.directory))
            actions.append(lambda: self.sandbox.terminate(wait=True))
        for action in actions:
            try:
                action()
            except Exception as exc:
                log(f'drain cleanup will be retried on exit: {exc}')
        if self.process is None:
            return
        self.stop_profile_workers()
        log(f'draining pending model answers for up to {self.args.drain_timeout_seconds}s')
        deadline = time.monotonic() + self.args.drain_timeout_seconds
        while self.process.poll() is None and time.monotonic() < deadline:
            self.interruptible_wait(0.2)

    def cleanup(self):
        errors = []
        def attempt(action):
            try:
                action()
            except Exception as exc:
                errors.append(str(exc))
                log(f'cleanup failed: {exc}')
        def stop_runner():
            if self.process is None:
                return
            signal_group(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_group(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            # Remove descendants even if the launcher itself exited first.
            signal_group(self.process.pid, signal.SIGKILL)
        attempt(stop_runner)
        attempt(lambda: remove_containers(self.metadata['container_label']))
        if self.sandbox is not None:
            if getattr(self.args, 'lock_gdn_bwd_reference', False) and self.sandbox.poll() is None:
                attempt(lambda: collect_reference_records(self.sandbox, self.directory))
            attempt(lambda: self.sandbox.terminate(wait=True))
        else:
            # Creation may have succeeded server-side before a client failure.
            def stop_by_name():
                try:
                    self.modal.Sandbox.from_name(self.args.app_name, self.metadata['sandbox_name']).terminate(wait=True)
                except self.modal.exception.NotFoundError:
                    pass
            attempt(stop_by_name)
        for reader in self.readers:
            reader.join(timeout=5)
        self.metadata['cleanup_errors'] = errors
        if errors:
            self.metadata['exit_code'] = 2
        self.metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(self.directory / 'modal.json', self.metadata)
        if not errors:
            self.active_path.unlink(missing_ok=True)
        return errors


def execute(args, modal, interrupted):
    with run_lock(args.state_dir):
        recover_orphan(args, modal)
        preflight_runner(args)
        attempt = Attempt(args, modal, interrupted)
        status = 2
        try:
            try:
                attempt.start()
                attempt.launch()
                status = attempt.monitor()
            except InfrastructureError as exc:
                attempt.metadata['failure'] = str(exc)
                log(str(exc))
                status = RETRY
                attempt.drain()
            except (modal.exception.AuthError, modal.exception.InvalidError, ConfigurationError) as exc:
                attempt.metadata['failure'] = str(exc)
                log(str(exc))
                status = 2
            except (OSError, modal.exception.Error) as exc:
                attempt.metadata['failure'] = str(exc)
                log(f'infrastructure exception: {exc}')
                status = RETRY
                attempt.drain()
        except SystemExit as exc:
            status = int(exc.code)
        finally:
            attempt.metadata['exit_code'] = status
            if attempt.cleanup():
                status = 2  # Keep active.json; don't allocate again until cleanup succeeds.
        return status


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--cleanup-state']:
        parser = argparse.ArgumentParser(description='Clean an interrupted owner before watcher auditing')
        parser.add_argument('--cleanup-state', type=Path, required=True)
        state_dir = parser.parse_args(argv).cleanup_state.expanduser().resolve()
        try:
            with run_lock(state_dir):
                import modal
                modal.Client.from_env()
                recover_orphan(argparse.Namespace(state_dir=state_dir), modal)
        except AlreadyRunning:
            return 73
        return 0
    args = parse_args(argv)
    validate(args)
    import modal
    modal.Client.from_env()
    docker('info', '--format', '{{json .SecurityOptions}}')
    interrupted = []
    def handle_signal(signum, _frame):
        # Record signals instead of raising inside Sandbox.create/Popen so their
        # returned resource IDs are captured and cleanup can always own them.
        if not interrupted:
            interrupted.append(signum)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return execute(args, modal, interrupted)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f'cannot launch profiling attempt: {exc}')
        raise SystemExit(2)
