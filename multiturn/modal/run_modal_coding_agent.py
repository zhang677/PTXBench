#!/usr/bin/env python3
"""Run a fresh Codex/AGY experiment with one Modal profiling GPU.

Usage: conda run --no-capture-output -n acc python run_modal_coding_agent.py CONFIG.json
Local PTXBench supplies the base Dockerfile; the sandbox clones its default
branch, installs the runtime, and generates blobs before starting the service.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid

PTXBENCH_GIT = 'https://github.com/zhang677/PTXBench.git'
from modal_profile_runtime import collect_reference_records, GPU_ARCH, MODAL_DOCKERFILE, SANDBOX_START, modal_base_image, modal_environment
import profile_dataset


def run(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, text=True, **kwargs)


def output(*args):
    return run(*args, stdout=subprocess.PIPE).stdout.strip()


def check_gateway_port(host, port):
    """Check gateway bind/listen readiness without rejecting old TIME_WAIT sockets."""
    try:
        with socket.socket() as sock:
            # Match the gateway server: closed connections may still be in TIME_WAIT.
            # SO_REUSEADDR permits their reuse but does not share a live listener.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(1)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        raise RuntimeError(
            f'gateway address {host}:{port} is already in use; '
            'wait for the previous gateway to stop or choose a different gateway_url'
        ) from exc


def read_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def wait_ready(url, predicate, alive, timeout):
    deadline = time.monotonic() + timeout
    last_error = 'no health response'
    while time.monotonic() < deadline:
        if not alive():
            raise RuntimeError(f'service exited before readiness: {url}; see runtime logs')
        try:
            if predicate(read_json(url + '/health')):
                return
        except (OSError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f'service readiness timed out: {url}: {last_error}')


def load_config(path, *, backend="modal"):
    from ptxbench_eval import launcher
    from ptxbench_eval.feedback import feedback_style_from_config
    from ptxbench_eval.models import validate_language_contract
    from ptxbench_eval.task_view import task_view_from_config

    config = launcher._load_config(path)
    validate_language_contract(config)
    feedback_style_from_config(config)
    task_view_from_config(config)
    config['agent'] = launcher._agent_kind(config)
    for key in (('profile_service', 'radix') if backend == 'modal' else ('modal',)) + ('profile_recovery', 'workload_manifest'):
        if key in config:
            raise ValueError(f'{key} is unsupported by the Modal launcher')
    for key in ('model', 'agent_image', 'gateway_image', 'experiment_root', 'gateway_url'):
        launcher._required_config_string(config, key)
    root = Path(config['experiment_root']).expanduser()
    if (not root.is_absolute() or root == Path('/')
            or (root.exists() and config['agent'] != 'codex-persistent')):
        raise ValueError('experiment_root must be a new absolute directory; runs do not resume')
    config['experiment_root'] = str(root.resolve())
    gateway = urlsplit(config['gateway_url'])
    if (gateway.scheme != 'http' or gateway.hostname not in ('127.0.0.1', 'localhost')
            or not gateway.port or gateway.username or gateway.password
            or gateway.path not in ('', '/') or gateway.query or gateway.fragment):
        raise ValueError('gateway_url must be http://127.0.0.1:PORT or http://localhost:PORT')
    config['gateway_url'] = config['gateway_url'].rstrip('/')
    for key, default in [('max_parallel', 1), ('status_interval_seconds', 30), ('token_budget', None)]:
        value = config.setdefault(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config.get('build_images')) is not bool:
        raise ValueError('build_images must be a boolean')
    if 'max_profiles' in config and (
            type(config['max_profiles']) is not int or config['max_profiles'] <= 0):
        raise ValueError('max_profiles must be a positive integer')
    if backend == 'modal':
        settings = config.setdefault('modal', {})
        if not isinstance(settings, dict) or set(settings) - {
                'gpu', 'timeout_seconds', 'ready_timeout_seconds', 'dataset', 'lock_gdn_bwd_reference'}:
            raise ValueError('modal accepts gpu, timeout_seconds, ready_timeout_seconds, '
                             'dataset, and lock_gdn_bwd_reference')
        gpu = settings.setdefault('gpu', 'B200')
        if gpu not in GPU_ARCH or config.get('target_hardware') != [gpu]:
            raise ValueError('modal.gpu must be B200, H100, or H200 and match target_hardware')
        modal_environment(gpu, settings.get('lock_gdn_bwd_reference', False))
        if config['language'] == 'cuda' and config['nvcc_gencode'] != (
                f'arch=compute_{GPU_ARCH[gpu].replace(".", "")},code=sm_{GPU_ARCH[gpu].replace(".", "")}'):
            raise ValueError('nvcc_gencode must match the selected GPU architecture')
        for key, default in [('timeout_seconds', 86400), ('ready_timeout_seconds', 3600)]:
            value = settings.setdefault(key, default)
            if type(value) is not int or not 0 < value <= 86400:
                raise ValueError(f'modal.{key} must be in 1..86400 seconds')
        if settings['ready_timeout_seconds'] > settings['timeout_seconds']:
            raise ValueError('Modal readiness timeout cannot exceed sandbox lifetime')
        if 'dataset' in settings:
            if not isinstance(settings['dataset'], dict):
                raise ValueError('modal.dataset must be a Hugging Face dataset object')
            settings['dataset'] = profile_dataset.dataset_settings({
                'profile_service': {'dataset': settings['dataset'],
                                    'restart_script': '/workspace/scripts/restart_profiling.sh'},
                'runs': config.get('runs'),
            })
    # The effective config lives under experiment_root; preserve source-relative paths.
    for key in ('prompt_hub', 'prompt_files_root', 'tvm_ffi_example', 'cutedsl_source',
                'cutedsl_example', 'triton_example', 'codex_auth_file'):
        if key in config:
            config[key] = str(launcher._resolve(config, config[key]).expanduser().resolve())
    if config['agent'] in {'codex', 'codex-persistent'}:
        launcher._codex_plan_auth(config)  # Check readability without logging credentials.
    elif not os.environ.get('GEMINI_API_KEY'):
        raise ValueError('GEMINI_API_KEY is not set')
    from ptxbench_eval.codex_persistent_runner import CapacitySettings
    CapacitySettings.from_config(config)
    launcher._agent_argv(config, 'validation')
    launcher._disable_web(config)
    launcher._add_goal_command(config, config['agent'])
    launcher._cutedsl_source(config)
    hub = launcher._load_prompt_hub(Path(config['prompt_hub']))
    example = launcher._example_key(config['language'])
    Path(launcher._required_config_string(config, example)).read_text()
    runs = config.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('runs must be a nonempty array')
    names = []
    for item in runs:
        for key in ('run_id', 'task_id', 'definition', 'workload_uuid', 'prompt_tag'):
            launcher._required_config_string(item, key)
        if 'workload_manifest' in item or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', item['run_id']):
            raise ValueError('each run needs a safe run_id and definition/workload_uuid selectors')
        names.append(launcher._container_name(item['run_id']))
        for tag in (item['prompt_tag'], item.get('extra_user_prompt_tag')):
            if tag:
                launcher._prompt_context_sources(tag, hub, Path(config['prompt_files_root']))
    if len(names) != len(set(names)):
        raise ValueError('run IDs must map to distinct Docker container names')
    config.pop('_config_dir', None)
    return config


def sandbox_start(config):
    """Reuse Radix's selected-workload downloader when a dataset is configured."""
    settings = config['modal'].get('dataset')
    if settings is None:
        return SANDBOX_START
    payload = dict(settings=settings, selected=profile_dataset.selectors(config))
    generation = '\n'.join([
        'env LD_LIBRARY_PATH= /workspace/acc/bin/python /workspace/dataset/generate_blobs.py ' + '\\',
        '    --dataset-root /workspace/dataset --device cuda:0', '',
    ])
    read_path = shlex.quote('import json; print(json.load(open("/workspace/modal-dataset.json"))["dataset_dir"])')
    prepare = '\n'.join([
        f'timeout --signal=TERM --kill-after=10 {settings["timeout_seconds"]} '
        '/workspace/acc/bin/python /workspace/profile_dataset.py --worker '
        "<<'MODAL_DATASET_CONFIG' > /workspace/modal-dataset.json",
        json.dumps(payload), 'MODAL_DATASET_CONFIG',
        'dataset_dir="$(/workspace/acc/bin/python -c ' + read_path + ')"',
        'export FIB_DATASET_DIR="$dataset_dir" FIB_DATASET_DIRS="$dataset_dir"', '',
    ])
    if SANDBOX_START.count(generation) != 1 or SANDBOX_START.count('--local /workspace/dataset') != 1:
        raise ValueError('Modal startup changed: cannot select the configured dataset')
    return SANDBOX_START.replace(generation, prepare).replace(
        '--local /workspace/dataset', '--local "$dataset_dir"')


def build_images(config):
    package = Path('ptxbench-eval')
    dockerfiles = [('codex' if config['agent'] in {'codex', 'codex-persistent'} else 'antigravity', config['agent_image']),
                   ('gateway', config['gateway_image'])]
    if config['build_images']:
        if subprocess.run(['docker', 'image', 'inspect', 'ptxbench-eval-base:latest'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            run('docker', 'build', '-f', package / 'docker/Dockerfile.gateway-base',
                '-t', 'ptxbench-eval-base:latest', PTXBENCH_GIT)
        for kind, image in dockerfiles:
            extra = ['--build-arg', 'CODEX_VERSION=' + config['codex_version']] if (
                kind == 'codex' and config.get('codex_version')) else []
            run('docker', 'build', *extra, '-f', package / f'docker/Dockerfile.{kind}',
                '-t', image, PTXBENCH_GIT)
    else:
        for _, image in dockerfiles:
            run('docker', 'image', 'inspect', image, stdout=subprocess.DEVNULL)
    if config['agent'] in {'codex', 'codex-persistent'} and config.get('codex_version'):
        version = output('docker', 'run', '--rm', '--network', 'none', '--entrypoint',
                         'codex', config['agent_image'], '--version')
        if version != 'codex-cli ' + config['codex_version']:
            raise ValueError(f'agent image Codex version mismatch: {version}')


def docker_wrapper(directory, label, rootless):
    """Label owned containers and preserve rootless bind-mount write access."""
    docker = shutil.which('docker')
    path = directory / 'docker'
    command = [docker, 'run', '--label', label]
    if rootless:
        command += ['--user', '0:0']
    path.write_text('#!/bin/bash\nif [[ "${1:-}" == run ]]; then\n  shift\n  exec '
                    + shlex.join(command) + ' "$@"\nfi\nexec ' + shlex.quote(docker) + ' "$@"\n')
    path.chmod(0o700)


def stream_logs(stream, path):
    try:
        with path.open('w') as log:
            for chunk in stream:
                log.write(chunk)
                log.flush()
                print(chunk, end='', flush=True)
    except Exception as exc:
        print(f'Modal log stream interrupted: {exc}', file=sys.stderr)


class ModalBackend:
    """One owned GPU sandbox per acquisition; reusable after verified release."""

    def __init__(self, config, modal):
        self.config = config
        self.folder = None
        self.modal = modal
        self.sandbox = None
        self.readers = []

    def acquire(self, folder):
        self.folder = folder
        import copy
        config = copy.deepcopy(self.config)
        settings = config['modal']
        modal = self.modal
        if self.sandbox is not None:
            raise RuntimeError('previous Modal sandbox has not been released')
        image = modal_base_image(modal)
        options = {}
        if settings.get('dataset'):
            image = image.add_local_file(profile_dataset.__file__, '/workspace/profile_dataset.py', copy=True)
            if os.environ.get('HF_TOKEN'):
                options['secrets'] = [modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']})]
        app = modal.App.lookup('ptxbench-profile', create_if_missing=True)
        with modal.enable_output():
            self.sandbox = modal.Sandbox.create(
                'bash', '-c', sandbox_start(config), app=app, image=image, gpu=settings['gpu'],
                cpu=8, memory=32768, timeout=settings['timeout_seconds'], encrypted_ports=[10000],
                env=modal_environment(settings['gpu'], settings.get('lock_gdn_bwd_reference', False)),
                **options)
        from ptxbench_eval.codex_persistent_runner import save_json
        metadata = {'sandbox_id': self.sandbox.object_id, 'gpu': settings['gpu'],
                    'dockerfile_sha256': hashlib.sha256(MODAL_DOCKERFILE.encode()).hexdigest()}
        save_json(folder / 'modal.json', metadata)
        print(f'Modal sandbox: {self.sandbox.object_id}', flush=True)
        for stream, name in [(self.sandbox.stdout, 'modal.stdout.log'), (self.sandbox.stderr, 'modal.stderr.log')]:
            reader = threading.Thread(target=stream_logs, args=(stream, folder / name), daemon=True)
            reader.start()
            self.readers.append(reader)
        config['profile_base_url'] = self.sandbox.tunnels()[10000].url.rstrip('/')
        def ready(health):
            workers = health.get('workers') or health.get('backends') or []
            return (health.get('status') == 'ok' and len(workers) == 1
                    and workers[0].get('healthy') is True and health.get('queue_size') == 0)
        wait_ready(config['profile_base_url'], ready, lambda: self.sandbox.poll() is None,
                   settings['ready_timeout_seconds'])
        for remote, target in [('modal-runtime.json', 'runtime.json'),
                               *([('modal-dataset.json', 'dataset.json')] if settings.get('dataset') else [])]:
            result = self.sandbox.exec('cat', '/workspace/' + remote, timeout=30)
            payload = json.loads(result.stdout.read())
            result.wait()
            if result.returncode:
                raise RuntimeError('could not read Modal runtime provenance')
            save_json(folder / target, payload)
            if remote == 'modal-dataset.json':
                source_path = Path(config['experiment_root']) / 'dataset_source.json'
                if source_path.exists() and json.loads(source_path.read_text()) != payload['source']:
                    raise RuntimeError('dataset provenance changed on reacquisition')
                save_json(source_path, payload['source'])
        return config

    def check(self):
        if self.sandbox is None or self.sandbox.poll() is not None:
            raise RuntimeError('Modal sandbox exited during the experiment')

    def release(self):
        capture_error = None
        if self.sandbox is not None:
            try:
                if self.config['modal'].get('lock_gdn_bwd_reference') and self.sandbox.poll() is None:
                    collect_reference_records(self.sandbox, self.folder)
            except Exception as exc:
                capture_error = exc
            finally:
                self.sandbox.terminate(wait=True)
            if self.sandbox.poll() is None:
                raise RuntimeError('Modal sandbox termination was not confirmed')
            self.sandbox = None
        for reader in self.readers:
            reader.join(timeout=5)
        self.readers.clear()
        if capture_error is not None:
            raise RuntimeError('could not preserve actual GDN reference configurations') from capture_error


def execute(config, modal):
    if config["agent"] == "codex-persistent":
        from leased_coding_agent import execute as execute_leased
        return execute_leased(config, ModalBackend(config, modal))
    from ptxbench_eval.feedback import feedback_style_from_config

    root = Path(config['experiment_root'])
    settings = config['modal']
    gateway = 'ptxbench-gateway-' + re.sub(r'[^A-Za-z0-9_.-]', '-', root.name)
    expected = {gateway, *('ptxbench-' + r['run_id'].replace('_', '-') for r in config['runs'])}
    conflicts = expected.intersection(output('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines())
    if conflicts:
        raise RuntimeError(f'conflicting Docker containers: {sorted(conflicts)}')
    address = urlsplit(config['gateway_url'])
    check_gateway_port(address.hostname, address.port)
    modal.Client.from_env()
    build_images(config)
    root.mkdir(parents=True)  # Atomic refusal if another launch created this root.
    (root / 'source_config.json').write_text(json.dumps(config, indent=2) + '\n')
    label = 'ptxbench.modal-launch=' + uuid.uuid4().hex
    sandbox = process = None
    readers = []
    cleanup_errors = []
    launcher = [sys.executable, '-m', 'ptxbench_eval.launcher']
    old_path = os.environ['PATH']
    with tempfile.TemporaryDirectory(prefix='ptxbench-modal-') as temporary:
        directory = Path(temporary)
        rootless = 'name=rootless' in output('docker', 'info', '--format', '{{json .SecurityOptions}}')
        docker_wrapper(directory, label, rootless)
        os.environ['PATH'] = str(directory) + os.pathsep + old_path
        try:
            # Modal ignores EXPOSE; port publication is specified on Sandbox.create.
            image = modal_base_image(modal)
            sandbox_options = {}
            if settings.get('dataset'):
                image = image.add_local_file(profile_dataset.__file__, '/workspace/profile_dataset.py', copy=True)
                if os.environ.get('HF_TOKEN'):
                    sandbox_options['secrets'] = [modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']})]
            app = modal.App.lookup('ptxbench-profile', create_if_missing=True)
            with modal.enable_output():
                sandbox = modal.Sandbox.create(
                    'bash', '-c', sandbox_start(config), app=app, image=image, gpu=settings['gpu'],
                    cpu=8, memory=32768, timeout=settings['timeout_seconds'],
                    encrypted_ports=[10000],
                    env=modal_environment(settings['gpu'], settings.get('lock_gdn_bwd_reference', False)),
                    **sandbox_options)
            metadata = {'sandbox_id': sandbox.object_id, 'gpu': settings['gpu'],
                        'dockerfile_sha256': hashlib.sha256(MODAL_DOCKERFILE.encode()).hexdigest()}
            metadata_path = root / 'modal.json'
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'Modal sandbox: {sandbox.object_id}; logs: {root}', flush=True)
            for stream, name in [(sandbox.stdout, 'modal.stdout.log'), (sandbox.stderr, 'modal.stderr.log')]:
                reader = threading.Thread(target=stream_logs, args=(stream, root / name), daemon=True)
                reader.start()
                readers.append(reader)
            config['profile_base_url'] = sandbox.tunnels()[10000].url.rstrip('/')
            effective = root / 'modal_config.json'
            effective.write_text(json.dumps(config, indent=2) + '\n')
            def profile_ready(health):
                workers = health.get('workers') or health.get('backends') or []
                return (health.get('status') == 'ok' and len(workers) == 1
                        and workers[0].get('healthy') is True and health.get('queue_size') == 0)
            wait_ready(config['profile_base_url'], profile_ready,
                       lambda: sandbox.poll() is None, settings['ready_timeout_seconds'])
            probe = sandbox.exec('cat', '/workspace/modal-runtime.json', timeout=30)
            metadata.update(json.loads(probe.stdout.read()))
            probe.wait()
            if probe.returncode:
                raise RuntimeError('could not read Modal runtime provenance')
            if settings.get('dataset'):
                probe = sandbox.exec('cat', '/workspace/modal-dataset.json', timeout=30)
                dataset = json.loads(probe.stdout.read())
                probe.wait()
                if probe.returncode:
                    raise RuntimeError('could not read Modal dataset provenance')
                (root / 'dataset_source.json').write_text(json.dumps(dataset['source'], indent=2) + '\n')
                metadata['dataset_dir'] = dataset['dataset_dir']
                metadata['dataset_preparer_sha256'] = hashlib.sha256(
                    Path(profile_dataset.__file__).read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            run(*launcher, 'prepare', effective)
            feedback = feedback_style_from_config(config)
            profile_args = (['--max-profiles', str(config['max_profiles'])]
                            if 'max_profiles' in config else [])
            run('docker', 'run', '--detach', '--name', gateway, '--network', 'host',
                '-v', f'{root}:{root}', config['gateway_image'], '--root', root,
                '--profile-base-url', config['profile_base_url'], '--host', address.hostname,
                '--port', address.port, '--feedback-style', feedback, *profile_args)
            wait_ready(config['gateway_url'], lambda h: (
                h.get('status') == 'ok' and h.get('root') == str(root)
                and h.get('profile_base_url') == config['profile_base_url']
                and h.get('feedback_style') == feedback
                and h.get('max_profiles') == config.get('max_profiles')),
                lambda: output('docker', 'inspect', '-f', '{{.State.Running}}', gateway) == 'true', 60)
            process = subprocess.Popen([*launcher, 'launch', str(effective)], start_new_session=True)
            while True:
                try:
                    status = process.wait(timeout=config['status_interval_seconds'])
                    break
                except subprocess.TimeoutExpired:
                    if sandbox.poll() is not None:
                        raise RuntimeError('Modal sandbox exited during the experiment')
                    run(*launcher, 'status', root)
            run(*launcher, 'status', root)
            return status
        finally:
            # One failed cleanup must not prevent the other resources being released.
            def cleanup(action):
                try:
                    action()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
                    print(f'Cleanup failed: {exc}', file=sys.stderr)
            def stop_launcher():
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
            def stop_containers():
                ids = output('docker', 'ps', '-aq', '--filter', 'label=' + label).split()
                if ids:
                    try:
                        with (root / 'gateway.log').open('w') as log:
                            subprocess.run(['docker', 'logs', gateway], stdout=log, stderr=log, timeout=15)
                    finally:
                        run('docker', 'rm', '-f', *ids, timeout=30)
            cleanup(stop_launcher)
            cleanup(stop_containers)
            if sandbox is not None:
                if settings.get('lock_gdn_bwd_reference') and sandbox.poll() is None:
                    cleanup(lambda: collect_reference_records(sandbox, root))
                cleanup(lambda: sandbox.terminate(wait=True))
            for reader in readers:
                reader.join(timeout=5)
            os.environ['PATH'] = old_path
            if cleanup_errors and sys.exc_info()[0] is None:
                raise RuntimeError('cleanup incomplete: ' + '; '.join(cleanup_errors))



def docker_namespace_pid():
    """Return the rootless Docker daemon PID when its network namespace differs."""
    security = json.loads(output('docker', 'info', '--format', '{{json .SecurityOptions}}'))
    if not any(item.split(',')[0] == 'name=rootless' for item in security or []):
        return None
    endpoint = os.environ.get('DOCKER_HOST', '')
    if not endpoint or os.environ.get('DOCKER_CONTEXT'):
        endpoint = output('docker', 'context', 'inspect', '--format', '{{.Endpoints.docker.Host}}')
    if not endpoint.startswith('unix://'):
        raise RuntimeError('rootless profiling requires a local Docker Unix socket')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(endpoint[len('unix://'):])
        pid, _, _ = struct.unpack('3i', client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if pid <= 0:
        raise RuntimeError('could not identify the rootless Docker daemon PID')
    proc = Path('/proc')
    if (proc / str(pid) / 'ns/net').stat().st_ino == (proc / 'self/ns/net').stat().st_ino:
        return None
    return pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    import ptxbench_eval  # installed from the PTXBench repository
    package = str(Path(ptxbench_eval.__file__).resolve().parents[1])
    os.environ['PYTHONPATH'] = package + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    path = args.config.resolve()
    config = load_config(path)
    pid = docker_namespace_pid()
    if pid is not None:
        os.execvp('nsenter', ['nsenter', '-t', str(pid), '-U', '-m', '-n', '--',
                             sys.executable, str(Path(__file__).resolve()), str(path)])
    import modal
    def interrupted(signum, _frame):
        # A second signal must not interrupt resource cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    return execute(config, modal)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
