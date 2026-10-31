"""Launch resumable multiturn Modal experiments from JSON.

Usage: python modal_profile_runtime.py CONFIG.json [--validate-only]
The GPU startup constants are also shared with the coding-agent launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import shlex
import shutil
import signal
import subprocess
import sys
import time

import fla_fp32_state_dot
import profile_dataset

GPU_ARCH = {'B200': '10.0a', 'H100': '9.0a', 'H200': '9.0a'}

# The sandbox clones PTXBench on startup; its base image needs only CUDA and
# tools for installing that clone. Keep this recipe beside the launchers.
MODAL_DOCKERFILE = '''FROM nvidia/cuda:13.0.2-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git python3-pip python3-dev curl tmux openssh-client ca-certificates libdw-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}" PROFILE_PORT=10000 WORKDIR=/workspace \
    FIB_DIR=flashinfer-bench-private FIB_DATASET_DIR=/workspace/dataset \
    FIB_DATASET_DIRS=/workspace/dataset UV_CACHE_DIR=/workspace/.cache/uv
WORKDIR /workspace
EXPOSE 10000
'''


def modal_base_image(modal):
    digest = hashlib.sha256(MODAL_DOCKERFILE.encode()).hexdigest()[:16]
    dockerfile = Path(tempfile.gettempdir()) / f'ptxbench-modal-{digest}.Dockerfile'
    if not dockerfile.exists() or dockerfile.read_text() != MODAL_DOCKERFILE:
        dockerfile.write_text(MODAL_DOCKERFILE)
    return modal.Image.from_dockerfile(dockerfile, context_dir=dockerfile.parent,
                                       ignore=lambda _: True)



def modal_environment(gpu, lock_gdn_bwd_reference=False, patch_fla_fp32_state_dot=False):
    """Validate the opt-in reference patches and construct the shared sandbox env."""
    if type(lock_gdn_bwd_reference) is not bool:
        raise ValueError('modal.lock_gdn_bwd_reference must be a boolean')
    if lock_gdn_bwd_reference and gpu != 'B200':
        raise ValueError('modal.lock_gdn_bwd_reference is supported only on B200')
    if type(patch_fla_fp32_state_dot) is not bool:
        raise ValueError('modal.patch_fla_fp32_state_dot must be a boolean')
    if lock_gdn_bwd_reference and patch_fla_fp32_state_dot:
        raise ValueError('modal.patch_fla_fp32_state_dot is independent of lock_gdn_bwd_reference; enable one')
    env = {'PROFILE_GPU': gpu, 'TVM_FFI_CUDA_ARCH_LIST': GPU_ARCH[gpu],
           'LOCK_GDN_BWD_REFERENCE': '1' if lock_gdn_bwd_reference else '0',
           'FLA_FP32_STATE_DOT': '1' if patch_fla_fp32_state_dot else '0'}
    return env


def environment_options(settings):
    """The modal_environment keywords from a modal config block or argparse namespace."""
    get = settings.get if isinstance(settings, dict) else lambda key, default: getattr(settings, key, default)
    return dict(lock_gdn_bwd_reference=get('lock_gdn_bwd_reference', False),
                patch_fla_fp32_state_dot=get('patch_fla_fp32_state_dot', False))


# Embed the validated source lock so all Modal launch paths use the same payload.
_GDN_HELPER = Path(__file__).with_name('modal_gdn_reference.py').read_text()
_GDN_MANIFEST = Path(__file__).with_name('gdn_reference_lock_b200.json').read_text()
B200_FLA_PATCH = ("""
if os.environ['PROFILE_GPU'] == 'B200' and os.environ.get('LOCK_GDN_BWD_REFERENCE') == '1':
    import site
    try:
        if 'B200' not in name:
            raise RuntimeError('GDN reference lock requires an actual B200')
        helper = Path(site.getsitepackages()[0]) / 'modal_gdn_reference.py'
        helper.write_text(""" + repr(_GDN_HELPER) + """)
        manifest_text = """ + repr(_GDN_MANIFEST) + """
        manifest = json.loads(manifest_text)
        Path('/workspace/gdn-reference-lock.json').write_text(manifest_text)
        from modal_gdn_reference import prepare_compiler, apply_lock, install_capture_hook
        compiler = prepare_compiler(manifest)
        info['fla_patches'] = [apply_lock(manifest, name)]
        info['gdn_reference_compiler'] = compiler
        info['gdn_reference_capture_hook'] = install_capture_hook(
            '/workspace/flashinfer-bench-private/flashinfer_bench/bench/evaluators/default.py')
    except Exception as exc:
        raise RuntimeError(f'MODAL_PROFILE_FATAL: {exc}') from exc
""")


def collect_reference_records(sandbox, directory):
    """Save actual worker selections before the owned sandbox is released."""
    code = ("import json; from pathlib import Path; "
            "root=Path('/workspace/gdn-reference-records'); "
            "manifest=Path('/workspace/gdn-reference-lock.json'); "
            "print(json.dumps({'records':[json.loads(p.read_text()) for p in sorted(root.glob('reference-*.json'))], "
            "'manifest':json.loads(manifest.read_text()) if manifest.exists() else None}))")
    process = sandbox.exec('python3', '-c', code, timeout=30)
    payload = json.loads(process.stdout.read())
    process.wait()
    if process.returncode:
        raise RuntimeError('could not collect actual GDN reference configurations')
    Path(directory, 'gdn_reference_records.json').write_text(json.dumps(payload, indent=2) + '\n')


# Use the same install-only prefix as radix/prepare_profile.sh. Do not duplicate
# package versions or modify PTXBench's entrypoint. Generation runs on the GPU.
SANDBOX_START = r'''set -euo pipefail
exec 2>&1
git clone --depth 1 https://github.com/zhang677/PTXBench.git /workspace/PTXBench
for part in flashinfer-bench-private dataset scripts; do
    ln -s "/workspace/PTXBench/fib-profile/$part" "/workspace/$part"
done
python3 - <<'PY'
from pathlib import Path
import tempfile
source = Path('/workspace/scripts/entrypoint.sh').read_text()
marker = 'exec "$HERE/restart_profiling.sh"'
if source.count(marker) != 1:
    raise RuntimeError('PTXBench entrypoint changed: cannot isolate runtime installation')
Path('/workspace/scripts/modal_install.sh').write_text(source.split(marker)[0])
PY
FIB_DATASET_MODE=existing bash /workspace/scripts/modal_install.sh
# The installer activates its venv in a child shell. The service's compiler
# subprocesses (notably ninja) also need that environment on PATH.
source /workspace/acc/bin/activate
/workspace/acc/bin/python - <<'PY'
import json, os, subprocess
from pathlib import Path
import tempfile
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected one CUDA GPU'
name = torch.cuda.get_device_name(0)
if os.environ['PROFILE_GPU'] not in name:
    raise RuntimeError(f"MODAL_PROFILE_FATAL: expected {os.environ['PROFILE_GPU']}, got {name}")
x = torch.ones(1, device='cuda:0')
assert (x + 1).item() == 2
info = dict(gpu=name, cuda_capability=torch.cuda.get_device_capability(0),
            torch=torch.__version__, ptxbench_revision=subprocess.check_output(
                ['git', '-C', '/workspace/PTXBench', 'rev-parse', 'HEAD'], text=True).strip())
''' + B200_FLA_PATCH + r'''
Path('/workspace/modal-runtime.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info), flush=True)
PY
# Apply or restore FLA's FP32 state dot (modal.patch_fla_fp32_state_dot) before any reference runs.
''' + fla_fp32_state_dot.shell_step() + r'''# The compiler alias exists only after preparation; export it for subsequent
# dataset/reference/candidate processes, including FlashInfer timing imports.
if [ "$PROFILE_GPU" = B200 ] && [ "${LOCK_GDN_BWD_REFERENCE:-0}" = 1 ]; then
    export TRITON_PTXAS_BLACKWELL_PATH=/workspace/gdn-reference-ptxas
    export TRITON_CACHE_DIR=/workspace/triton-gdn-reference-lock-v1
    export FLA_CACHE_MODE=disabled
fi
env LD_LIBRARY_PATH= /workspace/acc/bin/python /workspace/dataset/generate_blobs.py \
    --dataset-root /workspace/dataset --device cuda:0
exec /workspace/acc/bin/flashinfer-bench serve --local /workspace/dataset \
    --devices cuda:0 --host 0.0.0.0 --port 10000 \
    --config /workspace/flashinfer-bench-private/acc_config.yaml --timeout 30
'''


def sandbox_start(dataset=None, runs=()):
    """Reuse Radix's selected-workload downloader when a dataset is configured."""
    if dataset is None:
        return SANDBOX_START
    payload = dict(settings=dataset, selected=profile_dataset.selectors({'runs': list(runs)}))
    generation = '\n'.join([
        'env LD_LIBRARY_PATH= /workspace/acc/bin/python /workspace/dataset/generate_blobs.py ' + '\\',
        '    --dataset-root /workspace/dataset --device cuda:0', '',
    ])
    read_path = shlex.quote('import json; print(json.load(open("/workspace/modal-dataset.json"))["dataset_dir"])')
    prepare = '\n'.join([
        f'timeout --signal=TERM --kill-after=10 {dataset["timeout_seconds"]} '
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


def sandbox_image(modal, dataset=None):
    """The base image and Sandbox.create options; a dataset adds its preparer and HF_TOKEN."""
    image, options = modal_base_image(modal), {}
    if dataset:
        image = image.add_local_file(profile_dataset.__file__, '/workspace/profile_dataset.py', copy=True)
        if os.environ.get('HF_TOKEN'):
            options['secrets'] = [modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']})]
    return image, options


def read_dataset_provenance(sandbox):
    """The preparer's {dataset_dir, source} record from a ready sandbox."""
    probe = sandbox.exec('cat', '/workspace/modal-dataset.json', timeout=30)
    dataset = json.loads(probe.stdout.read())
    probe.wait()
    if probe.returncode:
        raise RuntimeError('could not read Modal dataset provenance')
    return dataset


HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent / 'run_parallel_v2.py'


def load_config(path):
    """Require explicit experiment settings; resolve paths beside the JSON file."""
    from run_modal_profile import ConfigurationError
    if str(RUNNER.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER.parent))
    from launcher_utils import load_config as load_prompt_config

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConfigurationError(f'duplicate config key: {key}')
            result[key] = value
        return result

    config = json.loads(path.read_text(), object_pairs_hook=unique_keys)

    def fields(value, expected, label, optional=()):
        if not isinstance(value, dict):
            raise ConfigurationError(f'{label} must be an object')
        missing, unknown = set(expected) - value.keys(), value.keys() - set(expected) - set(optional)
        if missing or unknown:
            raise ConfigurationError(f'{label}: missing fields {sorted(missing)}; '
                                     f'unknown fields {sorted(unknown)}')

    def positive(value, label):
        if type(value) is not int or value <= 0:
            raise ConfigurationError(f'{label} must be a positive integer')

    def string(value, label):
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f'{label} must be a nonempty string')

    def resolve(value, label):
        string(value, label)
        result = Path(value).expanduser()
        return str((path.parent / result).resolve())

    numbers = ('max_parallel', 'max_profiles', 'timeout_seconds', 'turn_timeout_seconds',
               'llm_api_timeout_seconds', 'model_retry_attempts', 'retry_interval_seconds')
    fields(config, (*numbers, 'run_name', 'model', 'gpu_arch', 'image', 'working_dir',
                    'modal', 'tasks'), 'config', optional=('language', 'dataset'))
    for key in numbers:
        positive(config[key], key)
    for key in ('run_name', 'model', 'gpu_arch', 'image'):
        string(config[key], key)
    if config.setdefault('language', 'cuda') not in ('cuda', 'triton'):
        raise ConfigurationError('language must be cuda or triton')
    config['working_dir'] = resolve(config['working_dir'], 'working_dir')
    modal = config['modal']
    modal_numbers = ('cpu', 'memory_mib', 'timeout_seconds', 'ready_timeout_seconds',
                     'poll_interval_seconds', 'health_failure_threshold', 'drain_timeout_seconds')
    fields(modal, (*modal_numbers, 'app_name', 'gpu', 'state_dir'), 'modal',
           optional=('lock_gdn_bwd_reference', 'patch_fla_fp32_state_dot'))
    for key in modal_numbers:
        positive(modal[key], f'modal.{key}')
    for key in ('app_name', 'gpu'):
        string(modal[key], f'modal.{key}')
    for key in ('state_dir',):
        modal[key] = resolve(modal[key], f'modal.{key}')
    if modal['gpu'] not in GPU_ARCH:
        raise ConfigurationError(f'modal.gpu must be one of {list(GPU_ARCH)}')
    try:
        modal_environment(modal['gpu'], **environment_options(modal))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    architecture = 'blackwell' if modal['gpu'] == 'B200' else 'hopper'
    if config['gpu_arch'] != architecture:
        raise ConfigurationError(f'gpu_arch must be {architecture} for {modal["gpu"]}')
    if not isinstance(config['tasks'], list) or not config['tasks']:
        raise ConfigurationError('tasks must be a nonempty list')
    for index, task in enumerate(config['tasks']):
        fields(task, ('definition', 'test_path', 'prompt_config', 'output_root'), f'tasks[{index}]',
               optional=('workload_uuid',))
        string(task['definition'], f'tasks[{index}].definition')
        for key in ('test_path', 'prompt_config', 'output_root'):
            task[key] = resolve(task[key], f'tasks[{index}].{key}')
        for key in ('test_path', 'prompt_config'):
            if not Path(task[key]).is_file():
                raise ConfigurationError(f'missing {key}: {task[key]}')
        # The runner checks the bundled docs only after the GPU is ready, and a retry cannot fix them.
        try:
            tags = {item['prompt_tag'] for item in load_prompt_config(Path(task['prompt_config']))}
        except (ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(f'tasks[{index}].prompt_config: {exc!r}') from exc
        missing = sorted(str(tag) for tag in tags
                         if not (RUNNER.parent / 'prompts/assembled' / f'{tag}.md').is_file())
        if missing:
            raise ConfigurationError(f'tasks[{index}]: missing bundled prompt docs {missing}')
        if ('workload_uuid' in task) != ('dataset' in config):
            raise ConfigurationError(f'tasks[{index}].workload_uuid is required with, '
                                     'and only with, a top-level dataset')
        if 'dataset' in config:
            # The sandbox serves only the selected workloads; the test submits its own.
            declared = dict(re.findall(r'^(DEFINITION_NAME|WORKLOAD_UUID) = "([^"\n]*)"$',
                                       Path(task['test_path']).read_text(), re.M))
            if declared != {'DEFINITION_NAME': task['definition'], 'WORKLOAD_UUID': task['workload_uuid']}:
                raise ConfigurationError(f'tasks[{index}]: test_path declares {declared}, '
                                         'not this definition and workload_uuid')
    if 'dataset' in config:
        if not isinstance(config['dataset'], dict):
            raise ConfigurationError('dataset must be a Hugging Face dataset object')
        try:
            config['dataset'] = profile_dataset.dataset_settings({
                'profile_service': {'dataset': config['dataset'],
                                    'restart_script': '/workspace/scripts/restart_profiling.sh'},
                'runs': dataset_runs(config),
            })
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
    return config


def dataset_runs(config):
    return [{'definition': task['definition'], 'workload_uuid': task['workload_uuid']}
            for task in config['tasks']]


def supervisor_args(config):
    """Translate the public JSON schema into the existing single-attempt API."""
    from run_modal_profile import URL_TOKEN

    tasks = config['tasks']
    command = ['env', f'LLM_API_TIMEOUT={config["llm_api_timeout_seconds"]}',
               f'MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT={config["model_retry_attempts"]}',
               sys.executable, str(RUNNER)]
    for flag, key in [('definitions', 'definition'), ('test-paths', 'test_path'),
                      ('configs', 'prompt_config'), ('output-roots', 'output_root')]:
        command += ['--' + flag, *(task[key] for task in tasks)]
    for flag, key in [('timeout', 'timeout_seconds'), ('model', 'model'), ('gpu-arch', 'gpu_arch'),
                      ('max-parallel', 'max_parallel'), ('max-profiles', 'max_profiles'),
                      ('turn-timeout', 'turn_timeout_seconds'), ('image', 'image'),
                      ('language', 'language')]:
        command += ['--' + flag, str(config[key])]
    command += ['--without-local-gpu', '--service-url', URL_TOKEN]
    settings = dict(config['modal'])
    for key in ('state_dir',):
        settings[key] = Path(settings[key])
    return argparse.Namespace(**settings, working_dir=Path(config['working_dir']),
                              definitions=[task['definition'] for task in tasks],
                              output_roots=[Path(task['output_root']) for task in tasks],
                              dataset=config.get('dataset'),
                              runs=dataset_runs(config) if 'dataset' in config else [],
                              command=command, experiment_config=config)


def running_eval_pids(config):
    """Avoid rewriting plans while an existing launcher or actor uses the roots."""
    roots = [task['output_root'].encode() for task in config['tasks']]
    hits = []
    for entry in Path('/proc').glob('[0-9]*'):
        if int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(Path(os.fsdecode(arg)).name in ('run_parallel_v2.py', 'run_v2.py') for arg in argv):
            if any(root in argv for root in roots):
                hits.append(int(entry.name))
    return hits


def refresh_plans(config):
    """Keep the prior watcher's plan refresh, without creating fresh output roots."""
    from run_modal_profile import log, write_json
    if str(RUNNER.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER.parent))
    from launcher_utils import load_config as load_prompt_config, materialize_run_fields, plan_from_experiments
    from run_parallel_v2 import expand_config

    for task in config['tasks']:
        root = Path(task['output_root'])
        if not root.exists():
            continue
        experiments = materialize_run_fields(
            expand_config(load_prompt_config(Path(task['prompt_config']))),
            [{'definition': task['definition'], 'test_path': task['test_path']}])
        plan = {'config': task['prompt_config'], 'plan': plan_from_experiments(experiments)}
        path = root / 'plan.json'
        if path.exists():
            if json.loads(path.read_text()) == plan:
                continue
            shutil.copy2(path, path.with_name(f'plan.json.{time.time_ns()}.bak'))
        write_json(path, plan)
        log(f'refreshed {path} from {task["prompt_config"]}')


def audit_roots(config):
    from watch_eval_audit import command_audit_roots
    return command_audit_roots(argparse.Namespace(
        labels=[task['definition'] for task in config['tasks']],
        output_roots=[task['output_root'] for task in config['tasks']])) == 0


def check_interrupted(interrupted):
    if interrupted:
        raise SystemExit(128 + interrupted[0])


def wait_for_retry(seconds, interrupted):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check_interrupted(interrupted)
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))


def watch(config, args, modal, interrupted, source_path):
    import run_modal_profile as supervisor

    # Hold a separate lock across audits and retries, while preserving the
    # supervisor's run.lock for compatibility with standalone/older owners.
    with supervisor.run_lock(args.state_dir, name='watch.lock'):
        source_bytes = source_path.read_bytes()
        supervisor.write_json(args.state_dir / 'effective_config.json', config)
        (args.state_dir / 'source_config.json').write_bytes(source_bytes)
        supervisor.log(f'watching {config["run_name"]}: model={config["model"]}, GPU={args.gpu}')
        while True:
            check_interrupted(interrupted)
            try:
                with supervisor.run_lock(args.state_dir):
                    supervisor.recover_orphan(args, modal)
            except supervisor.AlreadyRunning:
                supervisor.log('another Modal supervisor still owns this run')
                wait_for_retry(config['retry_interval_seconds'], interrupted)
                continue
            pids = running_eval_pids(config)
            if pids:
                supervisor.log(f'evaluation processes still running: {pids}')
                wait_for_retry(config['retry_interval_seconds'], interrupted)
                continue
            try:
                refresh_plans(config)
            except (OSError, ValueError) as exc:
                supervisor.log(f'plan refresh failed: {exc}; retrying')
                wait_for_retry(config['retry_interval_seconds'], interrupted)
                continue
            check_interrupted(interrupted)
            if audit_roots(config):
                supervisor.log('all roots complete')
                return 0
            try:
                status = supervisor.execute(args, modal, interrupted)
            except supervisor.AlreadyRunning:
                status = 73
            supervisor.log(f'Modal eval supervisor exited rc={status}')
            if status in (2, 130, 143):
                return status
            wait_for_retry(config['retry_interval_seconds'], interrupted)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--validate-only', action='store_true',
                        help='validate config and runner imports without allocating a GPU or creating run state')
    options = parser.parse_args(argv)
    import run_modal_profile as supervisor

    interrupted = []
    def handle_signal(signum, _frame):
        if not interrupted:
            interrupted.append(signum)
    previous = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        path = options.config.expanduser().resolve()
        config = load_config(path)
        args = supervisor_args(config)
        supervisor.validate(args)
        check_interrupted(interrupted)
        if options.validate_only:
            result = subprocess.run([*args.command, '--help'], cwd=args.working_dir,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                    timeout=120, stdout=subprocess.DEVNULL)
            check_interrupted(interrupted)
            return 0 if result.returncode == 0 else 2
        import modal
        modal.Client.from_env()
        supervisor.docker('info', '--format', '{{json .SecurityOptions}}')
        return watch(config, args, modal, interrupted, path)
    except supervisor.AlreadyRunning as exc:
        supervisor.log(str(exc))
        return 73
    except Exception as exc:
        check_interrupted(interrupted)
        supervisor.log(f'cannot launch experiment: {exc}')
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
