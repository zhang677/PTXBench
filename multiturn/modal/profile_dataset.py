#!/usr/bin/env python3
"""Prepare Hugging Face workloads for the shared leased-GPU launchers.

The remote container must already have PTXBench mounted at /workspace, including
/workspace/scripts/entrypoint.sh and uv. Downloads use an isolated cache and a
selected-workload dataset; PTXBench installs/starts it with dataset mode existing.
HF_TOKEN is forwarded through SSH stdin, never command arguments or saved config.

Local validation (no network, Docker, or writes):
    python profile_dataset.py CONFIG.json --validate-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys


def dataset_settings(config):
    settings = config.get('profile_service', {}).get('dataset')
    if settings is None:
        return None
    keys = {'mode', 'repo_id', 'revision', 'cache_dir', 'timeout_seconds'}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError('profile_service.dataset requires exactly: ' + ', '.join(sorted(keys)))
    if settings['mode'] != 'huggingface':
        raise ValueError('profile_service.dataset.mode must be huggingface')
    if not isinstance(settings['repo_id'], str) or not re.fullmatch(
            r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', settings['repo_id']):
        raise ValueError('dataset.repo_id must be a Hugging Face namespace/repository')
    for key in ('revision', 'cache_dir'):
        value = settings[key]
        if not isinstance(value, str) or not value.strip() or any(c in value for c in '\n\r\0'):
            raise ValueError(f'dataset.{key} must be a nonempty single-line string')
    path = PurePosixPath(settings['cache_dir'])
    if not path.is_absolute() or '..' in path.parts or ':' in str(path) or str(path) == '/':
        raise ValueError('dataset.cache_dir must be an absolute container directory without .. or :')
    timeout = settings['timeout_seconds']
    if type(timeout) is not int or timeout <= 0:
        raise ValueError('dataset.timeout_seconds must be a positive integer')
    if config['profile_service'].get('restart_script') != '/workspace/scripts/restart_profiling.sh':
        raise ValueError('Hugging Face preparation requires the PTXBench /workspace/scripts layout')
    selectors(config)
    return dict(settings)


def selectors(config):
    runs = config.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('runs must be a nonempty array')
    pairs = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError('each run must be an object')
        pair = tuple(run.get(key) for key in ('definition', 'workload_uuid'))
        if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', v) for v in pair):
            raise ValueError('each run needs a safe definition and workload_uuid')
        pairs.add(pair)
    return sorted(pairs)


def blob_paths(workload):
    paths = set()
    for group in ('inputs', 'outputs'):
        for spec in (workload.get(group) or {}).values():
            if spec.get('type') != 'safetensors':
                continue
            path = PurePosixPath(spec['path'])
            if path.is_absolute() or '..' in path.parts or path.parts[:1] != ('blob',):
                raise ValueError(f'unsafe dataset blob path: {spec["path"]!r}')
            paths.add(str(path))
    return paths


def prepare_dataset(settings, selected, token=None, *, api=None, download=None):
    """Run inside the profiler; resolve one revision, validate, then fetch blobs."""
    if api is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi(token=token)
        download = download or hf_hub_download
    info = api.dataset_info(settings['repo_id'], revision=settings['revision'],
                            files_metadata=True, timeout=30)
    if not re.fullmatch(r'[0-9a-f]{40}', info.sha or ''):
        raise ValueError('Hugging Face did not return a full dataset commit SHA')
    files = {item.rfilename: item for item in info.siblings}
    base = Path(settings['cache_dir']) / settings['repo_id'].replace('/', '--') / info.sha
    raw = base / 'metadata'
    selected = sorted(set(tuple(pair) for pair in selected))
    selection_id = hashlib.sha256(json.dumps(selected).encode()).hexdigest()[:20]
    target = base / 'selected' / selection_id

    def fetch(filename, directory):
        result = Path(download(repo_id=settings['repo_id'], repo_type='dataset',
                               revision=info.sha, filename=filename,
                               local_dir=str(directory), token=token))
        size = files[filename].size
        if not result.is_file() or (size is not None and result.stat().st_size != size):
            raise ValueError(f'incomplete dataset download: {filename}')
        return result

    definitions = {}
    rows = {}
    blobs = set()
    metadata_files = []
    for definition in sorted({pair[0] for pair in selected}):
        paths = {}
        for kind, suffix in (('definitions', '.json'), ('workloads', '.jsonl')):
            candidates = [name for name in files if name.startswith(kind + '/')
                          and PurePosixPath(name).name == definition + suffix]
            if len(candidates) != 1:
                raise ValueError(f'expected one {kind} file for {definition}, found {len(candidates)}')
            filename = candidates[0]
            if '..' in PurePosixPath(filename).parts:
                raise ValueError(f'unsafe metadata path: {filename}')
            paths[kind] = fetch(filename, raw)
            metadata_files.append(filename)
        value = json.loads(paths['definitions'].read_text())
        if value.get('name') != definition:
            raise ValueError(f'dataset returned the wrong definition for {definition}')
        definitions[definition] = paths['definitions']
        candidates = [json.loads(line) for line in paths['workloads'].read_text().splitlines()
                      if line.strip()]
        rows[definition] = []
        for _, workload_uuid in (pair for pair in selected if pair[0] == definition):
            matches = [row for row in candidates if row.get('definition') == definition
                       and row.get('workload', {}).get('uuid') == workload_uuid]
            if len(matches) != 1:
                raise ValueError(f'expected one workload {definition}/{workload_uuid}, found {len(matches)}')
            rows[definition].append(matches[0])
            blobs.update(blob_paths(matches[0]['workload']))
    missing = blobs - files.keys()
    if missing:
        raise ValueError('dataset is missing referenced blobs: ' + ', '.join(sorted(missing)))

    # Validate every requested UUID before transferring large blobs. The served
    # subset excludes unrelated workload rows whose blobs were not requested.
    for filename in sorted(blobs):
        fetch(filename, target)
    for kind in ('definitions', 'workloads'):
        (target / kind).mkdir(parents=True, exist_ok=True)
    for definition, source in definitions.items():
        shutil.copyfile(source, target / 'definitions' / (definition + '.json'))
        (target / 'workloads' / (definition + '.jsonl')).write_text(
            ''.join(json.dumps(row) + '\n' for row in rows[definition]))
    source = dict(repo_id=settings['repo_id'], requested_revision=settings['revision'],
                  resolved_revision=info.sha, workloads=selected,
                  metadata_files=sorted(metadata_files), blob_files=sorted(blobs))
    (target / 'dataset_source.json').write_text(json.dumps(source, indent=2) + '\n')
    return {'dataset_dir': str(target), 'source': source}


def remote_command(profile, argv):
    return ['ssh', *profile.get('ssh_options', []), profile['host'], shlex.join(argv)]


def prepare_and_start(config, *, run=subprocess.run):
    """Download in the existing container, then invoke PTXBench's real entrypoint."""
    settings = dataset_settings(config)
    if settings is None:
        raise ValueError('profile_service.dataset is required')
    profile = config['profile_service']
    timeout = settings['timeout_seconds']
    # timeout also runs remotely: losing the SSH client cannot leave an unbounded
    # download/install process running in the container.
    bounded = ['timeout', '--signal=TERM', '--kill-after=10', str(timeout)]
    command = ['docker', 'exec', '-i', profile['container'], *bounded,
               'uv', 'run', '--no-project', '--with', 'huggingface_hub',
               'python', '-c', Path(__file__).read_text(), '--worker']
    payload = dict(settings=settings, selected=selectors(config), token=os.environ.get('HF_TOKEN'))
    print(f'[profile-dataset] downloading {settings["repo_id"]}@{settings["revision"]}', flush=True)
    completed = run(remote_command(profile, command), input=json.dumps(payload),
                    stdout=subprocess.PIPE, text=True, check=True, timeout=timeout + 30)
    result = json.loads(completed.stdout)
    directory = result['dataset_dir']
    root = PurePosixPath(settings['cache_dir'])
    path = PurePosixPath(directory)
    if not path.is_relative_to(root) or '..' in path.parts or ':' in directory or '\n' in directory:
        raise ValueError('remote downloader returned an invalid dataset directory')
    env = {'FIB_DATASET_MODE': 'existing', 'FIB_DATASET_DIR': directory,
           'FIB_DATASET_DIRS': directory, 'FIB_DEVICES': profile['fib_devices'],
           'PROFILE_MAX_GPUS': str(profile['profile_max_gpus']),
           'PROFILE_PORT': str(profile['remote_port'])}
    if 'gpu_cooldown_sleep_s' in profile:
        env['FIB_GPU_COOLDOWN_SLEEP_S'] = str(profile['gpu_cooldown_sleep_s'])
    command = ['docker', 'exec']
    for key, value in env.items():
        command.extend(['-e', f'{key}={value}'])
    command.extend([profile['container'], *bounded, 'bash', '/workspace/scripts/entrypoint.sh'])
    print(f'[profile-dataset] starting PTXBench with {directory}', flush=True)
    run(remote_command(profile, command), text=True, check=True, timeout=timeout + 30)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, nargs='?')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        payload = json.load(sys.stdin)
        result = prepare_dataset(**payload)
        print(json.dumps(result))
    elif args.config and args.validate_only:
        config = json.loads(args.config.read_text())
        print(json.dumps({'dataset': dataset_settings(config), 'workloads': selectors(config)}, indent=2))
    else:
        parser.error('use CONFIG.json --validate-only; launch through run_modal_coding_agent.py')


if __name__ == '__main__':
    main()
