"""Capture and replay FLA configurations for the Modal B200 GDN backward oracle."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile

DEFINITION = 'gdn_bwd_qk8_v16_d128_k_last'
KERNELS = {
    'chunk_local_cumsum_scalar_kernel': 'fla.ops.utils.cumsum',
    'chunk_gated_delta_rule_fwd_kernel_h_blockdim64': 'fla.ops.common.chunk_delta_h',
    'chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64': 'fla.ops.common.chunk_delta_h',
    'chunk_bwd_kernel_dqkwg': 'fla.ops.common.chunk_o',
    'chunk_bwd_kernel_dv_local': 'fla.ops.common.chunk_o',
    'recompute_w_u_fwd_kernel': 'fla.ops.gated_delta_rule.wy_fast',
    'prepare_wy_repr_bwd_kernel': 'fla.ops.gated_delta_rule.wy_fast',
}
_recorded = set()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    try:
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def config_dict(config):
    return dict(kwargs=config.kwargs, num_warps=config.num_warps,
                num_stages=config.num_stages, num_ctas=config.num_ctas,
                maxnreg=config.maxnreg)


def tuners():
    result = {}
    for name, module_name in KERNELS.items():
        module = importlib.import_module(module_name)
        parent, attribute, obj = module, name, getattr(module, name)
        while not hasattr(obj, 'configs') and hasattr(obj, 'fn'):
            parent, attribute, obj = obj, 'fn', obj.fn
        if not hasattr(obj, 'best_config'):
            raise RuntimeError(f'GDN reference configuration not available for {name}')
        result[name] = (module, parent, attribute, obj)
    return result


def record_reference(definition, workload, cfg, device, runnable=None, inputs=None):
    if definition.name != DEFINITION or os.environ.get('PROFILE_GPU') != 'B200':
        return
    import torch
    import triton
    from importlib.metadata import version

    if 'B200' not in torch.cuda.get_device_name(device):
        raise RuntimeError('GDN reference capture expected an actual B200')
    from triton.backends.nvidia.compiler import get_ptxas
    compiler = get_ptxas(100)
    compiler_record = dict(path=compiler.path, version=compiler.version,
                           sha256=sha256(Path(compiler.path).read_bytes()))
    found = tuners()
    kernels = {}
    sources = {}
    for name, (module, _, _, tuner) in found.items():
        module_name = KERNELS[name]
        relative = module_name.replace('.', '/') + '.py'
        source = Path(module.__file__).read_bytes()
        sources[relative] = source
        kernels[name] = dict(module=module_name, source_relative=relative,
                             source_sha256=sha256(source), config=config_dict(tuner.best_config),
                             tuner_type=f'{type(tuner).__module__}.{type(tuner).__name__}',
                             cache_entries=[dict(key=repr(key), config=config_dict(value))
                                            for key, value in getattr(tuner, 'cache', {}).items()
                                            if hasattr(value, 'num_warps')])
    if os.environ.get('LOCK_GDN_BWD_REFERENCE') == '1':
        manifest = json.loads(Path(os.environ.get('GDN_REFERENCE_MANIFEST',
                              '/workspace/gdn-reference-lock.json')).read_text())
        if compiler_record['sha256'] != manifest['compiler']['sha256'] or compiler_record['version'] != manifest['compiler']['version']:
            raise RuntimeError('Actual Blackwell assembler differs from the reference lock')
        if workload.uuid != manifest['workload_uuid'] or workload.axes != manifest['axes']:
            raise RuntimeError('GDN reference lock received an unvalidated workload')
        for name, row in kernels.items():
            expected = manifest['kernels'][name]
            if row['config'] != expected['config'] or row['source_sha256'] != manifest['sources'][row['source_relative']]['patched_sha256']:
                raise RuntimeError(f'Actual GDN reference configuration differs from lock: {name}')
    signature = sha256(json.dumps(kernels, sort_keys=True).encode())
    if signature in _recorded:
        return
    record = dict(schema_version=1, definition=definition.name, workload_uuid=workload.uuid,
                  axes=workload.axes, pid=os.getpid(), gpu=torch.cuda.get_device_name(device),
                  cuda_capability=list(torch.cuda.get_device_capability(device)),
                  versions={'torch': torch.__version__, 'triton': triton.__version__,
                            'fla-core': version('fla-core')},
                  eval_config=cfg.model_dump(), kernels=kernels, compiler=compiler_record,
                  triton_cache_dir=os.environ.get('TRITON_CACHE_DIR'),
                  lock_enabled=os.environ.get('LOCK_GDN_BWD_REFERENCE') == '1')
    root = Path(os.environ.get('GDN_REFERENCE_RECORD_DIR', '/workspace/gdn-reference-records'))
    for relative, source in sources.items():
        atomic_write(root / 'sources' / relative, source)
    path = root / f'reference-{os.getpid()}-{signature[:16]}.json'
    atomic_write(path, (json.dumps(record, indent=2) + '\n').encode())
    # This also preserves selected configurations in streamed sandbox logs.
    print('MODAL_GDN_REFERENCE_CONFIG ' + json.dumps(record, sort_keys=True), flush=True)
    _recorded.add(signature)


def install_capture_hook(path):
    path = Path(path)
    before = path.read_bytes()
    marker = '        # Modal B200: record the actual reference worker before timing.\n'
    hook = marker + (
        '        if definition.name == "gdn_bwd_qk8_v16_d128_k_last":\n'
        '            from modal_gdn_reference import record_reference\n'
        '            record_reference(definition, workload, cfg, device, ref_runnable, inputs)\n\n')
    anchor = '        if cfg.profile_baseline:\n            latencies: List[float] = []\n'
    text = before.decode()
    if hook + anchor in text:
        return dict(status='already_applied', before_sha256=sha256(before), after_sha256=sha256(before))
    if marker in text or text.count(anchor) != 1:
        raise RuntimeError('PTXBench baseline capture hook source changed')
    after = text.replace(anchor, hook + anchor).encode()
    compile(after, str(path), 'exec')
    atomic_write(path, after)
    return dict(status='applied', before_sha256=sha256(before), after_sha256=sha256(after))


def patched_source(source, kernel_configs):
    """Freeze only the named autotune decorators, retaining their original keys/hooks."""
    text = source.decode()
    lines = text.splitlines(keepends=True)
    edits = []
    found = set()
    for node in ast.parse(text).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in kernel_configs:
            continue
        decorators = [d for d in node.decorator_list if isinstance(d, ast.Call) and (
            isinstance(d.func, ast.Name) and d.func.id == 'fla_cache_autotune' or
            isinstance(d.func, ast.Attribute) and d.func.attr == 'autotune')]
        if len(decorators) != 1:
            raise RuntimeError(f'Expected one autotune decorator for {node.name}')
        decorator = decorators[0]
        if decorator.args:
            raise RuntimeError(f'Unexpected positional autotune arguments for {node.name}')
        config = kernel_configs[node.name]
        arguments = ', '.join(f'{k}={v!r}' for k, v in config.items() if k != 'kwargs')
        retained = []
        for keyword in decorator.keywords:
            if keyword.arg in {'configs', 'cache_results'}:
                continue
            if keyword.arg is None:
                if not isinstance(keyword.value, ast.Name) or keyword.value.id != 'autotune_cache_kwargs':
                    raise RuntimeError(f'Unexpected expanded autotune arguments for {node.name}')
                continue
            retained.append(f'    {keyword.arg}={ast.unparse(keyword.value)},\n')
        replacement = (f'@{ast.unparse(decorator.func)}(\n'
                       f'    configs=[triton.Config({config["kwargs"]!r}, {arguments})],\n'
                       + ''.join(retained) + '    cache_results=False,\n)\n')
        edits.append((decorator.lineno - 1, decorator.end_lineno, replacement))
        found.add(node.name)
    if found != set(kernel_configs):
        raise RuntimeError(f'Unmatched FLA kernels: {sorted(set(kernel_configs) - found)}')
    for start, stop, replacement in sorted(edits, reverse=True):
        lines[start:stop] = [replacement]
    result = ''.join(lines).encode()
    compile(result, '<locked FLA source>', 'exec')
    return result


def apply_lock(manifest, gpu):
    """Validate the entire installed source set before replacing any file."""
    from importlib.metadata import distribution, version
    import torch
    import triton

    if 'B200' not in gpu or os.environ.get('PROFILE_GPU') != 'B200':
        raise RuntimeError('GDN reference lock requires B200')
    if os.environ.get('FLA_CACHE_MODE', 'disabled') != 'disabled':
        raise RuntimeError('GDN reference lock requires FLA_CACHE_MODE=disabled')
    versions = {'torch': torch.__version__, 'triton': triton.__version__, 'fla-core': version('fla-core')}
    if versions != manifest['versions']:
        raise RuntimeError(f'GDN reference lock runtime versions changed: {versions}')
    if set(manifest['kernels']) != set(KERNELS):
        raise RuntimeError('GDN reference lock kernel manifest is incomplete')
    distribution_root = distribution('fla-core')
    edits = []
    for relative, hashes in manifest['sources'].items():
        selected = {name: row['config'] for name, row in manifest['kernels'].items()
                    if row['source_relative'] == relative}
        if not selected or relative not in {name.replace('.', '/') + '.py' for name in KERNELS.values()}:
            raise RuntimeError(f'Unexpected GDN reference source: {relative}')
        path = Path(distribution_root.locate_file(relative))
        before = path.read_bytes()
        digest = sha256(before)
        if digest == hashes['original_sha256']:
            after = patched_source(before, selected)
            status = 'applied'
        elif digest == hashes['patched_sha256']:
            after = before
            status = 'already_applied'
        else:
            raise RuntimeError(f'GDN reference source changed: {relative} {digest}')
        if sha256(after) != hashes['patched_sha256']:
            raise RuntimeError(f'GDN reference patch hash mismatch: {relative}')
        edits.append((path, after, dict(source_relative=relative, before_sha256=digest,
                                      after_sha256=sha256(after), status=status)))
    expected = {row['source_relative'] for row in manifest['kernels'].values()}
    if set(manifest['sources']) != expected:
        raise RuntimeError('GDN reference source manifest is incomplete')
    for path, after, row in edits:
        if row['status'] == 'applied':
            atomic_write(path, after)
    return dict(id=manifest['id'], provenance=manifest['provenance'], versions=versions,
                sources=[row for _, _, row in edits],
                kernels={name: row['config'] for name, row in manifest['kernels'].items()})


def prepare_compiler(manifest, alias='/workspace/gdn-reference-ptxas'):
    """Use FlashInfer's supported override to prevent a mid-run compiler switch."""
    import triton
    binary = Path(triton.__file__).parent / 'backends/nvidia/bin/ptxas-blackwell'
    if sha256(binary.read_bytes()) != manifest['compiler']['sha256']:
        raise RuntimeError('Bundled Blackwell assembler differs from the validated reference')
    alias = Path(alias)
    if alias.exists() or alias.is_symlink():
        if not alias.is_symlink() or alias.resolve() != binary.resolve():
            raise RuntimeError('Unexpected existing Blackwell assembler alias')
    else:
        alias.symlink_to(binary)
    os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = str(alias)
    from triton.backends.nvidia.compiler import get_ptxas
    selected = get_ptxas(100)
    if selected.version != manifest['compiler']['version']:
        raise RuntimeError('Selected Blackwell assembler version changed')
    return dict(path=selected.path, version=selected.version,
                sha256=sha256(Path(selected.path).read_bytes()))
