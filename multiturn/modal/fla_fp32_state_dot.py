"""Accumulate FLA's chunk-end state dot (`tl.sum(b_h * b_dh)`) in FP32 for GDN/KDA backward.

Triton keeps BF16 as the `tl.sum` accumulator, so the reference `dg` depends on the
register layout (num_warps, BV). Casting both tiles first makes the sum an FP32
reduction.

The Modal startup runs shell_step() in the profiling sandbox before the service
starts, with the same code on every GPU. Patching requires every installed file to
hash to the known FLA 0.5.2 source first. The step always runs: with
FLA_FP32_STATE_DOT unset it restores a previously patched file and leaves
unrecognized sources alone. It is independent of, and not combinable with, the
B200 reference lock.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

ENV = 'FLA_FP32_STATE_DOT'
OLD, NEW = 'tl.sum(b_h * b_dh', 'tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32)'
# relative path -> (original line, original sha256, patched sha256); FLA 0.5.2.
PATCHES = {
    'fla/ops/common/chunk_o.py': (
        '            b_dg_last += (tl.sum(b_h * b_dh))\n',
        'f174d431ce67139204a0916f638b5af158dd5917e2cb37c0ca03f5ce735466a0',
        'af44ab2722921a8547f2d29348f448fe7e00dc94ce65b10b3075d916dc78df41'),
    'fla/ops/kda/chunk_bwd.py': (
        '            b_dgk += tl.sum(b_h * b_dh, axis=0)\n',
        '069395c9232647f5a4b82fa2dfe7ca077423016d4b641f2696f9d6328dc2e9be',
        '7651e9c3c219a49cd9bfbd43b86c50115f9a4d31f27654a9f2f29e7a6a1b2e2d'),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def rewrite(source, relative, patch):
    """Return (text, status) for the requested state; raise on any unknown source."""
    line, original, patched = PATCHES[relative]
    current, target = (original, patched) if patch else (patched, original)
    digest = sha256(source)
    if digest == target:
        return source, 'already_applied' if patch else 'original'
    if digest != current and not patch:
        return source, 'unrecognized'
    if digest != current:
        raise RuntimeError(f'FLA FP32 state-dot source changed: {relative} {digest}')
    old, new = line, line.replace(OLD, NEW)
    text = source.decode()
    after = (text.replace(old, new) if patch else text.replace(new, old)).encode()
    if sha256(after) != target:
        raise RuntimeError(f'FLA FP32 state-dot patch hash mismatch: {relative}')
    compile(after, relative, 'exec')
    return after, 'applied' if patch else 'reverted'


def apply(patch=True, root=None):
    """Validate every source before writing any, then patch (or restore) them in place."""
    if root is None:
        from importlib.metadata import distribution
        root = distribution('fla-core').locate_file('')
    edits = []
    for relative in PATCHES:
        path = Path(root, relative)
        before = path.read_bytes()
        after, status = rewrite(before, relative, patch)
        edits.append((path, after, dict(source_relative=relative, before_sha256=sha256(before),
                                        after_sha256=sha256(after), status=status)))
    for path, after, row in edits:
        if row['status'] in ('applied', 'reverted'):
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                stream.write(after)
            Path(stream.name).chmod(path.stat().st_mode & 0o777)
            Path(stream.name).replace(path)
    return dict(patch='fla_fp32_state_dot', enabled=patch, sources=[row for _, _, row in edits])


def shell_step(python='/workspace/acc/bin/python', target='/workspace/fla_fp32_state_dot.py'):
    """Bash that installs this file in the container and sets FLA to $FLA_FP32_STATE_DOT."""
    source = Path(__file__).read_text()
    if '\nFLA_FP32_HELPER\n' in source:
        raise RuntimeError('fla_fp32_state_dot.py contains its heredoc delimiter')
    return (f"cat > {target} <<'FLA_FP32_HELPER'\n{source}FLA_FP32_HELPER\n"
            f'{python} {target} "$([ "${{FLA_FP32_STATE_DOT:-0}}" = 1 ] && echo on || echo off)"\n')


if __name__ == '__main__':
    import json
    import sys
    if sys.argv[1:] not in (['on'], ['off']):
        raise SystemExit('usage: fla_fp32_state_dot.py on|off')
    print('FLA_FP32_STATE_DOT ' + json.dumps(apply(sys.argv[1] == 'on')), flush=True)
