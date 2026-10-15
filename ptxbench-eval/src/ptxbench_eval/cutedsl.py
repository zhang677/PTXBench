"""CuTeDSL source reference provenance and CPU-only submission preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .python_kernel import validation_error as validation_error


def source_snapshot(root: Path) -> dict[str, Any]:
    """Validate an exported installed-source snapshot before mounting it."""
    manifest_path = root / "source_manifest.json"
    encoded = manifest_path.read_bytes()
    manifest = json.loads(encoded)
    files = manifest.get("files")
    version = manifest.get("version")
    if (manifest.get("schema_version") != 1 or not isinstance(version, str)
            or not version or not isinstance(files, dict)
            or "cutlass/cute/__init__.py" not in files
            or manifest.get("packages", {}).get("nvidia-cutlass-dsl") != version):
        raise ValueError("invalid CuTeDSL source_manifest.json")
    actual = set()
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"CuTeDSL source contains a symlink: {path}")
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(root).as_posix()
        actual.add(relative)
        content = path.read_bytes()
        total_bytes += len(content)
        if hashlib.sha256(content).hexdigest() != files.get(relative):
            raise ValueError(f"CuTeDSL source hash mismatch: {relative}")
    if actual != set(files):
        raise ValueError("CuTeDSL source file set differs from source_manifest.json")
    return {
        "source": str(root.resolve()),
        "workspace_path": "cutedsl",
        "version": version,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }
