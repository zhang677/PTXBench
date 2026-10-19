"""Write Gemini-compatible success artifacts from canonical agent trajectories.

Used by Registry after every evaluation and by the CLI to backfill saved runs.
Only fully passing evaluations are exported; speedup is not a selection filter.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import uuid

from .util import atomic_write_json, file_lock, read_json


def _run_id(value: str) -> str:
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError(f"invalid run ID: {value!r}")
    return value


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _archived_sources(root: Path, run_id: str, actions: set[str]) -> dict[str, str]:
    """Recover exact bytes stripped by older versions of Registry.append_turn."""
    path = root / "raw_profile_events" / f"{run_id}.jsonl"
    if not actions or not path.exists():
        return {}
    sources: dict[str, str] = {}
    hashes: dict[str, str] = {}
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            action = event.get("action_id")
            if action not in actions:
                continue
            if event.get("event") == "evaluation_started":
                digest = event.get("source_sha256")
                if digest:
                    if action in hashes and hashes[action] != digest:
                        raise ValueError(f"conflicting source hashes for {action}")
                    hashes[action] = digest
            if event.get("event") != "profile_submit_request":
                continue
            items = event.get("request", {}).get("solution", {}).get("sources", [])
            if len(items) != 1 or not isinstance(items[0].get("content"), str):
                raise ValueError(f"expected one submitted source for {action}")
            source = items[0]["content"]
            if action in sources and sources[action] != source:
                raise ValueError(f"conflicting submitted sources for {action}")
            sources[action] = source
    for action, source in sources.items():
        if action in hashes and _digest(source) != hashes[action]:
            raise ValueError(f"submitted source hash mismatch for {action}")
    return sources


def collect_success(
    root: Path, run_id: str, trajectory: dict[str, Any], language: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    if language not in {"cuda", "triton", "cutedsl"}:
        raise ValueError(f"unsupported kernel language: {language!r}")
    suffix = ".cu" if language == "cuda" else ".py"
    messages = trajectory["messages"]
    passing = []
    turn = -1
    seen_actions = set()
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        turn += 1
        if index + 1 == len(messages):
            continue  # A submission without saved evaluation is not a success.
        feedback = messages[index + 1]
        extra = feedback.get("extra") or {}
        traces = extra.get("traces")
        if feedback.get("role") != "user" or extra.get("all_passed") is not True:
            continue
        if not isinstance(traces, list) or not traces or not all(
            isinstance(t, dict) and t.get("evaluation", {}).get("status") == "PASSED"
            for t in traces
        ):
            raise ValueError(f"turn {turn} claims success without fully passing traces")
        meta = (message.get("extra") or {}).get("ptxbench") or {}
        action = meta.get("action_id")
        result_meta = extra.get("ptxbench") or {}
        if not action or result_meta.get("action_id") != action or action in seen_actions:
            raise ValueError(f"turn {turn} has mismatched or duplicate action IDs")
        seen_actions.add(action)
        saved_turn = (extra.get("rollout") or {}).get("turn_idx", turn)
        if type(saved_turn) is not int or saved_turn != turn:
            raise ValueError(f"turn index mismatch at turn {turn}")
        passing.append((turn, message, extra, meta, result_meta))

    archived = _archived_sources(root, run_id, {
        meta["action_id"] for _, _, _, meta, _ in passing if "source_sha256" not in meta
    })
    records = []
    files = {}
    for turn, message, extra, meta, result_meta in passing:
        match = re.fullmatch(r"```(?:cpp|cuda|c\+\+|python|py)\n([\s\S]*)\n```", message["content"])
        if not match:
            raise ValueError(f"turn {turn} is not a canonical source block")
        source = match.group(1)
        if "source_sha256" in meta:
            if _digest(source) != meta["source_sha256"]:
                raise ValueError(f"source hash mismatch at turn {turn}")
        else:
            # Old canonical blocks used source.rstrip() + a newline. Prefer
            # recorded submissions, otherwise retain that canonical newline.
            source = archived.get(meta["action_id"], source + "\n")
            if source.rstrip() != match.group(1):
                raise ValueError(f"submitted source disagrees with trajectory at turn {turn}")
        traces = extra["traces"]
        timestamp = result_meta.get("evaluated_at") or next((
            t["evaluation"]["timestamp"] for t in reversed(traces)
            if t["evaluation"].get("timestamp")
        ), "")
        version = len(records)
        record = {"version": version, "turn": turn, "timestamp": timestamp, "traces": traces}
        if "verification" in extra:
            record["verification"] = extra["verification"]
        records.append(record)
        files[f"kernel_v{version}{suffix}"] = source.encode("utf-8")
    return records, files


def _atomic_write_source(path: Path, content: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_trajectory(
    root: Path, run_id: str, trajectory: dict[str, Any], language: str,
    *, backup_existing: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Export a snapshot; caller must hold the trajectory lock when writing."""
    _run_id(run_id)
    records, files = collect_success(root, run_id, trajectory, language)
    target = root / "success" / run_id
    conflict = False
    old_records = []
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ValueError(f"expected a regular directory: {target}")
        try:
            old_records = read_json(target / "record.json", [])
            conflict = not isinstance(old_records, list) or old_records != records[:len(old_records)]
            for path in target.iterdir():
                if path.name == "record.json" and path.is_file() and not path.is_symlink():
                    continue
                if (path.is_symlink() or not path.is_file() or path.name not in files
                        or path.read_bytes() != files[path.name]):
                    conflict = True
        except (ValueError, OSError):
            conflict = True
    if conflict and not backup_existing:
        raise ValueError(f"existing success artifacts differ at {target}; use --backup-existing to preserve and replace them")
    result = {"run_id": run_id, "success_dir": str(target), "kernels": len(records),
              "turns": [r["turn"] for r in records], "backup": None, "dry_run": dry_run}
    if dry_run:
        result["would_backup"] = conflict
        return result
    if conflict:
        # Stage a complete replacement before moving any existing evidence.
        stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=target.parent))
        backup = root / "success-backups" / f"{run_id}.{uuid.uuid4().hex}"
        try:
            for name, content in files.items():
                _atomic_write_source(stage / name, content)
            atomic_write_json(stage / "record.json", records)
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.rename(backup)
            try:
                stage.rename(target)
            except BaseException:
                backup.rename(target)
                raise
            result["backup"] = str(backup)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    elif records:
        target.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            if not (target / name).exists():
                _atomic_write_source(target / name, content)
        if old_records != records or not (target / "record.json").exists():
            atomic_write_json(target / "record.json", records)
    return result


def export_run(
    root: str | Path, run_id: str, *, backup_existing: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    run_id = _run_id(run_id)
    path = root / "trajectories" / f"{run_id}.json"
    if not path.is_file():
        raise ValueError(f"missing trajectory: {path}")
    with nullcontext() if dry_run else file_lock(path.with_suffix(".lock")):
        manifest = read_json(root / "manifests" / f"{run_id}.json")
        if not manifest:
            manifest = read_json(root / "runs" / f"{run_id}.json", {}).get("manifest", {})
        return export_trajectory(root, run_id, read_json(path), manifest.get("language"),
                                 backup_existing=backup_existing, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Experiment root containing trajectories/")
    parser.add_argument("--run-id", action="append", help="Select a run ID; repeat as needed (default: all)")
    parser.add_argument("--backup-existing", action="store_true", help="Move conflicting exports to success-backups/ before replacing")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files")
    args = parser.parse_args(argv)
    run_ids = args.run_id or [p.stem for p in sorted((args.run_dir / "trajectories").glob("*.json"))]
    if not run_ids:
        parser.error("no trajectories found")
    try:
        for run_id in run_ids:
            print(json.dumps(export_run(args.run_dir, run_id, backup_existing=args.backup_existing,
                                        dry_run=args.dry_run)))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"success export failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
