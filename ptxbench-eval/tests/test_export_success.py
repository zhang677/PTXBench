import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ptxbench_eval.export_success import export_run
from ptxbench_eval.registry import Registry


def make_run(root, language="cuda", harness="codex", feedback_style="multiturn"):
    manifest = {
        "schema_version": 1, "task_id": "task", "definition": "test", "workload_uuid": "uuid",
        "language": language, "target_hardware": ["H100"],
        "definition_payload": {"name": "test", "reference": "def run(x): return x"},
    }
    if language == "cuda":
        manifest.update(binding="tvm-ffi", nvcc_gencode="arch=compute_90a,code=sm_90a")
    registry = Registry(root)
    registry.create_run(run_id="exp_kda_bwd_000", manifest=manifest, agent_prompt="task",
                        token_budget=100, harness=harness, model="test", prompt_tag="test",
                        feedback_style=feedback_style)
    return registry


def trace(status="PASSED", speedup=0.25):
    return {"definition": "test", "workload": {"uuid": "uuid", "axes": {"B": 4}},
            "evaluation": {"status": status, "timestamp": "2026-10-18T12:00:00+00:00",
                           "performance": {"speedup_factor": speedup}, "log": "full diagnostics"}}


def append(registry, action, source="// kernel\n", *, traces=None, passed=True):
    registry.append_turn(run_id="exp_kda_bwd_000", action_id=action, source=source,
                         feedback="saved evaluation", traces=traces if traces is not None else [trace()],
                         all_passed=passed, min_speedup=0.25, usage_snapshot={}, replay_count=0)


@pytest.mark.parametrize("language,suffix", [("cuda", ".cu"), ("triton", ".py"), ("cutedsl", ".py")])
@pytest.mark.parametrize("harness", ["codex", "antigravity"])
@pytest.mark.parametrize("feedback_style", ["default", "multiturn"])
def test_live_exports_match_gemini_format(tmp_path, language, suffix, harness, feedback_style):
    registry = make_run(tmp_path, language, harness, feedback_style)
    target = tmp_path / "success/exp_kda_bwd_000"
    append(registry, "failed", traces=[trace("INCORRECT_NUMERICAL")], passed=False)
    assert not target.exists()
    source = "// café\r\n  \t\n" if language == "cuda" else "def run(x):\n    pass\n  \t\n"
    append(registry, "passing", source)
    append(registry, "failed-again", traces=[], passed=False)
    append(registry, "passing-again", source)
    # Same action must not become another turn or version on replay.
    append(registry, "passing-again", source)
    records = json.loads((target / "record.json").read_text())
    assert [(r["version"], r["turn"]) for r in records] == [(0, 1), (1, 3)]
    for record in records:
        assert set(record) == {"version", "turn", "timestamp", "traces"}
        assert record["traces"] == [trace()]
        assert record["timestamp"]
        assert (target / f'kernel_v{record["version"]}{suffix}').read_bytes() == source.encode()
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()}
    result = export_run(tmp_path, "exp_kda_bwd_000")
    assert result["kernels"] == 2
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in target.iterdir()}


def old_trajectory(registry, source="// original\n\n  "):
    append(registry, "failed", traces=[], passed=False)
    append(registry, "pass", source)
    path = registry.trajectory_path("exp_kda_bwd_000")
    data = json.loads(path.read_text())
    data["messages"][-2]["content"] = f"```cpp\n{source.rstrip()}\n```"
    del data["messages"][-2]["extra"]["ptxbench"]["source_sha256"]
    del data["messages"][-1]["extra"]["ptxbench"]["evaluated_at"]
    path.write_text(json.dumps(data))
    events = [
        {"action_id": "pass", "event": "evaluation_started", "source_sha256": hashlib.sha256(source.encode()).hexdigest()},
        {"action_id": "pass", "event": "profile_submit_request", "request": {"solution": {"sources": [{"content": source}]}}},
    ]
    registry.raw_events_path("exp_kda_bwd_000").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return source


def test_cli_migrates_legacy_export_and_preserves_backup(tmp_path):
    registry = make_run(tmp_path)
    source = old_trajectory(registry)
    target = tmp_path / "success/exp_kda_bwd_000"
    for path in target.iterdir():
        path.unlink()
    (target / "kernel_v2.cu").write_text(source)
    (target / "manifest.json").write_text('{"old": true}\n')
    (target / "ptx").mkdir()
    (target / "ptx/kernel_v2.ptx").write_text("PTX evidence")
    original = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="--backup-existing"):
        export_run(tmp_path, "exp_kda_bwd_000")
    result = export_run(tmp_path, "exp_kda_bwd_000", backup_existing=True, dry_run=True)
    assert result["would_backup"] and not (tmp_path / "success-backups").exists()
    cli = subprocess.run([sys.executable, "-m", "ptxbench_eval.export_success", "--run-dir", str(tmp_path),
                          "--backup-existing"], text=True, capture_output=True)
    assert cli.returncode == 0, cli.stderr
    result = json.loads(cli.stdout)
    backup = Path(result["backup"])
    assert original == {str(p.relative_to(backup)): p.read_bytes() for p in backup.rglob("*") if p.is_file()}
    assert sorted(p.name for p in target.iterdir()) == ["kernel_v0.cu", "record.json"]
    assert (target / "kernel_v0.cu").read_bytes() == source.encode()
    assert json.loads((target / "record.json").read_text()) == [
        {"version": 0, "turn": 1, "timestamp": trace()["evaluation"]["timestamp"], "traces": [trace()]}
    ]
    assert export_run(tmp_path, "exp_kda_bwd_000", backup_existing=True)["backup"] is None


@pytest.mark.parametrize("corruption", ["source", "hash", "action", "traces", "turn"])
def test_corrupt_evidence_never_replaces_existing_artifacts(tmp_path, corruption):
    registry = make_run(tmp_path)
    append(registry, "pass")
    path = registry.trajectory_path("exp_kda_bwd_000")
    data = json.loads(path.read_text())
    if corruption == "source":
        data["messages"][-2]["content"] = "```cpp\n// changed\n\n```"
    elif corruption == "hash":
        data["messages"][-2]["extra"]["ptxbench"]["source_sha256"] = "bad"
    elif corruption == "action":
        data["messages"][-1]["extra"]["ptxbench"]["action_id"] = "other"
    elif corruption == "traces":
        data["messages"][-1]["extra"]["traces"].append(trace("INCORRECT_NUMERICAL"))
    else:
        data["messages"][-1]["extra"]["rollout"]["turn_idx"] = 8
    path.write_text(json.dumps(data))
    target = tmp_path / "success/exp_kda_bwd_000"
    original = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(ValueError):
        export_run(tmp_path, "exp_kda_bwd_000", backup_existing=True)
    assert original == {p.name: p.read_bytes() for p in target.iterdir()}
    assert not (tmp_path / "success-backups").exists()


def test_old_export_checks_submitted_source_hash(tmp_path):
    registry = make_run(tmp_path)
    old_trajectory(registry)
    events = registry.raw_events_path("exp_kda_bwd_000")
    events.write_text(events.read_text().replace('"content": "// original', '"content": "// changed'))
    with pytest.raises(ValueError, match="hash mismatch"):
        export_run(tmp_path, "exp_kda_bwd_000", backup_existing=True)


def test_legacy_fallback_and_missing_source_repair(tmp_path):
    registry = make_run(tmp_path)
    old_trajectory(registry, "// original\n")
    registry.raw_events_path("exp_kda_bwd_000").unlink()
    export_run(tmp_path, "exp_kda_bwd_000", backup_existing=True)
    target = tmp_path / "success/exp_kda_bwd_000/kernel_v0.cu"
    target.unlink()
    export_run(tmp_path, "exp_kda_bwd_000")
    assert target.read_bytes() == b"// original\n"


def test_export_failure_does_not_discard_completed_evaluation(tmp_path, caplog):
    registry = make_run(tmp_path)
    target = tmp_path / "success/exp_kda_bwd_000"
    target.mkdir()
    (target / "kernel_v0.cu").write_text("unrelated artifact")
    append(registry, "pass")
    assert "Could not export success artifacts" in caplog.text
    assert json.loads(registry.trajectory_path("exp_kda_bwd_000").read_text())["messages"][-1]["extra"]["all_passed"]
    assert (target / "kernel_v0.cu").read_text() == "unrelated artifact"


def test_concurrent_backfills_do_not_duplicate_versions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    registry = make_run(tmp_path)
    append(registry, "pass")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: export_run(tmp_path, "exp_kda_bwd_000"), range(12)))
    assert all(r["kernels"] == 1 for r in results)
    assert len(json.loads((tmp_path / "success/exp_kda_bwd_000/record.json").read_text())) == 1
