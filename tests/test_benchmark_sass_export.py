import csv
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "mini-ptx-agent"
    / "benchmark"
    / "export_turn_correctness_arch.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_sass_exporter", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)

AGGREGATOR_PATH = SCRIPT_PATH.with_name("aggregate_sass_metrics.py")
AGGREGATOR_SPEC = importlib.util.spec_from_file_location("benchmark_sass_aggregator", AGGREGATOR_PATH)
assert AGGREGATOR_SPEC is not None and AGGREGATOR_SPEC.loader is not None
aggregator = importlib.util.module_from_spec(AGGREGATOR_SPEC)
sys.modules[AGGREGATOR_SPEC.name] = aggregator
AGGREGATOR_SPEC.loader.exec_module(aggregator)

COLLECTOR_PATH = (
    SCRIPT_PATH.parents[1] / "fib_runtime" / "multiturn" / "fix_kernels" / "collect_success_kernel_pairs.py"
)
COLLECTOR_SPEC = importlib.util.spec_from_file_location("benchmark_sass_pair_collector", COLLECTOR_PATH)
assert COLLECTOR_SPEC is not None and COLLECTOR_SPEC.loader is not None
collector = importlib.util.module_from_spec(COLLECTOR_SPEC)
sys.modules[COLLECTOR_SPEC.name] = collector
COLLECTOR_SPEC.loader.exec_module(collector)


def make_static_evidence(tag: str) -> exporter.StaticSassEvidence:
    return exporter.StaticSassEvidence(
        cubin_sass_arch_tag=tag,
        cubin_gmma_instruction_count=1 if tag == "H" else 0,
        cubin_tma_instruction_count=0,
        cubin_tcgen_instruction_count=1 if tag == "B" else 0,
        cubin_container_sha256="cubin-hash",
        matched_lines=("/*0010*/ HGMMA.X",) if tag else (),
        disassembly_line_count=10,
    )


def make_dynamic_evidence(tag: str = "H") -> exporter.DynamicSassEvidence:
    return exporter.DynamicSassEvidence(
        sass_arch_tag=tag,
        gmma_count=7 if tag == "H" else 0,
        tma_count=3 if tag == "H" else 0,
        tcgen_count=5 if tag == "B" else 0,
        gmma_pred_on_thread_count=224 if tag == "H" else 0,
        tma_pred_on_thread_count=3 if tag == "H" else 0,
        tcgen_pred_on_thread_count=160 if tag == "B" else 0,
        matched_lines=("0x10 HGMMA.X 7 224",),
        profile_line_count=20,
        task_id="task-1",
    )


def make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    trajectory_dir = run_dir / "trajectories"
    trajectory_dir.mkdir(parents=True)
    trajectory = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "initial prompt"},
            {"role": "assistant", "content": "```cpp\nincorrect\n```"},
            {"role": "user", "content": "RUNTIME_ERROR"},
            {"role": "assistant", "content": "```cpp\nno-target\n```"},
            {"role": "user", "content": "PASSED", "extra": {"min_speedup": 1.0}},
            {"role": "assistant", "content": "```cpp\ntarget\n```"},
            {"role": "user", "content": "PASSED", "extra": {"min_speedup": 1.1}},
        ]
    }
    (trajectory_dir / "exp_000.json").write_text(json.dumps(trajectory))
    return run_dir


def run_export(run_dir: Path, *, inspect_one, profile_one, verify_sass: bool = True) -> None:
    exporter.export_run(
        run_dir=run_dir,
        architecture="hopper",
        definition="definition",
        workload="workload",
        out_name=exporter.DEFAULT_OUTPUT_NAME,
        verify_sass=verify_sass,
        num_parallel=2,
        num_compile_parallel=2,
        force_static=False,
        force_profile=False,
        continue_on_static_error=False,
        continue_on_profile_error=False,
        inspect_one=inspect_one,
        profile_one=profile_one,
    )


def read_output(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "figures" / exporter.DEFAULT_OUTPUT_NAME).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_static_sass_uses_selected_hopper_and_blackwell_families() -> None:
    hopper = exporter.parse_static_sass(
        """/*0000*/ HGMMA.64x64x16.F32.BF16 R0, R0;
/*0010*/ UTMALDG.2D [UR8], [UR4];
/*0020*/ UTMACCTL.PF [UR12];
metadata mentions HGMMA but is not executable""",
        architecture="hopper",
    )
    assert hopper.cubin_sass_arch_tag == "H"
    assert hopper.cubin_gmma_instruction_count == 1
    assert hopper.cubin_tma_instruction_count == 1

    blackwell = exporter.parse_static_sass(
        """/*0000*/ UTCHMMA gdesc[UR1], tmem[UR2];
/*0010*/ LDTM.x4 R12, tmem[UR5];
/*0020*/ STTM.x4 tmem[UR5], R12;
/*0030*/ UTMALDG.2D [UR12], [UR20];""",
        architecture="blackwell",
    )
    assert blackwell.cubin_sass_arch_tag == "B"
    assert blackwell.cubin_tcgen_instruction_count == 3


def test_dynamic_tag_requires_positive_predicate_true_execution() -> None:
    evidence = exporter.parse_dynamic_sass(
        "0x100 HGMMA.X R0, R0 10 0\n0x110 UTMALDG.2D [UR8], [UR4] 2 2\n",
        architecture="hopper",
    )
    assert evidence.sass_arch_tag == "H"
    assert evidence.gmma_count == 0
    assert evidence.tma_count == 2


def test_export_writes_native_sass_arch_tag_and_skips_static_misses(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    inspected = []
    profiled = []

    def inspect_one(candidate):
        inspected.append(candidate.source)
        return make_static_evidence("H" if "target" in candidate.source and "no-target" not in candidate.source else "")

    def profile_one(candidate):
        profiled.append(candidate.source)
        return make_dynamic_evidence()

    run_export(run_dir, inspect_one=inspect_one, profile_one=profile_one)
    rows = read_output(run_dir)

    assert len(rows) == 3
    assert "sass_arch_tag" in rows[0]
    assert "arch_tag" not in rows[0]
    assert len(inspected) == 2
    assert len(profiled) == 1
    assert rows[0]["sass_verification_status"] == ""
    assert rows[1]["sass_verification_status"] == "cubin_sass_absent"
    assert rows[1]["sass_arch_tag"] == ""
    assert rows[2]["sass_verification_status"] == "dynamic_present"
    assert rows[2]["sass_arch_tag"] == "H"
    assert rows[2]["sass_gmma_count"] == "7"
    assert rows[2]["sass_profile_task_id"] == "task-1"


def test_export_reuses_static_and_dynamic_caches(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)

    def inspect_one(candidate):
        return make_static_evidence("H" if "target" in candidate.source and "no-target" not in candidate.source else "")

    run_export(
        run_dir,
        inspect_one=inspect_one,
        profile_one=lambda _candidate: make_dynamic_evidence(),
    )
    run_export(
        run_dir,
        inspect_one=lambda _candidate: (_ for _ in ()).throw(AssertionError("rebuilt cubin")),
        profile_one=lambda _candidate: (_ for _ in ()).throw(AssertionError("reprofiled kernel")),
    )
    assert read_output(run_dir)[2]["sass_arch_tag"] == "H"


def test_profile_cache_matches_accrl_v4_schema(tmp_path: Path) -> None:
    assert exporter.SASS_FIELDS == [
        "cubin_sass_arch_tag",
        "cubin_gmma_instruction_count",
        "cubin_tma_instruction_count",
        "cubin_tcgen_instruction_count",
        "cubin_container_sha256",
        "sass_arch_tag",
        "sass_gmma_count",
        "sass_tma_count",
        "sass_tcgen_count",
        "sass_gmma_thread_inst_executed_true",
        "sass_tma_thread_inst_executed_true",
        "sass_tcgen_thread_inst_executed_true",
        "sass_profile_task_id",
        "sass_verification_status",
    ]
    cache_path = tmp_path / "profile.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "architecture": "hopper",
                "definition": "definition",
                "workload": "workload",
                "source_sha256": "source-hash",
                "first_seen_at": {"trajectory_id": "exp_000", "turn": 0},
                "evidence": {
                    "sass_arch_tag": "H",
                    "gmma_count": 7,
                    "tma_count": 3,
                    "gmma_pred_on_thread_count": 224,
                    "tma_pred_on_thread_count": 3,
                    "matched_lines": ["0x10 HGMMA.X 7 224"],
                    "profile_line_count": 20,
                    "task_id": "task-accrl",
                    "tcgen_count": 0,
                    "tcgen_pred_on_thread_count": 0,
                },
            }
        )
    )

    evidence = exporter.load_cached_dynamic_evidence(cache_path)
    assert exporter.PROFILE_CACHE_SCHEMA_VERSION == 4
    assert evidence.sass_arch_tag == "H"
    assert evidence.task_id == "task-accrl"
    assert (
        exporter.profile_source_hash("source", "definition", "workload", "hopper")
        == "bebfc04539ff3ec7910c3064433027aa16dd03cfaf59588d20804374e5c9f76c"
    )


def test_correctness_only_mode_is_explicit(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    run_export(
        run_dir,
        verify_sass=False,
        inspect_one=lambda _candidate: (_ for _ in ()).throw(AssertionError("inspected cubin")),
        profile_one=lambda _candidate: (_ for _ in ()).throw(AssertionError("profiled kernel")),
    )
    rows = read_output(run_dir)
    assert [row["sass_verification_status"] for row in rows] == ["", "not_requested", "not_requested"]
    assert all(not row["sass_arch_tag"] for row in rows)


def test_cli_refuses_to_silently_reuse_non_native_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output = run_dir / "figures" / exporter.DEFAULT_OUTPUT_NAME
    write_csv(output, ["trajectory_id", "turn", "correctness", "arch_tag"], [])
    manifest = tmp_path / "experiments.csv"
    write_csv(
        manifest,
        ["arch", "exp_dir"],
        [{"arch": "hopper", "exp_dir": str(run_dir)}],
    )

    original_argv = sys.argv
    sys.argv = [
        str(SCRIPT_PATH),
        "--experiments-csv",
        str(manifest),
        "--skip-sass-verification",
    ]
    try:
        try:
            exporter.main()
        except ValueError as exc:
            assert "does not use the native SASS schema" in str(exc)
        else:
            raise AssertionError("non-native output was silently reused")
    finally:
        sys.argv = original_argv


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_aggregator_uses_only_native_dynamic_sass_tags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    turn_csv = run_dir / "figures" / "turn_correctness_arch.csv"
    fields = [
        "trajectory_id",
        "turn",
        "correctness",
        "sass_arch_tag",
        "sass_verification_status",
    ]
    write_csv(
        turn_csv,
        fields,
        [
            {
                "trajectory_id": "exp_000",
                "turn": 0,
                "correctness": "Correct",
                "sass_arch_tag": "H",
                "sass_verification_status": "dynamic_present",
            },
            {
                "trajectory_id": "exp_001",
                "turn": 0,
                "correctness": "Compilation error",
                "sass_arch_tag": "",
                "sass_verification_status": "",
            },
            {
                "trajectory_id": "exp_001",
                "turn": 1,
                "correctness": "Correct",
                "sass_arch_tag": "H",
                "sass_verification_status": "profile_error",
            },
        ],
    )
    rows = aggregator.aggregate_experiment(
        {
            "model": "example-model",
            "arch": "hopper",
            "definition": "gemm",
            "workload": "workload-id",
            "exp_dir": str(run_dir),
        },
        model_dates={"example-model": "2026-01-02"},
        turn_limits=(1, 2),
    )

    assert [row["correct_with_arch"] for row in rows] == [1, 1]
    assert [row["n_correct_turns"] for row in rows] == [1, 2]
    assert [row["n_unknown_correct_turns"] for row in rows] == [0, 1]
    assert all(row["tag_evidence"] == "dynamic_sass" for row in rows)
    assert all("exp_dir" not in row for row in rows)


def test_aggregator_rejects_legacy_arch_tag_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_csv(
        run_dir / "figures" / "turn_correctness_arch.csv",
        ["trajectory_id", "turn", "correctness", "arch_tag"],
        [],
    )

    try:
        aggregator.aggregate_experiment(
            {
                "model": "example-model",
                "arch": "hopper",
                "definition": "gemm",
                "workload": "workload-id",
                "exp_dir": str(run_dir),
            },
            model_dates={"example-model": "2026-01-02"},
            turn_limits=(1,),
        )
    except ValueError as exc:
        assert "not a native SASS export" in str(exc)
    else:
        raise AssertionError("legacy architecture tags were accepted")


def test_pair_collector_uses_only_turn_local_dynamic_tags(tmp_path: Path) -> None:
    turn_csv = tmp_path / "turn_correctness_arch.csv"
    write_csv(
        turn_csv,
        [
            "trajectory_id",
            "turn",
            "correctness",
            "sass_arch_tag",
            "sass_verification_status",
        ],
        [
            {
                "trajectory_id": "exp_000",
                "turn": 0,
                "correctness": "Correct",
                "sass_arch_tag": "H",
                "sass_verification_status": "dynamic_present",
            },
            {
                "trajectory_id": "exp_000",
                "turn": 1,
                "correctness": "Correct",
                "sass_arch_tag": "H",
                "sass_verification_status": "profile_error",
            },
        ],
    )

    assert collector.load_arch_sass_tags(turn_csv) == {("exp_000", 0): {"H"}}


def test_pair_collector_rejects_correctness_only_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_csv(
        run_dir / collector.TURN_CSV_REL,
        [
            "trajectory_id",
            "turn",
            "correctness",
            "sass_arch_tag",
            "sass_verification_status",
        ],
        [
            {
                "trajectory_id": "exp_000",
                "turn": 0,
                "correctness": "Correct",
                "sass_arch_tag": "",
                "sass_verification_status": "not_requested",
            }
        ],
    )

    try:
        collector.ensure_turn_csv(
            {"exp_dir": str(run_dir)},
            force=False,
            require_sass_tags=True,
            base_url=None,
        )
    except ValueError as exc:
        assert "complete dynamic SASS evidence" in str(exc)
    else:
        raise AssertionError("correctness-only export was accepted as SASS evidence")
