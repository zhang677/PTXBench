from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MINI_ROOT = ROOT / "packages" / "mini-ptx-agent"
CONSTRUCT_ROOT = MINI_ROOT / "fib_runtime" / "multiturn" / "construct_eval_scripts"
FIXIT_ROOT = ROOT / "experiments" / "fixit"
KERNELGEN_ROOT = ROOT / "experiments" / "kernelgen"
MULTITURN_ROOT = MINI_ROOT / "fib_runtime" / "multiturn"


def test_active_fixit_sources_have_no_legacy_absolute_roots() -> None:
    paths = [
        *FIXIT_ROOT.glob("*.sh"),
        ROOT / "benchmark" / "export_turn_correctness_arch.py",
        MINI_ROOT / "fib_runtime" / "multiturn" / "analyze_kernel_per_turn.py",
        CONSTRUCT_ROOT / "build_rebalanced_fixit_error_collection.py",
        CONSTRUCT_ROOT / "filter_fixit_error_kernels_by_prompt_tag.py",
        CONSTRUCT_ROOT / "fixit_downstream_process.py",
        CONSTRUCT_ROOT / "watch_eval_common.sh",
        MINI_ROOT
        / "fib_runtime"
        / "multiturn"
        / "fix_kernels"
        / "collect_success_kernel_pairs.py",
        MINI_ROOT
        / "fib_runtime"
        / "multiturn"
        / "fix_kernels"
        / "select_failed_kernels.py",
        MINI_ROOT / "fib_runtime" / "multiturn" / "run_parallel_v2.py",
        MINI_ROOT / "fib_runtime" / "multiturn" / "run_v2.py",
        MINI_ROOT / "fib_runtime" / "multiturn" / "common.py",
        MINI_ROOT / "accrl" / "distill" / "inspector.py",
        MINI_ROOT / "accrl" / "distill" / "sft" / "build_sft_dataset_kernelgen.py",
        MINI_ROOT
        / "fib_runtime"
        / "multiturn"
        / "task_to_correct_kernels"
        / "synthesize_correct_kernel_reasoning_openrouter.py",
        MINI_ROOT / "fib_runtime" / "multiturn" / "collect_notes" / "note_feedback.py",
        *KERNELGEN_ROOT.glob("*.sh"),
    ]
    legacy_roots = ("/home/ubuntu/AccRL", "/home/ubuntu/AccRL-exps")
    for path in paths:
        text = path.read_text()
        for root in legacy_roots:
            assert root not in text, f"{path.relative_to(ROOT)} still contains {root}"


def test_multiturn_regeneration_is_self_contained_and_current() -> None:
    prepare_script = (MULTITURN_ROOT / "prepare_scripts.sh").read_text()
    assert "/home/ubuntu/AccRL" not in prepare_script

    template = (MULTITURN_ROOT / "template_compile_measure_cuda.txt").read_text()
    gemm_rows = list(
        csv.DictReader(
            (MULTITURN_ROOT / "gemm-problems" / "gemm_problems.csv").open(
                newline=""
            )
        )
    )
    largest_gemms: dict[str, dict[str, str]] = {}
    for row in gemm_rows:
        definition = row["definition_name"]
        previous = largest_gemms.get(definition)
        if previous is None or int(row["M"]) > int(previous["M"]):
            largest_gemms[definition] = row

    suites = [
        (
            list(largest_gemms.values()),
            MULTITURN_ROOT / "gemm-problems",
        ),
        (
            list(
                csv.DictReader(
                    (MULTITURN_ROOT / "mha-with-lse-problems" / "mha_problems.csv").open(
                        newline=""
                    )
                )
            ),
            MULTITURN_ROOT / "mha-with-lse-problems",
        ),
        (
            list(
                csv.DictReader(
                    (
                        MULTITURN_ROOT
                        / "mha-bwd-problems"
                        / "mha_bwd_problems.csv"
                    ).open(newline="")
                )
            ),
            MULTITURN_ROOT / "mha-bwd-problems",
        ),
        (
            list(
                csv.DictReader(
                    (MULTITURN_ROOT / "fp8-mha-with-lse-problems" / "problems.csv").open(
                        newline=""
                    )
                )
            ),
            MULTITURN_ROOT / "fp8-mha-with-lse-problems",
        ),
        (
            list(
                csv.DictReader(
                    (MULTITURN_ROOT / "single_op_eval" / "problems.csv").open(
                        newline=""
                    )
                )
            ),
            MULTITURN_ROOT / "single_op_eval",
        ),
    ]

    generator_paths = [
        MULTITURN_ROOT / "gemm-problems" / "create_test.py",
        MULTITURN_ROOT / "mha-with-lse-problems" / "create_test.py",
        MULTITURN_ROOT / "mha-bwd-problems" / "create_test.py",
        MULTITURN_ROOT / "fp8-mha-with-lse-problems" / "scripts" / "create_test.py",
        MULTITURN_ROOT / "single_op_eval" / "create_test.py",
    ]
    for generator_path in generator_paths:
        generator = generator_path.read_text()
        assert "/home/ubuntu/AccRL" not in generator
        assert "perf.csv" not in generator

    for rows, suite_dir in suites:
        for row in rows:
            definition = row.get("definition_name", row.get("definition"))
            workload_uuid = row["workload_uuid"]
            generated_path = suite_dir / f"{definition}_{workload_uuid}.py"
            expected = template.replace("<definition_name>", definition).replace(
                "<workload_uuid>", workload_uuid
            )
            assert generated_path.read_text() == expected


def test_fixit_static_preflight() -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "smoke_fixit.sh"), "--check"],
        cwd=ROOT,
        env={
            **os.environ,
            "PTXBENCH_ROOT": str(ROOT),
            "MINI_PTX_AGENT_ROOT": str(MINI_ROOT),
            "PTXBENCH_DATA_ROOT": str(ROOT / "data"),
        },
        check=True,
    )


def test_path_resolver_uses_repo_layout() -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from mini_ptx_agent.paths import resolve_paths

    paths = resolve_paths()
    assert paths.repo_root == ROOT
    assert paths.mini_ptx_agent_root == MINI_ROOT
    assert paths.multiturn_root == MINI_ROOT / "fib_runtime" / "multiturn"
    assert paths.config_root == ROOT / "configs"
    assert paths.fixit_root == ROOT / "experiments" / "fixit"


def test_quickstart_report_distinguishes_attempt_from_correct_kernel(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from mini_ptx_agent.quickstart import build_result, write_result

    output_root = tmp_path / "quickstart"
    (output_root / "exp_000").mkdir(parents=True)
    (output_root / "exp_000" / "kernel.cu").write_text("generated but incorrect")
    (output_root / "trajectories").mkdir()
    (output_root / "trajectories" / "exp_000.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "candidate"},
                    {
                        "role": "user",
                        "content": "<output>test.py failed: compile error</output>",
                    },
                    {
                        "role": "exit",
                        "content": "LimitsExceeded",
                        "extra": {"exit_status": "LimitsExceeded"},
                    },
                ]
            }
        )
    )
    (output_root / "plan.json").write_text(
        json.dumps(
            {
                "plan": [
                    {
                        "exp_index": 0,
                        "definition": "gemm_n7168_k5120",
                        "num_turns": 1,
                    }
                ]
            }
        )
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "success": 1,
                "results": [
                    {
                        "exp_name": "exp_000",
                        "definition": "gemm_n7168_k5120",
                        "status": "success",
                    }
                ],
            }
        )
    )

    result = build_result(output_root)
    assert result["runner"]["completed_processes"] == 1
    assert result["outcome"] == {
        "generated_candidate_count": 1,
        "correct_kernel_count": 0,
        "target_achieved_count": 0,
    }
    assert result["experiments"][0]["feedback_excerpt"] == (
        "test.py failed: compile error"
    )
    result_path, written = write_result(output_root)
    assert json.loads(result_path.read_text()) == written


def test_quickstart_report_identifies_correct_target_kernel(tmp_path: Path) -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from mini_ptx_agent.quickstart import build_result

    output_root = tmp_path / "quickstart"
    success = output_root / "success" / "exp_000"
    success.mkdir(parents=True)
    (success / "kernel_v0.cu").write_text("correct kernel")
    (success / "record.json").write_text(json.dumps([{"version": 0, "turn": 0}]))
    trajectories = output_root / "trajectories"
    trajectories.mkdir()
    (trajectories / "exp_000.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {
                        "role": "exit",
                        "content": "Target speedup achieved.",
                        "extra": {"exit_status": "Submitted"},
                    },
                ]
            }
        )
    )

    result = build_result(output_root)
    assert result["outcome"]["generated_candidate_count"] == 0
    assert result["outcome"]["correct_kernel_count"] == 1
    assert result["outcome"]["target_achieved_count"] == 1
    assert result["experiments"][0]["correct_kernels"] == [
        "success/exp_000/kernel_v0.cu"
    ]


def test_parallel_summary_separates_completion_correctness_and_target(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(MINI_ROOT / "fib_runtime" / "multiturn"))
    from run_parallel_v2 import summarize_results

    results = [
        {
            "exp_index": 0,
            "exp_name": "exp_000",
            "prompt_tag": "hopper-no-hint",
            "status": "success",
            "duration": 1.0,
            "correct_kernel_count": 0,
            "target_met": False,
        },
        {
            "exp_index": 1,
            "exp_name": "exp_001",
            "prompt_tag": "hopper-no-hint",
            "status": "success",
            "duration": 2.0,
            "correct_kernel_count": 2,
            "target_met": True,
        },
    ]
    summarize_results(results, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] == 2
    assert summary["success"] == 2
    assert summary["correct_trajectory"] == 1
    assert summary["correct_kernel_versions"] == 2
    assert summary["target_met"] == 1
    assert "Legacy compatibility field" in summary["success_semantics"]


def test_construct_eval_directory_contains_only_shared_implementation() -> None:
    expected = {
        "README.md",
        "build_rebalanced_fixit_error_collection.py",
        "filter_fixit_error_kernels_by_prompt_tag.py",
        "fixit_downstream_process.py",
        "ptxbench_paths.sh",
        "watch_eval_audit.py",
        "watch_eval_common.sh",
    }
    actual = {
        str(path.relative_to(CONSTRUCT_ROOT))
        for path in CONSTRUCT_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert actual == expected


def test_compose_uses_one_collection_mount_and_dataset_roots_list() -> None:
    compose = (ROOT / "docker" / "compose.yaml").read_text()
    env_example = (ROOT / ".env.example").read_text()
    assert "PTXBENCH_TRACESETS_ROOT:?" in compose
    assert ":/workspace/trace-sets:ro" in compose
    assert "DATASET_ROOTS: ${DATASET_ROOTS:?" in compose
    assert "PTXBENCH_TRACESETS_ROOT=/home/ubuntu/PTXBench/data/datasets" in env_example
    assert "DATASET_ROOTS=/workspace/trace-sets/accrl-training" in env_example
    assert not (ROOT / "docker" / "compose.multitrace.yaml").exists()


def test_fibserve_verifier_imports_both_workspace_packages() -> None:
    verifier = (ROOT / "packages" / "fibserve" / "scripts" / "run_verify.sh").read_text()
    assert "$PTXBENCH_ROOT/packages/mini-ptx-agent:$PTXBENCH_ROOT/packages/fibserve" in verifier


def test_complete_fixit_source_preflight() -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "reproduce_fixit.sh"), "--check"],
        cwd=ROOT,
        env={
            **os.environ,
            "PTXBENCH_ROOT": str(ROOT),
            "MINI_PTX_AGENT_ROOT": str(MINI_ROOT),
            "PTXBENCH_DATA_ROOT": str(ROOT / "data"),
        },
        check=True,
    )


def test_fixit_from_scratch_stages_and_expert_guided_eval() -> None:
    driver = (ROOT / "scripts" / "reproduce_fixit.sh").read_text()
    for stage in [
        "source_00_watch_qwen36_linfo_mha.sh",
        "source_01_prepare_gemini_repairs.sh",
        "source_02_watch_gemini_repairs.sh",
        "source_03_collect_kernel_pairs.sh",
        "00_synthesize_qwen36-27b_reasoning.sh",
        "01_resynthesize_filtered_reasonings.sh",
        "02_build_full_parquet.sh",
        "03_train_sft_full.sh",
        "04_serve_remote_full.sh",
        "05_watch_5defs_eval.sh",
    ]:
        assert stage in driver
    assert len(list(csv.DictReader((FIXIT_ROOT / "source-runs.csv").open()))) == 8
    watcher = (FIXIT_ROOT / "05_watch_5defs_eval.sh").read_text()
    assert "mha-p4-mha-patched.json" in watcher
    assert "mha-bwd-p4-mha-patched.json" in watcher
    assert not list(FIXIT_ROOT.glob("0[67]_*.sh"))


def test_recipe_entrypoints_are_canonical_drivers() -> None:
    fixit = (ROOT / "scripts" / "reproduce_fixit.sh").read_text()
    kernelgen = (ROOT / "scripts" / "reproduce_kernelgen.sh").read_text()
    assert 'FIXIT_ROOT="$PTXBENCH_ROOT/experiments/fixit"' in fixit
    assert 'EXPERIMENT_ROOT="$PTXBENCH_ROOT/experiments/kernelgen"' in kernelgen
    assert "declare -a STAGES=(" in fixit
    assert "declare -a STAGES=(" in kernelgen


def test_complete_kernelgen_source_preflight() -> None:
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "reproduce_kernelgen.sh"), "--check"],
        cwd=ROOT,
        env={
            **os.environ,
            "PTXBENCH_ROOT": str(ROOT),
            "MINI_PTX_AGENT_ROOT": str(MINI_ROOT),
            "PTXBENCH_DATA_ROOT": str(ROOT / "data"),
        },
        check=True,
    )


def test_multiturn_child_driver_is_checked_by_both_recipes() -> None:
    relative_driver = (
        "packages/mini-ptx-agent/fib_runtime/multiturn/run_v2.py"
    )
    launcher = (
        MINI_ROOT / "fib_runtime" / "multiturn" / "run_parallel_v2.py"
    ).read_text()
    assert 'launch_script = SCRIPT_DIR / "run_v2.py"' in launcher
    assert (ROOT / relative_driver).is_file()
    for driver_path in (
        ROOT / "scripts" / "reproduce_fixit.sh",
        ROOT / "scripts" / "reproduce_kernelgen.sh",
    ):
        assert "$MINI_PTX_AGENT_ROOT/fib_runtime/multiturn/run_v2.py" in (
            driver_path.read_text()
        )


def test_sft_collector_has_no_hidden_kernel_extractor(tmp_path: Path) -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from fib_runtime.multiturn.task_to_correct_kernels.collect_correct_kernels import (
        ensure_kernels_dir,
    )

    missing_run = tmp_path / "run"
    try:
        ensure_kernels_dir(missing_run)
    except FileNotFoundError as exc:
        assert "extract the KernelGen source-data bundle" in str(exc)
    else:
        raise AssertionError("missing kernels directory was silently accepted")


def test_kernelgen_runnable_sources_are_present() -> None:
    assert len(list(csv.DictReader((KERNELGEN_ROOT / "source-runs.csv").open()))) == 12
    assert (MINI_ROOT / "accrl" / "distill" / "inspector.py").is_file()
    assert 'ptxbench-inspect = "accrl.distill.inspector:app"' in (
        MINI_ROOT / "pyproject.toml"
    ).read_text()


def test_kernelgen_builder_uses_first_prompt_and_selected_answer(
    tmp_path: Path, monkeypatch
) -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from accrl.distill.sft.build_sft_dataset_kernelgen import convert_record

    monkeypatch.setenv("PTXBENCH_DATA_ROOT", str(tmp_path))
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "base knowledge"},
                    {"role": "user", "content": "kernel task"},
                    {"role": "assistant", "content": "wrong first answer"},
                    {"role": "user", "content": "feedback"},
                    {"role": "assistant", "content": "selected correct answer"},
                ]
            }
        )
    )
    row = convert_record(
        {
            "reasoning": "derive the selected kernel",
            "metadata": {
                "trajectory_path": "${PTXBENCH_DATA_ROOT}/trajectory.json",
                "turn": 1,
                "run_id": "run",
                "trajectory_id": "exp_000",
                "definition_name": "mha",
            },
        },
        reasoning_field="reasoning",
        include_distill_system=False,
    )
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert row["messages"][0]["content"] == "base knowledge"
    assert row["messages"][1]["content"] == "kernel task"
    assert row["messages"][2]["content"] == (
        "<think>\nderive the selected kernel\n</think>\nselected correct answer"
    )


def test_kernelgen_bundle_csv_paths_are_relocatable(tmp_path: Path) -> None:
    import csv
    import importlib.util

    data_root = tmp_path / "legacy-data"
    mini_root = tmp_path / "legacy-mini"
    source = tmp_path / "enriched.csv"
    fieldnames = [
        "exp_dir",
        "test_path",
        "kernel_path",
        "correct_kernel_path",
        "turn_csv",
        "trajectory_path",
    ]
    with source.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "exp_dir": data_root / "eval_runs" / "run",
                "test_path": mini_root / "fib_runtime" / "test.py",
                "kernel_path": data_root / "eval_runs" / "run" / "kernel.cu",
                "correct_kernel_path": data_root
                / "eval_runs"
                / "run"
                / "kernel.cu",
                "turn_csv": data_root / "eval_runs" / "run" / "turn.csv",
                "trajectory_path": data_root / "eval_runs" / "run" / "exp.json",
            }
        )

    script = ROOT / "scripts" / "build_kernelgen_data_bundle.py"
    spec = importlib.util.spec_from_file_location("kernelgen_bundle", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rewritten = module.portable_csv(
        source, data_root=data_root, mini_root=mini_root
    ).decode()
    assert "${PTXBENCH_DATA_ROOT}/eval_runs/run/kernel.cu" in rewritten
    assert "${MINI_PTX_AGENT_ROOT}/fib_runtime/test.py" in rewritten


def test_training_stage_has_no_untracked_run_script_dependency() -> None:
    downstream = (CONSTRUCT_ROOT / "fixit_downstream_process.py").read_text()
    assert "run_qwen36-27b.sh" not in downstream
    assert "/home/ubuntu/tinker-cookbook" not in downstream
    assert "--remote-python" in downstream
    assert "accrl/distill/sft/tinker_sft_train.py" in downstream

    sys.path.insert(0, str(CONSTRUCT_ROOT))
    from fixit_downstream_process import build_train_command

    command = [
        str(part)
        for part in build_train_command(
            SimpleNamespace(
                base_model="Qwen/Qwen3.6-27B",
                train_run_tag="published-run",
                runs_dir=Path("/tmp/published-runs"),
                train_num_epochs=5,
                train_learning_rate=4.65e-4,
                train_load_checkpoint_path=None,
            ),
            Path("/tmp/published.parquet"),
        )
    ]
    assert "dataset_path=/tmp/published.parquet" in command
    assert "model_name=Qwen/Qwen3.6-27B" in command
    assert "num_epochs=5" in command
    assert "learning_rate=0.000465" in command
    assert "behavior_if_log_dir_exists=raise" in command


def test_fixit_path_resolver_expands_release_environment(
    tmp_path: Path, monkeypatch
) -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from accrl.distill.sft.build_sft_dataset_fixit import resolve_path
    from fib_runtime.multiturn.fix_kernels.synthesize_pair_reasoning_openrouter import (
        data_path,
    )

    monkeypatch.setenv("PTXBENCH_DATA_ROOT", str(tmp_path))
    portable_path = "${PTXBENCH_DATA_ROOT}/eval_runs/example/kernel.cu"
    expected = tmp_path / "eval_runs" / "example" / "kernel.cu"
    expected.parent.mkdir(parents=True)
    expected.write_text("kernel")
    assert resolve_path(portable_path, base_dir=Path("/unused")) == expected
    assert data_path(portable_path) == expected

    historical_path = "/old/release/root/eval_runs/example/kernel.cu"
    assert resolve_path(historical_path, base_dir=Path("/unused")) == expected
    assert data_path(historical_path) == expected


def test_fixit_builder_preserves_archival_paths_while_reading_relocated_data(
    tmp_path: Path, monkeypatch
) -> None:
    sys.path.insert(0, str(MINI_ROOT))
    from accrl.distill.sft.build_sft_dataset_fixit import build_row

    monkeypatch.setenv("PTXBENCH_DATA_ROOT", str(tmp_path))
    run = tmp_path / "eval_runs" / "run"
    wrong_kernel = run / "kernels" / "exp_001" / "kernel_t0.cu"
    wrong_log = run / "kernels" / "exp_001" / "log_t0.txt"
    trajectory = run / "trajectories" / "exp_001.json"
    correct_kernel = run / "success" / "exp_001" / "kernel_v0.cu"
    for path, content in (
        (wrong_kernel, "wrong kernel"),
        (wrong_log, "wrong feedback"),
        (correct_kernel, "correct kernel"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "wrong answer"},
                    {"role": "user", "content": "evaluation"},
                ]
            }
        )
    )
    historical_root = Path("/old/release/root/eval_runs/run")
    csv_row = {
        "exp_dir": str(historical_root),
        "trajectory_id": "exp_001",
        "wrong_kernel_path": str(
            historical_root / "kernels" / "exp_001" / "kernel_t0.cu"
        ),
        "wrong_log_path": str(
            historical_root / "kernels" / "exp_001" / "log_t0.txt"
        ),
        "wrong_trajectory_path": str(
            historical_root / "trajectories" / "exp_001.json"
        ),
        "wrong_turn": "0",
        "correct_kernel_path": str(
            historical_root / "success" / "exp_001" / "kernel_v0.cu"
        ),
        "correct_kernel_version": "0",
    }
    row = build_row(
        record={
            "reasoning": "repair reasoning",
            "metadata": {"correct_kernel_path": csv_row["correct_kernel_path"]},
        },
        csv_row=csv_row,
        csv_dir=tmp_path,
        source_label="test",
    )
    assert row["metadata"]["correct_kernel_path"] == csv_row["correct_kernel_path"]
    assert row["metadata"]["wrong_trajectory_path"] == csv_row[
        "wrong_trajectory_path"
    ]
    assert [message["content"] for message in row["messages"]] == [
        "system",
        "task",
        "wrong answer",
        "evaluation",
        "<think>\nrepair reasoning\n</think>\n\n```cpp\ncorrect kernel\n```",
    ]


def test_fixit_data_bundle_rewrites_paths_and_includes_implicit_inputs(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "legacy-data"
    mini_root = tmp_path / "legacy-mini"
    project = data_root / "sft_experiments" / "fixit"
    correct_run = data_root / "eval_runs" / "correct"
    wrong_run = data_root / "eval_runs" / "wrong"
    test_path = mini_root / "fib_runtime" / "multiturn" / "test.py"
    paths = {
        "test_path": test_path,
        "wrong_kernel_path": wrong_run / "kernels" / "exp_002" / "kernel_t0.cu",
        "wrong_log_path": wrong_run / "kernels" / "exp_002" / "log_t0.txt",
        "wrong_trajectory_path": wrong_run / "trajectories" / "exp_002.json",
        "correct_kernel_path": correct_run / "success" / "exp_001" / "kernel_v0.cu",
        "plan_path": correct_run / "plan.json",
        "turn_csv": correct_run / "figures" / "turn_correctness_arch.csv",
    }
    implicit_paths = (
        correct_run / "trajectories" / "exp_001.json",
        correct_run / "success" / "exp_001" / "record.json",
    )
    for path in (*paths.values(), *implicit_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {path.name}\n")

    pairs_csv = project / "fixit-v5-gemini-kernel-pairs.csv"
    pairs_csv.parent.mkdir(parents=True)
    fieldnames = [
        "exp_dir",
        "arch",
        "definition",
        "test_path",
        "trajectory_id",
        "prompt_tag",
        "arch_tag",
        "wrong_kernel_path",
        "wrong_log_path",
        "wrong_trajectory_path",
        "wrong_turn",
        "correct_kernel_path",
        "correct_kernel_version",
        "plan_path",
        "turn_csv",
    ]
    import csv

    with pairs_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "exp_dir": correct_run,
                "arch": "hopper",
                "definition": "test",
                "test_path": test_path,
                "trajectory_id": "exp_001",
                "prompt_tag": "hopper-00",
                "arch_tag": "H",
                "wrong_kernel_path": paths["wrong_kernel_path"],
                "wrong_log_path": paths["wrong_log_path"],
                "wrong_trajectory_path": paths["wrong_trajectory_path"],
                "wrong_turn": 0,
                "correct_kernel_path": paths["correct_kernel_path"],
                "correct_kernel_version": 0,
                "plan_path": paths["plan_path"],
                "turn_csv": paths["turn_csv"],
            }
        )

    archive_path = tmp_path / "fixit-data.tar.gz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_fixit_data_bundle.py"),
            "--pairs-csv",
            str(pairs_csv),
            "--data-root",
            str(data_root),
            "--mini-agent-root",
            str(mini_root),
            "--output",
            str(archive_path),
        ],
        cwd=ROOT,
        check=True,
    )

    with tarfile.open(archive_path) as archive:
        names = set(archive.getnames())
        portable_csv = archive.extractfile(
            "ptxbench-data/sft_experiments/fixit/fixit-v5-gemini-kernel-pairs.csv"
        )
        assert portable_csv is not None
        portable_text = portable_csv.read().decode()
    assert "${PTXBENCH_DATA_ROOT}/eval_runs/correct" in portable_text
    assert (
        "${MINI_PTX_AGENT_ROOT}/fib_runtime/multiturn/test.py" in portable_text
    )
    assert (
        "ptxbench-data/eval_runs/correct/trajectories/exp_001.json" in names
    )
    assert (
        "ptxbench-data/eval_runs/correct/success/exp_001/record.json" in names
    )
    assert "ptxbench-data/fixit-source-manifest.json" in names
