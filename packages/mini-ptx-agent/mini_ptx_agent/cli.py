"""PTXBench command-line utilities."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from .paths import resolve_paths
from .quickstart import preflight as quickstart_preflight
from .quickstart import print_result as print_quickstart_result
from .quickstart import run as run_quickstart
from .quickstart import write_result as write_quickstart_result

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def paths() -> None:
    """Print the resolved PTXBench filesystem layout."""
    typer.echo(json.dumps({key: str(value) for key, value in vars(resolve_paths()).items()}, indent=2))


@app.command()
def doctor() -> None:
    """Check the local source and command prerequisites without changing state."""
    resolved = resolve_paths()
    checks: list[tuple[str, bool, str]] = [
        ("repository", resolved.repo_root.is_dir(), str(resolved.repo_root)),
        ("mini-ptx-agent", resolved.mini_ptx_agent_root.is_dir(), str(resolved.mini_ptx_agent_root)),
        ("multiturn runner", (resolved.multiturn_root / "run_parallel_v2.py").is_file(), str(resolved.multiturn_root)),
        (
            "Fixit experiment",
            (resolved.fixit_v6_root / "05_watch_5defs_eval.sh").is_file(),
            str(resolved.fixit_v6_root),
        ),
        ("FIBServe source", (resolved.repo_root / "packages" / "fibserve" / "pyproject.toml").is_file(), "packages/fibserve"),
        ("docker", shutil.which("docker") is not None, shutil.which("docker") or "not found"),
        ("tmux", shutil.which("tmux") is not None, shutil.which("tmux") or "not found"),
    ]
    failed = False
    for label, ok, detail in checks:
        typer.echo(f"{'OK' if ok else 'FAIL':4} {label}: {detail}")
        failed |= not ok
    if failed:
        raise typer.Exit(1)


@app.command()
def quickstart(
    check: Annotated[
        bool,
        typer.Option("--check", help="Check live dependencies without running the model."),
    ] = False,
    run: Annotated[
        bool,
        typer.Option("--run", help="Run one three-turn GEMM trajectory."),
    ] = False,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Report an existing evaluation output root."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", envvar="MODEL_NAME", help="Model name understood by PTXBench."),
    ] = None,
    model_host: Annotated[
        str | None,
        typer.Option(
            "--model-host",
            envvar="ACCRL_MODEL_HOST",
            help="OpenAI-compatible local model endpoint as host:port.",
        ),
    ] = None,
    service_url: Annotated[
        str,
        typer.Option(
            "--service-url",
            envvar="SERVICE_URL",
            help="FIBServe base URL.",
        ),
    ] = "http://localhost:10000",
    eval_image: Annotated[
        str,
        typer.Option(
            "--eval-image",
            envvar="PTXBENCH_EVAL_IMAGE",
            help="Isolated CUDA evaluator Docker image.",
        ),
    ] = "ptxbench-eval:dev",
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="New directory for run artifacts."),
    ] = None,
) -> None:
    """Check or run PTXBench's smallest real kernel-agent example."""
    selected_modes = int(check) + int(run) + int(report is not None)
    if selected_modes > 1:
        typer.echo("Choose only one of --check, --run, or --report.", err=True)
        raise typer.Exit(2)
    if report is not None:
        if not report.is_dir():
            typer.echo(f"Evaluation output root does not exist: {report}", err=True)
            raise typer.Exit(1)
        result_path, result = write_quickstart_result(report)
        print_quickstart_result(result_path, result)
        return

    paths = resolve_paths()
    checks = quickstart_preflight(
        paths,
        model=model or "",
        model_host=model_host,
        service_url=service_url,
        eval_image=eval_image,
    )
    failed = False
    for item in checks:
        typer.echo(f"{'OK' if item.ok else 'FAIL':4} {item.label}: {item.detail}")
        failed |= not item.ok
    if failed:
        raise typer.Exit(1)
    if not run:
        typer.echo("Quickstart live preflight passed.")
        return

    if output_root is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output_root = paths.eval_runs_root / f"quickstart-{timestamp}-gemm"
    typer.echo(f"Starting one three-turn Hopper GEMM trajectory in {output_root}")
    returncode = run_quickstart(
        paths,
        model=model or "",
        service_url=service_url,
        eval_image=eval_image,
        output_root=output_root,
    )
    if returncode:
        raise typer.Exit(returncode)


if __name__ == "__main__":
    app()
