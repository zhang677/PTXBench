"""PTXBench command-line utilities."""

from __future__ import annotations

import json
import shutil

import typer

from .paths import resolve_paths

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
            "Fixit-v6 experiment",
            (resolved.fixit_v6_root / "07_watch_v6_full_5defs_eval.sh").is_file(),
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


if __name__ == "__main__":
    app()
