#!/usr/bin/env python3
"""Coordinate generic fix-it downstream stages.

This is the active reusable downstream runner for fix-it SFT workflows. It owns
the shared implementation for:

- pair-reasoning synthesis with pair-key coverage checking
- SFT parquet construction
- Tinker SFT launch
- checkpoint download/merge plus SGLang serve

Run-specific paths and names are supplied by wrapper scripts or CLI arguments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen


MINI_PTX_AGENT_ROOT = Path(
    os.environ.get("MINI_PTX_AGENT_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
PTXBENCH_ROOT = Path(
    os.environ.get("PTXBENCH_ROOT", MINI_PTX_AGENT_ROOT.parents[1])
).expanduser().resolve()
EXPS = Path(os.environ.get("PTXBENCH_DATA_ROOT", PTXBENCH_ROOT / "data")).expanduser().resolve()
ACCRL = MINI_PTX_AGENT_ROOT  # Compatibility name retained inside the ported implementation.
TINKER_COOKBOOK_ROOT = Path(
    os.environ.get("TINKER_COOKBOOK_ROOT", "/home/ubuntu/tinker-cookbook")
).expanduser().resolve()
DEFAULT_PROJECT = EXPS / "sft_experiments/test-fixit-qwen36-27b-gemini-glm"
DEFAULT_RUNS = DEFAULT_PROJECT / "runs"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_MODEL_PREFIX = "qwen36-27b-SFT"
DEFAULT_REASONING_MODEL = "openrouter/z-ai/glm-5.2"
DEFAULT_PROMPT_VERSION = "v1_trajectory_adopted"
DEFAULT_REMOTE = "ion-b200"
DEFAULT_CONTAINER = "sglang-genghan"
DEFAULT_REMOTE_PORTS = "9000,9001,9002,9003"
DEFAULT_LOCAL_PORTS = "30002,30012,30022,30032,30042,30052,30062,30072,30082,30092"
FREE_GPU_MAX_MEMORY_MIB = 1024
ALLOWED_STAGES = ("synthesize", "parquet", "train", "serve")


def env_text(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_path(name: str, default: str = "") -> Path | None:
    value = os.environ.get(name, default)
    return Path(value).expanduser() if value else None


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer list item: {item}") from exc
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def parse_stages(value: str) -> list[str]:
    if value == "all":
        return list(ALLOWED_STAGES)
    stages = [item.strip() for item in value.split(",") if item.strip()]
    bad = [stage for stage in stages if stage not in ALLOWED_STAGES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown stages: {bad}; allowed={list(ALLOWED_STAGES)} or all"
        )
    return stages


def require_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise SystemExit(f"missing required {label}")
    return path.expanduser()


def quote_cmd(args: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_date_from_run_dir(run_dir: Path) -> str:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$", run_dir.name)
    if not match:
        raise SystemExit(f"cannot derive RUN_DATE from run directory name: {run_dir}")
    year, month, day, hour, minute = match.groups()
    return f"{year}-{month}{day}-{hour}{minute}"


class Runner:
    def __init__(self, execute: bool, cwd: Path = EXPS):
        self.execute = execute
        self.cwd = cwd

    def log(self, command: str) -> None:
        prefix = "RUN" if self.execute else "DRY"
        print(f"\n[{prefix}] {command}", flush=True)

    def command(
        self,
        args: list[str | Path],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        self.log(quote_cmd(args))
        if not self.execute:
            return subprocess.CompletedProcess(args, 0)
        return subprocess.run([str(arg) for arg in args], cwd=str(cwd or self.cwd), check=check)

    def shell(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        self.log(command)
        if not self.execute:
            return subprocess.CompletedProcess(command, 0)
        return subprocess.run(command, cwd=str(cwd or self.cwd), shell=True, check=check)

    def capture(
        self,
        args: list[str | Path],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> str:
        self.log(quote_cmd(args))
        if not self.execute:
            return ""
        proc = subprocess.run(
            [str(arg) for arg in args],
            cwd=str(cwd or self.cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=check,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        return proc.stdout

    def tmux_new(self, session: str) -> None:
        self.command(["tmux", "new-session", "-Ad", "-s", session])

    def tmux_send(self, session: str, command: str) -> None:
        self.command(["tmux", "send-keys", "-t", session, command, "C-m"])

    def remote(self, remote: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess:
        return self.command(["ssh", remote, command], check=check)


def write_synth_config(args: argparse.Namespace) -> None:
    pairs_csv = require_path(args.pairs_csv, "--pairs-csv")
    output_jsonl = require_path(args.reasoning_jsonl, "--reasoning-jsonl")
    synth_config = require_path(args.synth_config, "--synth-config")

    synth_config.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"pairs_csv: {pairs_csv}",
        f"output_jsonl: {output_jsonl}",
    ]
    if args.provenance_json is not None:
        lines.append(f"provenance_json: {args.provenance_json}")
    lines.extend(
        [
            f"name: {args.synth_name}",
            f'description: "{args.synth_description}"',
            f"reasoning_model: {args.reasoning_model}",
            f"prompt_version: {args.prompt_version}",
            f"max_tokens: {args.max_tokens}",
            f"min_reasoning_chars: {args.min_reasoning_chars}",
            f"max_reasoning_chars: {args.max_reasoning_chars}",
            f"temperature: {args.temperature}",
            f"top_p: {args.top_p}",
            f"max_concurrent: {args.max_concurrent}",
            f"timeout: {args.timeout}",
            "limit:",
            f"overwrite: {'true' if args.overwrite else 'false'}",
            "",
        ]
    )
    synth_config.write_text("\n".join(lines))


def pair_key(row: dict[str, str]) -> str:
    return "\n".join(
        [
            row.get("exp_dir", ""),
            row.get("trajectory_id", ""),
            row.get("correct_kernel_path", ""),
            row.get("correct_kernel_version", ""),
        ]
    )


def pair_key_from_metadata(metadata: dict[str, object]) -> str:
    return "\n".join(
        str(part)
        for part in [
            metadata.get("exp_dir", ""),
            metadata.get("trajectory_id", "") or metadata.get("exp_id", ""),
            metadata.get("correct_kernel_path", ""),
            metadata.get("correct_kernel_version", ""),
        ]
    )


def reasoning_coverage(pairs_csv: Path, reasoning_jsonl: Path) -> tuple[int, int]:
    with pairs_csv.open(newline="") as f:
        expected = {pair_key(row) for row in csv.DictReader(f)}

    completed: set[str] = set()
    if reasoning_jsonl.is_file():
        with reasoning_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    continue
                key = metadata.get("pair_key")
                if key:
                    completed.add(str(key))
                normalized = pair_key_from_metadata(metadata)
                if normalized.strip():
                    completed.add(normalized)
    return len(expected & completed), len(expected)


def stage_synthesize(args: argparse.Namespace) -> None:
    pairs_csv = require_path(args.pairs_csv, "--pairs-csv")
    reasoning_jsonl = require_path(args.reasoning_jsonl, "--reasoning-jsonl")
    synth_config = require_path(args.synth_config, "--synth-config")
    if not pairs_csv.is_file():
        raise SystemExit(f"missing pairs CSV: {pairs_csv}")

    write_synth_config(args)
    previous: tuple[int, int] | None = None
    for attempt in range(1, args.max_passes + 1):
        covered, total = reasoning_coverage(pairs_csv, reasoning_jsonl)
        print(f"reasoning coverage before pass {attempt}: {covered}/{total}", flush=True)
        if covered == total:
            print("reasoning coverage complete", flush=True)
            return
        subprocess.run(
            [
                "python",
                str(ACCRL / "fib_runtime/multiturn/fix_kernels/synthesize_pair_reasoning_openrouter.py"),
                str(synth_config),
            ],
            check=True,
        )
        new_coverage = reasoning_coverage(pairs_csv, reasoning_jsonl)
        print(
            f"reasoning coverage after pass {attempt}: {new_coverage[0]}/{new_coverage[1]}",
            flush=True,
        )
        if previous == new_coverage:
            time.sleep(60)
        previous = new_coverage

    covered, total = reasoning_coverage(pairs_csv, reasoning_jsonl)
    raise SystemExit(f"reasoning coverage incomplete after {args.max_passes} passes: {covered}/{total}")


def stage_parquet(args: argparse.Namespace) -> None:
    pairs_csv = require_path(args.pairs_csv, "--pairs-csv")
    reasoning_jsonl = require_path(args.reasoning_jsonl, "--reasoning-jsonl")
    parquet = require_path(args.parquet, "--parquet")
    if not pairs_csv.is_file():
        raise SystemExit(f"missing pairs CSV: {pairs_csv}")
    if not reasoning_jsonl.is_file():
        raise SystemExit(f"missing reasoning JSONL: {reasoning_jsonl}")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        ACCRL / "accrl/distill/sft/build_sft_dataset_fixit.py",
        "--pairs-jsonl",
        reasoning_jsonl,
        "--kernel-pairs-csv",
        pairs_csv,
        "--output",
        parquet,
        "--tokenizer",
        args.tokenizer,
        "--max-tokens",
        str(args.parquet_max_tokens),
    ]
    if args.source_label:
        cmd.extend(["--source-label", args.source_label])
    if args.shuffle:
        cmd.extend(["--shuffle", "--shuffle-seed", str(args.shuffle_seed)])
    subprocess.run([str(part) for part in cmd], check=True)


def run_dir_glob_for_tag(args: argparse.Namespace) -> str:
    if args.train_run_tag:
        return f"{args.train_run_tag}-{args.base_model.replace('/', '-')}-*"
    return f"*-{args.base_model.replace('/', '-')}-*"


def latest_run_dir(args: argparse.Namespace, *, min_mtime: float | None = None) -> Path:
    pattern = run_dir_glob_for_tag(args)
    candidates = sorted(
        (
            path
            for path in args.runs_dir.glob(pattern)
            if path.is_dir() and (min_mtime is None or path.stat().st_mtime >= min_mtime)
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        detail = f" modified after {min_mtime}" if min_mtime is not None else ""
        tag_detail = f" for --train-run-tag {args.train_run_tag!r}" if args.train_run_tag else ""
        raise SystemExit(f"no run dirs matching {pattern!r}{tag_detail}{detail} found under {args.runs_dir}")
    return candidates[-1]


def stage_train(args: argparse.Namespace) -> None:
    parquet = require_path(args.parquet, "--parquet")
    if not parquet.is_file():
        raise SystemExit(f"missing parquet: {parquet}")
    train_command: list[str | Path] = [
        "bash",
        args.runs_dir / "run_qwen36-27b.sh",
        "--dataset-path",
        parquet,
        "--run-tag",
        args.train_run_tag,
    ]
    if args.train_num_epochs is not None:
        train_command.extend(["--num-epochs", str(args.train_num_epochs)])
    if args.train_learning_rate is not None:
        train_command.extend(["--learning-rate", str(args.train_learning_rate)])
    if args.train_load_checkpoint_path is not None:
        train_command.extend(
            ["--load-checkpoint-path", args.train_load_checkpoint_path]
        )
    command = quote_cmd(train_command)
    runner = Runner(True)
    runner.tmux_new(args.train_session)
    runner.tmux_send(args.train_session, command)
    print(f"training launched in tmux session {args.train_session}")


def wait_for_latest_checkpoint(args: argparse.Namespace, *, min_mtime: float | None) -> Path:
    deadline = time.monotonic() + args.train_timeout_s
    while time.monotonic() < deadline:
        try:
            run_dir = latest_run_dir(args, min_mtime=min_mtime)
        except SystemExit:
            time.sleep(args.poll_s)
            continue
        ckpt = run_dir / "checkpoints.jsonl"
        if ckpt.is_file() and ckpt.stat().st_size > 0:
            text = ckpt.read_text()
            if '"name": "final"' in text or "\tfinal\t" in text or "/weights/final" in text:
                return run_dir
        time.sleep(args.poll_s)
    raise SystemExit(f"timed out waiting for final checkpoint under {args.runs_dir}")


def remote_path_for_local(local_path: Path) -> str:
    local_str = str(local_path)
    prefix = str(EXPS)
    if not local_str.startswith(prefix):
        raise SystemExit(
            f"cannot map local path outside PTXBENCH_DATA_ROOT={EXPS} to /data02: {local_path}"
        )
    return "/data02" + local_str[len(prefix) :]


def copy_repo_to_container(runner: Runner, remote: str, container: str, repo: Path, dest: str) -> None:
    remote_cmd = f"docker exec -i {container} bash -lc 'mkdir -p {dest} && tar -C {dest} -xzf -'"
    command = (
        f"tar -C {shlex.quote(str(repo))} --exclude=.git -czf - . | "
        f"ssh {shlex.quote(remote)} {shlex.quote(remote_cmd)}"
    )
    runner.shell(command)


def transfer_checkpoint(runner: Runner, remote: str, container: str, checkpoints_jsonl: Path) -> str:
    remote_checkpoint = remote_path_for_local(checkpoints_jsonl)
    remote_dir = str(Path(remote_checkpoint).parent)
    remote_cmd = f"docker exec -i {container} bash -lc 'mkdir -p {remote_dir} && cat > {remote_checkpoint}'"
    runner.shell(
        f"ssh {shlex.quote(remote)} {shlex.quote(remote_cmd)} < {shlex.quote(str(checkpoints_jsonl))}"
    )
    if runner.execute:
        local_hash = sha256(checkpoints_jsonl)
        out = runner.capture(["ssh", remote, f"docker exec {container} sha256sum {shlex.quote(remote_checkpoint)}"])
        match = re.search(r"\b[0-9a-fA-F]{64}\b", out)
        remote_hash = match.group(0).lower() if match else ""
        if remote_hash != local_hash:
            raise SystemExit(f"checkpoint sha256 mismatch: local={local_hash} remote={remote_hash}")
    return remote_checkpoint


def prepare_remote_model(
    runner: Runner,
    *,
    remote: str,
    container: str,
    checkpoints_jsonl: Path,
    base_model: str,
) -> str:
    copy_repo_to_container(
        runner,
        remote,
        container,
        ACCRL,
        "/data02/PTXBench/packages/mini-ptx-agent",
    )
    copy_repo_to_container(
        runner,
        remote,
        container,
        TINKER_COOKBOOK_ROOT,
        "/data02/tinker-cookbook",
    )
    remote_checkpoint = transfer_checkpoint(runner, remote, container, checkpoints_jsonl)
    hf_output = f"{Path(remote_checkpoint).parent}/hf_merged_final"
    runner.remote(
        remote,
        (
            f"docker exec {container} bash -lc "
            f"{shlex.quote('/data02/tinker-cookbook/.venv/bin/python -c \"import tinker, tinker_cookbook\"')}"
        ),
    )
    downloader = (
        "export TINKER_API_KEY=$(cat /data02/TINKER_API_KEY); "
        "/data02/tinker-cookbook/.venv/bin/python "
        "/data02/PTXBench/packages/mini-ptx-agent/accrl/distill/sft/tinker_download_weights.py "
        f"--checkpoints-jsonl {shlex.quote(remote_checkpoint)} "
        f"--hf-output {shlex.quote(hf_output)} "
        "--checkpoint-name final "
        f"--base-model {shlex.quote(base_model)}"
    )
    runner.remote(remote, f"docker exec {container} bash -lc {shlex.quote(downloader)}")
    return hf_output


def remote_container_tmux_session_exists(runner: Runner, remote: str, container: str, session: str) -> bool:
    proc = subprocess.run(
        ["ssh", remote, f"docker exec {container} tmux has-session -t {shlex.quote(session)}"],
        cwd=str(runner.cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode == 0


def wait_for_remote_container_port_free(
    runner: Runner,
    remote: str,
    container: str,
    port: int,
    *,
    timeout_s: int,
    poll_s: int,
) -> None:
    probe = (
        "python -c "
        + shlex.quote(
            "import socket, sys; "
            "s=socket.socket(); "
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
            "port=int(sys.argv[1]); "
            "\ntry:\n"
            "    s.bind(('0.0.0.0', port))\n"
            "except OSError as exc:\n"
            "    print(exc)\n"
            "    sys.exit(1)\n"
            "finally:\n"
            "    s.close()\n"
        )
        + f" {port}"
    )
    command = f"docker exec {container} bash -lc {shlex.quote(probe)}"
    deadline = time.monotonic() + timeout_s
    last_output = ""
    while time.monotonic() < deadline:
        proc = subprocess.run(
            ["ssh", remote, command],
            cwd=str(runner.cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode == 0:
            print(f"remote container port free: {container}:{port}")
            return
        last_output = proc.stdout.strip()
        time.sleep(poll_s)
    raise SystemExit(f"remote container port is occupied: {container}:{port}: {last_output}")


def remote_container_port_is_free(runner: Runner, remote: str, container: str, port: int) -> bool:
    try:
        wait_for_remote_container_port_free(runner, remote, container, port, timeout_s=1, poll_s=1)
        return True
    except SystemExit:
        return False


def local_port_is_listening(port: int) -> bool:
    for host in ("127.0.0.1", "::1"):
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                if sock.connect_ex((host, port)) == 0:
                    return True
            except OSError:
                continue
    return False


def reusable_tunnel_process(local_port: int, remote_port: int, remote: str) -> str | None:
    proc = subprocess.run(
        ["ps", "-eo", "pid,args"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    mapping = f"{local_port}:localhost:{remote_port}"
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or "ps -eo pid,args" in stripped or "rg " in stripped:
            continue
        if "ssh" in stripped and "-L" in stripped and mapping in stripped and remote in stripped:
            return stripped[:300]
    return None


def wait_for_local_port_free(port: int, *, timeout_s: int, poll_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not local_port_is_listening(port):
            print(f"local port free: {port}")
            return
        time.sleep(poll_s)
    raise SystemExit(f"local port is occupied: {port}")


def local_port_for_remote_port(remote_port: int) -> int:
    if remote_port < 9000 or remote_port > 9009:
        raise SystemExit(f"cannot derive local tunnel port for remote port {remote_port}")
    return 30002 + (remote_port - 9000) * 10


def select_remote_port_and_local_port(
    runner: Runner,
    *,
    remote: str,
    container: str,
    candidate_remote_ports: list[int],
    candidate_local_ports: list[int],
) -> tuple[int, int]:
    for remote_port in candidate_remote_ports:
        if not remote_container_port_is_free(runner, remote, container, remote_port):
            print(f"remote container port busy: {container}:{remote_port}")
            continue
        local_candidates = [local_port_for_remote_port(remote_port)] + [
            port for port in candidate_local_ports if port != local_port_for_remote_port(remote_port)
        ]
        for local_port in local_candidates:
            tunnel = reusable_tunnel_process(local_port, remote_port, remote)
            if tunnel:
                print(f"reusing existing tunnel for remote={remote_port} local={local_port}: {tunnel}")
                return remote_port, local_port
            if local_port_is_listening(local_port):
                print(f"local tunnel port busy: {local_port} for remote port {remote_port}")
                continue
            print(f"selected ports: remote={remote_port} local={local_port}")
            return remote_port, local_port
    raise SystemExit(f"no free remote serve port found; checked remote ports {candidate_remote_ports}")


def select_two_free_gpus(runner: Runner, remote: str) -> str:
    gpu_out = runner.capture(
        [
            "ssh",
            remote,
            "nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits",
        ]
    )
    app_proc = subprocess.run(
        [
            "ssh",
            remote,
            "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits",
        ],
        cwd=str(runner.cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    busy_uuids = {line.strip() for line in app_proc.stdout.splitlines() if line.strip().startswith("GPU-")}
    free: list[int] = []
    for line in gpu_out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            used = int(parts[2])
        except ValueError:
            continue
        if parts[1] not in busy_uuids and used <= FREE_GPU_MAX_MEMORY_MIB:
            free.append(index)
    if len(free) < 2:
        raise SystemExit(f"no two free GPUs on {remote}; free GPUs under {FREE_GPU_MAX_MEMORY_MIB} MiB: {free}")
    selected = ",".join(str(index) for index in free[:2])
    print(f"selected free GPUs on {remote}: {selected}")
    return selected


def tmux_session_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).returncode == 0


def ensure_tunnel(runner: Runner, *, tunnel_session: str, remote: str, local_port: int, remote_port: int) -> None:
    if runner.execute:
        tunnel = reusable_tunnel_process(local_port, remote_port, remote)
        if tunnel:
            print(f"reusing existing tunnel: {tunnel}")
            return
        if tmux_session_exists(tunnel_session):
            raise SystemExit(f"local tmux tunnel session already exists, refusing to reuse it: {tunnel_session}")
        wait_for_local_port_free(local_port, timeout_s=60, poll_s=2)
    runner.shell(
        f"tmux new-session -d -s {shlex.quote(tunnel_session)} "
        f"{shlex.quote(f'ssh -N -L {local_port}:localhost:{remote_port} {remote}')}"
    )


def fetch_json(url: str, timeout_s: int) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_model(local_port: int, model_name: str, timeout_s: int, poll_s: int) -> None:
    url = f"http://localhost:{local_port}/v1/models"
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            data = fetch_json(url, 10)
            ids = [item.get("id") for item in data.get("data", [])]
            if model_name in ids:
                print(f"model endpoint ready: {url} includes {model_name}")
                return
            last_error = f"model ids={ids}"
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(poll_s)
    raise SystemExit(f"timed out waiting for {model_name} at {url}: {last_error}")


def start_sglang(
    runner: Runner,
    *,
    remote: str,
    container: str,
    serve_session: str,
    gpus: str,
    remote_port: int,
    served_model_name: str,
    hf_output: str,
) -> None:
    if runner.execute and remote_container_tmux_session_exists(runner, remote, container, serve_session):
        raise SystemExit(f"remote tmux session already exists, refusing to reuse it: {container}:{serve_session}")
    serve_cmd = (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpus)} "
        "sglang serve "
        f"--served-model-name {shlex.quote(served_model_name)} "
        f"--model-path {shlex.quote(hf_output)} "
        "--reasoning-parser qwen3 "
        "--host 0.0.0.0 "
        f"--port {remote_port} "
        "--tp 2"
    )
    if runner.execute:
        wait_for_remote_container_port_free(runner, remote, container, remote_port, timeout_s=60, poll_s=2)
    runner.remote(
        remote,
        f"docker exec {container} tmux new-session -d -s {shlex.quote(serve_session)} {shlex.quote(serve_cmd)}",
    )


def stage_serve(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    if run_dir is None:
        if args.wait_for_checkpoint:
            print(
                "waiting for final checkpoint: "
                f"train_run_tag={args.train_run_tag} timeout_s={args.train_timeout_s} "
                f"poll_s={args.poll_s}",
                flush=True,
            )
            run_dir = wait_for_latest_checkpoint(args, min_mtime=None)
            print(f"final checkpoint ready: {run_dir}", flush=True)
        else:
            run_dir = latest_run_dir(args)
    run_date = args.run_date or run_date_from_run_dir(run_dir)
    model_name = args.model_name or f"{args.model_prefix}-{run_date}"
    checkpoints_jsonl = run_dir / "checkpoints.jsonl"
    if args.execute_serve and not checkpoints_jsonl.is_file():
        raise SystemExit(f"missing checkpoints.jsonl: {checkpoints_jsonl}")

    runner = Runner(args.execute_serve)
    remote_port = args.remote_port
    local_port = args.local_port
    gpus = args.gpus
    if args.execute_serve:
        if remote_port is None:
            remote_port, selected_local_port = select_remote_port_and_local_port(
                runner,
                remote=args.remote,
                container=args.container,
                candidate_remote_ports=args.candidate_remote_ports,
                candidate_local_ports=args.candidate_local_ports,
            )
            if local_port is None:
                local_port = selected_local_port
        elif local_port is None:
            local_port = local_port_for_remote_port(remote_port)
            if local_port_is_listening(local_port):
                raise SystemExit(f"local tunnel port is occupied: {local_port}")
            if not remote_container_port_is_free(runner, args.remote, args.container, remote_port):
                raise SystemExit(f"remote container port is occupied: {args.container}:{remote_port}")
        if gpus is None:
            gpus = select_two_free_gpus(runner, args.remote)
    else:
        remote_port = remote_port or args.candidate_remote_ports[0]
        local_port = local_port or local_port_for_remote_port(remote_port)
        gpus = gpus or "<two-free-gpus>"

    print(
        "serve resource selection: "
        f"remote_port={remote_port} local_port={local_port} gpus={gpus} "
        f"serve_session={args.serve_session} tunnel_session={args.tunnel_session}"
    )
    hf_output = prepare_remote_model(
        runner,
        remote=args.remote,
        container=args.container,
        checkpoints_jsonl=checkpoints_jsonl,
        base_model=args.base_model,
    )
    start_sglang(
        runner,
        remote=args.remote,
        container=args.container,
        serve_session=args.serve_session,
        gpus=gpus,
        remote_port=remote_port,
        served_model_name=model_name,
        hf_output=hf_output,
    )
    ensure_tunnel(
        runner,
        tunnel_session=args.tunnel_session,
        remote=args.remote,
        local_port=local_port,
        remote_port=remote_port,
    )
    if runner.execute:
        wait_for_model(local_port, model_name, args.serve_timeout_s, args.poll_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=parse_stages, required=True)
    parser.add_argument("--pairs-csv", type=Path, default=env_path("PAIRS_CSV"))
    parser.add_argument("--synth-config", type=Path, default=env_path("SYNTH_CONFIG"))
    parser.add_argument("--reasoning-jsonl", type=Path, default=env_path("REASONING_JSONL"))
    parser.add_argument("--provenance-json", type=Path, default=env_path("PROVENANCE_JSON"))
    parser.add_argument("--parquet", type=Path, default=env_path("PARQUET"))
    parser.add_argument("--project-dir", type=Path, default=env_path("PROJECT_DIR", str(DEFAULT_PROJECT)))
    parser.add_argument("--runs-dir", type=Path, default=env_path("RUNS_DIR", str(DEFAULT_RUNS)))
    parser.add_argument("--base-model", default=env_text("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--model-prefix", default=env_text("MODEL_PREFIX", DEFAULT_MODEL_PREFIX))
    parser.add_argument("--synth-name", default=env_text("SYNTH_NAME", "fixit-pair-reasoning"))
    parser.add_argument("--synth-description", default=env_text("SYNTH_DESCRIPTION", "Synthesize reasoning for fix-it kernel pairs."))
    parser.add_argument("--reasoning-model", default=env_text("REASONING_MODEL", DEFAULT_REASONING_MODEL))
    parser.add_argument("--prompt-version", default=env_text("PROMPT_VERSION", DEFAULT_PROMPT_VERSION))
    parser.add_argument("--max-tokens", type=int, default=int(env_text("MAX_TOKENS", "131072")))
    parser.add_argument("--min-reasoning-chars", type=int, default=int(env_text("MIN_REASONING_CHARS", "2000")))
    parser.add_argument("--max-reasoning-chars", type=int, default=int(env_text("MAX_REASONING_CHARS", "190000")))
    parser.add_argument("--temperature", type=float, default=float(env_text("TEMPERATURE", "1.0")))
    parser.add_argument("--top-p", type=float, default=float(env_text("TOP_P", "0.95")))
    parser.add_argument("--max-concurrent", type=int, default=int(env_text("MAX_CONCURRENT", "8")))
    parser.add_argument("--timeout", type=float, default=float(env_text("TIMEOUT", "600.0")))
    parser.add_argument("--overwrite", action="store_true", default=env_text("OVERWRITE") == "1")
    parser.add_argument("--max-passes", type=int, default=int(env_text("MAX_PASSES", "20")))
    parser.add_argument("--tokenizer", default=env_text("TOKENIZER", DEFAULT_BASE_MODEL))
    parser.add_argument("--parquet-max-tokens", type=int, default=int(env_text("PARQUET_MAX_TOKENS", "65536")))
    parser.add_argument("--source-label", default=env_text("SOURCE_LABEL"))
    parser.add_argument("--shuffle", action="store_true", default=env_text("SHUFFLE") == "1")
    parser.add_argument("--shuffle-seed", type=int, default=int(env_text("SHUFFLE_SEED", "42")))
    parser.add_argument("--train-session", default=env_text("TRAIN_SESSION", "train-fixit"))
    parser.add_argument("--train-run-tag", default=env_text("TRAIN_RUN_TAG", "qwen36-27b-fixit"))
    parser.add_argument(
        "--train-num-epochs",
        type=int,
        default=(int(env_text("TRAIN_NUM_EPOCHS")) if env_text("TRAIN_NUM_EPOCHS") else None),
    )
    parser.add_argument(
        "--train-learning-rate",
        type=float,
        default=(
            float(env_text("TRAIN_LEARNING_RATE"))
            if env_text("TRAIN_LEARNING_RATE")
            else None
        ),
    )
    parser.add_argument(
        "--train-load-checkpoint-path",
        default=env_text("TRAIN_LOAD_CHECKPOINT_PATH") or None,
    )
    parser.add_argument("--train-timeout-s", type=int, default=int(env_text("TRAIN_TIMEOUT_S", str(24 * 3600))))
    parser.add_argument("--poll-s", type=int, default=int(env_text("POLL_S", "15")))
    parser.add_argument(
        "--wait-for-checkpoint",
        action="store_true",
        default=env_text("WAIT_FOR_CHECKPOINT") == "1",
        help="Poll for the latest run's final checkpoint before serving.",
    )
    parser.add_argument("--remote", default=env_text("REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--container", default=env_text("CONTAINER", DEFAULT_CONTAINER))
    parser.add_argument("--serve-session", default=env_text("SERVE_SESSION", "serve-fixit"))
    parser.add_argument("--tunnel-session", default=env_text("TUNNEL_SESSION", "connect-sglang-fixit"))
    parser.add_argument("--serve-timeout-s", type=int, default=int(env_text("SERVE_TIMEOUT_S", "1800")))
    parser.add_argument("--run-dir", type=Path, default=env_path("RUN_DIR"))
    parser.add_argument("--run-date", default=env_text("RUN_DATE"))
    parser.add_argument("--model-name", default=env_text("MODEL_NAME"))
    parser.add_argument("--gpus", default=env_text("GPUS") or None)
    parser.add_argument("--remote-port", type=int, default=int(env_text("REMOTE_PORT")) if env_text("REMOTE_PORT") else None)
    parser.add_argument("--local-port", type=int, default=int(env_text("LOCAL_PORT")) if env_text("LOCAL_PORT") else None)
    parser.add_argument("--profile-url", default=env_text("PROFILE_URL"))
    parser.add_argument("--candidate-remote-ports", type=parse_int_list, default=parse_int_list(env_text("CANDIDATE_REMOTE_PORTS", DEFAULT_REMOTE_PORTS)))
    parser.add_argument("--candidate-local-ports", type=parse_int_list, default=parse_int_list(env_text("CANDIDATE_LOCAL_PORTS", DEFAULT_LOCAL_PORTS)))
    parser.add_argument(
        "--execute-serve",
        action="store_true",
        default=env_text("EXECUTE") == "1",
        help="Actually run the remote serve stage. Defaults to true only when EXECUTE=1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path_attr in (
        "pairs_csv",
        "synth_config",
        "reasoning_jsonl",
        "provenance_json",
        "parquet",
        "project_dir",
        "runs_dir",
        "run_dir",
    ):
        path = getattr(args, path_attr)
        if path is not None:
            setattr(args, path_attr, path.expanduser())

    if args.train_num_epochs is not None and args.train_num_epochs <= 0:
        raise SystemExit("--train-num-epochs must be positive")
    if args.train_learning_rate is not None and (
        not math.isfinite(args.train_learning_rate) or args.train_learning_rate <= 0
    ):
        raise SystemExit("--train-learning-rate must be positive and finite")

    for stage in args.stages:
        if stage == "synthesize":
            stage_synthesize(args)
        elif stage == "parquet":
            stage_parquet(args)
        elif stage == "train":
            stage_train(args)
        elif stage == "serve":
            stage_serve(args)
        else:
            raise AssertionError(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
