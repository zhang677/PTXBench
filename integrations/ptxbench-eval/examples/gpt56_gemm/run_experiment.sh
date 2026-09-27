#!/usr/bin/env bash
set -Eeuo pipefail

# Complete launcher for the Codex GPT-5.6 Sol xhigh H100 GEMM experiment.
#
# Required:
#   codex login
#
# Common overrides:
#   EXPERIMENT_ROOT=/absolute/path/to/fresh-run
#   MODEL=gpt-5.6-sol
#   EFFORT=xhigh
#   CODEX_AUTH_FILE=/path/to/codex/auth.json
#   PROFILE_BASE_URL=http://127.0.0.1:10000
#   GATEWAY_PORT=18082
#   SKIP_BUILD=1
#   PROFILE_RECOVERY_COMMAND_JSON='["docker","exec","fib-profile","bash","-lc","bash /workspace/scripts/restart_profiling.sh"]'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd -- "$PACKAGE_ROOT/.." && pwd)
TEMPLATE_CONFIG="$SCRIPT_DIR/experiment.json"

RUN_STAMP=$(date -u +%Y%m%d-%H%M%S)
MODEL=${MODEL:-gpt-5.6-sol}
EFFORT=${EFFORT:-xhigh}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$REPO_ROOT/results/codex-gpt56-sol-${EFFORT}-h100-gemm-${RUN_STAMP}}
CODEX_AUTH_FILE=${CODEX_AUTH_FILE:-${CODEX_HOME:-$HOME/.codex}/auth.json}
PROFILE_BASE_URL=${PROFILE_BASE_URL:-http://127.0.0.1:10000}
GATEWAY_HOST=${GATEWAY_HOST:-127.0.0.1}
GATEWAY_PORT=${GATEWAY_PORT:-18082}
GATEWAY_URL="http://${GATEWAY_HOST}:${GATEWAY_PORT}"
AGENT_IMAGE=${AGENT_IMAGE:-ptxbench-codex:dev}
GATEWAY_IMAGE=${GATEWAY_IMAGE:-ptxbench-eval-gateway:dev}
GATEWAY_BASE_IMAGE=${GATEWAY_BASE_IMAGE:-ptxbench-eval-base:latest}
SKIP_BUILD=${SKIP_BUILD:-0}
PROFILE_RECOVERY_COMMAND_JSON=${PROFILE_RECOVERY_COMMAND_JSON:-}
PROFILE_RECOVERY_TIMEOUT=${PROFILE_RECOVERY_TIMEOUT:-720}
PROFILE_RECOVERY_RESTART_TIMEOUT=${PROFILE_RECOVERY_RESTART_TIMEOUT:-600}
PROFILE_RECOVERY_POLL_INTERVAL=${PROFILE_RECOVERY_POLL_INTERVAL:-1}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

for command_name in docker curl grep python realpath seq tr; do
    command -v "$command_name" >/dev/null 2>&1 \
        || die "required command not found: $command_name"
done

[[ -n "$MODEL" ]] || die "MODEL must not be empty"
[[ -n "$EFFORT" ]] || die "EFFORT must not be empty"
[[ -r "$CODEX_AUTH_FILE" ]] \
    || die "Codex plan auth file is not readable: $CODEX_AUTH_FILE (run: codex login)"
CODEX_AUTH_FILE=$(realpath -- "$CODEX_AUTH_FILE")
[[ "$EXPERIMENT_ROOT" = /* ]] || die "EXPERIMENT_ROOT must be absolute"
EXPERIMENT_ROOT=$(realpath -m -- "$EXPERIMENT_ROOT")
[[ "$EXPERIMENT_ROOT" != / ]] || die "refusing to use / as EXPERIMENT_ROOT"
[[ ! -e "$EXPERIMENT_ROOT" ]] \
    || die "experiment root already exists: $EXPERIMENT_ROOT"
[[ "$GATEWAY_PORT" =~ ^[0-9]+$ ]] \
    && ((GATEWAY_PORT >= 1 && GATEWAY_PORT <= 65535)) \
    || die "GATEWAY_PORT must be an integer from 1 to 65535"
[[ "$SKIP_BUILD" = 0 || "$SKIP_BUILD" = 1 ]] \
    || die "SKIP_BUILD must be 0 or 1"

python - "$PROFILE_RECOVERY_COMMAND_JSON" "$PROFILE_RECOVERY_TIMEOUT" \
    "$PROFILE_RECOVERY_RESTART_TIMEOUT" "$PROFILE_RECOVERY_POLL_INTERVAL" <<'PY'
import json
import sys

command_json, *timings = sys.argv[1:]
for value in timings:
    try:
        valid = float(value) > 0
    except ValueError:
        valid = False
    if not valid:
        raise SystemExit(f"profile recovery timing must be positive, got: {value}")
if command_json:
    try:
        command = json.loads(command_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid PROFILE_RECOVERY_COMMAND_JSON: {exc}") from exc
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise SystemExit(
            "PROFILE_RECOVERY_COMMAND_JSON must be a non-empty JSON array of strings"
        )
PY

docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"

readarray -t RUN_IDS < <(
    python - "$TEMPLATE_CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
for run in config["runs"]:
    print(run["run_id"])
PY
)
(( ${#RUN_IDS[@]} > 0 )) || die "template contains no runs"

for run_id in "${RUN_IDS[@]}"; do
    container_name="ptxbench-${run_id//_/-}"
    if docker ps -a --format '{{.Names}}' | grep -Fxq -- "$container_name"; then
        die "conflicting agent container already exists: $container_name"
    fi
done

python - "$GATEWAY_HOST" "$GATEWAY_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket() as sock:
    sock.settimeout(1)
    if sock.connect_ex((host, port)) == 0:
        raise SystemExit(f"gateway address is already in use: {host}:{port}")
PY

echo "Checking profiling service: $PROFILE_BASE_URL"
python - "$PROFILE_BASE_URL" "$TEMPLATE_CONFIG" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base_url = sys.argv[1].rstrip("/")
config = json.load(open(sys.argv[2], encoding="utf-8"))

def get_json(path):
    try:
        with urllib.request.urlopen(base_url + path, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"profiling request failed for {path}: {exc}") from exc

health = get_json("/health")
backends = health.get("backends") or health.get("workers") or []
healthy = (
    health.get("status") == "ok"
    and isinstance(backends, list)
    and len(backends) > 0
    and all(isinstance(item, dict) and item.get("healthy") is True for item in backends)
    and int(health.get("queue_size") or 0) == 0
)
if not healthy:
    raise SystemExit("profiling service is not ready: " + json.dumps(health, sort_keys=True))

definition = config["definition"]
expected_uuid = config["workload_uuid"]
workloads = get_json(f"/definitions/{definition}/workloads")
if not isinstance(workloads, list):
    raise SystemExit(f"unexpected workload response for {definition}")
available = {item.get("uuid") for item in workloads if isinstance(item, dict)}
if expected_uuid not in available:
    raise SystemExit(
        f"profiling service lacks workload {expected_uuid} for definition {definition}"
    )

print(
    f"Profiling ready: backends={len(backends)} queue=0 "
    f"definition={definition} workload={expected_uuid}"
)
PY

if [[ "$SKIP_BUILD" = 0 ]]; then
    echo "Building $AGENT_IMAGE"
    docker build \
        -f "$PACKAGE_ROOT/docker/Dockerfile.codex" \
        -t "$AGENT_IMAGE" \
        "$REPO_ROOT"

    if ! docker image inspect "$GATEWAY_BASE_IMAGE" >/dev/null 2>&1; then
        echo "Building gateway base image: $GATEWAY_BASE_IMAGE"
        docker build \
            -f "$PACKAGE_ROOT/docker/Dockerfile.gateway-base" \
            -t "$GATEWAY_BASE_IMAGE" \
            "$REPO_ROOT"
    fi

    echo "Building $GATEWAY_IMAGE"
    docker build \
        -f "$PACKAGE_ROOT/docker/Dockerfile.gateway" \
        --build-arg "BASE_IMAGE=$GATEWAY_BASE_IMAGE" \
        -t "$GATEWAY_IMAGE" \
        "$REPO_ROOT"
else
    docker image inspect "$AGENT_IMAGE" >/dev/null 2>&1 \
        || die "missing agent image: $AGENT_IMAGE"
    docker image inspect "$GATEWAY_IMAGE" >/dev/null 2>&1 \
        || die "missing gateway image: $GATEWAY_IMAGE"
fi

mkdir -p -- "$(dirname -- "$EXPERIMENT_ROOT")"
mkdir -- "$EXPERIMENT_ROOT"
RUN_CONFIG="$EXPERIMENT_ROOT/launch_config.json"

python - "$TEMPLATE_CONFIG" "$RUN_CONFIG" "$EXPERIMENT_ROOT" "$GATEWAY_URL" \
    "$AGENT_IMAGE" "$MODEL" "$EFFORT" "$CODEX_AUTH_FILE" \
    "$PROFILE_BASE_URL" <<'PY'
import json
import pathlib
import sys

template_path = pathlib.Path(sys.argv[1]).resolve()
output_path = pathlib.Path(sys.argv[2])
experiment_root = pathlib.Path(sys.argv[3]).resolve()
gateway_url = sys.argv[4]
agent_image = sys.argv[5]
model = sys.argv[6]
effort = sys.argv[7]
codex_auth_file = pathlib.Path(sys.argv[8]).resolve()
profile_base_url = sys.argv[9]

config = json.loads(template_path.read_text())
source_dir = template_path.parent

def absolute(value):
    path = pathlib.Path(value).expanduser()
    return str((path if path.is_absolute() else source_dir / path).resolve())

config["experiment_root"] = str(experiment_root)
config["gateway_url"] = gateway_url
config["profile_base_url"] = profile_base_url
config["agent"] = "codex"
config["agent_image"] = agent_image
config["model"] = model
config["effort"] = effort
config["codex_auth_file"] = str(codex_auth_file)
config["prompt_hub"] = absolute(config["prompt_hub"])
config["prompt_files_root"] = absolute(config["prompt_files_root"])
config["tvm_ffi_example"] = absolute(config["tvm_ffi_example"])

output_path.write_text(json.dumps(config, indent=2) + "\n")
PY

echo "Preparing fresh experiment: $EXPERIMENT_ROOT"
(
    cd "$PACKAGE_ROOT"
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PACKAGE_ROOT/src" \
        python -m ptxbench_eval.launcher prepare "$RUN_CONFIG"
)

gateway_basename=$(basename -- "$EXPERIMENT_ROOT")
gateway_suffix=$(printf '%s' "$gateway_basename" | tr -c 'A-Za-z0-9_.-' '-')
GATEWAY_CONTAINER="ptxbench-gateway-${gateway_suffix}"
if docker ps -a --format '{{.Names}}' | grep -Fxq -- "$GATEWAY_CONTAINER"; then
    die "conflicting gateway container already exists: $GATEWAY_CONTAINER"
fi

STATUS_WATCHER_PID=
PROFILE_RECOVERY_WORKER_PID=
PROFILE_RECOVERY_ENABLED=0
PROFILE_RECOVERY_DIR="$EXPERIMENT_ROOT/profile_recovery"
GATEWAY_RECOVERY_ARGS=()
if [[ -n "$PROFILE_RECOVERY_COMMAND_JSON" ]]; then
    PROFILE_RECOVERY_ENABLED=1
    mkdir -p -- "$PROFILE_RECOVERY_DIR"
    GATEWAY_RECOVERY_ARGS=(
        --profile-recovery-control-dir "$PROFILE_RECOVERY_DIR"
        --profile-recovery-timeout "$PROFILE_RECOVERY_TIMEOUT"
        --profile-recovery-poll-interval "$PROFILE_RECOVERY_POLL_INTERVAL"
        --max-profile-recoveries 1
    )
fi

cleanup_runtime() {
    if [[ -n "$STATUS_WATCHER_PID" ]]; then
        kill "$STATUS_WATCHER_PID" >/dev/null 2>&1 || true
        wait "$STATUS_WATCHER_PID" >/dev/null 2>&1 || true
    fi
    if docker ps -a --format '{{.Names}}' | grep -Fxq -- "$GATEWAY_CONTAINER"; then
        docker logs "$GATEWAY_CONTAINER" >"$EXPERIMENT_ROOT/gateway.log" 2>&1 || true
        echo "Stopping gateway container: $GATEWAY_CONTAINER"
        docker stop --time 10 "$GATEWAY_CONTAINER" >/dev/null 2>&1 || true
    fi
    if [[ -n "$PROFILE_RECOVERY_WORKER_PID" ]]; then
        kill "$PROFILE_RECOVERY_WORKER_PID" >/dev/null 2>&1 || true
        wait "$PROFILE_RECOVERY_WORKER_PID" >/dev/null 2>&1 || true
    fi
}

on_exit() {
    status=$?
    trap - EXIT INT TERM
    cleanup_runtime
    exit "$status"
}

on_signal() {
    trap - EXIT INT TERM
    cleanup_runtime
    exit 130
}

trap on_exit EXIT
trap on_signal INT TERM

if [[ "$PROFILE_RECOVERY_ENABLED" = 1 ]]; then
    echo "Starting host profile-recovery worker"
    (
        cd "$PACKAGE_ROOT"
        exec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PACKAGE_ROOT/src" \
            python -m ptxbench_eval.profile_recovery_worker \
                --control-dir "$PROFILE_RECOVERY_DIR" \
                --restart-command-json "$PROFILE_RECOVERY_COMMAND_JSON" \
                --restart-timeout "$PROFILE_RECOVERY_RESTART_TIMEOUT" \
                --poll-interval "$PROFILE_RECOVERY_POLL_INTERVAL"
    ) >"$EXPERIMENT_ROOT/profile_recovery_worker.log" 2>&1 &
    PROFILE_RECOVERY_WORKER_PID=$!
    recovery_worker_ready=0
    for _ in $(seq 1 50); do
        if [[ -f "$PROFILE_RECOVERY_DIR/worker.json" ]]; then
            recovery_worker_ready=1
            break
        fi
        if ! kill -0 "$PROFILE_RECOVERY_WORKER_PID" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ "$recovery_worker_ready" != 1 ]]; then
        cat "$EXPERIMENT_ROOT/profile_recovery_worker.log" >&2 || true
        die "profile-recovery worker did not become ready"
    fi
    echo "Automatic profile recovery enabled"
else
    echo "Automatic profile recovery disabled: PROFILE_RECOVERY_COMMAND_JSON is unset"
fi

echo "Starting gateway container: $GATEWAY_CONTAINER"
docker run --rm --detach \
    --name "$GATEWAY_CONTAINER" \
    --network host \
    -v "$EXPERIMENT_ROOT:$EXPERIMENT_ROOT" \
    "$GATEWAY_IMAGE" \
    --root "$EXPERIMENT_ROOT" \
    --profile-base-url "$PROFILE_BASE_URL" \
    --host "$GATEWAY_HOST" \
    --port "$GATEWAY_PORT" \
    "${GATEWAY_RECOVERY_ARGS[@]}" \
    >/dev/null

gateway_ready=0
for _ in $(seq 1 30); do
    if health_json=$(curl -fsS --max-time 2 "$GATEWAY_URL/health" 2>/dev/null); then
        if python - "$EXPERIMENT_ROOT" "$PROFILE_BASE_URL" \
            "$PROFILE_RECOVERY_ENABLED" "$health_json" <<'PY'
import json
import pathlib
import sys

expected_root = str(pathlib.Path(sys.argv[1]).resolve())
expected_profile = sys.argv[2].rstrip("/")
expected_recovery = sys.argv[3] == "1"
health = json.loads(sys.argv[4])
actual_root = str(pathlib.Path(health.get("root", "")).resolve())
actual_profile = str(health.get("profile_base_url", "")).rstrip("/")
if health.get("status") != "ok":
    raise SystemExit(1)
if actual_root != expected_root or actual_profile != expected_profile:
    raise SystemExit(1)
if bool((health.get("profile_recovery") or {}).get("enabled")) != expected_recovery:
    raise SystemExit(1)
PY
        then
            gateway_ready=1
            break
        fi
    fi
    if ! docker ps --format '{{.Names}}' | grep -Fxq -- "$GATEWAY_CONTAINER"; then
        break
    fi
    sleep 1
done

if [[ "$gateway_ready" != 1 ]]; then
    docker logs "$GATEWAY_CONTAINER" >&2 2>/dev/null || true
    die "gateway did not become ready with the expected root and profiling URL"
fi

echo "Gateway ready: $GATEWAY_URL -> $EXPERIMENT_ROOT"
echo "Launching ${#RUN_IDS[@]} agent runs"
echo "Status command: PYTHONPATH=$PACKAGE_ROOT/src python -m ptxbench_eval.launcher status $EXPERIMENT_ROOT"

(
    while sleep 30; do
        echo
        echo "Status at $(date -u +%Y-%m-%dT%H:%M:%SZ):"
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PACKAGE_ROOT/src" \
            python -m ptxbench_eval.launcher status "$EXPERIMENT_ROOT" || true
    done
) &
STATUS_WATCHER_PID=$!

cd "$PACKAGE_ROOT"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PACKAGE_ROOT/src" \
python -m ptxbench_eval.launcher launch "$RUN_CONFIG"

echo "Final status:"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PACKAGE_ROOT/src" \
    python -m ptxbench_eval.launcher status "$EXPERIMENT_ROOT"
