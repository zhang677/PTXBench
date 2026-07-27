#!/usr/bin/env bash

_WATCH_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_WATCH_COMMON_DIR/ptxbench_paths.sh"
unset _WATCH_COMMON_DIR

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

running_eval_pids() {
  python - "$WATCH_SCRIPT" "${OUTPUT_ROOTS[@]}" <<'PY'
import os
import subprocess
import sys

watch_script = sys.argv[1]
output_roots = sys.argv[2:]
out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
hits = []

for line in out.splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    pid, _, cmd = stripped.partition(" ")
    if pid == str(os.getpid()):
        continue
    if watch_script in cmd:
        continue
    if "run_parallel_v2.py" in cmd and any(root in cmd for root in output_roots):
        hits.append(pid)
    elif "run_v2.py" in cmd and any(root in cmd for root in output_roots):
        hits.append(pid)

if hits:
    print(" ".join(hits))
    sys.exit(0)
sys.exit(1)
PY
}

running_fixit_pids() {
  python - "$OUTPUT_ROOT" "$WATCH_SCRIPT" <<'PY'
import os
import subprocess
import sys

output_root = sys.argv[1]
watch_script = sys.argv[2]
out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
hits = []

for line in out.splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    pid, _, cmd = stripped.partition(" ")
    if pid == str(os.getpid()):
        continue
    if watch_script in cmd:
        continue
    if ("run_parallel_fix_v2.py" in cmd or "run_v2.py" in cmd) and output_root in cmd:
        hits.append(pid)

if hits:
    print(" ".join(hits))
    sys.exit(0)
sys.exit(1)
PY
}

audit_roots() {
  python "$WATCH_AUDIT" audit-roots \
    --labels "${DEFINITIONS[@]}" \
    --output-roots "${OUTPUT_ROOTS[@]}"
}

audit_root() {
  python "$WATCH_AUDIT" audit-roots \
    --labels "$(basename "$OUTPUT_ROOT")" \
    --output-roots "$OUTPUT_ROOT"
}

profile_restart_needed() {
  python "$WATCH_AUDIT" profile-restart-needed \
    --labels "${DEFINITIONS[@]}" \
    --output-roots "${OUTPUT_ROOTS[@]}"
}

refresh_existing_plans_from_configs() {
  python - "$PTXBENCH_MULTITURN_ROOT" "${#DEFINITIONS[@]}" "${DEFINITIONS[@]}" "${TEST_PATHS[@]}" "${CONFIGS[@]}" "${OUTPUT_ROOTS[@]}" <<'PY'
import json
import shutil
import sys
import time
from pathlib import Path

MULTITURN_DIR = Path(sys.argv[1])
sys.path.insert(0, str(MULTITURN_DIR))

from launcher_utils import load_config, materialize_run_fields, plan_from_experiments  # noqa: E402
from run_parallel_v2 import expand_config  # noqa: E402

n = int(sys.argv[2])
offset = 3
definitions = sys.argv[offset:offset + n]
offset += n
test_paths = sys.argv[offset:offset + n]
offset += n
configs = sys.argv[offset:offset + n]
offset += n
output_roots = sys.argv[offset:offset + n]

for definition, test_path, config, output_root in zip(definitions, test_paths, configs, output_roots):
    root = Path(output_root).expanduser().resolve()
    if not root.exists():
        print(f"{root}: missing root; runner will create fresh plan.json")
        continue

    config_path = Path(config).expanduser().resolve()
    experiments = expand_config(load_config(config_path))
    experiments = materialize_run_fields(
        experiments,
        [{"definition": definition, "test_path": test_path}],
    )
    plan_data = {"config": str(config_path), "plan": plan_from_experiments(experiments)}
    plan_path = root / "plan.json"

    old_data = None
    if plan_path.exists():
        old_data = json.loads(plan_path.read_text())
    if old_data == plan_data:
        print(f"{plan_path}: already current")
        continue

    if plan_path.exists():
        backup = plan_path.with_name(f"plan.json.{int(time.time())}.bak")
        shutil.copy2(plan_path, backup)
        backup_msg = f" backup={backup}"
    else:
        backup_msg = ""

    tmp_path = plan_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(plan_data, indent=2) + "\n")
    tmp_path.replace(plan_path)
    print(f"{plan_path}: refreshed from {config_path}{backup_msg}")
PY
}

refresh_existing_plan_from_config() {
  python - "$PTXBENCH_MULTITURN_ROOT" "$CONFIG" "$OUTPUT_ROOT" <<'PY'
import json
import shutil
import sys
import time
from pathlib import Path

MULTITURN_DIR = Path(sys.argv[1])
sys.path.insert(0, str(MULTITURN_DIR))

from launcher_utils import load_config, plan_from_experiments  # noqa: E402
from fix_kernels.run_parallel_fix_v2 import expand_config  # noqa: E402

config_path = Path(sys.argv[2]).expanduser().resolve()
root = Path(sys.argv[3]).expanduser().resolve()
if not root.exists():
    print(f"{root}: missing root; runner will create fresh plan.json")
    sys.exit(0)

experiments = expand_config(load_config(config_path))
plan_data = {"config": str(config_path), "plan": plan_from_experiments(experiments)}
plan_path = root / "plan.json"

old_data = None
if plan_path.exists():
    old_data = json.loads(plan_path.read_text())
if old_data == plan_data:
    print(f"{plan_path}: already current")
    sys.exit(0)

if plan_path.exists():
    backup = plan_path.with_name(f"plan.json.{int(time.time())}.bak")
    shutil.copy2(plan_path, backup)
    backup_msg = f" backup={backup}"
else:
    backup_msg = ""

tmp_path = plan_path.with_suffix(".json.tmp")
tmp_path.write_text(json.dumps(plan_data, indent=2) + "\n")
tmp_path.replace(plan_path)
print(f"{plan_path}: refreshed from {config_path}{backup_msg}")
PY
}

profile_health_ok() {
  python - "${SERVICE_URL}/health" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
except (OSError, urllib.error.URLError, json.JSONDecodeError):
    sys.exit(1)

backends = data.get("backends") or data.get("workers") or []
ok = (
    data.get("status") == "ok"
    and isinstance(backends, list)
    and len(backends) > 0
    and all(isinstance(item, dict) and item.get("healthy") is True for item in backends)
    and int(data.get("queue_size") or 0) == 0
)
sys.exit(0 if ok else 1)
PY
}

definition_health_ok() {
  curl -fsS --max-time 10 "${SERVICE_URL}/definitions/${PROFILE_DEFINITION}" >/dev/null
}

local_port_is_listening() {
  python - "$1" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

profile_host_exec() {
  local command="$1"
  case "${PROFILE_LOCAL:-0}" in
    1)
      bash -lc "$command"
      ;;
    0)
      if [[ -z "${PROFILE_HOST:-}" ]]; then
        log "PROFILE_HOST is required when PROFILE_LOCAL=0"
        return 1
      fi
      ssh "$PROFILE_HOST" "$command"
      ;;
    *)
      log "PROFILE_LOCAL must be 0 or 1, got: $PROFILE_LOCAL"
      return 1
      ;;
  esac
}

profile_host_label() {
  if [[ "${PROFILE_LOCAL:-0}" == "1" ]]; then
    printf 'local'
  else
    printf '%s' "${PROFILE_HOST:-unset}"
  fi
}

profile_forward_process() {
  local mapping="127.0.0.1:${PROFILE_TUNNEL_LOCAL_PORT}:localhost:${PROFILE_TUNNEL_REMOTE_PORT}"
  ps -eo pid=,args= | awk -v mapping="$mapping" -v host="$PROFILE_HOST" '
    index($0, "ssh ") && index($0, "-L " mapping) && index($0, host) { print; found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

ensure_profile_forward() {
  if [[ "${PROFILE_LOCAL:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -z "${PROFILE_TUNNEL_LOCAL_PORT:-}" ]]; then
    return 0
  fi

  local existing command attempt
  if existing="$(profile_forward_process)"; then
    log "reusing profile forward: ${existing# }"
    return 0
  fi

  if local_port_is_listening "$PROFILE_TUNNEL_LOCAL_PORT"; then
    log "cannot create profile forward: localhost:${PROFILE_TUNNEL_LOCAL_PORT} is occupied by another process"
    return 1
  fi

  if tmux has-session -t "$PROFILE_TUNNEL_SESSION" 2>/dev/null; then
    log "removing stale profile forward session: $PROFILE_TUNNEL_SESSION"
    tmux kill-session -t "$PROFILE_TUNNEL_SESSION" || return 1
  fi

  printf -v command \
    'exec ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -L 127.0.0.1:%q:localhost:%q %q' \
    "$PROFILE_TUNNEL_LOCAL_PORT" "$PROFILE_TUNNEL_REMOTE_PORT" "$PROFILE_HOST"
  log "creating profile forward localhost:${PROFILE_TUNNEL_LOCAL_PORT} -> ${PROFILE_HOST}:localhost:${PROFILE_TUNNEL_REMOTE_PORT} in tmux session $PROFILE_TUNNEL_SESSION"
  tmux new-session -d -s "$PROFILE_TUNNEL_SESSION" "$command" || return 1

  for ((attempt = 1; attempt <= 10; attempt++)); do
    if local_port_is_listening "$PROFILE_TUNNEL_LOCAL_PORT" && profile_forward_process >/dev/null; then
      log "profile forward ready: localhost:${PROFILE_TUNNEL_LOCAL_PORT} -> ${PROFILE_HOST}:localhost:${PROFILE_TUNNEL_REMOTE_PORT}"
      return 0
    fi
    if ! tmux has-session -t "$PROFILE_TUNNEL_SESSION" 2>/dev/null; then
      log "profile forward session exited before opening localhost:${PROFILE_TUNNEL_LOCAL_PORT}"
      return 1
    fi
    sleep 1
  done

  log "profile forward did not open localhost:${PROFILE_TUNNEL_LOCAL_PORT}"
  return 1
}

wait_for_profile_service() {
  local attempt
  for ((attempt = 1; attempt <= PROFILE_POLL_ATTEMPTS; attempt++)); do
    if profile_health_ok && definition_health_ok; then
      log "profile service ready: ${SERVICE_URL}"
      return 0
    fi
    log "waiting for profile service ${SERVICE_URL} (${attempt}/${PROFILE_POLL_ATTEMPTS})"
    sleep "$PROFILE_POLL_SLEEP"
  done
  log "profile service did not become healthy: ${SERVICE_URL}"
  return 1
}

profile_container_cuda_ok() {
  local min_gpus="${PROFILE_MIN_GPUS:-1}"
  local profile_python="${PROFILE_PYTHON:-python}"
  profile_host_exec \
    "docker exec $PROFILE_CONTAINER nvidia-smi >/dev/null && docker exec $PROFILE_CONTAINER bash -lc '$profile_python -c \"import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= int(\\\"$min_gpus\\\") else 1)\"'"
}

wait_for_profile_container_cuda() {
  local attempt profile_target
  profile_target="$(profile_host_label)"
  for ((attempt = 1; attempt <= PROFILE_CONTAINER_POLL_ATTEMPTS; attempt++)); do
    if profile_container_cuda_ok; then
      log "profile container CUDA ready: ${profile_target}/${PROFILE_CONTAINER}"
      return 0
    fi
    log "waiting for profile container CUDA ${profile_target}/${PROFILE_CONTAINER} (${attempt}/${PROFILE_CONTAINER_POLL_ATTEMPTS})"
    sleep "$PROFILE_CONTAINER_POLL_SLEEP"
  done
  log "profile container CUDA did not become ready: ${profile_target}/${PROFILE_CONTAINER}"
  return 1
}

restart_profile_container_if_needed() {
  local profile_target
  profile_target="$(profile_host_label)"
  if profile_container_cuda_ok; then
    log "profile container CUDA already ready: ${profile_target}/${PROFILE_CONTAINER}"
    return 0
  fi

  log "profile container CUDA unavailable; restarting ${profile_target}/${PROFILE_CONTAINER}"
  profile_host_exec "docker restart $PROFILE_CONTAINER >/dev/null" || return 1
  wait_for_profile_container_cuda
}

cleanup_profile_solution_runners() {
  log "cleaning stale profile solution runners on $(profile_host_label)/${PROFILE_CONTAINER}"
  profile_host_exec \
    "docker exec $PROFILE_CONTAINER bash -lc 'pattern=\"flashinfer_bench[.]agents[.]_solution_runner\"; pids=\$(pgrep -f \"\$pattern\" || true); if [ -z \"\$pids\" ]; then echo \"[watcher] no stale profile solution runners found\" >&2; exit 0; fi; echo \"[watcher] terminating stale profile solution runners: \$pids\" >&2; kill -TERM \$pids 2>/dev/null || true; for _ in \$(seq 1 10); do pids=\$(pgrep -f \"\$pattern\" || true); [ -z \"\$pids\" ] && exit 0; sleep 1; done; pids=\$(pgrep -f \"\$pattern\" || true); [ -z \"\$pids\" ] || kill -KILL \$pids 2>/dev/null || true'"
}

restart_profile_service() {
  local profile_max_gpus="${PROFILE_MAX_GPUS:-}"
  local restart_env=""

  if [[ -n "$profile_max_gpus" ]]; then
    if ! [[ "$profile_max_gpus" =~ ^[1-9][0-9]*$ ]]; then
      log "PROFILE_MAX_GPUS must be a positive integer, got: $profile_max_gpus"
      return 1
    fi
    restart_env+="PROFILE_MAX_GPUS=$profile_max_gpus "
  fi
  if [[ -n "${PROFILE_FIB_DEVICES:-}" ]]; then
    restart_env+="FIB_DEVICES=$PROFILE_FIB_DEVICES "
  fi

  ensure_profile_forward || return 1
  restart_profile_container_if_needed || return 1
  cleanup_profile_solution_runners || return 1
  log "restarting ${PROFILE_CONTAINER} on $(profile_host_label) for ${SERVICE_URL}"
  case "${PROFILE_RESTART_MODE:-container}" in
    container)
      profile_host_exec "docker restart $PROFILE_CONTAINER >/dev/null" || return 1
      ;;
    exec)
      local restart_script="${PROFILE_RESTART_SCRIPT:-/workspace/PTXBench/docker/restart_fibserve.sh}"
      profile_host_exec \
        "docker exec $PROFILE_CONTAINER bash -lc '${restart_env}bash $restart_script'" || return 1
      ;;
    *)
      log "PROFILE_RESTART_MODE must be container or exec, got: ${PROFILE_RESTART_MODE}"
      return 1
      ;;
  esac
  wait_for_profile_service
}

model_health_ok() {
  python - "http://${ACCRL_MODEL_HOST}/v1/models" "$MODEL_NAME" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
model_name = sys.argv[2]
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
except (OSError, urllib.error.URLError, json.JSONDecodeError):
    sys.exit(1)

models = data.get("data") or []
ids = {str(item.get("id")) for item in models if isinstance(item, dict)}
sys.exit(0 if model_name in ids else 1)
PY
}

wait_for_model() {
  local attempt
  for ((attempt = 1; attempt <= MODEL_POLL_ATTEMPTS; attempt++)); do
    if model_health_ok; then
      log "model endpoint ready: ${ACCRL_MODEL_HOST} includes ${MODEL_NAME}"
      return 0
    fi
    log "waiting for model endpoint ${ACCRL_MODEL_HOST} to include ${MODEL_NAME} (${attempt}/${MODEL_POLL_ATTEMPTS})"
    sleep "$MODEL_POLL_SLEEP"
  done
  log "model endpoint did not become ready: ${ACCRL_MODEL_HOST} model=${MODEL_NAME}"
  return 1
}

validate_static_inputs() {
  local item
  [[ ${#DEFINITIONS[@]} -eq ${#TEST_PATHS[@]} ]]
  [[ ${#DEFINITIONS[@]} -eq ${#CONFIGS[@]} ]]
  [[ ${#DEFINITIONS[@]} -eq ${#OUTPUT_ROOTS[@]} ]]
  if [[ "${PROFILE_LOCAL:-0}" != "0" && "${PROFILE_LOCAL:-0}" != "1" ]]; then
    log "PROFILE_LOCAL must be 0 or 1, got: $PROFILE_LOCAL"
    return 1
  fi
  if [[ "${PROFILE_LOCAL:-0}" == "0" && -z "${PROFILE_HOST:-}" ]]; then
    log "PROFILE_HOST is required when PROFILE_LOCAL=0"
    return 1
  fi
  if [[ -n "${PROFILE_MAX_GPUS:-}" ]] && ! [[ "$PROFILE_MAX_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    log "PROFILE_MAX_GPUS must be a positive integer, got: $PROFILE_MAX_GPUS"
    return 1
  fi

  for item in "${CONFIGS[@]}" "${TEST_PATHS[@]}"; do
    test -f "$item"
  done
}

validate_fixit_static_inputs() {
  test -f "$CONFIG"
}

run_eval_once() {
  local gpu_arch="${GPU_ARCH:-hopper}"
  local extra_args=()
  local env_args=(
    LLM_API_TIMEOUT=3600
    MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="${MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT:-1}"
  )
  if [[ -v ACCRL_MODEL_HOST ]]; then
    env_args+=(ACCRL_MODEL_HOST="$ACCRL_MODEL_HOST")
  fi
  if declare -p RUNNER_EXTRA_ARGS >/dev/null 2>&1; then
    extra_args=("${RUNNER_EXTRA_ARGS[@]}")
  fi

  (
    mkdir -p "$PTXBENCH_DATA_ROOT"
    cd "$PTXBENCH_DATA_ROOT"
    env "${env_args[@]}" \
    python "$RUNNER" \
      --definitions "${DEFINITIONS[@]}" \
      --test-paths "${TEST_PATHS[@]}" \
      --configs "${CONFIGS[@]}" \
      --output-roots "${OUTPUT_ROOTS[@]}" \
      --timeout "$TIMEOUT" \
      --without-local-gpu \
      --model "$MODEL_NAME" \
      --gpu-arch "$gpu_arch" \
      --max-parallel "$MAX_PARALLEL" \
      --max-profiles "$MAX_PROFILES" \
      --service-url "$SERVICE_URL" \
      --turn-timeout "$TURN_TIMEOUT" \
      "${extra_args[@]}"
  )
}

run_fixit_once() {
  local gpu_arch="${GPU_ARCH:-hopper}"
  local extra_args=()
  local args=(
    --output-root "$OUTPUT_ROOT"
    --model "$MODEL_NAME"
    --max-parallel "$MAX_PARALLEL"
    --max-profiles "$MAX_PROFILES"
    --timeout "$TIMEOUT"
    --turn-timeout "$TURN_TIMEOUT"
    --service-url "$SERVICE_URL"
    --gpu-arch "$gpu_arch"
  )
  if declare -p RUNNER_EXTRA_ARGS >/dev/null 2>&1; then
    extra_args=("${RUNNER_EXTRA_ARGS[@]}")
  fi

  if [[ -d "$OUTPUT_ROOT" ]]; then
    log "starting fixit runner in resume mode: OUTPUT_ROOT=$OUTPUT_ROOT"
    args=(--resume "${args[@]}")
  else
    log "starting fixit runner in fresh mode: OUTPUT_ROOT=$OUTPUT_ROOT CONFIG=$CONFIG"
    args=(--config "$CONFIG" "${args[@]}")
  fi

  (
    mkdir -p "$PTXBENCH_DATA_ROOT"
    cd "$PTXBENCH_DATA_ROOT"
    LLM_API_TIMEOUT=3600 \
    MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 \
    python "$RUNNER" "${args[@]}" "${extra_args[@]}"
  )
}
