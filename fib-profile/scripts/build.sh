#!/usr/bin/env bash
# Build the credential-free CUDA/uv base image for the mounted service.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-fib-profile:latest}"
if [ -n "${REMOTE:-}" ]; then
    echo "[build] streaming Dockerfile to $REMOTE, building $IMAGE" >&2
    remote_build_cmd=(docker build -t "$IMAGE" -)
    printf -v remote_build_cmd_quoted '%q ' "${remote_build_cmd[@]}"
    tar -czh -C "$HERE" Dockerfile | ssh "$REMOTE" "$remote_build_cmd_quoted"
else
    echo "[build] building $IMAGE" >&2
    docker build -t "$IMAGE" -f "$HERE/Dockerfile" "$HERE"
fi
