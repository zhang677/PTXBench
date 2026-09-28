# CUDA and system dependencies for the mounted profiling service.
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        vim git python3-pip python3-dev curl tmux openssh-client ca-certificates libdw-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

ENV PROFILE_PORT=10000 \
    WORKDIR=/workspace \
    FIB_DIR=flashinfer-bench-private \
    FIB_DATASET_DIR=/workspace/dataset \
    FIB_DATASET_DIRS=/workspace/dataset \
    UV_CACHE_DIR=/workspace/.cache/uv

WORKDIR /workspace
EXPOSE 10000
