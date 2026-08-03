FROM ptxbench/harbor:dev

COPY task.json /opt/ptxbench/task.json
COPY reference /opt/ptxbench/reference
COPY kernel.cu /workspace/kernel.cu

RUN chmod 0444 /opt/ptxbench/task.json \
    && chmod -R a-w /opt/ptxbench/reference \
    && chmod 0666 /workspace/kernel.cu

WORKDIR /workspace
