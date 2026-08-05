FROM ptxbench/harbor:dev

COPY task.json /opt/ptxbench/task.json
COPY reference /opt/ptxbench/reference

RUN chmod 0444 /opt/ptxbench/task.json \
    && chmod -R a-w /opt/ptxbench/reference

WORKDIR /workspace
