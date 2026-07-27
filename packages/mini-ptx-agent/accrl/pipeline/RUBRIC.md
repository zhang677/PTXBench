1. The problem should be challenging enough:
    - It should contain both matrix multiplication and vector operations
    - If it doesn't contain matrix multiplication, then the vector operations should be complex enough
    - The performance of PyTorch references should be 1 ms ~ 15 ms on every workload.
2. Check if the mathematical operation already exists in ~/flashinfer-trace/definitions or ~/accrl-training-data/definitions. If so, skip this kernel because we are collecting new kernels that are not already in our dataset.
3. The baseline solution if exists must be correct.