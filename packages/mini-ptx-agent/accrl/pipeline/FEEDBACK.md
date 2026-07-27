Check these items step by step
1. The mathematical operation of the definition should not exist in ~/flashinfer-trace/definitions or ~/accrl-training-data/definitions. 
If so, reject the definition because we are collecting new kernels that are not already in our dataset.
2. The problem should be challenging enough:
    - It should contain both matrix multiplication and vector operations
    - If it doesn't contain matrix multiplication, then the vector operations should be complex enough
If the definition meets neither of the two criteria, reject the definition
3. The performance of PyTorch references should be 1 ms ~ 15 ms on every workload.
Reject workloads that don't meet this requirement and ask the collector to replace it with a larger workload
4. If the collector also provides a baseline soluion, then the solution must be correct and faster than PyTorch reference.
If not, reject the baseline solution.