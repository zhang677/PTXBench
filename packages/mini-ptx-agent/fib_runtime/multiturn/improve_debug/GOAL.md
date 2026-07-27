Use error kernels with runtime error and kernel execution timeout from /home/ubuntu/AccRL-exps/eval_runs/2026-0624-0939-mha-bwd-d128/kernels. Select 20 representative error kernels

Change the flashinfer-bench locally

Mount the local flashinfer-bench to the flashinfer-bench-private of fib-profile; Use one GPU for fib-profile container; just launch one direct fib serve, no need for dispatcher

Give me a script that takes in an error kernel and output the returned message

Implement a debug endpoint based on https://vllm.ai/blog/2025-12-03-improved-cuda-debugging. The goal of this endpoint is to return the exact source lines and very precise error message. This should work with the current process.

Goal: iterate until I am statisfied by the returned message 