1. Use Helion's attention kernel as reference /home/ubuntu/helion/examples/attention.py I just need bf16 like the GEMM problems in /home/ubuntu/AccRL/fib_runtime/multiturn/2026-0413-1611
2. Use workloads in Triton bench /home/ubuntu/helion/benchmarks/run.py
3. Prepare json for definition and workloads in this folder
4. Verify reference across workloads
5. Ask me to inspect
6. After my permission, copy to /home/ubuntu/accrl-training and open a PR + upload to hf