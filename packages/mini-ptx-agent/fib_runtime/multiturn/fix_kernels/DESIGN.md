Infra failure:
1. Treat returncode 137 after Running memcheck 1/2... as infra failure.
The runner should classify this pattern separately from model/kernel failure:
```
<output>
test.py failed (returncode 137):
``` appears in the profiling output
2. Treat 
```
INFRA_TIMEOUT
```
also as infra failure.

Those mean the client was killed while blocked on the service.

2. Add a health check inside common.py.
Before submitting each new future, check /definitions/... or /health. If it fails twice, stop scheduling new experiments, warning infra_failed and sys.exit(1)

If any single experiment spots infra failure, warning infra_failed and sys.exit(1)


3. Add --resume after infra failure to run_parallel_fix_v2.py.
Resume handling that previously required separate scan, rerun, and merge
helpers is integrated into `run_parallel_fix_v2.py`. Check all trajectories,
remove turns starting from the first infrastructure failure, and remove a
trajectory if only one turn remains. After cleaning, ask for approval and then
restart from the resumed turn.

Update test.py
1. Add INFRA_TIMEOUT to /home/ubuntu/AccRL/fib_runtime/multiturn/template_compile_measure_cuda.txtand run `create_test.py` in /home/ubuntu/AccRL/fib_runtime/multiturn/gemm-problems, /home/ubuntu/AccRL/fib_runtime/multiturn/mha-with-lse-problems, and /home/ubuntu/AccRL/fib_runtime/multiturn/mha-bwd-problems

Other processes are also using run_v2.py and common.py; so please try those seperately. Maybe update those files after this workflow is proved.
