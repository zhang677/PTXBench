# Prompts
https://github.com/KnowingNothing/MatmulTutorial/examples/matmul
https://github.com/ByteDance-Seed/Triton-distributed/tree/main/python/little_kernel
https://veitner.bearblog.dev/sbo-and-lbo-explained-visually/
https://forums.developer.nvidia.com/t/cuda-error-invalid-value-when-creating-tensor-maps-with-swizzling/350966
https://docs.nvidia.com/cuda/parallel-thread-execution/
https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#shared-memory-bank-swizzling
https://gist.github.com/Ubospica/052990ed6eb80f726763f3de2de128da

# Model dates (model_release_dates.csv)
If there is no official cutoff date, I use the model's release date 

# FA dates (fa_release_dates.csv)
Use the Tri Dao's blog release dates. For FA4, it was code release date ("Since our initial code release 8 months ago," in https://tridao.me/blog/2026/flash4)

# PTX dates (ptx_release_dates.csv)
Use the release date of specific CUDA version:
1. The first PTX ISA version with official NVIDIA Hopper (SM90) support was PTX ISA 7.8, released with CUDA 11.8 on October 5, 2022
2. Blackwell PTX support was first publicly released with PTX ISA 8.7 in CUDA Toolkit 12.8, released on January 30–31, 2025. NVIDIA explicitly states that CUDA 12.8 is the first toolkit with full Blackwell support