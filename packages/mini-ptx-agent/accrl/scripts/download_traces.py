# import time
# from huggingface_hub import HfApi, hf_hub_download

# api = HfApi()
# files = api.list_repo_files("flashinfer-ai/flashinfer-trace", repo_type="dataset")

# REQS_PER_SEC = 5.0
# SLEEP = 1.0 / REQS_PER_SEC
# LOCAL_DIR = "/home/ubuntu/flashinfer-trace" # "/data/flashinfer-trace"

# for f in files:
#     hf_hub_download(
#         repo_id="flashinfer-ai/flashinfer-trace",
#         repo_type="dataset",
#         filename=f,
#         local_dir=LOCAL_DIR,
#     )
#     time.sleep(SLEEP)

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="flashinfer-ai/flashinfer-trace",
    repo_type="dataset",
    local_dir="/home/ubuntu/flashinfer-trace",
    local_dir_use_symlinks=False,   # real files in local_dir
    resume_download=True,
    max_workers=16,                 # increase if network/disk can handle it
    ignore_patterns=[".git*"],      # optional
)

# Run local: TVM_FFI_CUDA_ARCH_LIST="9.0a" flashinfer-bench serve --local /home/ubuntu/flashinfer-trace --port 10000 --devices cuda:0,cuda:1