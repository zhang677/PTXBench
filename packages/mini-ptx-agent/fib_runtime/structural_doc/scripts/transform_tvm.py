#!/usr/bin/env python3
"""Transform TVM-style bf16 device-only kernels into tvm-ffi host-wrapped kernels.

Reads each `gemm_vN.cu` from `examples_sm90_bf16/` (TVM-generated device code)
and emits a matching `gemm_vN.cu` in `examples_with_host_sm90_bf16/` containing
the full tvm-ffi host wrapper required by the harness at
`mini_swe_agent_docker/isolated/cuda_gemm_n7168_k5120/test.py`.

Concretely, the transformation:

  * Replaces the TVM preamble (`#include <cstdint>` ... `using TmaDescriptor`)
    with the full tvm-ffi preamble: extra headers, `CUDA_CHECK`/`CU_CHECK`,
    and the `create_tma_2d_descriptor_2B` helper tuned for bf16 (data type
    `BFLOAT16`, L2 promotion `L2_128B`, OOB fill `NONE`).
  * Wraps the device code in `namespace tvm_ffi_gemm_vN_kernel { ... }` and
    drops the TVM `extern "C" { ... }` marker so the kernel lives inside the
    namespace.
  * For v1 (no TMA), rewrites the global-memory B access from the TVM
    layout `B[gkb*N+gnb]` to the definition's actual `B[N,K]` layout
    (`B[gnb*K+gkb]`).
  * For v1-v7 (which emit float C in the TVM source), rewrites the C output
    type to `__nv_bfloat16*` and wraps the accumulator writes with
    `__float2bfloat16(...)`.
  * Appends a per-kernel `run()` host function that sets the CUDA device,
    builds the TMA descriptors required by that kernel, and launches with
    the correct grid, block, shared-memory size, and cluster configuration.

The list of kernels to emit is driven by the files present in
`examples_sm90_bf16/`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_BF16_DIR = ROOT / "examples_sm90_bf16"
DST_BF16_HOST_DIR = ROOT / "examples_with_host_sm90_bf16"


PREAMBLE = """\
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using TmaDescriptor = CUtensorMap;

#define CUDA_CHECK(call) do {                                        \\
    cudaError_t _e = (call);                                         \\
    if (_e != cudaSuccess) {                                         \\
        fprintf(stderr, "CUDA error %s at %s:%d\\n",                 \\
                cudaGetErrorString(_e), __FILE__, __LINE__);         \\
        exit(1);                                                     \\
    }                                                                \\
} while(0)

#define CU_CHECK(call) do {                                          \\
    CUresult _e = (call);                                            \\
    if (_e != CUDA_SUCCESS) {                                        \\
        const char* _s = nullptr;                                    \\
        cuGetErrorString(_e, &_s);                                   \\
        fprintf(stderr, "CUDA driver error %s at %s:%d\\n",          \\
                _s ? _s : "unknown", __FILE__, __LINE__);            \\
        exit(1);                                                     \\
    }                                                                \\
} while(0)

static void create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle,
    CUtensorMapFloatOOBfill oobFill) {
  uint64_t globalDim[2]     = {gmem_inner_dim, gmem_outer_dim};
  uint64_t globalStrides[1] = {gmem_inner_dim * 2};
  uint32_t boxDim[2]        = {smem_inner_dim, smem_outer_dim};
  uint32_t elementStrides[2]= {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress,
      globalDim, globalStrides, boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, oobFill));
}
"""

# Per-kernel host-side run() bodies. These encode the grid/block/smem/cluster
# configuration that each TVM kernel variant expects -- information that is
# not expressible in the device code alone.
RUN_BODIES: dict[str, str] = {
    "gemm_v1": """\
  dim3 grid((64 - 1 + N) / 64, (64 - 1 + M) / 64, 1);
  dim3 block(128, 1, 1);
  gemm_v1_kernel<<<grid, block, 0, stream>>>(a, b, c, M, N, K);
""",
    "gemm_v2": """\
  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  dim3 grid((64 - 1 + N) / 64, (64 - 1 + M) / 64, 1);
  dim3 block(128, 1, 1);
  gemm_v2_kernel<<<grid, block, 0, stream>>>(dA, dB, c, M, N, K);
""",
    "gemm_v3": """\
  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  dim3 grid((64 - 1 + N) / 64, (64 - 1 + M) / 64, 1);
  dim3 block(256, 1, 1);
  CUDA_CHECK(cudaFuncSetAttribute(gemm_v3_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 49200));
  gemm_v3_kernel<<<grid, block, 49200, stream>>>(dA, dB, c, M, N, K);
""",
    "gemm_v4": """\
  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  dim3 grid((128 - 1 + N) / 128, (128 - 1 + M) / 128, 1);
  dim3 block(256, 1, 1);
  CUDA_CHECK(cudaFuncSetAttribute(gemm_v4_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 65568));
  gemm_v4_kernel<<<grid, block, 65568, stream>>>(dA, dB, c, M, N, K);
""",
    "gemm_v5": """\
  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  int num_n_tiles  = (N + 128 - 1) / 128;
  int bm_cluster   = 128 * 2;
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  dim3 grid(num_n_tiles * 2, num_m_clusters, 1);
  dim3 block(256, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute(gemm_v5_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 65568));

  cudaLaunchConfig_t config = {};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 65568;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_v5_kernel, dA, dB, c, M, N, K));
""",
    "gemm_v6": """\
  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 64,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 256,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  int bm_cluster    = 128 * 2;
  int num_n_tiles   = (N + 256 - 1) / 256;
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  int num_tiles     = num_n_tiles * num_m_clusters;
  int num_clusters  = num_tiles;
  int num_ctas      = num_clusters * 2;
  dim3 grid(num_ctas, 1, 1);
  dim3 block(384, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute(gemm_v6_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98336));

  cudaLaunchConfig_t config = {};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 98336;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_v6_kernel,
      dA, dB, c, M, N, K,
      static_cast<int32_t>(num_tiles), static_cast<int32_t>(num_clusters)));
""",
    "gemm_v7": """\
  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  int bm_cluster    = 128 * 2;
  int num_n_tiles   = (N + 256 - 1) / 256;
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  int num_tiles     = num_n_tiles * num_m_clusters;
  int num_clusters  = num_tiles;
  int num_ctas      = num_clusters * 2;
  dim3 grid(num_ctas, 1, 1);
  dim3 block(384, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute(gemm_v7_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98336));

  cudaLaunchConfig_t config = {};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 98336;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_v7_kernel,
      dA, dB, c, M, N, K,
      static_cast<int32_t>(num_tiles), static_cast<int32_t>(num_clusters)));
""",
    "gemm_v8": """\
  CUtensorMap dA, dB, dC;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dC, c, N, M, 256, 64,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  int bm_cluster    = 128 * 2;
  int num_n_tiles   = (N + 256 - 1) / 256;
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  int num_tiles     = num_n_tiles * num_m_clusters;
  int num_clusters  = num_tiles;
  int num_ctas      = num_clusters * 2;
  dim3 grid(num_ctas, 1, 1);
  dim3 block(384, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute(gemm_v8_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 213040));

  cudaLaunchConfig_t config = {};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 213040;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_v8_kernel,
      dA, dB, dC, c, M, N, K,
      static_cast<int32_t>(num_tiles), static_cast<int32_t>(num_clusters)));
""",
    "gemm_v9": """\
  CUtensorMap dA, dB, dC;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dC, c, N, M, 256, 64,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  int bm_cluster    = 128 * 2;
  int num_n_tiles   = (N + 256 - 1) / 256;
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  int num_tiles     = num_n_tiles * num_m_clusters;
  int num_clusters  = num_tiles;
  int num_ctas      = num_clusters * 2;
  dim3 grid(num_ctas, 1, 1);
  dim3 block(384, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute(gemm_v9_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 213040));

  cudaLaunchConfig_t config = {};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 213040;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_v9_kernel,
      dA, dB, dC, c, M, N, K,
      static_cast<int32_t>(num_tiles), static_cast<int32_t>(num_clusters)));
""",
    "gemm_v10": """\
  CUtensorMap dA, dB, dC;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_2d_descriptor_2B(&dC, c, N, M, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  int bm_cluster    = 128 * 2;
  int num_n_tiles   = (N + 256 - 1) / 256;
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  int num_tiles     = num_n_tiles * num_m_clusters;
  int num_clusters  = num_tiles;
  int num_ctas      = num_clusters * 2;
  dim3 grid(num_ctas, 1, 1);
  dim3 block(384, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute(gemm_v10_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 213040));

  cudaLaunchConfig_t config = {};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 213040;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_v10_kernel,
      dA, dB, dC, c, M, N, K,
      static_cast<int32_t>(num_tiles), static_cast<int32_t>(num_clusters)));
""",
}


PREAMBLE_END_RE = re.compile(r"using\s+TmaDescriptor\s*=\s*CUtensorMap\s*;")

EXTERN_C_OPEN_RE = re.compile(r"^\s*extern\s+\"C\"\s*\{\s*$", re.MULTILINE)


def extract_kernel_body(source: str) -> tuple[str, str]:
    """Return (device_body, stem).

    device_body is the post-preamble portion with `extern "C" { ... }` stripped
    so it can be dropped inside a namespace.
    """
    m = PREAMBLE_END_RE.search(source)
    if not m:
        raise ValueError("source missing `using TmaDescriptor = CUtensorMap;`")
    body = source[m.end():]

    # Drop the `extern "C" {` opener and the matching closing `}`.
    open_match = EXTERN_C_OPEN_RE.search(body)
    if open_match is None:
        return body.strip() + "\n", ""
    start = open_match.start()
    after_open = body[open_match.end():]

    # The closing `}` is the last `}` in the file (matches `extern "C" {`).
    last_brace = after_open.rfind("}")
    if last_brace == -1:
        raise ValueError('unmatched `extern "C" {` in source')
    device_body = body[:start] + after_open[:last_brace]
    return device_body.strip() + "\n", ""


def derive_stem(source_filename: str) -> str:
    stem = Path(source_filename).stem
    if not stem.startswith("gemm_v"):
        raise ValueError(f"unexpected filename: {source_filename}")
    return stem


# Store-function replacements needed for v1-v7, whose TVM source emits float C.
# For each variant we replace both the signature of the device store helper
# and its in-body assignments so that writes go into bf16 C via
# __float2bfloat16.
_STORE_SIG_FLOAT_C_RES = [
    # (old, new) pairs. Multiple patterns because different store helpers
    # use different signatures (single-acc vs 4-acc).
    (
        re.compile(
            r"(store_acc64_global_f32_fn\s*\(\s*)float\*\s*C,",
            re.MULTILINE,
        ),
        r"\1__nv_bfloat16* C,",
    ),
    (
        re.compile(
            r"(store_acc_global_n256_fn\s*\(\s*)float\*\s*C,",
            re.MULTILINE,
        ),
        r"\1__nv_bfloat16* C,",
    ),
    (
        re.compile(
            r"(store_4acc_f32_fn\s*\(\s*)float\*\s*C,",
            re.MULTILINE,
        ),
        r"\1__nv_bfloat16* C,",
    ),
]


def _wrap_c_writes(body: str) -> str:
    """Wrap `C[...] = ac[r];` writes and friends with `__float2bfloat16(...)`.

    Only rewrites plain-register writes directly into global C; does not touch
    writes that already go through shared-memory stagers (those already use
    the correct type).
    """
    return re.sub(
        r"(C\[[^\]]+\]\s*=\s*)([a-zA-Z_][a-zA-Z0-9_]*\[r\])(\s*;)",
        r"\1__float2bfloat16(\2)\3",
        body,
    )


def _float_c_to_bf16(body: str) -> str:
    for pattern, repl in _STORE_SIG_FLOAT_C_RES:
        body = pattern.sub(repl, body)
    body = _wrap_c_writes(body)
    # Kernel signature: `float* C,` -> `__nv_bfloat16* C,`
    body = re.sub(
        r"(_kernel\s*\([^)]*?)\bfloat\*\s*C,",
        r"\1__nv_bfloat16* C,",
        body,
    )
    return body


def _fix_v1_b_layout(body: str) -> str:
    """The TVM v1 kernel indexes B as [K,N] (stride-1 along N) but the
    `gemm_n7168_k5120` definition stores B as [N,K] (stride-1 along K).
    Transpose the access to match the on-device layout."""
    return body.replace("B[((gkb * N) + gnb)]", "B[((gnb * K) + gkb)]")


def _C_is_float(body: str) -> bool:
    return re.search(r"_kernel\s*\([^)]*\bfloat\*\s*C,", body) is not None


def build_host_output(source: str, stem: str) -> str:
    device_body, _ = extract_kernel_body(source)

    if stem == "gemm_v1":
        device_body = _fix_v1_b_layout(device_body)

    if _C_is_float(device_body):
        device_body = _float_c_to_bf16(device_body)

    try:
        run_body = RUN_BODIES[stem]
    except KeyError as exc:
        raise ValueError(f"no run() body configured for {stem}") from exc

    namespace = f"tvm_ffi_{stem}_kernel"

    return (
        PREAMBLE
        + "\n"
        + f"namespace {namespace} {{\n\n"
        + device_body.rstrip()
        + "\n\n\n"
        + "void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {\n"
        + "  CUDA_CHECK(cudaSetDevice(A.device().device_id));\n"
        + "  int32_t M = static_cast<int32_t>(A.size(0));\n"
        + "  int32_t K = static_cast<int32_t>(A.size(1));\n"
        + "  int32_t N = static_cast<int32_t>(B.size(0));\n"
        + "  __nv_bfloat16* a = static_cast<__nv_bfloat16*>(A.data_ptr());\n"
        + "  __nv_bfloat16* b = static_cast<__nv_bfloat16*>(B.data_ptr());\n"
        + "  __nv_bfloat16* c = static_cast<__nv_bfloat16*>(C.data_ptr());\n"
        + "  cudaStream_t stream = static_cast<cudaStream_t>(\n"
        + "      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));\n\n"
        + run_body
        + "  CUDA_CHECK(cudaStreamSynchronize(stream));\n"
        + "}\n\n"
        + f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, {namespace}::run);\n\n"
        + f"}}  // namespace {namespace}\n"
    )


def transform_file(src_path: Path, dst_path: Path) -> None:
    stem = derive_stem(src_path.name)
    src_text = src_path.read_text()
    output = build_host_output(src_text, stem)
    dst_path.write_text(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=SRC_BF16_DIR,
        help="Directory containing bf16 device-only TVM source kernels.",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=DST_BF16_HOST_DIR,
        help="Destination directory for bf16 host-wrapped kernels.",
    )
    args = parser.parse_args()

    if not args.src_dir.is_dir():
        print(f"source dir not found: {args.src_dir}", file=sys.stderr)
        return 1
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for src_path in sorted(args.src_dir.glob("*.cu")):
        dst_path = args.dst_dir / src_path.name
        try:
            transform_file(src_path, dst_path)
        except Exception as exc:
            print(f"skip {src_path.name}: {exc}", file=sys.stderr)
            continue
        print(f"wrote {dst_path}")
        written += 1

    if written == 0:
        print("no files written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
