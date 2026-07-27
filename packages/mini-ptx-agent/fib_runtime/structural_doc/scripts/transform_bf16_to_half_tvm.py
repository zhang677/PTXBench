#!/usr/bin/env python3
"""Transform SM90 BF16 GEMM kernels to FP16 (half) with TVM FFI wrapping.

Reads .cu files from input dir, converts bf16->half, changes output from float
to half, adds TMA descriptor creation via create_tma_2d_descriptor_2B, and
wraps with TVM FFI for each kernel version (v1-v10).

The generated kernels compute C = A @ B^T where:
  A: (M, K) row-major half
  B: (N, K) row-major half  (NOT transposed)
  C: (M, N) row-major half  (destination-passing)

Usage:
    python transform_bf16_to_half_tvm.py [-i INPUT_DIR] [-o OUTPUT_DIR]
"""
import argparse
import os
import re
import glob
from dataclasses import dataclass, field
from typing import Optional


# ───────────────────────── per-version config ─────────────────────────

@dataclass
class KernelConfig:
    bm: int
    bn: int
    bk: int
    threads: int
    cluster_size: int
    has_tma_load: bool
    has_tma_store: bool
    grid_type: str            # '2d_simple', '2d_cluster', '1d_persistent'
    dynamic_smem: bool        # needs cudaFuncSetAttribute
    output_is_float: bool     # v1-v7: float* C -> half* C needed
    tma_a_smem_outer: int = 0
    tma_b_smem_outer: int = 0
    tma_c_smem_inner: int = 0
    tma_c_smem_outer: int = 0
    tma_c_swizzle: str = "CU_TENSOR_MAP_SWIZZLE_NONE"
    extra_kernel_args: str = ""  # e.g. "num_tiles, num_clusters"


CONFIGS = {
    1: KernelConfig(
        bm=64, bn=64, bk=64, threads=128, cluster_size=1,
        has_tma_load=False, has_tma_store=False,
        grid_type='2d_simple', dynamic_smem=False, output_is_float=True,
    ),
    2: KernelConfig(
        bm=64, bn=64, bk=64, threads=128, cluster_size=1,
        has_tma_load=True, has_tma_store=False,
        grid_type='2d_simple', dynamic_smem=False, output_is_float=True,
        tma_a_smem_outer=64, tma_b_smem_outer=64,
    ),
    3: KernelConfig(
        bm=64, bn=64, bk=64, threads=256, cluster_size=1,
        has_tma_load=True, has_tma_store=False,
        grid_type='2d_simple', dynamic_smem=True, output_is_float=True,
        tma_a_smem_outer=64, tma_b_smem_outer=64,
    ),
    4: KernelConfig(
        bm=128, bn=128, bk=64, threads=256, cluster_size=1,
        has_tma_load=True, has_tma_store=False,
        grid_type='2d_simple', dynamic_smem=True, output_is_float=True,
        tma_a_smem_outer=64, tma_b_smem_outer=64,
    ),
    5: KernelConfig(
        bm=128, bn=128, bk=64, threads=256, cluster_size=2,
        has_tma_load=True, has_tma_store=False,
        grid_type='2d_cluster', dynamic_smem=True, output_is_float=True,
        tma_a_smem_outer=64, tma_b_smem_outer=64,
    ),
    6: KernelConfig(
        bm=128, bn=256, bk=64, threads=384, cluster_size=2,
        has_tma_load=True, has_tma_store=False,
        grid_type='1d_persistent', dynamic_smem=True, output_is_float=True,
        tma_a_smem_outer=64, tma_b_smem_outer=256,
        extra_kernel_args="num_tiles, num_clusters",
    ),
    7: KernelConfig(
        bm=128, bn=256, bk=64, threads=384, cluster_size=2,
        has_tma_load=True, has_tma_store=False,
        grid_type='1d_persistent', dynamic_smem=True, output_is_float=True,
        tma_a_smem_outer=128, tma_b_smem_outer=128,
        extra_kernel_args="num_tiles, num_clusters",
    ),
    8: KernelConfig(
        bm=128, bn=256, bk=64, threads=384, cluster_size=2,
        has_tma_load=True, has_tma_store=True,
        grid_type='1d_persistent', dynamic_smem=True, output_is_float=False,
        tma_a_smem_outer=128, tma_b_smem_outer=128,
        tma_c_smem_inner=256, tma_c_smem_outer=64,
        tma_c_swizzle="CU_TENSOR_MAP_SWIZZLE_NONE",
        extra_kernel_args="num_tiles, num_clusters",
    ),
    9: KernelConfig(
        bm=128, bn=256, bk=64, threads=384, cluster_size=2,
        has_tma_load=True, has_tma_store=True,
        grid_type='1d_persistent', dynamic_smem=True, output_is_float=False,
        tma_a_smem_outer=128, tma_b_smem_outer=128,
        tma_c_smem_inner=256, tma_c_smem_outer=64,
        tma_c_swizzle="CU_TENSOR_MAP_SWIZZLE_NONE",
        extra_kernel_args="num_tiles, num_clusters",
    ),
    10: KernelConfig(
        bm=128, bn=256, bk=64, threads=384, cluster_size=2,
        has_tma_load=True, has_tma_store=True,
        grid_type='1d_persistent', dynamic_smem=True, output_is_float=False,
        tma_a_smem_outer=128, tma_b_smem_outer=128,
        tma_c_smem_inner=64, tma_c_smem_outer=128,
        tma_c_swizzle="CU_TENSOR_MAP_SWIZZLE_128B",
        extra_kernel_args="num_tiles, num_clusters",
    ),
}


# ───────────────────────── bf16 → half substitution ──────────────────

SUBSTITUTIONS = [
    # WGMMA PTX: bf16.bf16 → f16.f16
    (r"(wgmma\.mma_async\.sync\.aligned\.m\d+n\d+k\d+\.f32\.)bf16\.bf16", r"\1f16.f16"),
    # C++ types
    ("__nv_bfloat162", "__half2"),
    ("__nv_bfloat16", "half"),
    # Conversion intrinsics
    ("__floats2bfloat162_rn", "__floats2half2_rn"),
    ("__float2bfloat16", "__float2half"),
    # Remove cuda_bf16.h include (cuda_fp16.h already present)
    (r'#include <cuda_bf16\.h>\n', ""),
    # Cosmetic: remaining bf16 in function names → f16
    ("bf16", "f16"),
]


def apply_type_substitutions(code: str) -> str:
    for pat, repl in SUBSTITUTIONS:
        code = re.sub(pat, repl, code)
    return code


# ───────────────── float output → half output (v1-v7) ────────────────

def transform_store_to_half(code: str) -> str:
    """Change float* C output to half* C and add __float2half() around stores."""
    # 1. Store function signatures: float* C → half* C
    # Patterns: store_acc64_global_f32_fn(float* C, ...) etc.
    code = re.sub(
        r"(store_\w+_fn\s*\(\s*\n?\s*)float\*\s+C\b",
        r"\1half* C",
        code,
    )
    # Also direct: "float* C, float* a00" pattern in store_4acc_f32_fn
    code = re.sub(
        r"(store_4acc_f32_fn\s*\(\s*\n?\s*)float\*\s+C\b",
        r"\1half* C",
        code,
    )

    # 2. Kernel signature: float* C → half* C
    code = re.sub(
        r"(void\s+gemm_v\d+_kernel\s*\([^)]*?)float\*\s+C\b",
        r"\1half* C",
        code,
    )

    # 3. Store operations: C[expr] = value → C[expr] = __float2half(value)
    # Pattern: C[(expr)] = identifier[index];
    code = re.sub(
        r"C\[([^\]]+)\]\s*=\s*(\w+\[\w+\])\s*;",
        r"C[\1] = __float2half(\2);",
        code,
    )

    return code


# ────────────────── v1: fix B access (K×N → N×K) ─────────────────────

def fix_v1_b_access(code: str) -> str:
    """v1 accesses B[k*N+n] assuming K×N layout. Fix to B[n*K+k] for N×K."""
    # Original: B[((gkb * N) + gnb)]  → B[((gnb * K) + gkb)]
    code = code.replace(
        "B[((gkb * N) + gnb)]",
        "B[((gnb * K) + gkb)]",
    )
    return code


# ───────────────── strip header / extern "C" ─────────────────────────

def strip_generated_header(code: str) -> str:
    lines = code.split("\n")
    out = []
    skipping_header = True
    for line in lines:
        if skipping_header:
            if line.startswith("__device__") or line.startswith("__global__"):
                skipping_header = False
                out.append(line)
            continue
        out.append(line)
    # Remove 'extern "C" {' line
    result = []
    extern_c_removed = False
    for line in out:
        if not extern_c_removed and line.strip() == 'extern "C" {':
            extern_c_removed = True
            continue
        result.append(line)
    # Remove the matching closing '}'
    if extern_c_removed:
        for i in range(len(result) - 1, -1, -1):
            if result[i].strip() == "}":
                result.pop(i)
                break
    return "\n".join(result)


# ───────────────── extract metadata from code ─────────────────────────

def extract_kernel_name(code: str) -> str:
    m = re.search(r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)\s*\(", code)
    return m.group(1) if m else "unknown_kernel"


def extract_shared_mem(code: str) -> int:
    offsets = re.findall(
        r"__INTERNAL_DYN_SHMEM__\s*\+\s*(\d+)\);\s*/\*size\s*=\s*(\d+)\s*bytes\*/",
        code,
    )
    if not offsets:
        return 0
    return max(int(off) + int(sz) for off, sz in offsets)


# ─────────────────────── TVM FFI file template ───────────────────────

FILE_HEADER = """\
#include <cuda_fp16.h>
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
      d, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, globalAddress,
      globalDim, globalStrides, boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, oobFill));
}

"""


# ───────────────────── run() function generator ──────────────────────

def gen_run_v1(kname: str, cfg: KernelConfig) -> str:
    return f"""\
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {{
  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(0));
  half* a = static_cast<half*>(A.data_ptr());
  half* b = static_cast<half*>(B.data_ptr());
  half* c = static_cast<half*>(C.data_ptr());
  dim3 grid(({cfg.bn} - 1 + N) / {cfg.bn}, ({cfg.bm} - 1 + M) / {cfg.bm}, 1);
  dim3 block({cfg.threads}, 1, 1);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  {kname}<<<grid, block, 0, stream>>>(a, b, c, M, N, K);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}}
"""


def gen_run_tma_no_cluster(kname: str, cfg: KernelConfig, smem: int) -> str:
    smem_attr = ""
    if cfg.dynamic_smem:
        smem_attr = f"""\
  CUDA_CHECK(cudaFuncSetAttribute({kname},
      cudaFuncAttributeMaxDynamicSharedMemorySize, {smem}));
"""
    return f"""\
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {{
  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(0));
  half* a = static_cast<half*>(A.data_ptr());
  half* b = static_cast<half*>(B.data_ptr());
  half* c = static_cast<half*>(C.data_ptr());

  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, {cfg.tma_a_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, {cfg.tma_b_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

  dim3 grid(({cfg.bn} - 1 + N) / {cfg.bn}, ({cfg.bm} - 1 + M) / {cfg.bm}, 1);
  dim3 block({cfg.threads}, 1, 1);
{smem_attr}\
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  {kname}<<<grid, block, {smem}, stream>>>(dA, dB, c, M, N, K);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}}
"""


def gen_run_tma_2d_cluster(kname: str, cfg: KernelConfig, smem: int) -> str:
    """v5: 2D grid with cluster."""
    return f"""\
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {{
  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(0));
  half* a = static_cast<half*>(A.data_ptr());
  half* b = static_cast<half*>(B.data_ptr());
  half* c = static_cast<half*>(C.data_ptr());

  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, {cfg.tma_a_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, {cfg.tma_b_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

  int num_n_tiles  = (N + {cfg.bn} - 1) / {cfg.bn};
  int bm_cluster   = {cfg.bm} * {cfg.cluster_size};
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  dim3 grid(num_n_tiles * {cfg.cluster_size}, num_m_clusters, 1);
  dim3 block({cfg.threads}, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute({kname},
      cudaFuncAttributeMaxDynamicSharedMemorySize, {smem}));
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  cudaLaunchConfig_t config = {{}};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = {smem};
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = {cfg.cluster_size};
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, {kname}, dA, dB, c, M, N, K));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}}
"""


def gen_run_persistent_no_tma_store(kname: str, cfg: KernelConfig, smem: int) -> str:
    """v6, v7: 1D persistent grid, cluster, float/half* C output (register store)."""
    return f"""\
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {{
  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(0));
  half* a = static_cast<half*>(A.data_ptr());
  half* b = static_cast<half*>(B.data_ptr());
  half* c = static_cast<half*>(C.data_ptr());

  CUtensorMap dA, dB;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, {cfg.tma_a_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, {cfg.tma_b_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);

  int bm_cluster    = {cfg.bm} * {cfg.cluster_size};
  int num_n_tiles   = (N + {cfg.bn} - 1) / {cfg.bn};
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  int num_tiles     = num_n_tiles * num_m_clusters;
  int num_clusters  = num_tiles;
  int num_ctas      = num_clusters * {cfg.cluster_size};
  dim3 grid(num_ctas, 1, 1);
  dim3 block({cfg.threads}, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute({kname},
      cudaFuncAttributeMaxDynamicSharedMemorySize, {smem}));
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  cudaLaunchConfig_t config = {{}};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = {smem};
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = {cfg.cluster_size};
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, {kname},
      dA, dB, c, M, N, K,
      static_cast<int32_t>(num_tiles), static_cast<int32_t>(num_clusters)));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}}
"""


def gen_run_persistent_tma_store(kname: str, cfg: KernelConfig, smem: int) -> str:
    """v8-v10: 1D persistent grid, cluster, TMA store for C."""
    return f"""\
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {{
  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(0));
  half* a = static_cast<half*>(A.data_ptr());
  half* b = static_cast<half*>(B.data_ptr());
  half* c = static_cast<half*>(C.data_ptr());

  CUtensorMap dA, dB, dC;
  create_tma_2d_descriptor_2B(&dA, a, K, M, 64, {cfg.tma_a_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  create_tma_2d_descriptor_2B(&dB, b, K, N, 64, {cfg.tma_b_smem_outer},
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
  create_tma_2d_descriptor_2B(&dC, c, N, M, {cfg.tma_c_smem_inner}, {cfg.tma_c_smem_outer},
      {cfg.tma_c_swizzle}, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  int bm_cluster    = {cfg.bm} * {cfg.cluster_size};
  int num_n_tiles   = (N + {cfg.bn} - 1) / {cfg.bn};
  int num_m_clusters = (M + bm_cluster - 1) / bm_cluster;
  int num_tiles     = num_n_tiles * num_m_clusters;
  int num_clusters  = num_tiles;
  int num_ctas      = num_clusters * {cfg.cluster_size};
  dim3 grid(num_ctas, 1, 1);
  dim3 block({cfg.threads}, 1, 1);

  CUDA_CHECK(cudaFuncSetAttribute({kname},
      cudaFuncAttributeMaxDynamicSharedMemorySize, {smem}));
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  cudaLaunchConfig_t config = {{}};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = {smem};
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = {cfg.cluster_size};
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs    = attrs;
  config.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&config, {kname},
      dA, dB, dC, c, M, N, K,
      static_cast<int32_t>(num_tiles), static_cast<int32_t>(num_clusters)));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}}
"""


def generate_run(kname: str, version: int, cfg: KernelConfig, smem: int) -> str:
    if version == 1:
        return gen_run_v1(kname, cfg)
    elif cfg.grid_type == '2d_simple' and cfg.cluster_size <= 1:
        return gen_run_tma_no_cluster(kname, cfg, smem)
    elif cfg.grid_type == '2d_cluster':
        return gen_run_tma_2d_cluster(kname, cfg, smem)
    elif cfg.has_tma_store:
        return gen_run_persistent_tma_store(kname, cfg, smem)
    else:
        return gen_run_persistent_no_tma_store(kname, cfg, smem)


# ───────────────────── main transform pipeline ───────────────────────

def get_version(filename: str) -> int:
    m = re.search(r"gemm_v(\d+)", filename)
    return int(m.group(1)) if m else 0


def transform_file(input_path: str) -> str:
    version = get_version(os.path.basename(input_path))
    cfg = CONFIGS.get(version)
    if cfg is None:
        raise ValueError(f"No config for version {version}")

    with open(input_path) as f:
        code = f.read()

    # 1. bf16 → half
    code = apply_type_substitutions(code)

    # 2. float output → half output (v1-v7)
    if cfg.output_is_float:
        code = transform_store_to_half(code)

    # 3. v1: fix B access pattern (K×N → N×K)
    if version == 1:
        code = fix_v1_b_access(code)

    # 4. Extract metadata
    kname = extract_kernel_name(code)
    smem = extract_shared_mem(code)

    # 5. Strip header & extern "C"
    kernel_body = strip_generated_header(code)

    # 6. Generate run() function
    run_fn = generate_run(kname, version, cfg, smem)

    # 7. Assemble
    ns = f"tvm_ffi_{kname}"
    result = (
        FILE_HEADER
        + f"namespace {ns} {{\n\n"
        + kernel_body + "\n\n"
        + run_fn + "\n"
        + f"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, {ns}::run);\n\n"
        + f"}}  // namespace {ns}\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Transform SM90 BF16 GEMM kernels to FP16 half + TVM FFI")
    parser.add_argument(
        "-i", "--input-dir",
        default=os.path.join(os.path.dirname(__file__), "examples_sm90_bf16"),
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "examples_sm90_fp16"),
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "gemm_v*.cu")))
    if not files:
        print(f"No gemm_v*.cu files in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for path in files:
        name = os.path.basename(path)
        print(f"Transforming {name}...", end=" ")
        try:
            result = transform_file(path)
            out = os.path.join(args.output_dir, name)
            with open(out, "w") as f:
                f.write(result)
            print(f"OK -> {out}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
