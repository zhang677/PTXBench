#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__global__ void ScaleShiftKernel(const float* x, float* y, float scale, int32_t shift, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] * scale + static_cast<float>(shift);
  }
}

void run(tvm::ffi::TensorView x, float scale, int32_t shift, tvm::ffi::TensorView y) {
  CUDA_CHECK(cudaSetDevice(x.device().device_id)); // This is necessary to ensure the correct GPU is used, especially in multi-GPU setups.
  int64_t n = x.size(0); // BE CAREFUL: tvm::ffi::TensorView objects don't have .shape() or .empty() methods!!
  const float* x_data = static_cast<const float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

  // flashinfer-bench passes definition inputs with shape=null as Python scalars.
  // TVM FFI converts them to POD parameters here: float for float32 and int32_t for int32.
  ScaleShiftKernel<<<blocks, threads, 0, stream>>>(x_data, y_data, scale, shift, n);
  CUDA_CHECK(cudaGetLastError()); // Add `CUDA_CHECK(cudaGetLastError())` immediately after the kernel launch so the error could be reported at the launch site
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda
