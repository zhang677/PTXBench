#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ptxbench_gemm {

void run(
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView C) {
  cudaSetDevice(A.device().device_id);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  const size_t output_bytes =
      static_cast<size_t>(C.size(0)) * static_cast<size_t>(C.size(1)) *
      sizeof(__nv_bfloat16);
  // This deliberately incomplete starter is safe and fast to evaluate, but it
  // is numerically incorrect. Replace it with C = A @ B.T.
  cudaMemsetAsync(C.data_ptr(), 0, output_bytes, stream);
}

}  // namespace ptxbench_gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, ptxbench_gemm::run);
