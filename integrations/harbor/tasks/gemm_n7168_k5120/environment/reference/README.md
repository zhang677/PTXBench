# PTXBench CUDA ABI

Include the CUDA and TVM-FFI headers supplied by the environment:

```cpp
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
```

Define a single function named `run` with this signature:

```cpp
void run(
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView C);
```

Access tensor storage with `A.data_ptr()`, `B.data_ptr()`, and `C.data_ptr()`.
Select `A.device().device_id`, obtain the caller's CUDA stream with:

```cpp
cudaSetDevice(A.device().device_id);
cudaStream_t stream = static_cast<cudaStream_t>(
    TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
```

Export the function after its definition with:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
```

Do not allocate or return a new output tensor; write the result into `C`.
