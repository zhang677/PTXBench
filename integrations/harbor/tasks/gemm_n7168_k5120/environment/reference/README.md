# PTXBench CUDA ABI

Keep a single exported function named `run` with this signature:

```cpp
void run(
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView C);
```

Use `TVMFFIEnvGetStream` to launch work on the caller's CUDA stream and export
the function with `TVM_FFI_DLL_EXPORT_TYPED_FUNC`. Do not allocate or return a
new output tensor; write the result into `C`.
