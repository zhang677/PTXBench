TVM_FFI_DIR=/usr/local/lib/python3.12/dist-packages/tvm_ffi
rm -f kernel.so
GENCODE="${NVCC_GENCODE:-arch=compute_90a,code=sm_90a}"
nvcc -shared -O3 -gencode "$GENCODE" kernel.cu \
     -Xcompiler -fPIC,-fvisibility=hidden -lcuda \
     -I${TVM_FFI_DIR}/include -std=c++17 \
     -L${TVM_FFI_DIR}/lib -ltvm_ffi \
     -o kernel.so
