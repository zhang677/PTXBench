
TVM_FFI_DIR=/home/ubuntu/miniconda3/envs/acc/lib/python3.12/site-packages/tvm_ffi
GENCODE="${NVCC_GENCODE:-arch=compute_90a,code=sm_90a}"
nvcc -shared -O3 -gencode "$GENCODE" kernel.cu     -lineinfo --ptxas-options=-v      -Xcompiler -fPIC,-fvisibility=hidden -lcuda      -I${TVM_FFI_DIR}/include -std=c++17      -L${TVM_FFI_DIR}/lib -ltvm_ffi      -o kernel.so
