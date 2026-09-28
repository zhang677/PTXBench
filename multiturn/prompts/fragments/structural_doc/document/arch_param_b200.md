Below is the hardware parameter of NVIDIA B200 (SM100)
| Component | Value |
|-----------|---------------|
| SM Count | 148 |
| Peak MMA FP8 | ~4500 TFLOPS |
| Peak MMA BF16 | ~2250 TFLOPS (8192 ops/cycle/SM) |
| Peak MUFU (multifunction unit) | 16 ops/cycle/SM |
| Cluster Size | Portable: 8 CTAs; B200 nonportable: up to 16 CTAs |
| Max Registers / CTA | 65536 32-bit|
| Max Registers / Thread | 255 |
| Max CTA size (# of threads) | 1024 |

| Level | Capacity | Latency | Bandwidth |
|-------|----|------|---------|
| Global (HBM) | 180 GB | ~400 cycles | 8 TB/s |
| L2 Cache | 126 MB | ~100 cycles | ~16TB/s |
| Shared Memory | 0, 8, 16, 32, 64, 100, 132, 164, 196, and 228 KB per SM. A single CTA can address up to 227 KB of shared memory as 1 KB is reserved by CUDA | ~20 cycles | 128 B/cycle/SM |
| Tensor Memory (TMEM) | 256 KB (512 columns × 128 rows x 32-bit cells) / SM |  ~10 cycles | Tensor Core internal |
| Registers | 256 KB / SM  | 1 cycle | Unlimited (matched with compute) |