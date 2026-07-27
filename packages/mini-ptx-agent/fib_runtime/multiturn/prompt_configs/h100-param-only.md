Below is the hardware parameter of NVIDIA H100
| Component | Value |
|-----------|---------------|
| SM Count | 132 |
| Peak FP8 | ~1978 TFLOPS |
| Peak BF16 | ~989 TFLOPS |
| Cluster Size | Up to 16 SMs |
| Max Registers / CTA | 65536 |
| Max Registers / Thread | 255 |
| Max CTA size (# of threads) | 1024 |

| Level | Capacity | Latency | Bandwidth |
|-------|----|------|---------|
| Global (HBM) | 80 GB | ~500 cycles | 3.35 TB/s |
| L2 Cache | 50 MB | ~100 cycles | ~12TB/s |
| Shared Memory | 256 KB / SM (228 KB usable with 28KB L1 Cache) | ~30 cycles | 128 B/cycle/SM |
| Registers | 256 KB / SM  | 1 cycle | Unlimited (matched with compute) |

