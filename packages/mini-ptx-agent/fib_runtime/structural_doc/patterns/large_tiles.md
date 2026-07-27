# Optimization Pattern: Large Tiles for Better Arithmetic Intensity
BF16
| Tile Size | A Load | B Load | Compute | Intensity |
|-----------|--------|--------|---------|-----------|
| 64x64x64 | 8 KB | 8 KB | 524K FLOPs | 32 FLOPs/byte |
| 128x128x64 | 16 KB | 16 KB | 2.1M FLOPs | 65 FLOPs/byte |
| 128x256x64 | 16 KB | 32 KB | 4.2M FLOPs | 87 FLOPs/byte |

m64n256k16 needs 128 registers per accumulator - For BM=128, you need 2× m64n256k16 = 256 registers - This CAN cause register spill if not managed carefully. Process tiles sequentially to reduce register pressure

```cpp
// BAD: 256 registers live simultaneously → spill!
float acc0[128], acc1[128];
wgmma_m64n256k16_fn(acc0, ...);  // rows 0-63
wgmma_m64n256k16_fn(acc1, ...);  // rows 64-127
// GOOD: Reuse accumulator → only 128 registers
float acc[128];
// Process rows 0-63
for (k...) wgmma_m64n256k16_fn(acc, desc_a_low, ...);
// Store acc to output_low;
// Process rows 64-127
for (k...) wgmma_m64n256k16_fn(acc, desc_a_high, ...);
// Store acc to output_high;
```