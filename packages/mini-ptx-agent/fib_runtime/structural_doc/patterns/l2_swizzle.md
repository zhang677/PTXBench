# Optimization Pattern: L2 Cache Swizzle

Tile scheduling affects L2 cache hit rate:
```cpp
// Naive linear scheduling
tile_m = tile_idx / num_n_tiles;
tile_n = tile_idx % num_n_tiles;
// Problem: Adjacent tiles in M don't share L2 for B!
// GROUP_M swizzle (Triton-style)
int GROUP_M = 8;
int num_pid_in_group = GROUP_M * num_n_tiles;
int group_id = tile_idx / num_pid_in_group;
int first_pid_m = group_id * GROUP_M;
int group_size_m = min(GROUP_M, num_m_tiles - first_pid_m);
tile_m = first_pid_m + (tile_idx % group_size_m);
tile_n = (tile_idx % num_pid_in_group) / group_size_m;
```

Visual:
```
  Naive:                    GROUP_M=8 Swizzle:
  (0,0)(0,1)(0,2)...        (0,0)(1,0)(2,0)...(7,0)
  (1,0)(1,1)(1,2)...  →     (0,1)(1,1)(2,1)...(7,1)
  (2,0)(2,1)(2,2)...        ...

Consecutive tiles share B → L2 hits!
```