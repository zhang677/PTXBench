# Optimization Pattern: Split Output Ownership for Standard-Triton Attention Backward

Remove global gradient atomics by assigning exclusive output ownership to two disjoint program
regions. One region owns dQ tiles and reduces over all KV tiles. The other owns dK/dV tiles and
reduces over all query tiles. The regions can share one launch with a flat program-ID partition or
be separate kernel launches; they do not synchronize or exchange shared memory.

This is the faithful standard-Triton analogue of improving backward ownership. It is not a source-
level translation of FA4 two-CTA MMA, DSMEM exchange, TMEM column reuse, or deterministic semaphore
protocols.

## Flat-grid partition

For one batch item and KV head, reserve the first `num_kv_tiles` program IDs for dK/dV and the
remaining IDs for dQ across every associated query head:

```python
pid = tl.program_id(0)
num_kv_tiles = tl.cdiv(kv_length, BLOCK_N_DKDV)
num_q_tiles = tl.cdiv(query_length, BLOCK_M_DQ)

if pid < num_kv_tiles:
    kv_tile = pid
    # Own exactly one dK/dV tile and reduce over every contributing Q tile.
else:
    q_region_pid = pid - num_kv_tiles
    query_head = q_region_pid // num_q_tiles
    q_tile = q_region_pid % num_q_tiles
    # Own exactly one dQ tile and reduce over every contributing KV tile.
```

For GQA, each dK/dV owner loops over all query heads mapped to its KV head before storing. Each dQ
owner uses only its query head. Include batch broadcast rules in the mapping; do not let two program
IDs store the same logical output tile.

## dQ-owner region

Keep Q, dO, delta, and LSE resident while streaming K and V:

```python
dq = tl.zeros((BLOCK_M_DQ, HEAD_DIM_ROUNDED), tl.float32)
for kv_tile in range(0, tl.cdiv(kv_length, BLOCK_N_DQ)):
    scores = tl.dot(q, k.T) * softmax_scale
    p = tl.math.exp2(scores * RCP_LN2 - lse_log2[:, None])
    dp = tl.dot(do, v.T)
    ds = tl.where(
        valid_score,
        p * (dp - delta[:, None]) * softmax_scale,
        0.0,
    )
    dq += tl.dot(ds.to(q.dtype), k)

# Store the complete dQ tile once; no atomic is needed.
```

## dK/dV-owner region

Keep one K/V tile resident while streaming Q and dO tiles:

```python
dk = tl.zeros((BLOCK_N_DKDV, HEAD_DIM_ROUNDED), tl.float32)
dv = tl.zeros((BLOCK_N_DKDV, VALUE_DIM_ROUNDED), tl.float32)
for q_tile in range(0, tl.cdiv(query_length, BLOCK_M_DKDV)):
    scores_t = tl.dot(k, q.T) * softmax_scale
    p_t = tl.math.exp2(scores_t * RCP_LN2 - lse_log2[None, :])
    dv += tl.dot(p_t.to(q.dtype), do)
    dp_t = tl.dot(v, do.T)
    ds_t = tl.where(
        valid_score_t,
        p_t * (dp_t - delta[None, :]) * softmax_scale,
        0.0,
    )
    dk += tl.dot(ds_t.to(q.dtype), q)

# Store complete dK and dV tiles once; no atomic is needed.
```

Use descriptors for regular aligned blocks and pointer tensors for layouts or tails they cannot
describe safely. Loading LSE and delta before the first dot can shorten a later dependency stall.
Separate full blocks from partial/masked blocks when that removes meaningful per-iteration work.

## Tuning and validation

- Tune dQ and dK/dV tile shapes separately; their reduction directions and resident tensors differ.
- Compare one combined PID-partitioned launch with separate launches. Separate launches simplify
  control flow but add launch overhead.
- Keep FP32 accumulators through the full reduction, then convert once at the unique store.
- Validate the two regions independently and assert exact output-tile ownership over the whole grid.
- For deterministic output, exclusive ownership removes global atomic ordering from the gradient
  reduction, but floating-point results can still change with tile size and reduction order.
- Do not claim overlap between the two regions or a particular Blackwell instruction schedule from
  source structure alone; confirm generated code and profiler evidence when needed.
