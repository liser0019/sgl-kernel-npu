# sparse_kv_plan

`sgl_kernel_npu.operators.mem_cache.sparse_kv_plan` plans KV placement for ordered TopK positions. It updates cache metadata in place and produces the slots and the descriptors for missing KV. It performs metadata operations only. The caller schedules KV copies and attention after the plan on the stream or via events.

## Inputs

Let `R` be the allocated request count, `W` the query rows per request, `Q = R * W`, and `C` the hot-cache capacity in slots per cache row. The A5 (DAV_3510) path supports `W=1/2/4` with `C=4096/4096/8192` respectively, and `0 < max_token <= 358400`. `block_size` must divide `C`.

| Input | Shape and meaning |
| --- | --- |
| `req_ids` | INT64 `[R]`, unique live request identities. A changed identity resets its cache row. |
| `topk_indices` | INT32 `[Q, 2048]`, strictly ascending distinct logical token positions in each row. A contiguous `-1` run can pad the head or tail. |
| `stable_prefix_lens`, `visible_seq_lens` | INT32 `[Q]`, exclusive committed-token and query-visible bounds in the same logical coordinate space. |
| `token_to_req` | INT32 `[Q]`, indices into `req_ids`. An active request occupies one aligned, contiguous group of `W` rows. |
| `active_rows` | INT32 `[1]`, number of leading active rows, a multiple of `W`. |
| `state` | `SparseKvPlanState` from `allocate_sparse_kv_plan_state`. Reuse for one layer across decode steps. |

Device inputs and state must be contiguous, on the same device and non-overlapping. Allocate a separate state for concurrently executing plans. With asynchronous consumers, preserve the state's lifetime and order its reuse after those consumers.

## Outputs in state

- `current_slots[Q,2048]` gives local cache slots relative to `resident_rows[Q]` and `resident_block_table[Q,C/block_size]`. Invalid queries get `-1`.
- `miss_tokens[Q,2048]` and `miss_slots[Q,2048]` describe logical token and local destination slot. Consume only the first `miss_count[row]` entries in each row. Multi-row groups may share the leader's cache row and spread descriptors across the group's rows.
- `last_req_ids`, `slot_to_token`, `lru_slots`, and `row_modes` persist across calls. `row_modes` values 0, 1 and 2 designate a private row, group leader and follower. The capacity-shortage path may publish private rows, so use the actual `resident_rows` for each row.
- `status` is a sticky device view with INT32_MAX for success and `-1` on rejection. Check status at the caller's synchronization boundary and discard rejected results. Reinitialize the state before reusing it after a rejection. Other groups may have completed.

## Build and smoke test

Run from the repository root on Ascend 950 with CANN 9.2.1 and matching PyTorch and torch_npu. Install into your test environment.

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
python setup.py --arch-version 3510 --ops quant_lightning_indexer_ordered,sparse_kv_plan --without-third-party -j4
# Install the wheel printed by the build.
python -m pip install --no-deps <wheel-path>
python -m pytest -q tests/python/operators/indexer/test_quant_lightning_indexer_ordered.py \
  tests/python/operators/mem_cache/test_sparse_kv_plan.py --npu-device 0
```

## Decode call

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_ordered
from sgl_kernel_npu.operators.mem_cache import allocate_sparse_kv_plan_state, sparse_kv_plan

# Once per layer and shape, outside the decode loop.
state = allocate_sparse_kv_plan_state(
    requests, rows_per_request=4, max_token=358400, block_size=64, device=query.device
)

# On each decode step.
indices = quant_lightning_indexer_ordered(
    query, key, weights, q_scale, k_scale,
    actual_seq_lengths_query=query_ends,
    actual_seq_lengths_key=key_lengths,
    block_table=indexer_block_table,
).squeeze(1)
sparse_kv_plan(req_ids, indices, stable_prefix_lens, visible_seq_lens, token_to_req, active_rows, state)
```

`impl="torch_native"` is an independent correctness reference. The original Plan source is `op_test_kvoffload_simt_echo@ba4c5dd9bb6aaa432abf4822607ce224bec51862`. The device stages are imported from its tested two-file export with a CANN launch adapter. Source notices and license copies are included in the wheel.
