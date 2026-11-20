# quant_lightning_indexer_ordered

`sgl_kernel_npu.operators.indexer.quant_lightning_indexer_ordered` selects KV tokens by score and returns the selected **logical token positions in ascending order**. Equal scores at the cutoff prefer earlier positions. The result has a valid prefix followed by `-1` padding.

| Argument | Contract |
| --- | --- |
| `query` | TND `[Q, H, D]` |
| `key` | PA_BSND `[pages, page_size, 1, D]` |
| `weights`, `query_dequant_scale` | `[Q, H]` |
| `key_dequant_scale` | `[pages, page_size, 1]` |
| `actual_seq_lengths_query` | INT32 cumulative query row ends, one per request |
| `actual_seq_lengths_key` | INT32 visible key lengths, one per request |
| `block_table` | INT32 logical-to-physical page table, one row per request |
| `topk` | 1 through 2048, defaults to 2048 |
| `sparse_mode` | 3 for causal visibility (default), 0 for full visibility |
| Return | INT32 `[Q, 1, topk]` |

The device path targets Ascend 950 with CANN 9.2.1. The supported dtype pairs are INT8 query/key with FP16 weights and scales, or FP8 E4M3FN query/key with BF16 weights and FP32 scales. `impl="torch_native"` is a synchronous reference for tests.

The native build compiles the checked-in Ascend C source and PyTorch adapter into the wheel. Source provenance is recorded in the [implementation README](../../../../../csrc/indexer/quant_lightning_indexer_ordered/README.md). Build and smoke-test it with [Sparse KV Plan](../mem_cache/sparse_kv_plan.md).
