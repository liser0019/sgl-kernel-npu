# quant_lightning_indexer_ordered

`sgl_kernel_npu.operators.indexer.quant_lightning_indexer_ordered` 按分数选择 TopK，再按**逻辑 token 位置**升序输出。分数在截断边界并列时优先保留位置较早的 token。结果以有效位置开头，尾部填充 `-1`。

| 参数 | 约定 |
| --- | --- |
| `query` | TND `[Q, H, D]` |
| `key` | PA_BSND `[页数, 页大小, 1, D]` |
| `weights`、`query_dequant_scale` | `[Q, H]` |
| `key_dequant_scale` | `[页数, 页大小, 1]` |
| `actual_seq_lengths_query` | INT32，每个请求的 query 行结束位置，按 batch 累计 |
| `actual_seq_lengths_key` | INT32，每个请求的可见 key 长度 |
| `block_table` | INT32，每个请求的逻辑页到物理页映射 |
| `topk` | 1～2048，默认 2048 |
| `sparse_mode` | 3 为因果可见性（默认），0 为全量可见 |
| 返回值 | INT32 `[Q, 1, topk]` |

设备实现面向 Ascend 950 和 CANN 9.2.1。INT8 版使用 FP16 权重和 scale。FP8 E4M3FN 版使用 BF16 权重和 FP32 scale。`impl="torch_native"` 提供同步参考实现。

构建时从本仓源码编译 Ascend C kernel 和 PyTorch 绑定。源码来源见[实现说明](../../../../../csrc/indexer/quant_lightning_indexer_ordered/README.md)。构建和冒烟命令见 [Sparse KV Plan](../mem_cache/sparse_kv_plan.md)。
