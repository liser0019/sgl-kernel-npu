# sparse_kv_plan

`sgl_kernel_npu.operators.mem_cache.sparse_kv_plan` 接收按 token 位置升序排列的 TopK，原地更新 KV 缓存元数据，输出 attention 槽位和 miss 描述符。调用方根据描述符搬运 KV，再通过 stream 顺序或事件启动 attention。

## 输入

`R` 为分配的请求数，`W` 为每请求 query 行数，`Q = R × W`，`C` 为每个缓存行的槽容量。Ascend 950 路径支持 `W=1/2/4`，对应 `C=4096/4096/8192`。要求 `0 < max_token <= 358400`，且 `C` 能被 `block_size` 整除。

| 输入 | 形状及语义 |
| --- | --- |
| `req_ids` | INT64 `[R]`，有效请求的唯一身份。身份变化时重置对应缓存行。 |
| `topk_indices` | INT32 `[Q,2048]`，每行有效逻辑 token 位置严格递增。允许连续 `-1` 填充放在行首或行尾。 |
| `stable_prefix_lens`、`visible_seq_lens` | INT32 `[Q]`，分别是已提交前缀和本行因果可见位置的排他上限。 |
| `token_to_req` | INT32 `[Q]`，每行对应的 `req_ids` 下标。一个有效请求占连续对齐的 `W` 行。 |
| `active_rows` | INT32 `[1]`，从首行开始的有效行数，须为 `W` 的整数倍。 |
| `state` | 通过 `allocate_sparse_kv_plan_state` 分配，每层跨 decode step 复用。 |

传入同一设备上的连续、互不重叠张量。并发 Plan 使用独立状态。跨流消费时保持状态存活，并等待消费者完成后再复用。

## 输出

- `current_slots[Q,2048]` 是 `resident_rows[Q]` 对应缓存行内的局部设备槽。使用 `resident_block_table[Q,C/block_size]` 解析对应页。无效 query 的槽号为 `-1`。
- `miss_tokens` 和 `miss_slots` 分别给出逻辑 token 和目的局部槽。每行消费 `miss_count[row]` 项。联合组可将描述符分布在多行，使用输出的 `resident_rows` 找实际缓存行。
- `last_req_ids`、`slot_to_token`、`lru_slots`、`row_modes` 跨调用保留。模式 0/1/2 分别代表独立行、联合 leader、follower。槽不足时可能发布独立行。
- `status` 是持久设备状态视图。成功值为 INT32_MAX，拒绝值为 `-1`。在已有同步边界检查，丢弃被拒组的结果，并在复用前重新初始化状态。其他组可能已经完成。

## 构建和调用

在 Ascend 950 上加载 CANN 9.2.1 环境和配套 PyTorch、torch_npu。从仓库根目录运行构建，把生成的 wheel 安装到测试环境。

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
python setup.py --arch-version 3510 --ops quant_lightning_indexer_ordered,sparse_kv_plan --without-third-party -j4
python -m pip install --no-deps <wheel-path>
python -m pytest -q tests/python/operators/indexer/test_quant_lightning_indexer_ordered.py \
  tests/python/operators/mem_cache/test_sparse_kv_plan.py --npu-device 0
```

[英文接口页](../../../en/operators/mem_cache/sparse_kv_plan.md)包含两个独立算子串接示例。`impl="torch_native"` 提供独立正确性参考。设备实现来源为 `op_test_kvoffload_simt_echo@ba4c5dd9bb6aaa432abf4822607ce224bec51862` 的两文件导出版本，保留源码声明与许可证。
