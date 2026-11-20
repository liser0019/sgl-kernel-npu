# Ordered QuantLightningIndexer

This operator preserves the selection, tiling and device computation of the
standalone Ascend 950 implementation in
[ops-transformer at 7b434c4](https://gitcode.com/sparse/ops-transformer/tree/7b434c429b7b5af85246f12040bb3a79531bfa87/attention/quant_lightning_indexer_ordered).
The upstream operator tree is `2f3532304e78af4325d77c7160c50d9d24cbf776`.

`op_kernel/shared` contains the transitive device headers used by the operator
from upstream `attention/lightning_indexer` and `attention/lightning_indexer_v2`.
Headers retain their source notices. The [CANN license](../../../python/sgl_kernel_npu/sgl_kernel_npu/_licenses/quant_lightning_indexer_ordered/CANN-2.0.txt)
is included in the wheel.

The port registers the CANN OpType `SglQuantLightningIndexerOrdered`, uses one
plain tiling struct between Host and device, and uses the repository's native
Torch adapter and vendor. See the [public interface](../../../docs/api/en/operators/indexer/quant_lightning_indexer_ordered.md)
for the TND/PA_BSND contract and the two-operator build.
