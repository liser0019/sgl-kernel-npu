// SPDX-License-Identifier: Apache-2.0

#include "aclnn_runtime.h"
#include "aclnn_sgl_quant_lightning_indexer_ordered.h"

#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>

namespace {

aclDataType AclDtype(at::ScalarType dtype)
{
    switch (dtype) {
        case at::kFloat8_e4m3fn:
            return ACL_FLOAT8_E4M3FN;
        case at::kChar:
            return ACL_INT8;
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        default:
            TORCH_CHECK(false, "Unsupported ordered indexer dtype: ", dtype);
    }
}

int64_t PagedOuterStride(const at::Tensor &tensor, const char *name)
{
    if (tensor.is_contiguous()) {
        return -1;
    }
    int64_t rowElements = 1;
    for (int64_t axis = tensor.dim() - 1; axis > 0; --axis) {
        TORCH_CHECK(tensor.stride(axis) == rowElements || tensor.size(axis) == 1, name,
                    " permits padding only between physical pages");
        rowElements *= tensor.size(axis);
    }
    TORCH_CHECK(tensor.stride(0) >= rowElements && tensor.stride(0) <= std::numeric_limits<uint32_t>::max(), name,
                " has an unsupported physical page stride");
    return tensor.stride(0);
}

at::Tensor OrderedIndexerMeta(const at::Tensor &query, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                              const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                              int64_t topK, int64_t)
{
    TORCH_CHECK(query.dim() == 3, "Ordered Indexer query must be TND");
    return at::empty_symint({query.sym_size(0), 1, topK}, query.options().dtype(at::kInt));
}

at::Tensor OrderedIndexer(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                          const at::Tensor &queryScale, const at::Tensor &keyScale, const at::Tensor &queryEnds,
                          const at::Tensor &keyLengths, const at::Tensor &blockTable, int64_t topK, int64_t sparseMode)
{
    const auto device = query.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "Ordered Indexer requires NPU tensors");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && key.size(2) == 1 && keyScale.dim() == 3,
                "Ordered Indexer expects TND queries and PA_BSND keys with a single key head");
    TORCH_CHECK(topK > 0 && topK <= 2048 && (sparseMode == 0 || sparseMode == 3),
                "Ordered Indexer requires 0 < topk <= 2048 and sparse_mode 0 or 3");
    const auto keyStride = PagedOuterStride(key, "key");
    const auto scaleStride = PagedOuterStride(keyScale, "key_dequant_scale");
    const c10::DeviceGuard guard(device);
    auto output = at::empty({query.size(0), 1, topK}, query.options().dtype(at::kInt));
    const std::array<at::Tensor, 9> tensors{query,     key,        weights,    queryScale, keyScale,
                                            queryEnds, keyLengths, blockTable, output};
    std::array<std::shared_ptr<aclTensor>, tensors.size()> descriptors;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        TORCH_CHECK(tensor.device() == device, "Ordered Indexer tensors must be on the same NPU");
        TORCH_CHECK(i == 1 || i == 4 || tensor.is_contiguous(),
                    "Ordered Indexer requires contiguous tensors except for page strides in key and key_dequant_scale");
        descriptors[i] = sgl_cann::tensor_descriptor(tensor, AclDtype(tensor.scalar_type()));
    }

    static const auto getWorkspace = reinterpret_cast<decltype(&aclnnSglQuantLightningIndexerOrderedGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglQuantLightningIndexerOrderedGetWorkspaceSize"));
    static const auto execute = reinterpret_cast<decltype(&aclnnSglQuantLightningIndexerOrdered)>(
        sgl_cann::op_api_symbol("aclnnSglQuantLightningIndexerOrdered"));
    constexpr auto full = std::numeric_limits<int64_t>::max();
    auto queryLayout = std::make_shared<std::string>("TND");
    auto keyLayout = std::make_shared<std::string>("PA_BSND");
    const auto &d = descriptors;
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = getWorkspace(d[0].get(), d[1].get(), d[2].get(), d[3].get(), d[4].get(), d[5].get(), d[6].get(),
                                     d[7].get(), 0, 0, queryLayout->data(), keyLayout->data(), topK, sparseMode, full,
                                     full, keyStride, scaleStride, d[8].get(), &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "Ordered Indexer workspace query failed: ", status, " ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= static_cast<uint64_t>(full), "Ordered Indexer workspace is too large");
    const auto workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors) {
        tensor.record_stream(stream);
    }
    workspace.record_stream(stream);
    const auto aclStream = stream.stream(false);
    at_npu::native::OpCommand::RunOpApi("aclnnSglQuantLightningIndexerOrdered", [=]() -> int {
        // Keep descriptor storage and the layout strings alive through the queued host launch.
        (void)tensors;
        (void)descriptors;
        (void)queryLayout;
        (void)keyLayout;
        const auto result = execute(bytes ? workspace.data_ptr() : nullptr, bytes, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "Ordered Indexer launch failed: ", result, " ", aclGetRecentErrMsg());
        return result;
    });
    return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sgl_quant_lightning_indexer_ordered(Tensor query, Tensor key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor key_dequant_scale, Tensor actual_seq_lengths_query, "
        "Tensor actual_seq_lengths_key, Tensor block_table, int topk=2048, int sparse_mode=3) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_quant_lightning_indexer_ordered", &OrderedIndexer);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sgl_quant_lightning_indexer_ordered", &OrderedIndexerMeta);
}
