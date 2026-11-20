// SPDX-License-Identifier: Apache-2.0

#include "aclnn_runtime.h"
#include "aclnn_sgl_sparse_kv_plan.h"
#include "op_kernel/sparse_kv_plan_tiling.h"

#include <algorithm>
#include <array>
#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>

namespace {

struct Buffer {
    const char *name;
    const at::Tensor &tensor;
    at::ScalarType dtype;
    std::initializer_list<int64_t> shape;
};

struct StorageRange {
    uintptr_t begin;
    uintptr_t end;
};

void SparseKvPlan(const at::Tensor &reqIds, const at::Tensor &topkIndices, const at::Tensor &stablePrefixLens,
                  const at::Tensor &visibleSeqLens, const at::Tensor &tokenToReq, const at::Tensor &activeRows,
                  const at::Tensor &lastReqIds, const at::Tensor &slotToToken, const at::Tensor &lruSlots,
                  const at::Tensor &currentSlots, const at::Tensor &missCount, const at::Tensor &missTokens,
                  const at::Tensor &missSlots, const at::Tensor &planWorkspace, const at::Tensor &rowModes,
                  const at::Tensor &residentRows, const at::Tensor &residentBlockTable, int64_t capacity,
                  int64_t maxToken, int64_t blockSize)
{
    const auto device = reqIds.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "sparse_kv_plan requires NPU tensors");
    TORCH_CHECK(reqIds.dim() == 1 && topkIndices.dim() == 2, "expected req_ids [R] and topk_indices [Q, 2048]");
    const auto requests = reqIds.size(0);
    const auto rows = topkIndices.size(0);
    uint64_t workspaceWords = 0;
    SparseKvPlanTiling tiling{};
    TORCH_CHECK(blockSize > 0 && blockSize <= INT32_MAX && rows <= std::numeric_limits<int64_t>::max() / 1024 &&
                    sparse_kv_plan_config::MakeJointTiling(requests, rows, capacity, maxToken,
                                                           static_cast<int32_t>(blockSize), tiling, workspaceWords),
                "unsupported sparse KV Plan shape: expected 1, 2 or 4 rows per request with capacity "
                "4096, 4096 or 8192, capacity divisible by block_size, and 0 < max_token <= 358400");

    const std::array<Buffer, 17> buffers{{
        {"req_ids", reqIds, at::kLong, {requests}},
        {"topk_indices", topkIndices, at::kInt, {rows, 2048}},
        {"stable_prefix_lens", stablePrefixLens, at::kInt, {rows}},
        {"visible_seq_lens", visibleSeqLens, at::kInt, {rows}},
        {"token_to_req", tokenToReq, at::kInt, {rows}},
        {"active_rows", activeRows, at::kInt, {1}},
        {"last_req_ids", lastReqIds, at::kLong, {rows}},
        {"slot_to_token", slotToToken, at::kInt, {rows, capacity}},
        {"lru_slots", lruSlots, at::kInt, {rows, capacity}},
        {"current_slots", currentSlots, at::kInt, {rows, 2048}},
        {"miss_count", missCount, at::kInt, {rows}},
        {"miss_tokens", missTokens, at::kInt, {rows, 2048}},
        {"miss_slots", missSlots, at::kInt, {rows, 2048}},
        {"plan_workspace", planWorkspace, at::kInt, {static_cast<int64_t>(workspaceWords)}},
        {"row_modes", rowModes, at::kInt, {rows}},
        {"resident_rows", residentRows, at::kInt, {rows}},
        {"resident_block_table", residentBlockTable, at::kInt, {rows, capacity / blockSize}},
    }};
    std::array<StorageRange, buffers.size()> ranges;
    std::array<at::Tensor, buffers.size()> tensors;
    std::array<std::shared_ptr<aclTensor>, buffers.size()> descriptors;
    const c10::DeviceGuard guard(device);
    for (size_t i = 0; i < buffers.size(); ++i) {
        const auto &buffer = buffers[i];
        const auto &tensor = buffer.tensor;
        TORCH_CHECK(tensor.device() == device && tensor.is_contiguous() && tensor.scalar_type() == buffer.dtype &&
                        tensor.sizes() == at::IntArrayRef(buffer.shape),
                    buffer.name, " must be contiguous on ", device, " with dtype ", buffer.dtype, " and shape ",
                    at::IntArrayRef(buffer.shape));
        const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        ranges[i] = {begin, begin + tensor.nbytes()};
        tensors[i] = tensor;
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const auto &left, const auto &right) { return left.begin < right.begin; });
    for (size_t i = 1; i < ranges.size(); ++i) {
        TORCH_CHECK(ranges[i - 1].end <= ranges[i].begin, "sparse_kv_plan buffers must not overlap");
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        descriptors[i] = sgl_cann::tensor_descriptor(tensors[i], buffers[i].dtype == at::kLong ? ACL_INT64 : ACL_INT32);
    }
    static const auto query = reinterpret_cast<decltype(&aclnnSglSparseKvPlanGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSparseKvPlanGetWorkspaceSize"));
    static const auto execute =
        reinterpret_cast<decltype(&aclnnSglSparseKvPlan)>(sgl_cann::op_api_symbol("aclnnSglSparseKvPlan"));
    const auto &d = descriptors;
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    const auto result =
        query(d[0].get(), d[1].get(), d[2].get(), d[3].get(), d[4].get(), d[5].get(), capacity, maxToken, blockSize,
              d[6].get(), d[7].get(), d[8].get(), d[9].get(), d[10].get(), d[11].get(), d[12].get(), d[13].get(),
              d[14].get(), d[15].get(), d[16].get(), &bytes, &executor);
    TORCH_CHECK(result == ACL_SUCCESS, "sparse_kv_plan workspace query failed: ", result, " ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "Plan workspace too large");
    const auto workspace = at::empty({static_cast<int64_t>(bytes)}, reqIds.options().dtype(at::kByte));
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors) {
        tensor.record_stream(stream);
    }
    workspace.record_stream(stream);
    const auto aclStream = stream.stream(false);
    at_npu::native::OpCommand::RunOpApi("aclnnSglSparseKvPlan", [=]() -> int {
        // Keep descriptors and their storage alive until the queued host launch.
        (void)tensors;
        (void)descriptors;
        const auto status = execute(bytes ? workspace.data_ptr() : nullptr, bytes, executor, aclStream);
        TORCH_CHECK(status == ACL_SUCCESS, "sparse_kv_plan launch failed: ", status, " ", aclGetRecentErrMsg());
        return status;
    });
}

void SparseKvPlanMeta(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                      const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                      const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                      const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                      const at::Tensor &, int64_t, int64_t, int64_t)
{}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sgl_sparse_kv_plan(Tensor req_ids, Tensor topk_indices, Tensor stable_prefix_lens, "
        "Tensor visible_seq_lens, Tensor token_to_req, Tensor active_rows, Tensor(a!) last_req_ids, "
        "Tensor(b!) slot_to_token, Tensor(c!) lru_slots, Tensor(d!) current_slots, Tensor(e!) miss_count, "
        "Tensor(f!) miss_tokens, Tensor(g!) miss_slots, Tensor(h!) plan_workspace, Tensor(i!) row_modes, "
        "Tensor(j!) resident_rows, Tensor(k!) resident_block_table, int capacity, int max_token, "
        "int block_size) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_sparse_kv_plan", &SparseKvPlan);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sgl_sparse_kv_plan", &SparseKvPlanMeta);
}
