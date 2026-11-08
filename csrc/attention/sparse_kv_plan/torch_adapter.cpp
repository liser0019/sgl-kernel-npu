// SPDX-License-Identifier: Apache-2.0
#include "sparse_kv_runtime.h"
#include "aclnn_sgl_sparse_kv_plan.h"

namespace {
void sparse_kv_plan(
    const at::Tensor& reqIds, const at::Tensor& topkIndices, const at::Tensor& stablePrefixLens,
    const at::Tensor& visibleSeqLens, const at::Tensor& tokenToReq, const at::Tensor& blockTable,
    const at::Tensor& activeRows, const at::Tensor& lastReqIds, const at::Tensor& slotToToken,
    const at::Tensor& lruSlots, const at::Tensor& currentSlots, const at::Tensor& missCount,
    const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& compactWorkspace,
    const at::Tensor& rowModes, const at::Tensor& residentRows, const at::Tensor& residentBlockTable,
    int64_t topk, int64_t capacity, int64_t maxToken,
    int64_t blockSize, int64_t hostNumBlocks, int64_t jointMtp)
{
    static auto query = reinterpret_cast<decltype(&aclnnSglSparseKvPlanGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSparseKvPlanGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglSparseKvPlan)>(
        sgl_cann::op_api_symbol("aclnnSglSparseKvPlan"));
    sgl_cann::run_sparse_kv("aclnnSglSparseKvPlan", {
        reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable,
        activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount,
        missTokens, missSlots, compactWorkspace, rowModes, residentRows, residentBlockTable
    }, execute, [&](const auto& desc, uint64_t* bytes, aclOpExecutor** executor) {
        return query(desc[0].get(), desc[1].get(), desc[2].get(), desc[3].get(), desc[4].get(),
                desc[5].get(), desc[6].get(), desc[7].get(), desc[8].get(), desc[9].get(),
                desc[10].get(), desc[11].get(), desc[12].get(), desc[13].get(), desc[14].get(),
                desc[15].get(), desc[16].get(), desc[17].get(), topk, capacity,
                maxToken, blockSize, hostNumBlocks, jointMtp, bytes,
                executor);
    });
}

void meta(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t)
{
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sgl_sparse_kv_plan(Tensor reqIds, Tensor topkIndices, Tensor stablePrefixLens, Tensor "
        "visibleSeqLens, Tensor tokenToReq, Tensor blockTable, Tensor activeRows, Tensor(h!) lastReqIds, "
        "Tensor(i!) slotToToken, Tensor(j!) lruSlots, Tensor(k!) currentSlots, Tensor(l!) missCount, "
        "Tensor(m!) missTokens, Tensor(n!) missSlots, Tensor(o!) compactWorkspace, Tensor(p!) rowModes, "
        "Tensor(q!) residentRows, Tensor(r!) residentBlockTable, int topk, int capacity, int maxToken, "
        "int blockSize, int hostNumBlocks, int jointMtp) -> ()"
    );
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_sparse_kv_plan", &sparse_kv_plan);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sgl_sparse_kv_plan", &meta);
}
