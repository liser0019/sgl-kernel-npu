// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include "sparse_kv_runtime.h"
#include "aclnn_sgl_sparse_kv_transfer.h"

namespace {
void sparse_kv_transfer(
    const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots,
    const at::Tensor& tokenToReq, const at::Tensor& blockTable, const at::Tensor& hostCacheBases,
    const at::Tensor& activeRows, const at::Tensor& residentK, const at::Tensor& residentV,
    const at::Tensor& residentRows, int64_t topk, int64_t capacity,
    int64_t blockSize, int64_t hostNumBlocks, int64_t tokenSizeBytesK,
    int64_t tokenSizeBytesV)
{
    static auto query = reinterpret_cast<decltype(&aclnnSglSparseKvTransferGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSparseKvTransferGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglSparseKvTransfer)>(
        sgl_cann::op_api_symbol("aclnnSglSparseKvTransfer"));
    sgl_cann::run_sparse_kv("aclnnSglSparseKvTransfer", {
        missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases,
        activeRows, residentK, residentV, residentRows
    }, execute, [&](const auto& desc, uint64_t* bytes, aclOpExecutor** executor) {
        return query(desc[0].get(), desc[1].get(), desc[2].get(), desc[3].get(), desc[4].get(),
                desc[5].get(), desc[6].get(), desc[7].get(), desc[8].get(), desc[9].get(),
                topk, capacity, blockSize, hostNumBlocks, tokenSizeBytesK,
                tokenSizeBytesV, bytes, executor);
    });
}

void meta(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t)
{
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sgl_sparse_kv_transfer(Tensor missCount, Tensor missTokens, Tensor missSlots, Tensor tokenToReq, "
        "Tensor blockTable, Tensor hostCacheBases, Tensor activeRows, Tensor(h!) residentK, Tensor(i!) "
        "residentV, Tensor residentRows, int topk, int capacity, int blockSize, int hostNumBlocks, int "
        "tokenSizeBytesK, int tokenSizeBytesV) -> ()"
    );
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_sparse_kv_transfer", &sparse_kv_transfer);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sgl_sparse_kv_transfer", &meta);
}
