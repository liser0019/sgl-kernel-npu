// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aclnn_runtime.h"

#include <c10/core/DeviceGuard.h>
#include <limits>
#include <vector>

namespace sgl_cann {

inline aclDataType sparse_kv_dtype(at::ScalarType dtype)
{
    switch (dtype) {
        case at::kInt: return ACL_INT32;
        case at::kLong: return ACL_INT64;
        case at::kBFloat16: return ACL_BF16;
        default: TORCH_CHECK(false, "Unexpected sparse KV dtype: ", dtype);
    }
}

template <typename Execute, typename Query>
void run_sparse_kv(const char* name, std::vector<at::Tensor> tensors, Execute execute, Query query)
{
    const auto device = tensors.front().device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, name, " requires NPU tensors");
    const c10::DeviceGuard guard(device);
    std::vector<std::shared_ptr<aclTensor>> descriptors;
    descriptors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device() == device && tensor.is_contiguous(),
                    name, " requires contiguous tensors on the same NPU");
        descriptors.push_back(tensor_descriptor(tensor, sparse_kv_dtype(tensor.scalar_type())));
    }

    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    auto status = query(descriptors, &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, name, " workspace query failed: ", status, " ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                name, " workspace exceeds the tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, tensors.front().options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& tensor : tensors) {
        tensor.record_stream(stream);
    }
    workspace.record_stream(stream);
    auto acl_stream = stream.stream(false);
    auto launch = [=]() -> int {
        // Retain the input handles and descriptors while the launch is queued.
        (void)tensors;
        (void)descriptors;
        const auto result = execute(bytes ? workspace.data_ptr() : nullptr, bytes, executor, acl_stream);
        TORCH_CHECK(result == ACL_SUCCESS, name, " execution failed: ", result, " ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi(name, launch);
}

}  // namespace sgl_cann
