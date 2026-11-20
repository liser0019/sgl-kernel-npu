/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef SGL_SPARSE_KV_PLAN_TILING_H
#define SGL_SPARSE_KV_PLAN_TILING_H

#include <climits>
#include <cstdint>

namespace sparse_kv_plan_config {

constexpr uint32_t PLAN_THREADS = 1024U;
constexpr uint32_t PLAN_WARP_SIZE = 32U;
constexpr uint32_t PLAN_WARP_COUNT = PLAN_THREADS / PLAN_WARP_SIZE;
constexpr uint32_t PLAN_CONTROL_ELEMENTS = 8U;
constexpr uint32_t PLAN_SCAN_ELEMENTS = 2U * PLAN_WARP_COUNT + PLAN_CONTROL_ELEMENTS;
constexpr uint32_t PLAN_SCAN_BYTES = (PLAN_SCAN_ELEMENTS * sizeof(int32_t) + 31U) & ~31U;
constexpr uint32_t PLAN_SCAN_STORAGE_ELEMENTS = PLAN_SCAN_BYTES / sizeof(int32_t);
constexpr uint32_t PLAN_ROW_UB_LIMIT_BYTES = 112U * 1024U;
constexpr uint32_t PLAN_ROW_UB_LIMIT_ELEMENTS = PLAN_ROW_UB_LIMIT_BYTES / sizeof(int32_t);
constexpr uint32_t PLAN_MAX_DYNAMIC_UB_BYTES = PLAN_SCAN_BYTES + PLAN_ROW_UB_LIMIT_BYTES;

static_assert(PLAN_THREADS > 0U, "Sparse KV Plan requires at least one thread");
static_assert(PLAN_THREADS % PLAN_WARP_SIZE == 0U, "Sparse KV Plan threads must contain whole warps");
static_assert(PLAN_SCAN_BYTES % 32U == 0U, "Sparse KV Plan scan workspace must be 32-byte aligned");
static_assert(PLAN_ROW_UB_LIMIT_BYTES % sizeof(int32_t) == 0U, "Sparse KV Plan row UB limit must be int32 aligned");
static_assert(PLAN_MAX_DYNAMIC_UB_BYTES <= 120U * 1024U, "Sparse KV Plan dynamic UB exceeds the full-DCache budget");

}  // namespace sparse_kv_plan_config

// Shared byte layout for CANN runtime tiling and the device kernel.
struct SparseKvPlanTiling {
    uint64_t hashCapacity;
    uint64_t workspaceRowElements;
    int64_t maxRows;
    int64_t topk;
    int64_t capacity;
    int64_t maxToken;
    int64_t maxRequests;
    int32_t blockSize;
    uint32_t localMemoryBytes;
    uint64_t jointHashCapacity;
    uint64_t scratchElements;
};
static_assert(sizeof(SparseKvPlanTiling) == 80, "joint tiling ABI");
namespace sparse_kv_plan_config {

inline bool MakeJointTiling(int64_t maxRequests, int64_t maxRows, int64_t capacity, int64_t maxToken, int32_t blockSize,
                            SparseKvPlanTiling &tiling, uint64_t &workspaceInt32)
{
    if (maxRequests <= 0 || maxRows <= 0 || maxRows % maxRequests != 0 || maxToken <= 0 || maxToken > 358400 ||
        blockSize <= 0) {
        return false;
    }
    const int64_t rowsPerRequest = maxRows / maxRequests;
    if ((rowsPerRequest != 1 && rowsPerRequest != 2 && rowsPerRequest != 4) ||
        capacity != (rowsPerRequest == 4 ? 8192 : 4096) || capacity % blockSize != 0 ||
        maxRows > INT32_MAX / (capacity / blockSize)) {
        return false;
    }

    constexpr uint64_t topK = 2048;
    constexpr uint64_t hashCapacity = 4096;
    const uint64_t residentCapacity = static_cast<uint64_t>(capacity);
    const uint64_t mergeWorkspaceWords = rowsPerRequest == 4 ? 32768U : 24576U;
    const uint64_t scratchWords = 128U + residentCapacity / 2U + residentCapacity + mergeWorkspaceWords;
    const uint32_t dynamicUb = PLAN_SCAN_BYTES + static_cast<uint32_t>(scratchWords * sizeof(int32_t));
    tiling = {hashCapacity, 3U * hashCapacity + residentCapacity + topK,
              maxRows,      topK,
              capacity,     maxToken,
              maxRequests,  blockSize,
              dynamicUb,    2U * residentCapacity,
              scratchWords};
    workspaceInt32 = static_cast<uint64_t>(maxRows) * topK / 2U;
    return true;
}

}  // namespace sparse_kv_plan_config

#endif
