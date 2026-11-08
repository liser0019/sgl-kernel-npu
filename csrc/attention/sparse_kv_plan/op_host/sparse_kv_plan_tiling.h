/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef SGL_SPARSE_KV_PLAN_TILING_H
#define SGL_SPARSE_KV_PLAN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseKvPlanTilingData)
TILING_DATA_FIELD_DEF(uint64_t, hashCapacity);
TILING_DATA_FIELD_DEF(uint64_t, workspaceRowElements);
TILING_DATA_FIELD_DEF(int64_t, maxRows);
TILING_DATA_FIELD_DEF(int64_t, topk);
TILING_DATA_FIELD_DEF(int64_t, capacity);
TILING_DATA_FIELD_DEF(int64_t, maxToken);
TILING_DATA_FIELD_DEF(int64_t, maxRequests);
TILING_DATA_FIELD_DEF(int64_t, maxNumBlocks);
TILING_DATA_FIELD_DEF(int64_t, hostNumBlocks);
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, localMemoryBytes);
TILING_DATA_FIELD_DEF(uint64_t, jointHashCapacity);
TILING_DATA_FIELD_DEF(uint64_t, scratchElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SglSparseKvPlan, SparseKvPlanTilingData)
struct SparseKvPlanCompileInfo {};
}  // namespace optiling
#endif
