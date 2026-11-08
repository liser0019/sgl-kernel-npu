/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sparse_kv_plan_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <cstdio>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sparse_kv_plan_config.h"

namespace optiling {
namespace {
constexpr uint64_t MIN_HASH_CAPACITY = 32U;
constexpr uint32_t PLAN_BLOCKS = 64U;

uint64_t HashCapacity(int64_t topk) {
  uint64_t target = static_cast<uint64_t>(topk) * 2U;
  uint64_t capacity = MIN_HASH_CAPACITY;
  while (capacity < target) {
    capacity <<= 1U;
  }
  return capacity;
}

ge::graphStatus Fail(gert::TilingContext* context, const char* message) {
  std::fprintf(stderr, "%s: %s\n", context->GetNodeName(), message);
  return ge::GRAPH_FAILED;
}
}  // namespace

static ge::graphStatus SparseKvPlanTiling(gert::TilingContext* context) {
  const gert::StorageShape* reqShape = context->GetInputShape(0);
  const gert::StorageShape* topkShape = context->GetInputShape(1);
  const gert::StorageShape* blockTableShape = context->GetInputShape(5);
  const auto* attrs = context->GetAttrs();
  if (reqShape == nullptr || topkShape == nullptr || blockTableShape == nullptr || attrs == nullptr) {
    return Fail(context, "SglSparseKvPlan requires shapes and attributes");
  }
  const gert::Shape& reqOrigin = reqShape->GetOriginShape();
  const gert::Shape& topkOrigin = topkShape->GetOriginShape();
  const gert::Shape& blockTableOrigin = blockTableShape->GetOriginShape();
  if (reqOrigin.GetDimNum() != 1 || topkOrigin.GetDimNum() != 2 || blockTableOrigin.GetDimNum() != 2) {
    return Fail(context, "SglSparseKvPlan expects req_ids rank 1 and topk/block_table rank 2");
  }

  const int64_t maxRows = topkOrigin.GetDim(0);
  const int64_t maxRequests = reqOrigin.GetDim(0);
  const int64_t maxNumBlocks = blockTableOrigin.GetDim(1);
  const int64_t topk = *attrs->GetInt(0);
  const int64_t capacity = *attrs->GetInt(1);
  const int64_t maxToken = *attrs->GetInt(2);
  const int64_t blockSize = *attrs->GetInt(3);
  const int64_t hostNumBlocks = *attrs->GetInt(4);
  if (maxRows <= 0 || maxRequests <= 0 || topk <= 0 || capacity <= 0 || maxToken <= 0 || maxNumBlocks <= 0 ||
      blockSize <= 0 || hostNumBlocks <= 0) {
    return Fail(context, "SglSparseKvPlan received a non-positive shape or attribute");
  }
  if (blockTableOrigin.GetDim(0) != maxRequests || topkOrigin.GetDim(1) != topk) {
    return Fail(context, "SglSparseKvPlan input shapes do not match request/topk attributes");
  }
  if (topk > capacity || capacity % blockSize != 0) {
    return Fail(context, "SglSparseKvPlan requires topk <= capacity and capacity divisible by blockSize");
  }
  if (topk > std::numeric_limits<int32_t>::max() / 2 || capacity > std::numeric_limits<int32_t>::max() / 2 ||
      maxToken > std::numeric_limits<int32_t>::max()) {
    return Fail(context, "SglSparseKvPlan shape exceeds int32 kernel limits");
  }

  const uint64_t hashCapacity = HashCapacity(topk);
  const uint64_t rowElements = 3U * hashCapacity + static_cast<uint64_t>(capacity) + static_cast<uint64_t>(topk);
  const bool jointMtp = *attrs->GetInt(5) != 0;
  const uint64_t jointHashCapacity = jointMtp ? HashCapacity(capacity) : 0U;
  const uint64_t sortedElements = 64U + ((4U * static_cast<uint64_t>(topk) + capacity + 7U) & ~uint64_t{7U}) +
                                  std::max<uint64_t>(3U * capacity, 64U);
  const uint64_t scratchElements = jointMtp ? std::max(rowElements, sortedElements) : rowElements;
  const uint64_t rowBytes = scratchElements * sizeof(int32_t);
  const uint64_t alignedRowBytes = (rowBytes + 31U) & ~static_cast<uint64_t>(31U);
  const uint64_t chunk = (std::min<uint64_t>(768U, std::max<uint64_t>(64U, 3U * capacity) / 16U) / 4U) * 4U;
  const uint64_t stageBytes = jointMtp ? (64U + 16U * chunk) * sizeof(int32_t) : 0U;
  const uint64_t localBytes =
      sparse_kv_plan_config::PLAN_SCAN_BYTES +
      (alignedRowBytes <= sparse_kv_plan_config::PLAN_ROW_UB_LIMIT_BYTES ? alignedRowBytes : stageBytes);
  if (localBytes > std::numeric_limits<uint32_t>::max()) {
    return Fail(context, "SglSparseKvPlan local memory size overflow");
  }
  const uint32_t localMemoryBytes = static_cast<uint32_t>(localBytes);
  if (context->SetDynUBufSize(localMemoryBytes) != ge::GRAPH_SUCCESS) {
    return Fail(context, "SglSparseKvPlan failed to set dynamic UB size");
  }

  SparseKvPlanTilingData tiling;
  tiling.set_jointHashCapacity(jointHashCapacity);
  tiling.set_scratchElements(scratchElements);
  tiling.set_hashCapacity(hashCapacity);
  tiling.set_workspaceRowElements(rowElements);
  tiling.set_maxRows(maxRows);
  tiling.set_topk(topk);
  tiling.set_capacity(capacity);
  tiling.set_maxToken(maxToken);
  tiling.set_maxRequests(maxRequests);
  tiling.set_maxNumBlocks(maxNumBlocks);
  tiling.set_hostNumBlocks(hostNumBlocks);
  tiling.set_blockSize(static_cast<int32_t>(blockSize));
  tiling.set_localMemoryBytes(localMemoryBytes);

  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return Fail(context, "SglSparseKvPlan cannot query platform information");
  }
  const platform_ascendc::PlatformAscendC platform(platformInfo);
  const uint32_t blockDim = std::min<uint32_t>(PLAN_BLOCKS, platform.GetCoreNumAiv());
  if (blockDim == 0) {
    return Fail(context, "SglSparseKvPlan found zero AIV cores");
  }
  context->SetBlockDim(blockDim);
  size_t* workspaceSize = context->GetWorkspaceSizes(1);
  workspaceSize[0] = platform.GetLibApiWorkSpaceSize();
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseSparseKvPlan(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(SglSparseKvPlan).Tiling(SparseKvPlanTiling).TilingParse<SparseKvPlanCompileInfo>(ParseSparseKvPlan);
}  // namespace optiling
