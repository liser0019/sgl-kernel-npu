/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef SGL_SPARSE_KV_TRANSFER_TILING_H
#define SGL_SPARSE_KV_TRANSFER_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseKvTransferTilingData)
TILING_DATA_FIELD_DEF(int64_t, maxRows);
TILING_DATA_FIELD_DEF(int64_t, maxRequests);
TILING_DATA_FIELD_DEF(int64_t, maxNumBlocks);
TILING_DATA_FIELD_DEF(int64_t, topk);
TILING_DATA_FIELD_DEF(int64_t, capacity);
TILING_DATA_FIELD_DEF(int64_t, hostNumBlocks);
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, tokenSizeBytesK);
TILING_DATA_FIELD_DEF(int32_t, tokenSizeBytesV);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SglSparseKvTransfer, SparseKvTransferTilingData)
struct SparseKvTransferCompileInfo {};
}  // namespace optiling
#endif
