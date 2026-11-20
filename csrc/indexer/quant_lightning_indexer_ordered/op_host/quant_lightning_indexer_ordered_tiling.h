/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_ordered_tiling.h
 * \brief
 */

#ifndef quant_lightning_indexer_ordered_TILING_H_
#define quant_lightning_indexer_ordered_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "../op_kernel/quant_lightning_indexer_ordered_tiling_data.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "platform/soc_spec.h"
namespace optiling {
// ------------------公共定义--------------------------
struct QLIOrderedTilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct QLIOrderedTilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class QLIOrderedDataLayout : uint32_t { BSND = 0, TND = 1, PA_BSND = 2 };

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t QUERY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 6;
constexpr uint32_t BLOCK_TABLE_INDEX = 7;
constexpr uint32_t quant_lightning_indexer_ordered = 0;
// Attributes Index
constexpr uint32_t ATTR_QUERY_QUANT_MODE_INDEX = 0;
constexpr uint32_t ATTR_KEY_QUANT_MODE_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 5;
constexpr uint32_t ATTR_PRE_TOKENS_INDEX = 6;
constexpr uint32_t ATTR_NEXT_TOKENS_INDEX = 7;
constexpr uint32_t ATTR_KEY_STRIDE0_INDEX = 8;
constexpr uint32_t ATTR_KEY_DEQUANT_SCALE_STRIDE0_INDEX = 9;
// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t G_SIZE_LIMIT = 64;
constexpr uint32_t G_SIZE_LIMIT_950 = 24;
constexpr uint32_t G_SIZE_LIMIT_32_950 = 32;
constexpr uint32_t G_SIZE_LIMIT_16_950 = 16;  // ascend950只支持 G = 8/16/24/32/64
constexpr uint32_t G_SIZE_LIMIT_8_950 = 8;
constexpr uint32_t BLOCK_SIZE_LIMIT = 1024;
constexpr uint32_t BLOCK_SIZE_FACTOR = 16;
constexpr uint32_t SPARSE_MODE_LOWER = 3;

// -----------算子TilingData定义---------------

// -----------算子CompileInfo定义-------------------
struct QLIOrderedCompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct QLIOrderedParaInfo {
    QLIOrderedTilingRequiredParaInfo query = {nullptr, nullptr};
    QLIOrderedTilingRequiredParaInfo key = {nullptr, nullptr};
    QLIOrderedTilingRequiredParaInfo weights = {nullptr, nullptr};
    QLIOrderedTilingRequiredParaInfo query_dequant_scale = {nullptr, nullptr};
    QLIOrderedTilingRequiredParaInfo key_dequant_scale = {nullptr, nullptr};
    QLIOrderedTilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    QLIOrderedTilingOptionalParaInfo actualSeqLengthsK = {nullptr, nullptr};
    QLIOrderedTilingOptionalParaInfo blockTable = {nullptr, nullptr};
    QLIOrderedTilingRequiredParaInfo attenOut = {nullptr, nullptr};

    const int32_t *queryQuantMode = nullptr;
    const int32_t *keyQuantMode = nullptr;
    const char *layOutQuery = nullptr;
    const char *layOutKey = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *sparseMode = nullptr;
    const int32_t *sparseCount = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    int64_t keyStride0 = 0;
    int64_t keyDequantScaleStride0 = 0;
};

// -----------算子Tiling入参信息类---------------
class QLIOrderedTilingInfo
{
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QLIOrderedParaInfo opParamInfo;
    // Base Param
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t tSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t gSize = 0;
    uint32_t qkHeadDim = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    int64_t keyStride0 = 0;
    int64_t keyDequantScaleStride0 = 0;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    QLIOrderedDataLayout inputQLayout = QLIOrderedDataLayout::BSND;
    QLIOrderedDataLayout inputKLayout = QLIOrderedDataLayout::PA_BSND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class QLIOrderedInfoParser
{
public:
    explicit QLIOrderedInfoParser(gert::TilingContext *context) : context_(context) {}
    ~QLIOrderedInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckTensorShapes() const;
    ge::graphStatus CheckTensorDescriptions() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckQliRequiredParameters() const;
    ge::graphStatus GetActualSeqLenSize(int64_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName) const;
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus CheckAttrParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus CheckScaleShape();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetActualSeqInfo();
    void GenerateInfo(QLIOrderedTilingInfo &QLIInfo);
    ge::graphStatus ParseAndCheck(QLIOrderedTilingInfo &QLIInfo);
    size_t GetTensorDimNum(const uint32_t tensorIdx);
    int64_t GetTensorDim(const uint32_t tensorIdx, const size_t idx);
    ge::graphStatus CheckContiguous();

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    QLIOrderedParaInfo qliParams_;

    // BaseParams
    int64_t bSize_ = 0;
    int64_t tSize_ = 0;
    int64_t n1Size_ = 0;
    int64_t n2Size_ = 0;
    int64_t gSize_ = 0;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t headDim_ = 0;
    // Layout
    QLIOrderedDataLayout qLayout_ = QLIOrderedDataLayout::BSND;
    QLIOrderedDataLayout kLayout_ = QLIOrderedDataLayout::PA_BSND;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType inputQueryScaleType_ = ge::DT_FLOAT16;
    ge::DataType inputKeyScaleType_ = ge::DT_FLOAT16;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
};

// ---------------算子Tiling类---------------
class QuantLightningIndexerOrderedTiling
{
public:
    explicit QuantLightningIndexerOrderedTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus DoTiling(QLIOrderedTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    QLIOrderedTilingData tilingData_;
};

}  // namespace optiling
#endif  // quant_lightning_indexer_ordered_TILING_H_
