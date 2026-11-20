// SPDX-License-Identifier: Apache-2.0

#include "../op_kernel/sparse_kv_plan_tiling.h"
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>
#include <limits>

namespace {

constexpr std::array<const char *, 6> kInputs{"req_ids",          "topk_indices", "stable_prefix_lens",
                                              "visible_seq_lens", "token_to_req", "active_rows"};
constexpr std::array<const char *, 11> kOutputs{
    "last_req_ids", "slot_to_token",  "lru_slots", "current_slots", "miss_count",          "miss_tokens",
    "miss_slots",   "plan_workspace", "row_modes", "resident_rows", "resident_block_table"};

ge::graphStatus PlanTiling(gert::TilingContext *context)
{
    const auto *requests = context->GetInputShape(0);
    const auto *topk = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const auto *capacity = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *maxToken = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    const auto *blockSize = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    if (!requests || !topk || !capacity || !maxToken || !blockSize || *blockSize <= 0 || *blockSize > INT32_MAX ||
        !context->GetPlatformInfo() || !workspace || !raw || !raw->GetData() ||
        raw->GetCapacity() < sizeof(SparseKvPlanTiling)) {
        return ge::GRAPH_FAILED;
    }

    const auto &requestShape = requests->GetStorageShape();
    const auto &topkShape = topk->GetStorageShape();
    if (requestShape.GetDimNum() != 1 || topkShape.GetDimNum() != 2 || topkShape.GetDim(1) != 2048) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<SparseKvPlanTiling>();
    uint64_t workspaceWords = 0;
    if (!tiling ||
        !sparse_kv_plan_config::MakeJointTiling(requestShape.GetDim(0), topkShape.GetDim(0), *capacity, *maxToken,
                                                static_cast<int32_t>(*blockSize), *tiling, workspaceWords)) {
        return ge::GRAPH_FAILED;
    }
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const auto cores = std::min<int64_t>(64, platform.GetCoreNumAiv());
    if (cores <= 0 || context->SetDynUBufSize(tiling->localMemoryBytes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(static_cast<uint32_t>(std::min(tiling->maxRequests, cores)));
    context->SetTilingKey(0);
    workspace[0] = 0;
    raw->SetDataSize(sizeof(*tiling));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PlanInferShape(gert::InferShapeContext *context)
{
    const auto *topk = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const auto *capacity = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *blockSize = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    if (!topk || topk->GetDimNum() != 2 || !capacity || !blockSize || *blockSize <= 0 ||
        (*capacity != 4096 && *capacity != 8192) || *capacity % *blockSize != 0) {
        return ge::GRAPH_FAILED;
    }
    const auto rows = topk->GetDim(0);
    if (rows > std::numeric_limits<int64_t>::max() / 1024) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape rowShape{rows};
    const gert::Shape cacheShape{rows, *capacity};
    const gert::Shape workspaceShape{rows < 0 ? -1 : rows * 1024};
    const gert::Shape blockShape{rows, *capacity / *blockSize};
    const std::array<gert::Shape, kOutputs.size()> shapes{rowShape, cacheShape, cacheShape, *topk,
                                                          rowShape, *topk,      *topk,      workspaceShape,
                                                          rowShape, rowShape,   blockShape};
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto *output = context->GetOutputShape(i);
        if (!output) {
            return ge::GRAPH_FAILED;
        }
        *output = shapes[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PlanInferDataType(gert::InferDataTypeContext *context)
{
    for (size_t i = 0; i < kOutputs.size(); ++i) {
        context->SetOutputDataType(i, i == 0 ? ge::DT_INT64 : ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace

namespace ops {

class SglSparseKvPlan : public OpDef
{
public:
    explicit SglSparseKvPlan(const char *name) : OpDef(name)
    {
        for (size_t i = 0; i < kInputs.size(); ++i) {
            this->Input(kInputs[i])
                .ParamType(REQUIRED)
                .DataType({i == 0 ? ge::DT_INT64 : ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (size_t i = 0; i < kOutputs.size(); ++i) {
            this->Output(kOutputs[i])
                .ParamType(REQUIRED)
                .DataType({i == 0 ? ge::DT_INT64 : ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Attr("capacity").AttrType(REQUIRED).Int();
        this->Attr("max_token").AttrType(REQUIRED).Int();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        this->SetInferShape(PlanInferShape).SetInferDataType(PlanInferDataType);
        this->AICore().SetTiling(PlanTiling);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};
OP_ADD(SglSparseKvPlan);

}  // namespace ops
