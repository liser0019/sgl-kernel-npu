/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_LIGHTNING_INDEXER_ORDERED_TOPK_H
#define QUANT_LIGHTNING_INDEXER_ORDERED_TOPK_H

#include <type_traits>
#include "kernel_operator.h"

namespace qliOrderedTopk {
namespace Reg = AscendC::Reg;

// The final radix histogram is a CDF within the selected byte prefix.
// Preserve its residual rank before FindKth overwrites it with the score key.
__simd_vf__ inline void FindTieQuotaVFImpl(__ubuf__ uint32_t *quotaBuf, __ubuf__ uint32_t *histogramsBuf,
                                           __ubuf__ uint32_t *rankBuf)
{
    Reg::MaskReg mask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg eligible;
    Reg::RegTensor<uint32_t> rank;
    Reg::RegTensor<uint32_t> counts;
    Reg::RegTensor<uint32_t> candidates;
    Reg::RegTensor<uint32_t> limit;
    Reg::RegTensor<uint32_t> minimum;
    Reg::RegTensor<uint32_t> selectedCount;
    Reg::RegTensor<uint32_t> quota;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(rank, rankBuf);
    Reg::Duplicate(limit, static_cast<uint32_t>(0xffffffffU));
    Reg::Duplicate(minimum, static_cast<uint32_t>(0xffffffffU));
    for (uint16_t i = 0; i < 4; ++i) {
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(counts, histogramsBuf + i * 64);
        Reg::Compare<uint32_t, AscendC::CMPMODE::GE>(eligible, counts, rank, mask);
        Reg::Select(candidates, counts, limit, eligible);
        Reg::Min(minimum, minimum, candidates, mask);
    }
    Reg::Reduce<Reg::ReduceType::MIN, uint32_t, uint32_t>(selectedCount, minimum, mask);
    Reg::Duplicate(quota, selectedCount, mask);
    // CDF(threshold) - residualRank + 1 == K - count(score > threshold).
    Reg::Sub(quota, quota, rank, mask);
    Reg::Adds(quota, quota, static_cast<uint32_t>(1), mask);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(quotaBuf, quota, mask);
}

// outputIdxBuf has candidate capacity: first use it for EQ indices, then for Top-K.
// vfLoop covers the aligned candidate buffer, including its original padding.
template <typename T>
__simd_vf__ inline void SelectOrderedIndicesVFImpl(__ubuf__ T *outputIdxBuf, __ubuf__ T *inputValueBuf,
                                                   __ubuf__ uint32_t *kValue, __ubuf__ uint32_t *quotaBuf,
                                                   uint16_t vfLoop)
{
    using IndexType = typename std::conditional<sizeof(T) == sizeof(uint16_t), int16_t, int32_t>::type;
    constexpr uint16_t repeatSize = 256 / sizeof(T);
    constexpr Reg::LoadDist broadcast =
        sizeof(T) == sizeof(uint16_t) ? Reg::LoadDist::DIST_BRC_B16 : Reg::LoadDist::DIST_BRC_B32;
    Reg::MaskReg mask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> kthValue;
    Reg::RegTensor<uint32_t> quota;
    Reg::LoadAlign<uint32_t, broadcast>(kthValue, kValue);
    Reg::LoadAlign<uint32_t, broadcast>(quota, quotaBuf);
    __ubuf__ T *scratchBase = outputIdxBuf;
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore equalStore;
    Reg::RegTensor<T> values;
    Reg::RegTensor<IndexType> indices;
    Reg::RegTensor<T> packed;
    Reg::MaskReg equal;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::Arange(indices, i * repeatSize);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(values, inputValueBuf + i * repeatSize);
        Reg::Compare<T, AscendC::CMPMODE::EQ>(equal, values, (Reg::RegTensor<T> &)kthValue, mask);
        Reg::Squeeze<T, Reg::GatherMaskMode::STORE_REG>(packed, (Reg::RegTensor<T> &)indices, equal);
        Reg::StoreUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, packed, equalStore);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, equalStore);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

    // A Kth threshold guarantees 1 <= quota <= count(EQ). Gather its cutoff once.
    Reg::RegTensor<T> one;
    Reg::RegTensor<T> lookup;
    Reg::RegTensor<T> cutoff;
    Reg::Duplicate(one, static_cast<T>(1));
    Reg::Sub(lookup, (Reg::RegTensor<T> &)quota, one, mask);
    Reg::Gather(cutoff, scratchBase, lookup, mask);
    // Capture the cutoff before overwriting the same UB storage with final indices.
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
    outputIdxBuf = scratchBase;
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore store;
    Reg::MaskReg greater;
    Reg::MaskReg beforeTieEnd;
    Reg::MaskReg selected;
    for (uint16_t i = 0; i < vfLoop; ++i) {
        Reg::Arange(indices, i * repeatSize);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(values, inputValueBuf + i * repeatSize);
        Reg::Compare<T, AscendC::CMPMODE::GT>(greater, values, (Reg::RegTensor<T> &)kthValue, mask);
        Reg::Compare<T, AscendC::CMPMODE::EQ>(equal, values, (Reg::RegTensor<T> &)kthValue, mask);
        Reg::Compare<T, AscendC::CMPMODE::LE>(beforeTieEnd, (Reg::RegTensor<T> &)indices, cutoff, mask);
        Reg::And(equal, equal, beforeTieEnd, mask);
        Reg::Or(selected, greater, equal, mask);
        Reg::Squeeze<T, Reg::GatherMaskMode::STORE_REG>(packed, (Reg::RegTensor<T> &)indices, selected);
        Reg::StoreUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(outputIdxBuf, packed, store);
    }
    Reg::StoreUnAlignPost(outputIdxBuf, store);
}
}  // namespace qliOrderedTopk

#endif
