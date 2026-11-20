/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_topk.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_TOPK_H
#define QUANT_LIGHTNING_INDEXER_TOPK_H

#include "kernel_operator.h"
#include "../../shared/lightning_indexer/lightning_indexer_common.h"
#include "../../shared/lightning_indexer_v2/arch35/common/vf/lightning_indexer_v2_topk_base.h"
#include "../../shared/lightning_indexer/arch35/vf/vf_topk_gather.h"
#include "../../shared/lightning_indexer/arch35/vf/common/vf_topk_16_gather.h"
#include "quant_lightning_indexer_ordered_topk.h"

namespace topk {
template <typename T>
class LITopk
{
public:
    __aicore__ inline void operator()(LocalTensor<uint32_t> &outputIdxLocal, LocalTensor<T> &inputLocal,
                                      uint32_t s2SeqLen)
    {}
};
template <>
class LITopk<uint32_t>
{
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        return liV2TopkCommon::GetGatherTmpBufferSize<uint32_t, liV2TopkCommon::B32_RADIX_BUFFER_NUM>(qliTopKCount,
                                                                                                      qliTrunkLength);
    }

    __aicore__ inline void Init(uint32_t qliTopKCount, uint32_t qliTrunkLength)
    {
        this->qliTopKCount = qliTopKCount;
        this->qliTrunkLength = qliTrunkLength;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t> &qliSharedBuffer, LocalTensor<uint32_t> &qliIndicesOut)
    {
        LocalTensor<uint32_t> hisIndex1 = qliIndicesOut;
        LocalTensor<uint32_t> hisIndex2 = qliSharedBuffer[0];
        hisIndexLocal[0] = hisIndex1;
        hisIndexLocal[1] = hisIndex2;
        histogramsLocal = hisIndex2[QLICommon::Align(qliTopKCount, (uint32_t)256)];  // 256:topk对齐256
        idxLocal0 = histogramsLocal[256];                                            // 256: 本地内存对齐基线
        idxLocal1 = idxLocal0[256];                                                  // 256: 同上
        idxLocal2 = idxLocal1[256];                                                  // 256: 同上
        idxLocal3 = idxLocal2[256];                                                  // 256: 同上
        nkValueLocal = idxLocal3[256];                                               // 256: 同上
        tmpIndexLocal = nkValueLocal[64];  // 64: 单核/单线程输出元素容量（G维度切分阈值）
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t> &mrgValueLocal, LocalTensor<uint32_t> &qliIndicesOut,
                                      LocalTensor<uint32_t> &hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx,
                                      uint32_t s2LoopNum)
    {
        if (s2LoopNum == 1) {
            SelectOrderedTopK<false>(mrgValueLocal, hisValueLocal, s2SeqLen);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(qliIndicesOut, tmpIndexLocal,
                              QLICommon::Align(qliTopKCount, (uint32_t)256));  // 256:topk对齐256
        } else {
            if (loopIdx == 0) {
                SelectOrderedTopK<true>(mrgValueLocal, hisValueLocal, s2SeqLen);
                PipeBarrier<PIPE_V>();
                AscendC::DataCopy(hisIndexLocal[(loopIdx + 1) % 2],                               // 2:pingpong
                                  tmpIndexLocal, QLICommon::Align(qliTopKCount, (uint32_t)256));  // 256:topk对齐256
            } else {
                // Scores are consumed only by a subsequent chunk.
                if (loopIdx == s2LoopNum - 1) {
                    SelectOrderedTopK<false>(mrgValueLocal, hisValueLocal, s2SeqLen);
                } else {
                    SelectOrderedTopK<true>(mrgValueLocal, hisValueLocal, s2SeqLen);
                }
                PipeBarrier<PIPE_V>();
                topkb32gather::LiTopKGatherVF(
                    hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, mrgValueLocal,  // 2:pingpong
                    tmpIndexLocal, hisIndexLocal[loopIdx % 2],                       // 2:pingpong
                    qliTopKCount,
                    loopIdx * qliTrunkLength - QLICommon::Align(qliTopKCount, (uint32_t)256),  // 256:topk对齐256
                    s2SeqLen);
                if (loopIdx == s2LoopNum - 1) {
                    PipeBarrier<PIPE_V>();
                    if ((loopIdx + 1) % 2 == 1) {                                           // 2:pingpong
                        AscendC::DataCopy(qliIndicesOut, hisIndexLocal[(loopIdx + 1) % 2],  // 2:pingpong
                                          QLICommon::Align(qliTopKCount, (uint32_t)256));   // 256:topk对齐256
                    }
                }
            }
        }
    }

private:
    template <bool OUTPUT_VALUE>
    __aicore__ inline void SelectOrderedTopK(LocalTensor<uint32_t> &input, LocalTensor<uint32_t> &values,
                                             uint32_t validLen)
    {
        auto *inputBuf = (__ubuf__ uint32_t *)input.GetPhyAddr();
        auto *histBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
        auto *idx0 = (__ubuf__ uint32_t *)idxLocal0.GetPhyAddr();
        auto *idx1 = (__ubuf__ uint32_t *)idxLocal1.GetPhyAddr();
        auto *idx2 = (__ubuf__ uint32_t *)idxLocal2.GetPhyAddr();
        auto *idx3 = (__ubuf__ uint32_t *)idxLocal3.GetPhyAddr();
        auto *rankBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();
        uint16_t histogramLoops = (validLen + 255) / 256;
        topkb32gather::HistogramsFirstVFImpl<uint32_t>(histBuf, inputBuf, histogramLoops, true);
        topkb32gather::FindFirstTargetBinVFImpl(idx0, rankBuf, histBuf, validLen - qliTopKCount + 1);
        topkb32gather::HistogramsSecondVFImpl<uint32_t>(histBuf, inputBuf, idx0, histogramLoops, true);
        topkb32gather::FindSecondTargetBinVFImpl(idx1, rankBuf, rankBuf, histBuf);
        topkb32gather::HistogramsThirdVFImpl<uint32_t>(histBuf, inputBuf, idx0, idx1, histogramLoops, true);
        topkb32gather::FindThirdTargetBinVFImpl(idx2, rankBuf, rankBuf, histBuf);
        topkb32gather::HistogramsLastVFImpl<uint32_t>(histBuf, inputBuf, idx0, idx1, idx2, histogramLoops, true);
        // Only idx0[0] remains live. Reuse a disjoint vector for the tie quota.
        qliOrderedTopk::FindTieQuotaVFImpl(idx0 + 64, histBuf, rankBuf);
        topkb32gather::FindKthVFImpl(rankBuf, histBuf, idx0, idx1, idx2, idx3);
        uint32_t tailCount = QLICommon::Align(qliTopKCount, (uint32_t)64) - qliTopKCount / 64 * 64;
        AscendC::Duplicate(tmpIndexLocal[qliTopKCount / 64 * 64], (uint32_t)0, tailCount);
        PipeBarrier<PIPE_V>();
        qliOrderedTopk::SelectOrderedIndicesVFImpl<uint32_t>((__ubuf__ uint32_t *)tmpIndexLocal.GetPhyAddr(), inputBuf,
                                                             rankBuf, idx0 + 64, (validLen + 63) / 64);
        if constexpr (OUTPUT_VALUE) {
            // The next chunk must see historical scores paired with the ordered indices.
            PipeBarrier<PIPE_V>();
            topkb32gather::FindValueOutputVFImpl(
                (__ubuf__ uint32_t *)values.GetPhyAddr(), (__ubuf__ uint32_t *)input.GetPhyAddr(),
                (__ubuf__ uint32_t *)tmpIndexLocal.GetPhyAddr(), (qliTopKCount + 63) / 64);
        }
    }

    LocalTensor<uint32_t> hisIndexLocal[2];  // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;   // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal0;         // 输入数据第1个8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal1;         // 输入数据第2个8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal2;         // 输入数据第3个8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLocal3;         // 输入数据第4个8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;      // next_k 暂存Buf 64 * 4B
    LocalTensor<uint32_t> tmpIndexLocal;     // 每trunkLen + topK的临时index
    uint32_t qliTopKCount = 512U;
    uint32_t qliTrunkLength = 8192;
};

template <>
class LITopk<uint16_t>
{
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        return liV2TopkCommon::GetGatherTmpBufferSize<uint16_t, liV2TopkCommon::B16_RADIX_BUFFER_NUM>(qliTopKCount,
                                                                                                      qliTrunkLength);
    }

    __aicore__ inline void Init(uint32_t qliTopK, uint32_t qliTrunkLen)
    {
        this->qliTopKCount = qliTopK;
        this->qliTrunkLength = qliTrunkLen;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t> &qliSharedBuffer, LocalTensor<uint32_t> &qliIndicesOut)
    {
        // 256: 将 topK 实际分配容量向上取整至 256 的倍数
        LocalTensor<uint32_t> hisIndexLocal1 = qliIndicesOut;
        LocalTensor<uint32_t> hisIndexLocal2 = qliSharedBuffer[0];
        hisIndexLocal[0] = hisIndexLocal1;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = hisIndexLocal2[QLICommon::Align(qliTopKCount, (uint32_t)256)];  // 256: 同上
        idxHighLocal = histogramsLocal[256];                        // 256: 本地内存对齐基线步长。
        idxLowLocal = idxHighLocal[256];                            // 256: 延续对齐基线步长
        nkValueLocal = idxLowLocal[256];                            // 256: 固定偏移步长
        LocalTensor<uint32_t> tmpIndexLocalTmp = nkValueLocal[64];  // 64: 单核/单线程输出容量
        tmpIndexLocal = tmpIndexLocalTmp.template ReinterpretCast<uint16_t>();
    }

    __aicore__ inline void operator()(LocalTensor<uint16_t> &mrgValueLocal, LocalTensor<uint32_t> &qliIndicesOut,
                                      LocalTensor<uint16_t> &hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx,
                                      uint32_t s2LoopNum)
    {
        if (s2LoopNum == 1) {
            SelectOrderedTopK<false>(mrgValueLocal, hisValueLocal, s2SeqLen);
            PipeBarrier<PIPE_V>();
            Cast(qliIndicesOut, tmpIndexLocal, RoundMode::CAST_NONE, qliTopKCount);
            return;
        }

        if (loopIdx == 0) {
            SelectOrderedTopK<true>(mrgValueLocal, hisValueLocal, s2SeqLen);
            PipeBarrier<PIPE_V>();
            Cast(hisIndexLocal[(loopIdx + 1) % 2], tmpIndexLocal, RoundMode::CAST_NONE,
                 qliTopKCount);  // 2: 双缓冲的“对侧”Bank
        } else {
            // Scores are consumed only by a subsequent chunk.
            if (loopIdx == s2LoopNum - 1) {
                SelectOrderedTopK<false>(mrgValueLocal, hisValueLocal, s2SeqLen);
            } else {
                SelectOrderedTopK<true>(mrgValueLocal, hisValueLocal, s2SeqLen);
            }
            PipeBarrier<PIPE_V>();
            topkb16gather::LiTopKGatherVF(hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal,  // 2: 双缓冲的Bank
                                          mrgValueLocal, tmpIndexLocal, hisIndexLocal[loopIdx % 2],  // 2: 双缓冲的Bank
                                          qliTopKCount,
                                          loopIdx * qliTrunkLength - QLICommon::Align(qliTopKCount, (uint32_t)256),
                                          s2SeqLen);  // 256: 硬件对齐粒度
            if (loopIdx == s2LoopNum - 1) {
                PipeBarrier<PIPE_V>();
                if ((loopIdx + 1) % 2 == 1) {  // 1: 直接访问 Bank 1; 2: 双缓冲总数
                    AscendC::DataCopy(
                        qliIndicesOut,
                        hisIndexLocal[(loopIdx + 1) % 2],                // 1: 直接访问 Bank 1; 2: 双缓冲总数
                        QLICommon::Align(qliTopKCount, (uint32_t)256));  // 256: 硬件对齐粒度; 2: 双缓冲的Bank
                }
            }
        }
    }

private:
    template <bool OUTPUT_VALUE>
    __aicore__ inline void SelectOrderedTopK(LocalTensor<uint16_t> &input, LocalTensor<uint16_t> &values,
                                             uint32_t validLen)
    {
        auto *inputBuf = (__ubuf__ uint16_t *)input.GetPhyAddr();
        auto *histBuf = (__ubuf__ uint32_t *)histogramsLocal.GetPhyAddr();
        auto *idxHigh = (__ubuf__ uint32_t *)idxHighLocal.GetPhyAddr();
        auto *idxLow = (__ubuf__ uint32_t *)idxLowLocal.GetPhyAddr();
        auto *rankBuf = (__ubuf__ uint32_t *)nkValueLocal.GetPhyAddr();
        uint16_t histogramLoops = (validLen + 255) / 256;
        topkb16gather::HistogramsHighVFImpl<uint16_t>(histBuf, inputBuf, histogramLoops, true);
        topkb16gather::FindHighTargetBinVFImpl(idxHigh, rankBuf, histBuf, validLen - qliTopKCount + 1);
        topkb16gather::HistogramsLowVFImpl<uint16_t>(histBuf, inputBuf, idxHigh, histogramLoops, true);
        // Only idxHigh[0] remains live. Keep quota separate from output-tail initialization.
        qliOrderedTopk::FindTieQuotaVFImpl(idxHigh + 64, histBuf, rankBuf);
        topkb16gather::FindKthVFImpl(rankBuf, histBuf, idxHigh, idxLow);
        uint32_t tailCount = QLICommon::Align(qliTopKCount, (uint32_t)128) - qliTopKCount / 128 * 128;
        AscendC::Duplicate(tmpIndexLocal[qliTopKCount / 128 * 128], (uint16_t)0, tailCount);
        PipeBarrier<PIPE_V>();
        qliOrderedTopk::SelectOrderedIndicesVFImpl<uint16_t>((__ubuf__ uint16_t *)tmpIndexLocal.GetPhyAddr(), inputBuf,
                                                             rankBuf, idxHigh + 64, (validLen + 127) / 128);
        if constexpr (OUTPUT_VALUE) {
            PipeBarrier<PIPE_V>();
            topkb16gather::FindValueOutputVFImpl(
                (__ubuf__ uint16_t *)values.GetPhyAddr(), (__ubuf__ uint16_t *)input.GetPhyAddr(),
                (__ubuf__ uint16_t *)tmpIndexLocal.GetPhyAddr(), (qliTopKCount + 127) / 128);
        }
    }

    LocalTensor<uint32_t> hisIndexLocal[2];  // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;   // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idxHighLocal;      // 输入数据高8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLowLocal;       // 输入数据低8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;      // next_k 暂存Buf 64 * 4B
    LocalTensor<uint16_t> tmpIndexLocal;     // 每trunkLen + topK的临时index
    uint32_t qliTopKCount = 512U;
    uint32_t qliTrunkLength = 16384;
};
}  // namespace topk
#endif
