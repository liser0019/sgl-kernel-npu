# SPDX-License-Identifier: Apache-2.0
"""Score-based reference for the standalone ordered indexer."""

import torch


def ordered_indexer_reference(
    query,
    key,
    weights,
    q_scale,
    k_scale,
    query_ends,
    key_lengths,
    block_table,
    topk,
    mode,
):
    q = query.cpu().float() * q_scale.cpu().float().unsqueeze(-1)
    k = key.cpu().float().squeeze(2) * k_scale.cpu().float().reshape(
        key.shape[:2]
    ).unsqueeze(-1)
    weights = weights.cpu().float()
    block_table = block_table.cpu()
    result = torch.full((q.shape[0], 1, topk), -1, dtype=torch.int32, device="cpu")
    start = 0
    for request, (end, length) in enumerate(
        zip(query_ends.cpu().tolist(), key_lengths.cpu().tolist())
    ):
        token = torch.arange(length, device="cpu")
        blocks = block_table[request, token // key.shape[1]].long()
        keys = k[blocks, token % key.shape[1]]
        for row in range(start, end):
            visible = length if mode == 0 else max(0, length - (end - row - 1))
            if visible == 0:
                continue
            scores = (
                (q[row] @ keys[:visible].T).relu() * weights[row].unsqueeze(-1)
            ).sum(0)
            count = min(topk, visible)
            selected = scores.argsort(descending=True, stable=True)[:count]
            result[row, 0, :count] = selected.sort().values
        start = end
    return result.to(query.device)
