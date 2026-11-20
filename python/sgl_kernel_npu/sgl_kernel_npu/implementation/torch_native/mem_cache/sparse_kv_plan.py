# SPDX-License-Identifier: Apache-2.0
"""Independent, eager reference for sparse KV Plan."""

import torch


def _classify(cache, lru, selected, stable_limit, max_token):
    hits, victims, owner = [], [], {}
    for slot in lru:
        token = cache[slot]
        if stable_limit <= token < max_token:
            cache[slot] = -1
        if 0 <= token < stable_limit and token in selected:
            hits.append(slot)
            owner[token] = slot
        else:
            victims.append(slot)
    return hits, victims, owner


def _plan(cache, lru, tokens, stable_limit, max_token, allow_shortage):
    hits, victims, owner = _classify(cache, lru, set(tokens), stable_limit, max_token)
    missing = [token for token in tokens if token not in owner]
    if len(missing) > len(victims) and not allow_shortage:
        return None
    descriptors = list(zip(missing, victims))
    for token, slot in descriptors:
        cache[slot] = token
        owner[token] = slot
    lru = victims[len(descriptors) :] + victims[: len(descriptors)] + hits
    return cache, lru, owner, descriptors


def sparse_kv_plan_reference(
    req_ids, topk_indices, stable_lens, visible_lens, token_to_req, active_rows, state
):
    inputs = [
        tensor.cpu().tolist()
        for tensor in (req_ids, topk_indices, stable_lens, visible_lens, token_to_req)
    ]
    requests, queries, stable, visible, request_for_row = inputs
    rows, topk = topk_indices.shape
    capacity = state.slot_to_token.shape[1]
    pages = capacity // state.block_size
    width = rows // len(requests)
    active = int(active_rows.item())
    status = state.status
    if not 0 <= active <= rows or active % width:
        status[0] = -1
        return

    outputs = [
        tensor.cpu().tolist()
        for tensor in (
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            state.current_slots,
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            state.row_modes,
            state.resident_rows,
            state.resident_block_table,
        )
    ]
    last, cache, lru, slots, counts, miss_ids, miss_slots, modes, owners, table = (
        outputs
    )

    def plan_rows(first, end, req_id, expected_mode, allow_shortage=False):
        reset = last[first] != req_id or modes[first] != expected_mode
        old_cache = [-1] * capacity if reset else cache[first].copy()
        old_lru = list(range(capacity)) if reset else lru[first]
        if not reset and sorted(old_lru) != list(range(capacity)):
            return False
        selections = [
            {
                value
                for value in queries[r]
                if 0 <= value < min(visible[r], state.max_token)
            }
            for r in range(first, end)
        ]
        selected = sorted(set().union(*selections))
        stable_limit = max(0, min(state.max_token, *stable[first:end]))
        result = _plan(
            old_cache, old_lru, selected, stable_limit, state.max_token, allow_shortage
        )
        if result is None:
            return None
        cache[first], lru[first], token_to_slot, descriptors = result
        last[first] = req_id
        for row in range(first, end):
            row_descriptors = descriptors[
                (row - first) * topk : (row - first + 1) * topk
            ]
            counts[row] = count = len(row_descriptors)
            miss_ids[row] = [token for token, _ in row_descriptors] + [-1] * (
                topk - count
            )
            miss_slots[row] = [slot for _, slot in row_descriptors] + [-1] * (
                topk - count
            )
            slots[row] = [
                token_to_slot.get(token, -1) if token in selections[row - first] else -1
                for token in queries[row]
            ]
            owners[row] = first
            modes[row] = expected_mode if row == first else 2
            table[row] = list(range(first * pages, (first + 1) * pages))
        return True

    for first in range(0, active, width):
        end = first + width
        request = request_for_row[first]
        if (
            not 0 <= request < len(requests)
            or any(v != request for v in request_for_row[first:end])
            or (first > 0 and request_for_row[first - 1] == request)
            or (end < active and request_for_row[end] == request)
        ):
            status[first] = -1
            continue
        req_id = requests[request]
        result = plan_rows(first, end, req_id, int(width > 1))
        if result is None:
            for row in range(first, end):
                plan_rows(row, row + 1, req_id, 0, allow_shortage=True)
        elif not result:
            status[first] = -1

    tensors = (*state._tensor_args()[:7], *state._tensor_args()[8:])
    for tensor, values in zip(tensors, outputs):
        tensor.copy_(torch.tensor(values, dtype=tensor.dtype, device=tensor.device))
