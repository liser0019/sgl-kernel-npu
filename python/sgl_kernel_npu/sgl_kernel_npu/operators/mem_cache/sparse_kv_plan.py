# SPDX-License-Identifier: Apache-2.0
"""Allocate reusable state and plan sparse KV placement on Ascend 950."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

TOP_K = 2048
MAX_TOKEN = 358400
_OK = torch.iinfo(torch.int32).max
_CAPACITY = {1: 4096, 2: 4096, 4: 8192}


@dataclass(frozen=True)
class SparseKvPlanState:
    """Fixed-shape cache state and output buffers for one layer and stream."""

    last_req_ids: torch.Tensor
    slot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    current_slots: torch.Tensor
    miss_count: torch.Tensor
    miss_tokens: torch.Tensor
    miss_slots: torch.Tensor
    workspace: torch.Tensor
    row_modes: torch.Tensor
    resident_rows: torch.Tensor
    resident_block_table: torch.Tensor
    max_token: int
    block_size: int

    @property
    def status(self) -> torch.Tensor:
        """Sticky device status per query row, INT32_MAX on success."""
        return self.workspace[: self.last_req_ids.numel()]

    def _tensor_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.last_req_ids,
            self.slot_to_token,
            self.lru_slots,
            self.current_slots,
            self.miss_count,
            self.miss_tokens,
            self.miss_slots,
            self.workspace,
            self.row_modes,
            self.resident_rows,
            self.resident_block_table,
        )


def allocate_sparse_kv_plan_state(
    requests: int,
    rows_per_request: int,
    *,
    max_token: int,
    block_size: int,
    device: torch.device | str,
) -> SparseKvPlanState:
    """Allocate cold state for one layer. Reuse it across decode steps."""
    capacity = _CAPACITY.get(rows_per_request)
    if requests <= 0 or capacity is None:
        raise ValueError(
            "requests must be positive and rows_per_request must be 1, 2 or 4"
        )
    if block_size <= 0 or capacity % block_size or not 0 < max_token <= MAX_TOKEN:
        raise ValueError(
            "capacity must be divisible by block_size and 0 < max_token <= 358400"
        )
    rows = requests * rows_per_request
    pages = capacity // block_size
    if rows * pages > _OK:
        raise ValueError("cache page indices exceed the int32 range")

    def full(shape, value, dtype=torch.int32):
        return torch.full(shape, value, dtype=dtype, device=device)

    return SparseKvPlanState(
        last_req_ids=full((rows,), -1, torch.int64),
        slot_to_token=full((rows, capacity), -1),
        lru_slots=torch.arange(capacity, dtype=torch.int32, device=device).repeat(
            rows, 1
        ),
        current_slots=full((rows, TOP_K), -1),
        miss_count=full((rows,), 0),
        miss_tokens=full((rows, TOP_K), -1),
        miss_slots=full((rows, TOP_K), -1),
        workspace=full((rows * TOP_K // 2,), _OK),
        row_modes=full((rows,), 0),
        resident_rows=full((rows,), -1),
        resident_block_table=full((rows, pages), -1),
        max_token=max_token,
        block_size=block_size,
    )


@register_impl(
    "mem_cache.sparse_kv_plan", "auto", supported_generations=[Generation.ARCH35]
)
@register_impl(
    "mem_cache.sparse_kv_plan", "ascendc", supported_generations=[Generation.ARCH35]
)
def _ascendc_plan(
    req_ids,
    topk_indices,
    stable_prefix_lens,
    visible_seq_lens,
    token_to_req,
    active_rows,
    state,
):
    torch.ops.npu.sgl_sparse_kv_plan(
        req_ids,
        topk_indices,
        stable_prefix_lens,
        visible_seq_lens,
        token_to_req,
        active_rows,
        *state._tensor_args(),
        state.slot_to_token.shape[1],
        state.max_token,
        state.block_size,
    )


@register_impl("mem_cache.sparse_kv_plan", "torch_native")
def _reference_plan(*args):
    from ...implementation.torch_native.mem_cache.sparse_kv_plan import (
        sparse_kv_plan_reference,
    )

    sparse_kv_plan_reference(*args)


@register_operator("mem_cache.sparse_kv_plan")
def sparse_kv_plan(
    req_ids: torch.Tensor,
    topk_indices: torch.Tensor,
    stable_prefix_lens: torch.Tensor,
    visible_seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    active_rows: torch.Tensor,
    state: SparseKvPlanState,
    *,
    impl: str = "auto",
) -> None:
    """Update cache state and write attention slots and miss descriptors in place.

    Valid TopK token positions must be strictly ascending in each row. Keep
    state alive and order KV copies after the Plan before using attention slots.
    """
    get_impl_by_name("mem_cache.sparse_kv_plan", impl)(
        req_ids,
        topk_indices,
        stable_prefix_lens,
        visible_seq_lens,
        token_to_req,
        active_rows,
        state,
    )
