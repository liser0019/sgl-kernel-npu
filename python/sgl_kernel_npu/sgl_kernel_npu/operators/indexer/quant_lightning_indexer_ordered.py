# SPDX-License-Identifier: Apache-2.0
"""Ordered INT8/FP8 Lightning Indexer for paged KV on Ascend 950."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP = "indexer.quant_lightning_indexer_ordered"
_MAX_TOKENS = (1 << 63) - 1


@register_impl(_OP, "auto", supported_generations=[Generation.ARCH35])
@register_impl(_OP, "ops_transformer", supported_generations=[Generation.ARCH35])
def _ordered_indexer(
    query,
    key,
    weights,
    q_scale,
    k_scale,
    query_ends,
    key_lengths,
    block_table,
    topk,
    sparse_mode,
):
    try:
        from ..._third_party.sgl_ops_transformer_ordered_indexer.ops import (
            quant_lightning_indexer_ordered as execute,
        )
    except ModuleNotFoundError as exc:
        package = "sgl_kernel_npu._third_party.sgl_ops_transformer_ordered_indexer"
        if exc.name not in (package, "sgl_kernel_npu._third_party"):
            raise
        raise RuntimeError(
            "Ordered Indexer requires a CANN 9.2.1 wheel built with "
            "--third-party ops_transformer_ordered_indexer"
        ) from exc

    return execute(
        query,
        key,
        weights,
        q_scale,
        k_scale,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_key=key_lengths,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=sparse_mode,
        pre_tokens=_MAX_TOKENS,
        next_tokens=_MAX_TOKENS,
    )


@register_impl(_OP, "torch_native")
def _reference(*args):
    from ...implementation.torch_native.indexer.quant_lightning_indexer_ordered import (
        ordered_indexer_reference,
    )

    return ordered_indexer_reference(*args)


@register_operator(_OP)
def quant_lightning_indexer_ordered(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    topk: int = 2048,
    sparse_mode: int = 3,
    impl: str = "auto",
) -> torch.Tensor:
    """Return score-selected logical KV positions in ascending order with -1 padding.

    The paged key uses PA_BSND and queries use TND. Query lengths are
    cumulative row ends, and key lengths are per-request token counts.
    """
    if not 1 <= topk <= 2048:
        raise ValueError("topk must be in [1, 2048]")
    if sparse_mode not in (0, 3):
        raise ValueError("sparse_mode must be 0 (full) or 3 (causal)")
    if query.ndim != 3 or key.ndim != 4 or key.shape[2] != 1:
        raise ValueError("expected TND queries and PA_BSND keys with a single key head")
    return get_impl_by_name(_OP, impl)(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        topk,
        sparse_mode,
    )
