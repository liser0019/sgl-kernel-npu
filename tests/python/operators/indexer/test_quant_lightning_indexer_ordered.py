# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.python.utils.common import get_test_impls, make_executor, require_test_impl
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Resolve bundled vendor paths before the device fixture runs.
import sgl_kernel_npu  # noqa: F401


@pytest.fixture(autouse=True)
def require_ordered_operator():
    if not hasattr(torch.ops.npu, "sgl_quant_lightning_indexer_ordered"):
        pytest.skip("wheel was built without the native Ordered Indexer")


def generate_data(length, rows, dtype=torch.float8_e4m3fn):
    page_size = 128
    pages = (length + page_size - 1) // page_size
    # Each physical page is in reverse order, so passing physical slot IDs will fail.
    logical = torch.zeros(pages * page_size, dtype=torch.float32)
    logical[:4] = 0.5
    logical[length - 5 : length] = 1.0
    key = logical.view(pages, page_size).flip(0)[:, :, None, None]
    if dtype == torch.int8:
        key = key * 2
    key = key.expand(-1, -1, 1, 128).contiguous().to(dtype)
    query = torch.ones((rows, 16, 128), dtype=torch.float32).to(dtype)
    weight_dtype = torch.float16 if dtype == torch.int8 else torch.bfloat16
    scale_dtype = torch.float16 if dtype == torch.int8 else torch.float32
    weights = torch.ones((rows, 16), dtype=weight_dtype)
    q_scale = torch.ones((rows, 16), dtype=scale_dtype)
    k_scale = torch.ones((pages, page_size, 1), dtype=scale_dtype)
    args = query, key, weights, q_scale, k_scale
    kwargs = {
        "actual_seq_lengths_query": torch.tensor([rows], dtype=torch.int32),
        "actual_seq_lengths_key": torch.tensor([length], dtype=torch.int32),
        "block_table": torch.arange(pages - 1, -1, -1, dtype=torch.int32).view(1, -1),
    }
    return args, kwargs


@pytest.mark.parametrize(
    "length,rows,topk,dtype,padded_pages",
    [
        (32, 1, 64, torch.float8_e4m3fn, False),
        (16385, 4, 2048, torch.float8_e4m3fn, False),
        (256, 2, 64, torch.int8, False),
        (256, 2, 64, torch.float8_e4m3fn, True),
    ],
)
def test_ordered_indexer_logical_token_order(length, rows, topk, dtype, padded_pages):
    from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_ordered

    impl, impl_ref = get_test_impls()
    require_test_impl(quant_lightning_indexer_ordered, impl)
    args, kwargs = generate_data(length, rows, dtype)
    expected = quant_lightning_indexer_ordered(
        *args, **kwargs, topk=topk, impl=impl_ref
    )
    device_args = [value.npu() for value in args]
    if padded_pages:
        for index in (1, 4):
            data = device_args[index]
            storage = torch.empty(
                (data.shape[0] * 2, *data.shape[1:]),
                device=data.device,
                dtype=data.dtype,
            )
            storage[1::2].copy_(data)
            device_args[index] = storage[1::2]
            assert not device_args[index].is_contiguous()
    device_kwargs = {name: value.npu() for name, value in kwargs.items()}
    actual = quant_lightning_indexer_ordered(
        *device_args, **device_kwargs, topk=topk, impl=impl
    )
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    check_performance(
        make_executor(
            quant_lightning_indexer_ordered,
            device_args,
            kwargs={**device_kwargs, "topk": topk},
            accuracy_checked=True,
        ),
        enable_perf=False,
    )


def test_ordered_indexer_to_plan():
    from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_ordered
    from sgl_kernel_npu.operators.mem_cache import (
        allocate_sparse_kv_plan_state,
        sparse_kv_plan,
    )

    require_test_impl(quant_lightning_indexer_ordered, "auto")
    require_test_impl(sparse_kv_plan, "auto")
    if not hasattr(torch.ops.npu, "sgl_sparse_kv_plan"):
        pytest.skip("wheel was built without Sparse KV Plan")
    length, width = 32, 4
    args, kwargs = generate_data(length, width)
    indices = quant_lightning_indexer_ordered(
        *(value.npu() for value in args),
        **{name: value.npu() for name, value in kwargs.items()}
    ).squeeze(1)
    state = allocate_sparse_kv_plan_state(
        1, width, max_token=length, block_size=64, device=indices.device
    )
    request = torch.tensor([100], dtype=torch.int64, device=indices.device)
    visible = torch.arange(
        length - width + 1, length + 1, dtype=torch.int32, device=indices.device
    )
    stable = torch.full_like(visible, length)
    token_to_req = torch.zeros_like(visible)
    active = torch.tensor([width], dtype=torch.int32, device=indices.device)

    for expected_misses in ([length, 0, 0, 0], [0, 0, 0, 0]):
        sparse_kv_plan(request, indices, stable, visible, token_to_req, active, state)
        assert state.miss_count.cpu().tolist() == expected_misses
        assert state.resident_rows.cpu().tolist() == [0] * width
        assert (state.status.cpu() == torch.iinfo(torch.int32).max).all()
        # Cold slots follow ascending position. The same slots serve next-step hits.
        torch.testing.assert_close(
            state.current_slots.cpu(), indices.cpu(), rtol=0, atol=0
        )
