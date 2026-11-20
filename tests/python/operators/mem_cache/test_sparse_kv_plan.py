# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.python.utils.common import get_test_impls, make_executor, require_test_impl
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Resolve bundled vendor paths before the device fixture runs.
import sgl_kernel_npu  # noqa: F401


def generate_data(requests, width, context):
    rows = requests * width
    req_ids = torch.arange(100, 100 + requests, dtype=torch.int64, device="npu")
    topk = torch.arange(2048, dtype=torch.int32, device="npu").repeat(rows, 1)
    # Exercise one partial stream and cross-row duplicates.
    topk[:, :1024] += 17
    topk[:, 1024:] += context - 2048
    if width > 1:
        topk[1::width, -3:] = -1
    stable = torch.full((rows,), context, dtype=torch.int32, device="npu")
    visible = stable.clone()
    token_to_req = torch.arange(
        requests, dtype=torch.int32, device="npu"
    ).repeat_interleave(width)
    active = torch.tensor([rows], dtype=torch.int32, device="npu")
    return req_ids, topk, stable, visible, token_to_req, active


def check_state(actual, expected):
    for actual_tensor, expected_tensor in zip(
        actual._tensor_args(), expected._tensor_args()
    ):
        torch.testing.assert_close(
            actual_tensor.cpu(), expected_tensor.cpu(), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "requests,width,context", [(1, 1, 102400), (2, 2, 262144), (4, 4, 358400)]
)
def test_sparse_kv_plan_cold_hot_and_request_reuse(requests, width, context):
    from sgl_kernel_npu.operators.mem_cache import (
        allocate_sparse_kv_plan_state,
        sparse_kv_plan,
    )

    impl, impl_ref = get_test_impls()
    require_test_impl(sparse_kv_plan, impl)
    args = generate_data(requests, width, context)
    state = allocate_sparse_kv_plan_state(
        requests, width, max_token=context, block_size=64, device="npu"
    )
    golden = allocate_sparse_kv_plan_state(
        requests, width, max_token=context, block_size=64, device="cpu"
    )
    for step in range(3):
        if step == 2:
            args[0][0] = 500
        before = [value.cpu().clone() for value in args]
        sparse_kv_plan(*args, state, impl=impl)
        sparse_kv_plan(*before, golden, impl=impl_ref)
        check_state(state, golden)
        for value, snapshot in zip(args, before):
            torch.testing.assert_close(value.cpu(), snapshot)
        if step == 1:
            assert not state.miss_count.cpu().any()
    # Additional samples reuse the warmed state and measure the all-hit path.
    check_performance(
        make_executor(
            sparse_kv_plan,
            (*args, state),
            accuracy_checked=True,
        ),
        enable_perf=False,
    )


@pytest.mark.parametrize("width", [2, 4])
def test_sparse_kv_plan_duplicate_resident_fallback(width):
    from sgl_kernel_npu.operators.mem_cache import (
        allocate_sparse_kv_plan_state,
        sparse_kv_plan,
    )

    impl, impl_ref = get_test_impls()
    require_test_impl(sparse_kv_plan, impl)
    context = 102400
    args = list(generate_data(1, width, context))
    args[1] = torch.arange(width * 2048, dtype=torch.int32, device="npu").reshape(
        width, 2048
    )
    states = [
        allocate_sparse_kv_plan_state(
            1, width, max_token=context, block_size=64, device=device
        )
        for device in ("npu", "cpu")
    ]
    for state in states:
        state.last_req_ids[0] = 100
        state.row_modes[0] = 1
        # The union fills capacity; protecting both copies of token 0 leaves one slot short.
        state.slot_to_token[0, :2] = 0
    sparse_kv_plan(*args, states[0], impl=impl)
    sparse_kv_plan(*(value.cpu() for value in args), states[1], impl=impl_ref)
    check_state(*states)
    assert states[0].row_modes.cpu().tolist() == [0] * width


def test_sparse_kv_plan_partial_group_is_rejected():
    from sgl_kernel_npu.operators.mem_cache import (
        allocate_sparse_kv_plan_state,
        sparse_kv_plan,
    )

    impl, _ = get_test_impls()
    require_test_impl(sparse_kv_plan, impl)
    state = allocate_sparse_kv_plan_state(
        1, 4, max_token=102400, block_size=64, device="npu"
    )
    args = generate_data(1, 4, 102400)
    args[-1].fill_(3)
    sparse_kv_plan(*args, state, impl=impl)
    assert state.status.cpu()[0] == -1
    assert (state.slot_to_token.cpu() == -1).all()
    assert (state.last_req_ids.cpu() == -1).all()


def test_sparse_kv_plan_rejects_overlapping_outputs():
    from dataclasses import replace

    from sgl_kernel_npu.operators.mem_cache import (
        allocate_sparse_kv_plan_state,
        sparse_kv_plan,
    )

    require_test_impl(sparse_kv_plan, "ascendc")
    state = allocate_sparse_kv_plan_state(
        1, 1, max_token=102400, block_size=64, device="npu"
    )
    aliased = replace(state, miss_slots=state.miss_tokens)
    with pytest.raises(RuntimeError, match="buffers must not overlap"):
        sparse_kv_plan(*generate_data(1, 1, 102400), aliased, impl="ascendc")
