"""整代查询步幅逐行分解的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from table_diffevo.step_diagnostics import (
    compute_row_query_deltas,
    decompose_query_step,
)


def test_compute_row_query_deltas_for_conjunctions():
    current = pd.DataFrame({
        "a": [0, 0, 1],
        "b": [0, 1, 1],
    })
    proposal = pd.DataFrame({
        "a": [1, 1, 1],
        "b": [1, 0, 0],
    })
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
    ]

    result = compute_row_query_deltas(current, proposal, queries)

    np.testing.assert_array_equal(
        result,
        np.asarray([
            [1, 1, 1],
            [1, -1, 0],
            [0, -1, -1],
        ], dtype=np.int8),
    )


def test_decomposition_with_positive_cross_term():
    result = decompose_query_step(
        np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        np.asarray([3.0, -2.0]),
    )

    np.testing.assert_array_equal(result["delta_q"], [2.0, 0.0])
    assert result["linear_gain"] == 6.0
    assert result["self_penalty"] == 1.0
    assert result["cross_penalty"] == 1.0
    assert result["quadratic_penalty"] == 2.0
    assert result["net_gain"] == 4.0


def test_decomposition_with_negative_cross_term():
    result = decompose_query_step(
        np.asarray([[1.0], [-1.0]]),
        np.asarray([5.0]),
    )

    np.testing.assert_array_equal(result["delta_q"], [0.0])
    assert result["linear_gain"] == 0.0
    assert result["self_penalty"] == 1.0
    assert result["cross_penalty"] == -1.0
    assert result["quadratic_penalty"] == 0.0
    assert result["net_gain"] == 0.0


def test_empty_rows_have_zero_step():
    result = decompose_query_step(
        np.empty((0, 2), dtype=float),
        np.asarray([1.0, -1.0]),
    )

    np.testing.assert_array_equal(result["delta_q"], [0.0, 0.0])
    for key in (
        "linear_gain",
        "self_penalty",
        "cross_penalty",
        "quadratic_penalty",
        "net_gain",
    ):
        assert result[key] == 0.0


@pytest.mark.parametrize(
    "current,proposal,message",
    [
        (pd.DataFrame({"a": [0]}), pd.DataFrame({"a": [0, 1]}), "行数"),
        (pd.DataFrame({"a": [0]}), pd.DataFrame({"b": [0]}), "列名"),
        ("not-a-frame", pd.DataFrame({"a": [0]}), "DataFrame"),
    ],
)
def test_compute_row_query_deltas_rejects_mismatched_frames(
    current, proposal, message
):
    with pytest.raises(ValueError, match=message):
        compute_row_query_deltas(current, proposal, [])


@pytest.mark.parametrize(
    "deltas,residual",
    [
        (np.asarray([1.0, 2.0]), np.asarray([1.0])),
        (np.asarray([["bad"]]), np.asarray([1.0])),
        (np.asarray([[np.nan]]), np.asarray([1.0])),
        (np.asarray([[1.0, 2.0]]), np.asarray([1.0])),
        (np.asarray([[1.0]]), np.asarray([np.inf])),
    ],
)
def test_decomposition_rejects_invalid_arrays(deltas, residual):
    with pytest.raises(ValueError):
        decompose_query_step(deltas, residual)
