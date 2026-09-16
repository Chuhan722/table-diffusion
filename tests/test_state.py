"""表状态与账本的锚点测试。"""
import numpy as np
import pytest

from conftest import FOUR_FEATURES, FOUR_STATES
from resevo.state import (
    check_state_ids,
    features_from_predicates,
    make_workload,
    table_answers,
    table_loss,
    table_residual,
)


def test_answers_residual_loss(four_record):
    workload, ids = four_record
    assert np.array_equal(table_answers(workload, ids), [2.0, 2.0, 1.0])
    assert np.array_equal(table_residual(workload, ids), [1.0, -1.0, 0.0])
    assert table_loss(workload, ids) == 1.0


def test_manual_step_anchor(four_record):
    # 手算锚，把 01 改成 10 后 q=(3,1,1)，损失 0，该步真实增益为 1
    workload, ids = four_record
    stepped = ids.copy()
    stepped[1] = 2
    assert np.array_equal(table_answers(workload, stepped), [3.0, 1.0, 1.0])
    assert table_loss(workload, stepped) == 0.0
    assert table_loss(workload, ids) - table_loss(workload, stepped) == 1.0


def test_row_permutation_invariance(four_record):
    workload, ids = four_record
    rng = np.random.default_rng(20260911)
    for _ in range(5):
        perm = rng.permutation(ids)
        assert np.array_equal(
            table_answers(workload, perm), table_answers(workload, ids)
        )
        assert table_loss(workload, perm) == table_loss(workload, ids)


def test_features_from_predicates_matches_anchor():
    predicates = [
        lambda s: s[0] == 1,
        lambda s: s[1] == 1,
        lambda s: s[0] == 1 and s[1] == 1,
    ]
    a = features_from_predicates(FOUR_STATES, predicates)
    assert np.array_equal(a, FOUR_FEATURES)


def test_workload_arrays_readonly(four_record):
    workload, _ = four_record
    with pytest.raises(ValueError):
        workload.features[0, 0] = 9.0
    with pytest.raises(ValueError):
        workload.target[0] = 9.0


def test_bad_inputs_rejected():
    good_a = FOUR_FEATURES
    y = np.array([3.0, 1.0, 1.0])
    w = np.ones(3)
    with pytest.raises(ValueError):
        make_workload(np.array([[np.nan, 0, 0]]), y, w)
    with pytest.raises(ValueError):
        make_workload(good_a, np.array([3.0, np.inf, 1.0]), w)
    with pytest.raises(ValueError):
        make_workload(good_a, y, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        make_workload(good_a, np.array([3.0, 1.0]), w)
    workload = make_workload(good_a, y, w)
    with pytest.raises(ValueError):
        check_state_ids(workload, np.array([0, 1, 2, 4]))
    with pytest.raises(ValueError):
        check_state_ids(workload, np.array([0.5, 1.0]))
    with pytest.raises(ValueError):
        check_state_ids(workload, np.array([], dtype=np.int64))
