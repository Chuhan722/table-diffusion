"""转移增益的锚点测试，含 16 项暴力对拍与双行块补偿例子。"""
import numpy as np
import pytest

from conftest import FOUR_GAIN_MATRIX
from resevo.gain import block_deltas, block_gains, gain_by_replay
from resevo.state import make_workload, table_residual


ALL_OUTCOMES = ((0,), (1,), (2,), (3,))


def gain_matrix(workload, ids):
    """按行算 4×4 增益矩阵，行=源状态，列=目标状态。"""
    e = table_residual(workload, ids)
    rows = []
    for i in range(4):
        src = (int(ids[i]),)
        outcomes = (src,) + tuple(u for u in ALL_OUTCOMES if u != src)
        d = block_deltas(workload.features, src, outcomes)
        g = block_gains(d, e, workload.weights)
        row = np.empty(4)
        for u, gv in zip(outcomes, g):
            row[u[0]] = gv
        rows.append(row)
    return np.array(rows)


def test_gain_matrix_matches_anchor(four_record):
    workload, ids = four_record
    assert np.allclose(gain_matrix(workload, ids), FOUR_GAIN_MATRIX, atol=1e-12)


def test_source_gain_is_zero(four_record):
    workload, ids = four_record
    g = gain_matrix(workload, ids)
    assert np.array_equal(np.diag(g), np.zeros(4))


def test_all_16_entries_match_brute_force(four_record):
    # 公式版与暴力重放版互相独立，16 项必须逐项一致
    workload, ids = four_record
    g = gain_matrix(workload, ids)
    for i in range(4):
        for u in range(4):
            replay = gain_by_replay(workload, ids, (i,), (u,))
            assert abs(g[i, u] - replay) < 1e-12


def test_flipped_residual_breaks_anchor(four_record):
    # 残差符号写反时增益矩阵必须对不上锚点
    workload, ids = four_record
    e = -table_residual(workload, ids)
    d = block_deltas(workload.features, (int(ids[0]),), ALL_OUTCOMES)
    g = block_gains(d, e, workload.weights)
    assert not np.allclose([g[2]], [FOUR_GAIN_MATRIX[0, 2]], atol=1e-12)


def test_dropping_quadratic_term_breaks_anchor(four_record):
    # 删掉二次项的线性打分必须对不上锚点
    workload, ids = four_record
    e = table_residual(workload, ids)
    d = block_deltas(workload.features, (int(ids[0]),), ALL_OUTCOMES)
    linear_only = d @ (workload.weights * e)
    assert not np.allclose(linear_only, FOUR_GAIN_MATRIX[0], atol=1e-12)


def test_one_dimensional_counterexample():
    # 一维反例，e=0.2 动作加 1，线性项 0.2 为正，真实增益 -0.3 为负
    a = np.array([[0.0], [1.0]])
    workload = make_workload(a, np.array([0.2]), np.ones(1))
    ids = np.array([0], dtype=np.int64)
    e = table_residual(workload, ids)
    d = block_deltas(a, (0,), ((0,), (1,)))
    g = block_gains(d, e, workload.weights)
    linear = d @ (workload.weights * e)
    assert abs(linear[1] - 0.2) < 1e-15
    assert abs(g[1] - (-0.3)) < 1e-15
    assert abs(gain_by_replay(workload, ids, (0,), (1,)) - (-0.3)) < 1e-12


def test_two_row_block_compensation_anchor():
    # 双行块补偿例子，单行增益全负，联合动作增益 1/2
    a = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    workload = make_workload(a, np.array([1.0, 1.0, 0.0]), np.array([2.0, 2.0, 1.0]))
    ids = np.array([0, 3], dtype=np.int64)
    e = table_residual(workload, ids)

    single_anchors = {
        (0, (1,)): -1.0,
        (0, (2,)): -1.0,
        (0, (3,)): -3.5,
        (1, (0,)): -1.5,
        (1, (1,)): -0.5,
        (1, (2,)): -0.5,
    }
    for (row, outcome), expected in single_anchors.items():
        src = (int(ids[row]),)
        d = block_deltas(a, src, (src, outcome))
        g = block_gains(d, e, workload.weights)
        assert abs(g[1] - expected) < 1e-12
        assert abs(gain_by_replay(workload, ids, (row,), outcome) - expected) < 1e-12

    # 联合动作 (00,11)->(01,10)，G12=G1+G2-d1^T W d2=1/2
    src = (0, 3)
    joint = (1, 2)
    d = block_deltas(a, src, (src, joint))
    g = block_gains(d, e, workload.weights)
    assert abs(g[1] - 0.5) < 1e-12
    assert abs(gain_by_replay(workload, ids, (0, 1), joint) - 0.5) < 1e-12

    d1 = a[1] - a[0]
    d2 = a[2] - a[3]
    cross = float(np.dot(d1 * workload.weights, d2))
    assert abs(cross - (-2.0)) < 1e-15
    g1 = single_anchors[(0, (1,))]
    g2 = single_anchors[(1, (2,))]
    assert abs((g1 + g2 - cross) - 0.5) < 1e-12
