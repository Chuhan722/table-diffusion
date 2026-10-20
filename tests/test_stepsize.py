"""整代矩与解析步长的锚点测试。"""
import numpy as np
import pytest

from conftest import build_four_record_blocks
from resevo.stepsize import (
    analytic_step,
    apply_row_budget,
    block_drifts,
    block_leave_rates,
    cross_interaction,
    cross_interaction_slow,
    expected_changed_rows_at_unit_step,
    rates_from_probabilities,
)
from resevo.tilt import calibrate_beta


def four_record_moments():
    workload, ids, blocks = build_four_record_blocks()
    gains = [g for _, _, g in blocks]
    refs = [nb.reference for nb, _, _ in blocks]
    deltas = [d for _, d, _ in blocks]
    outcomes = [nb.outcomes for nb, _, _ in blocks]
    tilt = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    rates = rates_from_probabilities(tilt.probabilities)
    return workload, tilt, rates, deltas, outcomes


def test_leave_rates_anchor():
    _, _, rates, _, _ = four_record_moments()
    b = block_leave_rates(rates)
    anchor = np.array([0.213419, 0.696332, 0.000107527, 0.0358140])
    assert np.allclose(b, anchor, atol=1e-6)


def test_interaction_and_step_anchor():
    workload, tilt, rates, deltas, _ = four_record_moments()
    drifts = block_drifts(rates, deltas)
    C = cross_interaction(drifts, workload.weights)
    assert abs(C - 0.1548345932) < 1e-9
    b = block_leave_rates(rates)
    h = analytic_step(float(b.max()), tilt.direction_gain, C)
    # D/(2C) 约 2.42 大于可行上限，步长取 h_feas=1/max b
    assert tilt.direction_gain / (2 * C) > 1.0 / b.max()
    assert abs(h - 1.4360970925) < 1e-9
    assert abs(h - 1.0 / b.max()) < 1e-15


def test_fast_slow_interaction_agree():
    workload, _, rates, deltas, _ = four_record_moments()
    drifts = block_drifts(rates, deltas)
    fast = cross_interaction(drifts, workload.weights)
    slow = cross_interaction_slow(drifts, workload.weights)
    assert abs(fast - slow) < 1e-12
    rng = np.random.default_rng(20260911)
    for _ in range(20):
        vs = [rng.normal(size=3) for _ in range(5)]
        w = rng.uniform(0.5, 2.0, size=3)
        assert abs(cross_interaction(vs, w) - cross_interaction_slow(vs, w)) < 1e-10


def test_interaction_can_be_negative():
    # 两块漂移反向时 C 为负，此时步长只受可行上限约束
    vs = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    w = np.ones(2)
    C = cross_interaction(vs, w)
    assert C == -1.0
    assert abs(cross_interaction_slow(vs, w) - C) < 1e-15
    assert analytic_step(0.5, 1.0, C) == 2.0


def test_three_row_rate_example():
    # 三行速率例子，D=3/2 C=3 h_feas=1，h*=1/4，E[L']=L-hD+h^2C=5/16
    h = analytic_step(1.0, 1.5, 3.0)
    assert h == 0.25
    L = 0.5
    assert abs((L - h * 1.5 + h * h * 3.0) - 5.0 / 16.0) < 1e-15


def test_tiny_gain_step_is_two():
    # 微小增益反例，b=1/2 C=0 h_feas=2，非保持概率 h*r=1，变化概率恒为 1
    gains = [np.array([0.0, 1e-3])]
    refs = [np.array([0.9, 0.1])]
    deltas = [np.array([[0.0], [1.0]])]
    tilt = calibrate_beta(gains, refs, old_loss=1.0, alpha=0.5)
    rates = rates_from_probabilities(tilt.probabilities)
    b = block_leave_rates(rates)
    assert abs(b[0] - 0.5) < 1e-9
    C = cross_interaction(block_drifts(rates, deltas), np.ones(1))
    assert C == 0.0
    h = analytic_step(float(b.max()), tilt.direction_gain, C)
    assert abs(h - 2.0) < 1e-8
    assert abs(h * b[0] - 1.0) < 1e-8


def test_row_budget_anchor():
    # B_max=0.25 变体，h=min(h*, B_max/H)，期望改行数正好 0.25
    workload, tilt, rates, deltas, outcomes = four_record_moments()
    H = expected_changed_rows_at_unit_step(rates, outcomes)
    b = block_leave_rates(rates)
    # 全部单行块时 c=1，H 与 sum b 一致
    assert abs(H - b.sum()) < 1e-12
    C = cross_interaction(block_drifts(rates, deltas), workload.weights)
    h_star = analytic_step(float(b.max()), tilt.direction_gain, C)
    h = apply_row_budget(h_star, H, 0.25)
    assert abs(h - 0.2643622352) < 1e-9
    assert abs(h * H - 0.25) < 1e-12


def test_multi_row_block_changed_count():
    # 双行块里只改一行的后继 c=1，两行都改 c=2
    rates = [np.array([0.0, 0.1, 0.2])]
    outcomes = [(((0, 3)), (0, 2), (1, 2))]
    H = expected_changed_rows_at_unit_step(rates, outcomes)
    assert abs(H - (0.1 * 1 + 0.2 * 2)) < 1e-15


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        analytic_step(1.0, 1.0, 0.0, damping=0.0)
    with pytest.raises(FloatingPointError):
        analytic_step(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        apply_row_budget(1.0, 1.0, 0.0)
